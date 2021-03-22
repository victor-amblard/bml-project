import argparse
import os, sys
import time
import tabulate

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
sys.path.append("/home/victor/Data/Mines-ParisTech/M2/mva/S2/understandingbdl/")
from swag import data, models, utils, losses
from swag.posteriors import SWAG
import yaml 

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--config_file', type=str, default=None, required=True, help='training directory (default: None)')
parser_args = parser.parse_args()

args =None 

with open(parser_args.config_file, 'r') as fl:
    try:
        args = yaml.safe_load(fl)
    except yaml.YAMLError as exc:
        print(exc)
print(args)
args['subspace'] = 'covariance'
args['seed'] = 1
args['device'] = None
args['batch_size'] = 64
args['num_workers'] = 5
args['save_freq'] = 50
args['eval_freq'] = 50
args['swag_c_epochs'] = 1

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

print('Preparing directory %s' % args['dir'])
os.makedirs(args['dir'], exist_ok=True)
with open(os.path.join(args['dir'], 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

print('Using model %s' % args['model'])
model_cfg = getattr(models, args['model'])
print(model_cfg.kwargs)
print('Loading dataset %s from %s' % (args['dataset'], args['data_path']))
loaders, num_classes = data.loaders(
    args['dataset'],
    args['data_path'],
    args['batch_size'],
    args['num_workers'],
    model_cfg.transform_train,
    model_cfg.transform_test)

print('Preparing model')

print(*model_cfg.args)
model = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
model.to(args['device'])

args['no_cov_mat'] = False

print('SWAG training')
swag_model = SWAG(model_cfg.base, 
                subspace_type=args['subspace'],
                *model_cfg.args, **model_cfg.kwargs)
swag_model.to(args['device'])


def schedule(epoch):
    t = (epoch) / (args['swag_start'])
    lr_ratio = args['swag_lr'] / args['lr_init']
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args['lr_init'] * factor


criterion = losses.mse_loss
regularizer = lambda model: sum([p.norm()**2 for p in model.parameters()]) / (2 * args['prior_var'] * len(loaders['train'].dataset))

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args['lr_init']
)

start_epoch = 0

columns = ['ep', 'lr', 'tr_loss','time']
columns = columns[:-2] + ['swa_tr_loss'] + columns[-2:]
swag_res = {'loss': None}

utils.save_checkpoint(
    args['dir'],
    start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict()
)

sgd_ens_preds = None
sgd_targets = None
n_ensembled = 0.

for epoch in range(start_epoch, args['epochs']):
    time_ep = time.time()

    lr = schedule(epoch)
    utils.adjust_learning_rate(optimizer, lr)
    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, regularizer=regularizer)
    
    if epoch == 0 or epoch % args['eval_freq'] == args['eval_freq'] - 1 or epoch == args['epochs'] - 1:
        test_res = utils.eval(loaders['train'], model, criterion)
    else:
        test_res = {'loss': None}

    if  (epoch + 1) > args['swag_start'] and (epoch + 1 - args['swag_start']) % args['swag_c_epochs'] == 0:
        sgd_preds, sgd_targets = utils.predictions(loaders["train"], model)
        sgd_res = utils.predict(loaders["train"], model)
        sgd_preds = sgd_res["predictions"]
        sgd_targets = sgd_res["targets"]
        # print("updating sgd_ens")
        if sgd_ens_preds is None:
            sgd_ens_preds = sgd_preds.copy()
        else:
            #TODO: rewrite in a numerically stable way
            sgd_ens_preds +=  (sgd_preds - sgd_ens_preds)/ (n_ensembled + 1)

        sgd_ens_acc = (np.argmax(sgd_ens_preds, axis=1) == sgd_targets).mean()
        n_ensembled += 1
        swag_model.collect_model(model)
        if epoch == 0 or epoch % args['eval_freq'] == args['eval_freq'] - 1 or epoch == args['epochs'] - 1:
            swag_model.set_swa()
            utils.bn_update(loaders['train'], swag_model)
            swag_res = utils.eval(loaders['train'], swag_model, criterion)
        else:
            swag_res = {'loss': None}

    if (epoch + 1) % args['save_freq'] == 0:
        print("save")
        utils.save_checkpoint(
            args['dir'],
            epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        utils.save_checkpoint(
                args['dir'],
                epoch + 1,
                name='swag',
                state_dict=swag_model.state_dict(),
        )

    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'],  time_ep]
    values = values[:-1] + [swag_res['loss']] + values[-1:]
    
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if (epoch - start_epoch) % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

if args['epochs'] % args['save_freq'] != 0:
    utils.save_checkpoint(
        args['dir'],
        args['epochs'],
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )
    if epochs > args['swag_start']:
        utils.save_checkpoint(
            args['dir'],
            epochs,
            name='swag',
            state_dict=swag_model.state_dict(),
        )

np.savez(os.path.join(args['dir'], "sgd_ens_preds.npz"), predictions=sgd_ens_preds, targets=sgd_targets)