import argparse
import os, sys 
import time
import tabulate
import yaml 

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
sys.path.append('../../')
from swag import data, models, utils, losses
from swag.posteriors import SWAG

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--config_file', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--n_seeds', type=int, default=None, required=True, help='training directory (default: None)')
parser_args = parser.parse_args()

args =None 

with open(parser_args.config_file, 'r') as fl:
    try:
        args = yaml.safe_load(fl)
    except yaml.YAMLError as exc:
        print(exc)
print(args)

args['inference'] = 'low_rank_gaussian'
args['subspace'] = 'covariance'
args['no_cov_mat'] = False
args['batch_size'] = 64
args['num_workers'] = 4
args['savedir'] = os.path.join(args['dir'], "multiswag")

args['device'] = None
if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')


torch.backends.cudnn.benchmark = True
model_cfg = getattr(models, args['model'])

print('Loading dataset %s from %s' % (args['dataset'], args['data_path']))
loaders, num_classes = data.loaders(
    args['dataset'],
    args['data_path'],
    args['batch_size'],
    args['num_workers'],
    model_cfg.transform_train,
    model_cfg.transform_test
    )


print('Preparing model')
model = model_cfg.base(*model_cfg.args,
                       **model_cfg.kwargs)
model.to(args['device'])
print("Model has {} parameters".format(sum([p.numel() for p in model.parameters()])))


swag_model = SWAG(model_cfg.base,
                args['subspace'], 
                *model_cfg.args,
                **model_cfg.kwargs)
swag_model.to(args['device'])


columns = ['swag', 'sample', 'te_loss', 'te_acc', 'ens_loss', 'ens_acc']

n_ensembled = 0.
multiswag_probs = None
all_probs = []

swag_ckpts = [os.path.join(args['dir'],"swag_{}".format(seed), "swag-{}.pt".format(args['epochs'])) for seed in range(parser_args.n_seeds)]
print(swag_ckpts)
for ckpt_i, ckpt in enumerate(swag_ckpts):
    print("Checkpoint {}".format(ckpt))
    checkpoint = torch.load(ckpt)
    swag_model.subspace.rank = torch.tensor(0)
    swag_model.load_state_dict(checkpoint['state_dict'])
    all_probs_sample = []

    for sample in range(args['swag_samples']):
        all_params_split = []
        offset = 0
        for p in swag_model.base_model.parameters():
            all_params_split.append(p.data.mean())
        idx = 1
        print("{:.8f}".format(all_params_split[0]))

        swag_model.sample(.5)
        res = utils.test(loaders['test'], swag_model)
        probs = res['predictions']

        all_probs_sample.append(probs.copy())
        if multiswag_probs is None:
            multiswag_probs = probs.copy()
        else:
            #TODO: rewrite in a numerically stable way
            multiswag_probs +=  (probs - multiswag_probs)/ (n_ensembled + 1)
        n_ensembled += 1

    all_probs.append(all_probs_sample.copy())

print('Preparing directory %s' % args['savedir'])
os.makedirs(args['savedir'], exist_ok=True)
with open(os.path.join(args['savedir'], 'eval_command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

np.savez(os.path.join(args['savedir'], "multiswag_output.npz"),
         predictions=multiswag_probs, all_predictions =all_probs)
