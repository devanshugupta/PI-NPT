import pandas as pd
import numpy as np
import torch
import pickle
import wandb
from npt.configs import build_parser
import json
from npt.utils.model_init_utils import (
    init_model_opt_scaler)
metadata_path = '/Users/devu/PycharmProjects/PI-NPT/data/ode/ssl__True/np_seed=42__n_cv_splits=1__exp_num_runs=1/dataset__metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
print(metadata)
parser = build_parser()
args = parser.parse_args()
wandb_args = dict(
        project=args.project,
        entity=args.entity,
        dir=args.wandb_dir,
        reinit=True,
        name=args.exp_name,
        group=args.exp_group)
wandb_run = wandb.init(**wandb_args)
args.cv_index = 0
wandb.config.update(args, allow_val_change=True)
c = wandb.config
model, optimizer, scaler = init_model_opt_scaler(
            c, metadata=metadata,
            device='cpu')
best_model_path = '/Users/devu/PycharmProjects/PI-NPT/data/ode/ssl__True/np_seed=42__n_cv_splits=1__exp_num_runs=1/model_checkpoints/model_540.pt'
# Load from checkpoint, populate state dicts
checkpoint = torch.load(best_model_path, map_location='cpu')
# Strict setting -- allows us to load saved attention maps
# when we wish to visualize them
model.load_state_dict(checkpoint['model_state_dict'],
                      strict=(not c.viz_att_maps))
path = '/Users/devu/PycharmProjects/PI-NPT/data/ode/ssl__True/np_seed=42__n_cv_splits=1__exp_num_runs=1/dataset__split=0.pkl'
with open(path, 'rb') as f:
    data_dict = pickle.load(file=f)
masked_tensors = data_dict['masked_tensors']
extra_args = {}
#print('masked tensor (input to model) ------------', masked_tensors)
with torch.no_grad():
    output = model(masked_tensors, **extra_args)
print(output)
