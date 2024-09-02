# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import json
import wandb
from npt.configs import build_parser
from npt.utils.model_init_utils import (
    init_model_opt_scaler)

# PATHS
cache_path = './data/ode/ssl__True/np_seed=42__n_cv_splits=1__exp_num_runs=1/'
model_checkpoint = cache_path + 'model_checkpoints/' + os.listdir(cache_path+ 'model_checkpoints/')[0]

metadata_path = cache_path + 'dataset__metadata.json'

with open(metadata_path, 'r') as f:
    metadata =  json.load(f)

# CONFIG parameters
parser = build_parser()
args = parser.parse_args()
wandb_args = dict(
        project=args.project,
        entity=args.entity,
        dir=args.wandb_dir,
        reinit=True,
        name=args.exp_name,
        group=args.exp_group)
wandb_run = wandb.init(**wandb_args, mode='offline')
args.cv_index = 0
wandb.config.update(args, allow_val_change=True)
c = wandb.config

# MODEL
model, optimizer, scaler = init_model_opt_scaler(
            c, metadata=metadata,
            device=c.exp_device)

checkpoint = torch.load(model_checkpoint)

model.load_state_dict(checkpoint['model_state_dict'],
                      strict=True)

extra_args = {}

x = np.linspace(0, 2 * np.pi, 256)
t = np.linspace(0, 1, 100)
X, T = np.meshgrid(x, t)

X_flat = X.flatten()[:, None]
T_flat = T.flatten()[:, None]
U_flat = torch.zeros(X_flat.shape, dtype=torch.float32)
beta = torch.ones(X_flat.shape, dtype=torch.float32) * 40  # Set beta to 40
rho = torch.zeros(X_flat.shape, dtype=torch.float32)
nu = torch.zeros(X_flat.shape, dtype=torch.float32)
X_flat = torch.tensor(X_flat, dtype=torch.float32)
T_flat = torch.tensor(T_flat, dtype=torch.float32)

# OUTPUT
with torch.no_grad():
    U_flat = model([X_flat, T_flat, U_flat, beta, rho, nu], **extra_args)

U = U_flat.reshape(X.shape)

plt.figure(figsize=(8, 6))

plt.contourf(T, X, U, levels=50, cmap='viridis')  # Contour plot
plt.colorbar(label='u(x,t)')
plt.xlabel('t')
plt.ylabel('x')
plt.show()