# IMPORTS
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import seaborn as sns
import pandas as pd
import pickle
import json
import os
import torch

import wandb
from npt.configs import build_parser
import json
from npt.utils.model_init_utils import (
    init_model_opt_scaler)
from npt.utils.encode_utils import get_torch_dtype, get_torch_tensor_type

# PATHS
cache_path = './data/ode/ssl__True/np_seed=42__n_cv_splits=1__exp_num_runs=1/'
model_checkpoint = cache_path + 'model_checkpoints/' + os.listdir(cache_path+ 'model_checkpoints/')[0]
dataset_path = cache_path + 'dataset__split=0.pkl'
metadata_path = cache_path + 'dataset__metadata.json'

with open(metadata_path, 'r') as f:
    metadata =  json.load(f)
with open(dataset_path, 'rb') as f:
    data_dict = pickle.load(file=f)

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

# DATA TABLES and Tensors
scalers = data_dict['scalers']
data_table = data_dict['data_table']
fixed_test_index = data_dict['fixed_test_set_index']

ground_truth_x, ground_truth_t, ground_truth_u = data_table[0].flatten()[fixed_test_index:], data_table[1].flatten()[fixed_test_index:], data_table[2].flatten()[fixed_test_index:]
ground_truth_b = data_table[3].flatten()[fixed_test_index:]

missing_matrix = data_dict['missing_matrix']

data_arrs =  []
for col in data_table:
    data_arrs.append(torch.tensor(col))

for col, data_arr in enumerate(data_arrs):
    # Get boolean 'mask' selection mask for this column.
    mask_col = missing_matrix[:, col]

    # If there are no masks in this column, continue.
    if mask_col.sum() == 0:
        continue

    # Zero out indices corresponding to
    # bert masking and bert random assignment.
    data_arr[mask_col, :] = 0

    # If there is no bert randomization,
    # all mask entries should be given a '1' mask token,
    data_arr[mask_col, -1] = 1
    # and we are done with masking for this column
    continue
data_dtype = get_torch_tensor_type(c.data_dtype)

masked_tensors = [
        masked_arr.type(data_dtype) for masked_arr in data_arrs]
if c.data_set_on_cuda:
    masked_tensors = [
        masked_arr.type(data_dtype).to(device=c.exp_device)
        for masked_arr in data_arrs]

# MODEL
'''
model, optimizer, scaler = init_model_opt_scaler(
            c, metadata=metadata,
            device=c.exp_device)


checkpoint = torch.load(model_checkpoint)
        # Strict setting -- allows us to load saved attention maps
        # when we wish to visualize them
model.load_state_dict(checkpoint['model_state_dict'],
                      strict=True)
'''

extra_args = {}

# OUTPUT

'''with torch.no_grad():
    output = model(masked_tensors, **extra_args)

output_x, output_t, output_u = output[0].flatten()[fixed_test_index:], output[1].flatten()[fixed_test_index:], output[2].flatten()[fixed_test_index:]
'''
path = './dataset'
test_data = pd.read_csv(f'{path}/{c.pde_type}/test/test_{c.target_coeff_1}_{c.pde_type}.csv').drop(['beta', 'rho', 'nu'], axis=1)

ground_truth_x, ground_truth_t, ground_truth_u = test_data['x_data'].to_numpy(), test_data['t_data'].to_numpy(), test_data['u_data'].to_numpy()

# Step 1: Create a uniform mesh grid
x = ground_truth_x
t = ground_truth_t
u = ground_truth_u

x_d = np.linspace(0, max(x), 50)
t_d = np.linspace(0, 1, 50)

#X, T = np.meshgrid(x_d, t_d)
X, T = np.meshgrid(x, t)

U = griddata((x, t), u, (X, T), method = 'linear')
print(np.sum(np.isnan(U))) # print indexes of nan values in the U grid --> np.argwhere()

# Step 3: Plot the results
plt.figure(figsize=(8, 6))

#plt.imshow(U, extent=(x.min(), x.max(), t.min(), t.max()), origin='lower', aspect = 'auto')
plt.contourf(T, X, U, levels=50, cmap='viridis')  # Contour plot
plt.colorbar(label='u(x,t)')
plt.xlabel('t')
plt.ylabel('x')
plt.show()