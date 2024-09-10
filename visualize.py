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
print(model_checkpoint)
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

#checkpoint = torch.load(model_checkpoint)

#model.load_state_dict(checkpoint['model_state_dict'],strict=True)

extra_args = {}

def convection_diffusion_discrete_solution(u0 = '1+sin(x)', nu = 1, beta = 40, source=0, xgrid=256, nt=100):
    N = xgrid
    h = 2 * np.pi / N
    x = np.arange(0, 2*np.pi, h) # not inclusive of the last point
    t = np.linspace(0, 1, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)


    # call u0 this way so array is (n, ), so each row of u should also be (n, )
    u0 = lambda x: 1 + np.sin(x)
    u0 = u0(x)
    G = (np.copy(u0)*0)+source # G is the same size as u0

    IKX_pos =1j * np.arange(0, N/2+1, 1)
    IKX_neg = 1j * np.arange(-N/2+1, 0, 1)
    IKX = np.concatenate((IKX_pos, IKX_neg))
    IKX2 = IKX * IKX

    uhat0 = np.fft.fft(u0)
    nu_factor = np.exp(nu * IKX2 * T - beta * IKX * T)
    A = uhat0 - np.fft.fft(G)*0 # at t=0, second term goes away
    uhat = A*nu_factor + np.fft.fft(G)*T # for constant, fft(p) dt = fft(p)*T
    u = np.real(np.fft.ifft(uhat))

    u_vals = u.flatten()
    return u_vals, u

'''
x = np.linspace(0, 2*np.pi, 256, endpoint=False).reshape(-1, 1) # not inclusive
t = np.linspace(0, 1, 100).reshape(-1, 1)
X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data
t_noinitial = t[1:]
x_noboundary = x[1:]
X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))
'''
# sample collocation points only from the interior (where the PDE is enforced)
#X_f_train, idx_test, idx_val = sample_random(X_star_noinitial_noboundary, args.N_f)

u_vals, u_v = convection_diffusion_discrete_solution('1+sin(x)', 0.0, 0.0, 0, 256, 100)
h = 2 * np.pi / 256
x = np.arange(0, 2*np.pi, h) # not inclusive of the last point
t = np.linspace(0, 1, 100).reshape(-1, 1)
X, T = np.meshgrid(x, t)

X_flat = X.flatten()[:,None]
T_flat = T.flatten()[:,None]
U_flat = torch.zeros(X_flat.shape, dtype=torch.float32)
beta = torch.ones(X_flat.shape, dtype=torch.float32) # Set beta to 40
rho = torch.zeros(X_flat.shape, dtype=torch.float32)
nu = torch.zeros(X_flat.shape, dtype=torch.float32)
X_flat = torch.tensor(X_flat, dtype=torch.float32)
T_flat = torch.tensor(T_flat, dtype=torch.float32)

masked_tensor = [X_flat, T_flat, U_flat, beta, rho, nu]
for i in range(len(masked_tensor)):
    encoded_col = masked_tensor[i]
    if i == 2:
        masked_tensor[i] = torch.hstack([encoded_col, torch.ones((len(encoded_col),1))])
    else:
        masked_tensor[i] = torch.hstack([encoded_col, torch.zeros((len(encoded_col),1))])


# OUTPUT
'''with torch.no_grad():
    output = model(masked_tensor, **extra_args)'''
output = model(masked_tensor, **extra_args)

U_flat = output[2].flatten()
U = U_flat.reshape(X.shape)

plt.figure(figsize=(8, 6))

plt.contourf(T, X, U, levels=50, cmap='viridis')  # Contour plot
plt.colorbar(label='u(x,t)')
plt.xlabel('t')
plt.ylabel('x')
plt.show()