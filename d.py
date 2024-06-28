import pandas as pd
import numpy as np
import torch
import pickle
file_train = 'npt/datasets/train_u_0_convection.csv'
file_test = 'npt/datasets/test_0_convection.csv'

train = pd.read_csv(file_train, header=0)
test = pd.read_csv(file_test, header=0)

data_table = pd.concat([train,test]).drop(['beta','nu','rho'], axis = 1).to_numpy()
N = data_table.shape[0]
D = data_table.shape[1]

missing_matrix = np.ones((N, D))
missing_matrix = torch.tensor(missing_matrix.astype(dtype=np.bool_))

cat_features = []
num_features = list(range(D))
dataset_path = 'data/ode/ssl__True/np_seed=42__n_cv_splits=1__exp_num_runs=1/dataset__split=0.pkl'
with open(dataset_path, 'rb') as f:
    data_dict = pickle.load(file=f)
print(data_dict)
#print(data_table, N, D, num_features, missing_matrix.shape)
