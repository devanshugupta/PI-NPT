from pathlib import Path

import numpy as np
import pandas as pd
import torch


from npt.datasets.base import BaseDataset
from npt.utils.data_loading_utils import download
import random

class OrdinaryDifferentialEquationDataset(BaseDataset):
    def __init__(self, c):
        super(OrdinaryDifferentialEquationDataset, self).__init__(
            fixed_test_set_index=None)
        self.c = c
        self.fixed_test_set_index = None

    def load_and_preprocess_ode_dataset(self, c):
        """Class imbalance is [357, 212]."""
        path = c.dataset_path

        ###################### Dataset #######################
        train_data_f = pd.read_csv(f'{path}/{c.pde_type}/train/train_f_{c.target_coeff_1}_{c.pde_type}.csv')
        train_data_u = pd.read_csv(f'{path}/{c.pde_type}/train/train_u_{c.target_coeff_1}_{c.pde_type}.csv')
        train_data_bd = pd.read_csv(f'{path}/{c.pde_type}/train/train_boundary_{c.target_coeff_1}_{c.pde_type}.csv')
        test_data = pd.read_csv(f'{path}/{c.pde_type}/test/test_{c.target_coeff_1}_{c.pde_type}.csv')

        for i in range(c.start_coeff_1, c.end_coeff_1):
            f_sample = pd.read_csv(f'{path}/{c.pde_type}/train/train_f_{i + 1}_{c.pde_type}.csv')
            u_sample = pd.read_csv(f'{path}/{c.pde_type}/train/train_u_{i + 1}_{c.pde_type}.csv')
            bd_sample = pd.read_csv(f'{path}/{c.pde_type}/train/train_boundary_{i + 1}_{c.pde_type}.csv')
            test_sample = pd.read_csv(f'{path}/{c.pde_type}/test/test_{i + 1}_{c.pde_type}.csv')

            train_data_f = pd.concat([train_data_f, f_sample], ignore_index=True)
            train_data_u = pd.concat([train_data_u, u_sample], ignore_index=True)
            train_data_bd = pd.concat([train_data_bd, bd_sample], ignore_index=True)
            test_data = pd.concat([test_data, test_sample], ignore_index=True)
        '''
        # Create separate dataframes for lb and ub data with same coefficients
        data_ub = train_data_bd[['x_data_ub', 't_data_ub', 'beta', 'nu', 'rho']].copy()
        data_ub.columns = ['x_data', 't_data', 'beta', 'nu', 'rho']

        data_lb = train_data_bd[['x_data_lb', 't_data_lb', 'beta', 'nu', 'rho']].copy()
        data_lb.columns = ['x_data', 't_data', 'beta', 'nu', 'rho']

        # There is no u_data for boundary condition and to get u_data observing the initial conditions,
        # we might use dritchlet equation 

        # Concat both dataframes
        train_data_bd = pd.concat([data_lb, data_ub], ignore_index=True)
        # u_data for boundary is None values for now
        '''
        train_data_f['beta'] = train_data_f['beta'].apply(lambda x: 1.0)

        # Combine train and test datasets and remove columns
        data_table = pd.concat([train_data_f,train_data_u,test_data], ignore_index=True)

        # Get number of rows for each dataset
        len_train_f,len_train_u,len_train_bd,len_test = train_data_f.shape[0],train_data_u.shape[0],train_data_bd.shape[0],test_data.shape[0]
        print(len_train_f,len_train_u,len_train_bd, len_test)
        test_index = data_table.shape[0] - len_test
        self.fixed_test_set_index = test_index
        # Convert data table to numpy
        data_table = data_table.to_numpy()
        N = data_table.shape[0]
        D = data_table.shape[1]

        indexes_train_u = [i for i in range(len_train_f, len_train_u+len_train_f)]
        k = int(0.1*len_train_u)
        k_indexes_train_u = random.choices(indexes_train_u, k=k)

        print(f'ODE Dataset has {N} rows')
        # Create missing matrix for combined data
        missing_matrix = np.zeros((N, D), dtype=bool)

        # Create the masks
        missing_matrix[test_index: , 2] = True
        missing_matrix[:len_train_f, 2] = True
        missing_matrix[k_indexes_train_u, 2] = True

        # Prepare feature indices
        cat_features = []
        num_features = list(range(D))

        return data_table, N, D, cat_features, num_features, missing_matrix

    '''
    def load_and_preprocess_ode_dataset(self, c):
        """Class imbalance is [357, 212]."""
        path = Path(c.data_path) / c.data_set

        file_train = 'npt/datasets/train_u_0_convection.csv'
        file_test = 'npt/datasets/test_0_convection.csv'

        # Read dataset
        train = pd.read_csv(file_train, header=0)
        test = pd.read_csv(file_test, header=0)

        data_table = pd.concat([train, test]).drop(['beta', 'nu', 'rho'], axis=1).to_numpy()
        print(data_table)
        N = data_table.shape[0]
        D = data_table.shape[1]

        if c.exp_smoke_test:
            print('Running smoke test -- building simple ode dataset.')
            dm = data_table[data_table[:, 0] == 'M'][:8, :5]
            db = data_table[data_table[:, 0] == 'B'][:8, :5]
            data_table = np.concatenate([dm, db], 0)
            N = data_table.shape[0]
            D = data_table.shape[1]

            # Speculate some spurious missing features
            missing_matrix = np.zeros((N, D))
            missing_matrix[0, 1] = 1
            missing_matrix[2, 2] = 1
            missing_matrix = missing_matrix.astype(dtype=np.bool_)
        else:
            missing_matrix = np.zeros((N, D))
            missing_matrix = missing_matrix.astype(dtype=np.bool_)

        cat_features = []
        num_features = list(range(D))
        return data_table, N, D, cat_features, num_features, missing_matrix

    '''
    '''def get_data_dict(self, force_disable_auroc=None):
        if not self.is_data_loaded:
            self.load()

        self.auroc_setting = self.use_auroc(force_disable_auroc)
        return self.__dict__'''

    def load(self):
        (self.data_table, self.N, self.D, self.cat_features, self.num_features,
            self.missing_matrix) = self.load_and_preprocess_ode_dataset(
            self.c)

        # For breast cancer, target index is the first column
        self.num_target_cols = [2]
        self.cat_target_cols = []

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['wdbc.data']

        # overwrite missing
        if (p := self.c.exp_artificial_missing) > 0:
            self.missing_matrix = self.make_missing(p)
            # this is not strictly necessary with our code, but safeguards
            # against bugs
            # TODO: maybe replace with np.nan
            self.data_table[self.missing_matrix] = 0






class ODEDebugClassificationDataset(BaseDataset):
    """For debugging row interactions. Add two columns for index tracking."""
    def __init__(self, c):
        super(OrdinaryDifferentialEquationDataset, self).__init__(
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        raise
        # need to augment table and features and and and
        # (to contain the index rows!! can already write index rows as long
        # as permutation is random!)

        (self.data_table, self.N, self.D, self.cat_features, self.num_features,
            self.missing_matrix) = load_and_preprocess_ode_dataset(
            self.c)

        # For breast cancer, target index is the first column
        self.num_target_cols = [2]
        self.cat_target_cols = []

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['wdbc.data']

        # overwrite missing
        if (p := self.c.exp_artificial_missing) > 0:
            self.missing_matrix = self.make_missing(p)
            # this is not strictly necessary with our code, but safeguards
            # against bugs
            # TODO: maybe replace with np.nan
            self.data_table[self.missing_matrix] = 0
