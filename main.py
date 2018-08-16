

import sys, os
# add directories in src/ to path
sys.path.insert(0, '/home/zty/lyx/SpectralNet/src/')

# import run_net and get_data
from src.applications.spectralnet import run_net
from src.core.data import get_data
import time
import loaddata
import numpy as np
import operator  
from itertools import chain


# define hyperparameters
params = {
    'dset': 'breakfast',
    'val_set_fraction': 0.1,
    'siam_batch_size': 128,
    'n_clusters': 48,
    'affinity': 'siamese',
    'n_nbrs': 3,
    'scale_nbr': 2,
    'siam_k': 2,
    'siam_ne': 100,
    'spec_ne': 100,
    'siam_lr': 1e-3,
    'spec_lr': 1e-3,
    'siam_patience': 10,
    'spec_patience': 10,
    'siam_drop': 0.1,
    'spec_drop': 0.1,
    'batch_size': 128,
    'siam_reg': None,
    'spec_reg': None,
    'siam_n': None,
    'siamese_tot_pairs': 600000,
    'arch': [
        {'type': 'relu', 'size': 1024},
        {'type': 'relu', 'size': 512},
        {'type': 'relu', 'size': 256},
        {'type': 'relu', 'size': 48},
        ],
    'use_approx': False,
    }
    
# load dataset
data_dir = '/home/zty/lyx/SpectralNet/data/'
breakfast_data = loaddata.breakfast_dataset(data_dir)

splits=['s1', 's2', 's3', 's4']
for split in splits:
            print('Start:', split)
            split_start = time.time()

            # Load data for each split
            x_train_tmp, y_train_tmp = breakfast_data.get_split(split, "train")
            x_test_tmp, y_test_tmp = breakfast_data.get_split(split, "test")

            y_train=np.array(list(chain(*y_train_tmp)))
            x_train=np.array(list(chain(*x_train_tmp)))
            y_test=np.array(list(chain(*y_test_tmp)))
            x_test=np.array(list(chain(*x_test_tmp)))

            print('-------data profile-------',end='')
            print('x_train=',x_train.shape,'y_train=',y_train.shape)

            new_dataset_data = (np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test))

            # preprocess dataset
            data = get_data(params, new_dataset_data)

            # run spectral net
            x_spectralnet, y_spectralnet = run_net(data, params)