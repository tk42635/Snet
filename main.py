

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
import random
import argparse
import math
import numba
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# define hyperparameters
params = {
    'dset': 'breakfast',
    'val_set_fraction': 0.1,
    'siam_batch_size': 128,
    'n_clusters': 5,
    'affinity': 'siamese',
    'n_nbrs': 20,
    'scale_nbr': 5,
    'siam_k': 20,
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
    'siamese_tot_pairs': 6000000,
    'arch': [
        {'type': 'relu', 'size': 512},
        {'type': 'relu', 'size': 256},
        {'type': 'relu', 'size': 64},
        {'type': 'relu', 'size': 5},
        ],
    'use_approx': False,
    'generalization_metrics': True
    }


def selection(index, y_train_new, x_train_new, y_train, x_train):
    for i in index:
        y_train_new=np.row_stack((y_train_new,y_train[i]))
        x_train_new=np.row_stack((x_train_new,x_train[i]))
    x_train_new=np.delete(x_train_new,0,0)
    y_train_new=np.delete(y_train_new,0,0)
    x_train=x_train_new
    y_train=y_train_new

    return x_train,y_train

def onehot(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

   
    
# load dataset
parser = argparse.ArgumentParser()
parser.add_argument('dset', default='larger', choices = ['less', 'larger'])
parser.add_argument('affnity',  default='s', choices = ['s', 'full'])
sample_rate=float(input('sample_rate='))
args = parser.parse_args()
if args.affnity == 'full':
    params['affinity']={'full'}
if args.dset == 'larger':        
    data_dir = '/home/zty/lyx/SpectralNet/data/'        
else:
    data_dir = '/home/zty/lyx/SpectralNet/data2/'

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
            
            index=random.sample(range(x_train.shape[0]),math.floor(x_train.shape[0]*sample_rate))
            
            y_train_new = np.array([0])
            x_train_new = np.zeros((1,64))

            x_train, y_train = selection(index, y_train_new, x_train_new, y_train, x_train)
           
            y_train=onehot(y_train)
            y_test=onehot(y_test)
            
            print('-------data profile-------\n')
            print('x_train=',x_train.shape,'y_train=',y_train.shape)
          #  params['n_clusters']={int(y_train.shape[1])}
            new_dataset_data = (np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test))

            # preprocess dataset
            data = get_data(params, new_dataset_data)

            # run spectral net
            x_spectralnet, y_spectralnet = run_net(data, params)