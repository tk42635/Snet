#load data from breakfast dataset
from __future__ import division


import numpy as np
import os
import scipy.io as sio
import glob
from tqdm import tqdm

# ----------- initialize -------------
class breakfast_dataset():
    def __init__(self, dir):
        print('loading data from:', dir)
        all = ["P%02d" % i for i in range(3, 56)]
        splits = [all[:13], all[13:26], all[26:39], all[39:52]]

        features = glob.glob(os.path.join(dir, 'breakfast_data', 's1', '*', '*.txt'))
        labels = glob.glob(os.path.join(dir, 'segmentation_coarse', '*', '*.txt'))
        features.sort()
        labels.sort()

        self.data = [[] for i in range(8)]  # x,y for 4 splits
        actions = []

        for f, l in tqdm(zip(features, labels)):
            _x = np.loadtxt(f)
            _y_raw = open(l).readlines()
            _y = np.repeat(0, int(_y_raw[-1].split()[0].split('-')[1]))
            for act in _y_raw:
                tm, lb = act.split()
                if lb not in actions:
                    actions.append(lb)
                _y[(int(tm.split('-')[0]) - 1):int(tm.split('-')[1])] = actions.index(lb)
            n = min(len(_x), len(_y))  # make sure the same length

            for i in range(4):
                if sum([k in f for k in splits[i]]) > 0:  # in the split
                    self.data[2 * i].append(_x[5:n, 1:])  # exclude index and first 5 frames (all-zero feature)
                    self.data[2 * i + 1].append(_y[5:n])
                    break

    def get_split(self, split, type):
            split = int(split[1]) - 1
            if type == 'train':
                x = []
                y = []
                for i in range(4):
                    if i != split:
                        x += self.data[2 * i]
                        y += self.data[2 * i + 1]
                return x, y
            elif type == 'test':
                return self.data[2 * split], self.data[2 * split + 1]

