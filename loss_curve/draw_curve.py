from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import numpy as np
import matplotlib.pyplot as plt

def cal_mean_std(data):
    """
    data: shape [time, n]
          dtype [np.array]
    """
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    return mean, std

def read_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        flist = [row for row in reader]
        return flist[0], np.array(flist[1:], dtype=np.float)

def merge_col(data, col):
    cols = [d[:, col] for d in data]
    return np.transpose(np.array(cols), [1, 0])

"""
Examples:
    
    f_num = 3
    head = read_csv('loss_curve/loss_%d_test.csv' % 0)[0]
    data = [read_csv('loss_curve/loss_%d_test.csv' % i)[1] for i in range(f_num)]
    t_num, head_num = data[0].shape
    merge_data = [merge_col(data, i) for i in range(head_num)]
    m, s = cal_mean_std(merge_data[3])
    plt.plot(merge_data[0][:, 0], m)
    for i in range(t_num):
        plt.errorbar(merge_data[0][i, 0], m[i], yerr=s[i], fmt='o')
    plt.savefig('test.png', dpi=300, facecolor='w', edgecolor='w', pad_inches=0.4, bbox_inches='tight')

"""