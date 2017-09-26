from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def read_lines(path):
    ftype = path.split('.')[-1]
    if ftype == 'txt':
        with open(path, 'r') as f:
            flist = f.readlines()
            flist = [row.split(' ') for row in flist]
            return flist
    elif ftype == 'csv':
        with open(path, 'r') as f:
            reader = csv.reader(f)
            flist = [row for row in reader]
            return flist[1:]
    else:
        with open(path, 'r') as f:
            flist = f.readlines()
            return flist

def list2npArray(flist):
    farray = np.array(flist)
    x = farray[:, 0]
    y = np.array(farray[:, 1], dtype=np.int)
    return x, y
     
def auto_batch_max(N, max_batch=-1):
    if max_batch < 1:
        max_batch = np.int(np.sqrt(N))
        
    for i in range(max_batch, 1, -1):
        if N % i == 0:
            return i
    return 1

"""
## for training

import tensorlayer as tl

for batch in tl.iterate.minibatches(inputs=X, targets=y, batch_size=batch_size, shuffle=True):
    xbatch, ybatch = batch

## for testing

for batch in tl.iterate.minibatches(inputs=X, targets=y, batch_size=auto_batch_max(X.shape[0])):
    xbatch, ybatch = batch

"""