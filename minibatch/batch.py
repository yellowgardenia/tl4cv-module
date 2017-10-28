from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import cv2
import csv

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
    
def Ximread(x, path):
    """ ori image read """
    im = cv2.imread(os.path.join(path, x[0]+'.tif'))
    w, h, c = im.shape
    k = x.shape[0]
    X = np.zeros((k, w, h, c))
    for i in range(k):
        im = cv2.imread(os.path.join(path, x[i]+'.tif'))
        X[i, :, :, :] = im
    return X

def MaskImread(x, path):
    """ mask [0, 1] image read """
    im = cv2.imread(os.path.join(path, x[0]+'.tif'), 0)
    w, h = im.shape
    k = x.shape[0]
    X = np.zeros((k, w, h, 1))
    for i in range(k):
        im = cv2.imread(os.path.join(path, x[i]+'.tif'), 0)>0
        im = np.reshape(im, [w, h, 1])
        X[i, :, :, :] = im
    return X
	
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