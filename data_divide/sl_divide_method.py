#! /usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import os
import cv2
import csv
import random
import datetime
import xml.dom.minidom

class data_divide(object):
    def __init__(self, num_classes=5, num_parts=2):
        self._divide_list = np.zeros((num_parts, num_classes))
        self._max_list = np.zeros((num_parts, num_classes))
        self._mean = np.zeros(num_parts)
        
    def get_max_list(self, max_list):
        self._max_list = max_list
        
    def flag_add(self, save_flag):
        k, _ = self._divide_list.shape
        save_flag += 1
        return save_flag % k
    
    def count(self, save_flag, label, mean):
        k, _ = self._divide_list.shape
        for i in range(k):
            if self._divide_list[save_flag, label] < self._max_list[save_flag, label]:
                self._divide_list[save_flag, label] += 1
                self._mean[save_flag] += mean
                return save_flag
            else:
                save_flag = self.flag_add(save_flag)
                
        self._divide_list[save_flag, label] += 1
        self._mean[save_flag] += mean
        return save_flag
    
    def update_mean(self):
        self._mean /= np.sum(self._divide_list, axis=1)
        
    def array2str(self, arr):
        txt = ','.join(np.str(i) for i in arr)
        return txt
    
    def show(self):
        k, _ = self._divide_list.shape
        print('Image number for data list:')
        for i in range(k):
            print('List', i, ':', self._divide_list[i], sum(self._divide_list[i]), '', self._mean[i])
        print('Totol :\t', sum(self._max_list), sum(sum(self._max_list)))
        
    def save_info(self, path):
        #f = open(path, 'w')
        #f.close()
        #k, _ = self._divide_list.shape
        #f = open(path, 'a+')
        #for i in range(k):
        #    f.write('List_'+str(i)+'\t'+str(self._divide_list[i])+'\t'+ \
        #            str(sum(self._divide_list[i]))+'\t'+str(self._mean[i])+'\n')
        #f.write('Total\t'+str(sum(self._max_list)))
        #f.close()
        k, num_classes = self._divide_list.shape
        # create an empty file
        doc = xml.dom.minidom.Document()
        # create root
        root = doc.createElement('DivideInfomation')
        # add root attribute
        now = datetime.datetime.now()
        root.setAttribute('Date', now.strftime('%Y-%m-%d %H:%M:%S'))
        root.setAttribute('n_classes', np.str(num_classes))
        root.setAttribute('n_list', np.str(k))
        # add root to file
        doc.appendChild(root)
        
        for i in range(k):
            nodeobj = doc.createElement('List_'+np.str(i))
            root.appendChild(nodeobj)
            nodeList = doc.createElement('EachClass')
            nodeList.appendChild(doc.createTextNode(self.array2str(self._divide_list[i])))
            nodeobj.appendChild(nodeList)
            nodeSum = doc.createElement('Sum')
            nodeSum.appendChild(doc.createTextNode(np.str(sum(self._divide_list[i]))))
            nodeobj.appendChild(nodeSum)
            nodeMean = doc.createElement('Mean')
            nodeMean.appendChild(doc.createTextNode(np.str(self._mean[i])))
            nodeobj.appendChild(nodeMean)

        # write xml
        with open(path, 'w') as fp:
            doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding="utf-8")

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
    
def divide_into_two_parts(image_root, list_path, save_path, **kwargs):
    default_ratio = 0.5
    try:
        ratio = np.float(kwargs['ratio'])
    except:
        ratio = default_ratio
    
    num_classes = kwargs['num_classes']
    data_div = data_divide(num_classes=num_classes, num_parts=2)

    fheader = ['name', 'label']
    for i in range(2):
        f = open(os.path.join(save_path, 'list_'+str(i)+'.csv'), 'w')
        f.write(fheader[0]+','+fheader[1]+'\n')
        f.close()

    #f = open(list_path, 'r')
    #f_list = f.readlines()
    #f.close()
    f_list = read_lines(list_path)
    label_max = np.zeros((1, num_classes))
    for line in f_list:
        label = int(line[1])
        label_max[0, label] += 1
    max_list = np.zeros((2, num_classes))
    max_list[0][:] = label_max*ratio
    max_list[1][:] = label_max*(1-ratio)
    data_div.get_max_list(max_list)
    
    for line in f_list:
        img = cv2.imread(os.path.join(image_root, line[0]+'.tif'))
        label = int(line[1])
        mean = img.mean()

        save_flag = random.randint(0, 1)
        save_flag = data_div.count(save_flag, label, mean)

        with open(os.path.join(save_path, 'list_'+str(save_flag)+'.csv'), 'a+') as f:
            f.write(line[0]+','+np.str(np.int(line[1]))+'\n')
            
    data_div.update_mean()
    data_div.show()
    data_div.save_info(os.path.join(save_path, 'list_info.xml'))
    

def divide_into_n_parts(image_root, list_path, save_path, **kwargs):
    default_r = '1:1'
    try:
        r = kwargs['r']
    except:
        r = default_r
    
    ratio = np.array(r.split(':'), dtype=np.float)
    k = ratio.shape[0]
    num_classes = kwargs['num_classes']
    data_div = data_divide(num_classes=num_classes, num_parts=k)

    fheader = ['name', 'label']
    for i in range(k):
        f = open(os.path.join(save_path, 'list_'+str(i)+'.csv'), 'w')
        f.write(fheader[0]+','+fheader[1]+'\n')
        f.close()

    f_list = read_lines(list_path)
    label_max = np.zeros((1, num_classes))
    for line in f_list:
        label = int(line[1])
        label_max[0, label] += 1
    max_list = np.zeros((k, num_classes))
    #max_list[:] = label_max / k
    for i in range(k):
        max_list[i, :] = label_max * ratio[i] / np.sum(ratio)
        
    data_div.get_max_list(max_list)

    for line in f_list:
        img = cv2.imread(os.path.join(image_root, line[0]+'.tif'))
        label = int(line[1])
        mean = img.mean()

        save_flag = random.randint(0, k-1)
        save_flag = data_div.count(save_flag, label, mean)

        with open(os.path.join(save_path, 'list_'+str(save_flag)+'.csv'), 'a+') as f:
            f.write(line[0]+','+np.str(np.int(line[1]))+'\n')
            
    data_div.update_mean()
    data_div.show()
    data_div.save_info(os.path.join(save_path, 'list_info.xml'))
    

def divide_into_k_folds(image_root, list_path, save_path, **kwargs):
    default_k = 10
    try:
        k = np.int(kwargs['k'])
    except:
        k = default_k
    
    num_classes = kwargs['num_classes']
    data_div = data_divide(num_classes=num_classes, num_parts=k)

    fheader = ['name', 'label']
    for i in range(k):
        f = open(os.path.join(save_path, 'list_'+str(i)+'.csv'), 'w')
        f.write(fheader[0]+','+fheader[1]+'\n')
        f.close()

    f_list = read_lines(list_path)
    label_max = np.zeros((1, num_classes))
    for line in f_list:
        label = int(line[1])
        label_max[0, label] += 1
    max_list = np.zeros((k, num_classes))
    max_list[:] = label_max / k
    data_div.get_max_list(max_list)

    for line in f_list:
        img = cv2.imread(os.path.join(image_root, line[0]+'.tif'))
        label = int(line[1])
        mean = img.mean()

        save_flag = random.randint(0, k-1)
        save_flag = data_div.count(save_flag, label, mean)

        with open(os.path.join(save_path, 'list_'+str(save_flag)+'.csv'), 'a+') as f:
            f.write(line[0]+','+np.str(np.int(line[1]))+'\n')
            
    data_div.update_mean()
    data_div.show()
    data_div.save_info(os.path.join(save_path, 'list_info.xml'))
    
method_map = {
    'divide_into_two_parts': divide_into_two_parts,
    'divide_into_n_parts': divide_into_n_parts,
    'k_fold_cross_validation': divide_into_k_folds,
}

def get_method_fn(name):
    if name not in method_map:
        raise ValueError('Name of method function is unknow %s' % name)
    func = method_map[name]
    @functools.wraps(func)
    def method_fn(image_root, list_path, save_path, **kwargs):
        return func(image_root, list_path, save_path, **kwargs)
    
    return method_fn