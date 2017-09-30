#! /usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2
import os
import numpy as np
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
        
def encode_tf(savep, listp, datap):
    flist = read_lines(listp)
    writer = tf.python_io.TFRecordWriter(savep)
    for line in flist:
        img = cv2.imread(os.path.join(datap, line[0]+'.tif'))
        img_raw = img.tobytes()
        index = np.int(line[1])
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())
    writer.close()
    
def encode_tf_with_seg(savep, listp, datap, segp):
    flist = read_lines(listp)
    writer = tf.python_io.TFRecordWriter(savep)
    for line in flist:
        img = cv2.imread(os.path.join(datap, line[0]+'.tif'))
        seg = cv2.imread(os.path.join(segp, line[0]+'.tif'), 0)
        
        w, h, _ = img.shape
        seg = seg.reshape(w, h, 1)
        img = np.append(img, seg, axis=2)
        
        img_raw = img.tobytes()
        index = np.int(line[1])
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())
    writer.close()
    
def decode_tf(filename, imshape):
    # generate a queue with a given file name
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)     # return the file and the name of file
    features = tf.parse_single_example(serialized_example,  # see parse_single_sequence_example for sequence example
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    w, h, c = imshape
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [w, h, c])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    return img, label

def decode_tf_with_seg(filename, imshape):
    # generate a queue with a given file name
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)     # return the file and the name of file
    features = tf.parse_single_example(serialized_example,  # see parse_single_sequence_example for sequence example
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    w, h, c = imshape
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [w, h, c+1])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    return img, label