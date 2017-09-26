#! /usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sl_divide_method as dm
import time

tf.app.flags.DEFINE_string('divide_method', 'divide_into_two_parts, ratio=0.5', 'The method of dataset divide.')
tf.app.flags.DEFINE_integer('num_classes', 1, 'The number of classes. [1]')
tf.app.flags.DEFINE_string('save_path', None, 'The save path of the divided list files. [csv]')
tf.app.flags.DEFINE_string('list_path', None, 'The path of the origin list file. [txt|csv]')
tf.app.flags.DEFINE_string('data_path', None, 'The path of the origin data.')

FLAGS = tf.app.flags.FLAGS

def get_dict(p_str, p_dict):
    p_ = p_str.split(', ')
    name = p_[0]
    p_.remove(name)
    if p_ is not None:
        for s in p_:
            key = s.split('=')[0]
            value = s.split('=')[1]
            p_dict[key] = value
    return name, p_dict

def divide_dataset():
    p_dict = {'num_classes': FLAGS.num_classes}
    name, p_dict = get_dict(FLAGS.divide_method, p_dict)
    method_fn = dm.get_method_fn(name)
    method_fn(FLAGS.data_path, FLAGS.list_path, FLAGS.save_path, **p_dict)

def main(_):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    time1 = time.time()
    print(now, 'Divide data ...')
    
    divide_dataset()

    time2 = time.time()
    print('Time cost:', round(time2-time1, 3), 'sec')
    print('Done!')

if __name__ == '__main__':
    tf.app.run()