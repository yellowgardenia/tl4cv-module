#! /usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tfrecords import encode_tf, encode_tf_with_seg
import time

tf.app.flags.DEFINE_string('save_path', None, 'The save path of the tfrecords file. [tfrecords]')
tf.app.flags.DEFINE_string('list_path', None, 'The path of the list file. [csv|txt]')
tf.app.flags.DEFINE_string('data_path', None, 'The path of the origin image.')
tf.app.flags.DEFINE_string('seg_path', None, 'The path of the seg image.')

FLAGS = tf.app.flags.FLAGS

def main(_):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    time1 = time.time()
    print(now, 'Make tfrecords files ...')
    
    if FLAGS.seg_path is None:
        encode_tf(FLAGS.save_path, FLAGS.list_path, FLAGS.data_path)
    else:
        encode_tf_with_seg(FLAGS.save_path, FLAGS.list_path, FLAGS.data_path, FLAGS.seg_path)

    time2 = time.time()
    print('Time cost:', round(time2-time1, 3), 'sec')
    print('Done!')

if __name__ == '__main__':
    tf.app.run()