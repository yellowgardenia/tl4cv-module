#! /usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl

"""

tensorboard : boolean
    if True summary data will be stored to the log/ direcory for visualization with tensorboard.
    See also detailed tensorboard_X settings for specific configurations of features. (default False)
    Also runs tl.layers.initialize_global_variables(sess) internally in fit() to setup the summary nodes, see Note:

tensorboard_epoch_freq : int
    how many epochs between storing tensorboard checkpoint for visualization to log/ directory (default 5)

tensorboard_weight_histograms : boolean
    if True updates tensorboard data in the logs/ directory for visulaization
    of the weight histograms every tensorboard_epoch_freq epoch (default True)

tensorboard_graph_vis : boolean
    if True stores the graph in the tensorboard summaries saved to log/ (default True)
    
"""


if(tensorboard):
    print("Setting up tensorboard ...")
    #Set up tensorboard summaries and saver
    tl.files.exists_or_mkdir('log/')

    #Only write summaries for more recent TensorFlow versions
    if hasattr(tf, 'summary') and hasattr(tf.summary, 'FileWriter'):
        if tensorboard_graph_vis:
            train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        else:
            train_writer = tf.summary.FileWriter('logs/train')

    #Set up summary nodes
    if(tensorboard_weight_histograms):
        for param in network.all_params:
            if hasattr(tf, 'summary') and hasattr(tf.summary, 'histogram'):
                print('Param name ', param.name)
                tf.summary.histogram(param.name, param)

    if hasattr(tf, 'summary') and hasattr(tf.summary, 'histogram'):
        tf.summary.scalar('cost', cost)

    merged = tf.summary.merge_all()

    #Initalize all variables and summaries
    tl.layers.initialize_global_variables(sess)
    print("Finished! use $tensorboard --logdir=logs/ to start server")

tensorboard_train_index = 0
for epoch in range(n_epoch):
    if tensorboard and hasattr(tf, 'summary'):
        if epoch+1 == 1 or (epoch+1) % tensorboard_epoch_freq == 0:
            for X_train_a, y_train_a in iterate.minibatches(
                                        X_train, y_train, batch_size, shuffle=True):
                dp_dict = dict_to_one( network.all_drop )    # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                result = sess.run(merged, feed_dict=feed_dict)
                train_writer.add_summary(result, tensorboard_train_index)
                tensorboard_train_index += 1