from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Layer, list_remove_repeat

class ResizeBilinearLayer(Layer):
    """
    The :class:`ResizeBilinearLayer` class is bilinear resize of a neural network.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    shape : a tf.shape [dims=2] or None
        Size of resize op output.
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
        self,
        layer = None,
        shape = None,
        name ='resize_bilinear_layer',
    ):
        Layer.__init__(self, name=name)
        
        self.inputs = layer.outputs
        if shape == None:
            shape = tf.shape(self.inputs)[1:3,]
            
        self.outputs = tf.image.resize_bilinear(self.inputs, shape)
        print("  [TL] ResizeBilinearLayer  %s: %s" % (self.name, self.outputs.get_shape()))
        
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)