#! /usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pydensecrf.densecrf as dcrf

def dense_crf(probs, img=None, n_iters=10,
              sxy_gaussian=(1, 1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.
       
    Args:
      probs: class probabilities per pixel. [softmax float32 (bs, w, h, nc)]
      img: if given, the pairwise bilateral potential on raw RGB values will be computed. [uint8 (bs, w, h, c)]
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).

    Returns:
      Refined predictions after MAP inference.
    """
    bs, w, h, nc = probs.shape
    
    preds = np.zeros((bs, w, h, nc))
    for i in range(bs):
        prob = probs[i].transpose(2, 0, 1).copy(order='C') # Need a contiguous array.
        
        d = dcrf.DenseCRF2D(h, w, nc)          # Define DenseCRF model.
        U = -np.log(prob, dtype=np.float32)    # Unary potential.
        U = U.reshape((nc, -1))                # Needs to be flat.
        d.setUnaryEnergy(U)
        
        d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                              kernel=kernel_gaussian, normalization=normalisation_gaussian)
        
        if img is not None:
            assert(img.shape[1:3] == (w, h)), "The image height and width must coincide with dimensions of the logits."
            img = np.array(img, dtype=np.uint8)
            d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                                   kernel=kernel_bilateral, normalization=normalisation_bilateral,
                                   srgb=srgb_bilateral, rgbim=img[i])
        
        Q = d.inference(n_iters)
        pred = np.array(Q, dtype=np.float32).reshape((nc, w, h)).transpose(1, 2, 0)
        
        preds[i, :, :, :] = pred
    return preds

"""

Use py func with tf:
    preds = tf.py_func(dense_crf, [tf.nn.softmax(probs), img], tf.float32)

If num_classes == 1:???
    probs = tf.concat([probs, blank_class], axis=3)
    preds = func...(probs...)
    preds = preds[0]
    
If for some reason you want to run the inference loop manually, you can do so:
    Q, tmp1, tmp2 = d.startInference()
    for i in range(5):
        print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
        d.stepInference(Q, tmp1, tmp2)
        
"""