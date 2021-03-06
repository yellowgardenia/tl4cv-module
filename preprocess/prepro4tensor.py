from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

def im_crop_central(img, **kwargs):
    try:
        size = kwargs['size']
    except:
        size = kwargs['default_size']

    img = tf.image.resize_image_with_crop_or_pad(img, size, size)
    return img

def im_random_crop(img, **kwargs):
    try:
        size = kwargs['size']
    except:
        size = kwargs['default_size']
    
    _, _, nz = img.get_shape().as_list()
    img = tf.random_crop(img, [size, size, nz])
    return img

def im_random_flip_h(img, **kwargs):
    img = tf.image.random_flip_left_right(img)
    return img

def im_random_flip_v(img, **kwargs):
    img = tf.image.random_flip_up_down(img)
    return img

def im_random_brightness(img, **kwargs):
    default_max_delta = 63.
    try:
        max_delta = kwargs['max_delta']
    except:
        max_delta = default_max_delta
    
    with_seg = kwargs['with_seg']
    if with_seg:
        nx, ny, nz = img.get_shape().as_list()
        img3 = img[:, :, 0:nz-1]
        seg = img[:, :, nz-1]
        seg = tf.reshape(seg, [nx, ny, 1])
        img3 = tf.image.random_brightness(img3, max_delta=max_delta)
        img = tf.concat([img3, seg], axis=2)
    else:
        img = tf.image.random_brightness(img, max_delta=max_delta)
    return img

def im_random_contrast(img, **kwargs):
    default_lower = 0.2
    default_upper = 1.8
    try:
        lower = kwargs['lower']
    except:
        lower = default_lower
        
    try:
        upper = kwargs['upper']
    except:
        upper = default_upper

    with_seg = kwargs['with_seg']
    if with_seg:
        nx, ny, nz = img.get_shape().as_list()
        img3 = img[:, :, 0:nz-1]
        seg = img[:, :, nz-1]
        seg = tf.reshape(seg, [nx, ny, 1])
        img3 = tf.image.random_contrast(img3, lower=lower, upper=upper)
        img = tf.concat([img3, seg], axis=2)
    else:
        img = tf.image.random_contrast(img, lower=lower, upper=upper)
    return img

def im_standardization(img, **kwargs):
    with_seg = kwargs['with_seg']
    if with_seg:
        nx, ny, nz = img.get_shape().as_list()
        img3 = img[:, :, 0:nz-1]
        seg = img[:, :, nz-1]
        seg = tf.reshape(seg, [nx, ny, 1])
        img3 = tf.image.per_image_standardization(img3)
        img = tf.concat([img3, seg], axis=2)
    else:
        img = tf.image.per_image_standardization(img)
    return img

def im_reduce_mean(img, **kwargs):
    default_mean = 91.0
    try:
        mean = kwargs['mean']
    except:
        mean = default_mean

    with_seg = kwargs['with_seg']
    if with_seg:
        nx, ny, nz = img.get_shape().as_list()
        img3 = img[:, :, 0:nz-1]
        seg = img[:, :, nz-1]
        seg = tf.reshape(seg, [nx, ny, 1])
        img3 = tf.cast(img3, tf.float32) - mean
        img = tf.concat([img3, seg], axis=2)
    else:
        img = tf.cast(img, tf.float32) - mean
    return img
    
def im_resize(img, **kwargs):
    try:
        size = int(kwargs['size'])
    except:
        size = kwargs['default_size']

    img = tf.image.resize_images(img, [size, size])
    return img

prepro_map = {
    'crop_central': im_crop_central,
    'random_crop': im_random_crop,
    'random_flip_h': im_random_flip_h,
    'random_flip_v': im_random_flip_v,
    'random_brightness': im_random_brightness,
    'random_contrast': im_random_contrast,
    'standardization': im_standardization,
    'reduce_mean': im_reduce_mean,
    'resize': im_resize,
}

def get_prepro_fn(name):
    if name not in prepro_map:
        raise ValueError('Name of preprocess function is unknow %s' % name)
    func = prepro_map[name]
    @functools.wraps(func)
    def prepro_fn(images, **kwargs):
        return func(images, **kwargs)
    
    return prepro_fn

def get_dict(p_str, p_dict):
    p_ = p_str.split(', ')
    name = p_[0]
    p_.remove(name)
    if p_ is not None:
        for s in p_:
            key = s.split('=')[0]
            value = float(s.split('=')[1])
            p_dict[key] = value
    return name, p_dict
    
class preprocess(object):
    def __init__(self, file_path):
        self.path = file_path
        f = open(self.path, 'r')
        l = f.readlines()
        f.close()
        train_b = l.index('train\n')
        test_b = l.index('test\n')
        self.train = []
        self.test = []
        for i in range(train_b+1, test_b):
            self.train += [l[i].split('\n')[0]]
        for i in range(test_b+1, len(l)):
            self.test += [l[i].split('\n')[0]]
            
    def im_train_preprocess(self, img, im_output_size, with_seg=False):
        for p in self.train:
            train_dict = {'default_size': im_output_size, 'with_seg': with_seg}
            name, train_dict = get_dict(p, train_dict)
            #print(train_dict)
            prepro_fn = get_prepro_fn(name)
            img = prepro_fn(img, **train_dict)
        return img

    def im_test_preprocess(self, img, im_output_size, with_seg=False):
        for p in self.train:
            test_dict = {'default_size': im_output_size, 'with_seg': with_seg}
            name, test_dict = get_dict(p, test_dict)
            prepro_fn = get_prepro_fn(name)
            img = prepro_fn(img, **test_dict)
        return img