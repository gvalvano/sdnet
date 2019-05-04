"""
Module for data_augmentation.

Author: Gabriele Valvano

_____
Example:

def _data_augmentation_ops(x_train, y_train):
    x_train, y_train = data_aug.add_rotation(x_batch=x_train, y_batch=y_train, max_angle=360., cval=0)
    x_train, y_train = data_aug.add_warping(x_batch=x_train, y_batch=y_train, scale_limits=(1, 1),
                                            shear_limits=(-0.10, 0.10),
                                            translation_limits=(20, 20),
                                            xcval=0,
                                            ycval=0)

    x_train, y_train = data_aug.add_left_righ_flipping(x_batch=x_train, y_batch=y_train)
    x_train, y_train = data_aug.add_up_down_flipping(x_batch=x_train, y_batch=y_train)
    # X_train, Y_train = data_aug.add_random_blurring(x_batch=X_train, y_batch=Y_train, sigma_max=0.2)

    # double linear --> quadratic bias
    x_train = data_aug.add_linear_bias(x_batch=np.copy(x_train), cval=[0.5, 1.5])
    x_train = data_aug.add_linear_bias(x_batch=np.copy(x_train), cval=[0.5, 1.5])

    x_train = data_aug.add_random_noise(x_batch=x_train, cval=0.015*x_train.max(), negative_values=False)  # cval=0.02
    return x_train, y_train
"""

import scipy.ndimage
import numpy as np


def _random_linear_bias(x_batch, cval):
    """
    Add data augmentation to the samples. Add linear bias in a random direction with values in range cval.
    Assuming image in grayscale.
    :param x_batch: input samples
    :param cval: values used to scale image values along a random direction.
    :return: transformed data
    """
    n_samples = len(x_batch)
    x = np.linspace(cval[0], cval[1], x_batch[0].shape[0])
    mesh_2d = np.repeat(x[:, np.newaxis], x_batch[0].shape[1], axis=1)
    mesh = np.repeat(mesh_2d[:, :, np.newaxis], x_batch[0].shape[2], axis=2)
    for i in range(n_samples):
        # assuming image in grayscale
        angle = np.random.uniform(0, 360)
        rotated_mesh = scipy.ndimage.interpolation.rotate(mesh, angle, reshape=False, mode='nearest')
        x_batch[i] *= rotated_mesh.reshape(x_batch[i].shape)
    return x_batch


def add_linear_bias(x_batch, cval):
    """ Add random linear bias to the data. """
    return _random_linear_bias(x_batch=x_batch, cval=cval)
