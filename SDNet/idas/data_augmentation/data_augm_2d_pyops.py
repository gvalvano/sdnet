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

import random
import scipy.ndimage
import numpy as np
from skimage.transform import AffineTransform, warp


def _random_flip_left_right(x_batch, y_batch=None):
    """
    Add data augmentation to the samples. Random flipping is applied to the data, axis: left-right.
    :param x_batch: input samples
    :param y_batch: output samples (optional)
    :return: transformed data
    """
    for i in range(len(x_batch)):
        if bool(random.getrandbits(1)):
            x_batch[i] = np.fliplr(x_batch[i])
            if y_batch is not None:
                y_batch[i] = np.fliplr(y_batch[i])
    if y_batch is not None:
        return x_batch, y_batch
    return x_batch


def _random_flip_up_down(x_batch, y_batch=None):
    """
    Add data augmentation to the samples. Random flipping is applied to the data, axis: up-down.
    :param x_batch: input samples
    :param y_batch: output samples (optional)
    :return: transformed data
    """
    for i in range(len(x_batch)):
        if bool(random.getrandbits(1)):
            x_batch[i] = np.flipud(x_batch[i])
            if y_batch is not None:
                y_batch[i] = np.flipud(y_batch[i])
    if y_batch is not None:
        return x_batch, y_batch
    return x_batch


def _random_blur(x_batch, sigma_max=0.2, y_batch=None):
    """
    Add data augmentation to the samples. Random gaussian blurring is applied to the data.
    :param x_batch: input samples
    :param y_batch: output samples (optional)
    :return: transformed data
    """
    for i in range(len(x_batch)):
        if bool(random.getrandbits(1)):
            # Random sigma
            sigma = random.uniform(0., sigma_max)
            x_batch[i] = scipy.ndimage.filters.gaussian_filter(x_batch[i], sigma)
            if y_batch is not None:
                y_batch[i] = scipy.ndimage.filters.gaussian_filter(y_batch[i], sigma)
    if y_batch is not None:
        return x_batch, y_batch
    return x_batch


def _apply_warping(img, tform, out_shape, cval, order, mode='constant'):
    """ Utility to apply warping. Makes sure that if input are float then are in [-1, +1] during the transformation. """
    if np.issubdtype(img.dtype, np.float):
        mmax = np.max(np.abs(img))
        if mmax != 0:
            img /= mmax

    # Apply transformation:
    img_warped = warp(img, tform.inverse, output_shape=out_shape, cval=cval, order=order, mode='edge')  # mode per il padding

    # if float, rescaling:
    if np.issubdtype(img.dtype, np.float):
        img_warped *= mmax
    return img_warped


def _random_warping(x_batch, scale_limits, shear_limits, translation_limits, xcval=0, y_batch=None, ycval=0):
    """
    Add data augmentation to the samples. Random warping is applyed by translating, shearing and scaling input data.
    If y_batch is not None, the same trasformation is applyed also to the output samples.
    If the input data x_batch has more then one channel the same warping is applyed to every channel.
    :param x_batch: input samples, assumed to have shape [n_samples, N, M, n_channels]
    :param scale_limits: tuple or list, contains min and max values of scaling factors along 1 and 2 axis of x_batch
    :param shear_limits: tuple or list, contains min and max values of shearing factors along 1 and 2 axis of x_batch 
    :param translation_limits: tuple or list, contains min and max number of pixels for translation along 1 and 2 axis of x_batch  
    :param xcval: value to use for the padding on the input contained into x_batch
    :param y_batch: output samples (optional), assumed to have shape [n_samples, N, M, n_labels] 
    :param ycval: value to use for the padding on the mask: pixels are set to 1 in the relative channel
    :return: randomly warped input
    """
    n_samples, N, M, n_channels = x_batch.shape
    if y_batch is not None:
        n_labels = y_batch.shape[-1]

    for i in range(n_samples):
        # random sampling of all warping factors:
        scale_x = np.random.uniform(low=scale_limits[0], high=scale_limits[1])
        scale_y = np.random.uniform(low=scale_limits[0], high=scale_limits[1])
        shear_factor = np.random.uniform(low=shear_limits[0], high=shear_limits[1])
        trsl_x = np.random.uniform(low=-translation_limits[0], high=translation_limits[0])
        trsl_y = np.random.uniform(low=-translation_limits[1], high=translation_limits[1])

        # Affine Transform definition
        tform = AffineTransform(scale=(scale_x, scale_y), rotation=0, shear=shear_factor, translation=(trsl_x, trsl_y))

        for c in range(n_channels):
            # applying warping on each channel
            _data_slice = x_batch[i][:, :, c]
            xcval = np.min(_data_slice)
            x_batch[i][:, :, c] = _apply_warping(_data_slice, tform, out_shape=(N, M), cval=xcval, order=1)

        if y_batch is not None:
            for l in range(n_labels):
                _data_slice = y_batch[i][:, :, l]
                if l == ycval:
                    y_batch[i][:, :, l] = _apply_warping(_data_slice, tform, out_shape=(N, M), cval=1, order=0)
                else:
                    y_batch[i][:, :, l] = _apply_warping(_data_slice, tform, out_shape=(N, M), cval=ycval, order=0)
    if not y_batch is None:
        return x_batch, y_batch
    return x_batch


def _random_rotation(x_batch, max_angle, cval=0.0, y_batch=None):
    # TODO: possibilità di mettere k su x_batch e comunque label 0 su maschera y_batch (cval = tupla)
    """
    Add data augmentation to the samples. Random rotation in range [-max_angle, max_angle]
    If y_batch is not None, the same trasformation is applyed also to the output samples.
    :param x_batch: input samples
    :param y_batch: output samples (optional)
    :param max_angle: max absolute value of matrix rotation
    :param cval: value used for points outside the boundaries of the input. Default is 0.0
    :return: transformed data
    """
    n_samples = len(x_batch)
    for i in range(n_samples):
        if bool(random.getrandbits(1)):
            # Random angle
            angle = random.uniform(-max_angle, max_angle)
            x_batch[i] = scipy.ndimage.interpolation.rotate(x_batch[i], angle, reshape=False, mode='nearest')
            if y_batch is not None:
                #y_batch[i] = scipy.ndimage.interpolation.rotate(y_batch[i], angle, reshape=False, cval=cval, mode='nearest', order=0)
                n_channels = y_batch.shape[-1]
                for ch in range(n_channels):
                    if ch == cval:  # TODO: brutto
                        y_batch[i, :, :, ch] = scipy.ndimage.interpolation.rotate(y_batch[i, :, :, ch], angle, reshape=False, cval=1, order=0, mode='nearest')
                    else:
                        y_batch[i, :, :, ch] = scipy.ndimage.interpolation.rotate(y_batch[i, :, :, ch], angle, reshape=False, cval=0, order=0, mode='nearest')
    if not y_batch is None:
        return x_batch, y_batch
    return x_batch


def _random_noise(x_batch, cval, negative_values=True):
    """
    Add data augmentation to the samples. Add random noise to the samples to improve regularization.
    Noise values are floats in the half-open interval cval*[0.0, 1.0) if negative values are contemplate, cval*(-1.0, 1.0) otherwise.
    :param x_batch: input samples
    :param cval: absolute limit value for the random noise
    :param negative_values: if True random noise is contemplate, otherwise only positive noise is considered.
    :return: transformed data
    """
    n_samples, x, y, z = x_batch.shape
    for i in range(n_samples):
        noise = np.random.random_sample((x, y, z))
        if negative_values: noise = 2*(noise - 0.5)
        x_batch[i] += cval*noise
    return x_batch


def _random_linear_bias(x_batch, cval):
    """
    Add data augmentation to the samples. Add linear bias in a random direction with values in range cval
    :param x_batch: input samples
    :param cval: values used to scale image values along a random direction.
    :return: transformed data
    """
    # TODO: np.copy(x_batch)
    n_samples = len(x_batch)
    x = np.linspace(cval[0], cval[1], x_batch[0].shape[0])
    mesh = np.repeat(x[:, np.newaxis], x_batch[0].shape[1], axis=1)
    for i in range(n_samples):
        angle = np.random.uniform(0, 360)
        # TODO: immagine assunta in scala di grigi
        x_batch[i] *= scipy.ndimage.interpolation.rotate(mesh, angle, reshape=False, mode='nearest').reshape(x_batch[i].shape)
    return x_batch


def _random_right_left_translation(x_batch, max_shift, cval=0.0, y_batch=None):
    # TODO: aggiungere possibilità di mettere k su x_batch e comunque label 0 su maschera y_batch (cval = tupla)
    """
    Add data augmentation to the samples. Random translation in range [0, max_shift], axis right-left
    If y_batch is not None, the same trasformation is applyed also to the output samples.
    :param x_batch: input samples
    :param y_batch: output samples (optional)
    :param max_shift: max absolute value of matrix translation
    :param cval: value used for points outside the boundaries of the input. Default is 0.0
    :return: transformed data
    """
    n_samples = len(x_batch)
    m = cval * np.ones(x_batch.shape)
    if not y_batch is None:
        my = cval * np.ones(x_batch.shape)
    for i in range(n_samples):
        # Random shift
        shift = random.randint(1, max_shift)
        if bool(random.getrandbits(1)):
            # left translation
            m[i][:, :-shift, :] = np.array(x_batch[i][:, shift:, :])
            m[i][:, -shift:, :] = cval
            if not y_batch is None:
                my[i][:, :-shift] = np.array(y_batch[i][:, shift:])
                my[i][:, -shift:] = cval
        else:
            # right translation
            m[i][:, shift:, :] = np.array(x_batch[i][:, :-shift, :])
            m[i][:, :shift, :] = cval
            if not y_batch is None:
                my[i][:, shift:] = np.array(y_batch[i][:, :-shift])
                my[i][:, :shift] = cval
    if not y_batch is None:
        return m, my
    return m


def _random_up_down_translation(x_batch, max_shift, cval=0.0, y_batch=None):
    # TODO: aggiungere possibilità di mettere k su x_batch e comunque label 0 su maschera y_batch (cval = tupla)
    """
    Add data augmentation to the samples. Random translation in range [0, max_shift], axis up-down
    If y_batch is not None, the same trasformation is applyed also to the output samples.
    :param x_batch: input samples
    :param y_batch: output samples (optional)
    :param max_shift: max absolute value of matrix translation
    :param cval: value used for points outside the boundaries of the input. Default is 0.0
    :return: transformed data
    """
    n_samples = len(x_batch)
    m = cval * np.ones(x_batch.shape)
    if not y_batch is None:
        my = cval * np.ones(x_batch.shape)
    for i in range(n_samples):
        # Random shift
        shift = random.randint(1, max_shift)
        if bool(random.getrandbits(1)):
            # left translation
            m[i][:-shift, :, :] = np.array(x_batch[i][shift:, :, :])
            m[i][-shift:, :, :] = cval
            if not y_batch is None:
                my[i][:-shift, :] = y_batch[i][shift:, :]
                my[i][-shift:, :] = cval
        else:
            # right translation
            m[i][shift:, :, :] = np.array(x_batch[i][:-shift, :, :])
            m[i][:shift, :, :] = cval
            if not y_batch is None:
                my[i][shift:, :] = y_batch[i][:-shift, :]
                my[i][:shift, :] = cval
    if y_batch is not None:
        return m, my
    return m


def add_warping(x_batch, y_batch=None, scale_limits=(1, 1), shear_limits=(0, 0), translation_limits=(0, 0), xcval=0, ycval=0):
    """ Add random warping to the data, via random scaling and shearing. """
    return _random_warping(x_batch=x_batch, y_batch=y_batch, scale_limits=scale_limits, shear_limits=shear_limits,
                           translation_limits=translation_limits, xcval=xcval, ycval=ycval)


def add_rotation(x_batch, max_angle, cval, y_batch=None):
    """ Add random rotation to the data. """
    return _random_rotation(x_batch=x_batch, max_angle=max_angle, cval=cval, y_batch=y_batch)


def add_linear_bias(x_batch, cval):
    """ Add random linear bias to the data. """
    return _random_linear_bias(x_batch=x_batch, cval=cval)


def add_random_noise(x_batch, cval, negative_values=True):
    """ Add random noise to the data. """
    return _random_noise(x_batch=x_batch, cval=cval, negative_values=negative_values)


def add_right_left_translation(x_batch, max_shift, cval, y_batch=None):
    """ Add random translation to the data, axis: left-right. """
    return _random_right_left_translation(x_batch=x_batch, max_shift=max_shift, cval=cval, y_batch=y_batch)


def add_up_down_translation(x_batch, max_shift, cval, y_batch=None):
    """ Add random translation to the data, axis: up-down. """
    return _random_up_down_translation(x_batch=x_batch, max_shift=max_shift, cval=cval, y_batch=y_batch)


def add_up_down_flipping(x_batch, y_batch=None):
    """ Add random flipping to the data, axis: up-down. """
    return _random_flip_up_down(x_batch=x_batch, y_batch=y_batch)


def add_left_righ_flipping(x_batch, y_batch=None):
    """ Add random flipping to the data, axis: left-right. """
    return _random_flip_left_right(x_batch=x_batch, y_batch=y_batch)


def add_random_blurring(x_batch, sigma_max, y_batch=None):
    """ Add random gaussian blurring to the data. """
    return _random_blur(x_batch=x_batch, sigma_max=sigma_max, y_batch=y_batch)
