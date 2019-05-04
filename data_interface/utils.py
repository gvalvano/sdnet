"""
utility functions for the data interfaces
"""
import tensorflow as tf
from math import pi


def one_hot_encode(y, nb_classes):
    # TODO: could it be preferable embed this method into the computation graph rather than in the input pipeline?
    y_shape = list(y.shape)
    y_shape.append(nb_classes)
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            res = sess.run(tf.one_hot(indices=y, depth=nb_classes))
    return res.reshape(y_shape)


def standardize(incoming):
    """ Linearly scales image to have zero mean and unit variance.
    :param incoming: incoming tensor
    """
    return tf.image.per_image_standardization(incoming)


def rescale(incoming, w, h):
    """
    Rescales to a common size all the data using a bi-linear interpolation.
    :param incoming: incoming tensor
    :param w: output width
    :param h: output height
    :return: scaled output
    """
    return tf.image.resize_images(incoming, size=[w, h], method=tf.image.ResizeMethod.BILINEAR)


def data_augmentation_ops(x_train):
    """
    Data augmentation pipeline for the training samples.
    :param x_train: training samples batch
    """
    angles = tf.random_uniform((1, 1), minval=-pi/8, maxval=pi/8)
    x_train = tf.contrib.image.rotate(x_train, angles[0], interpolation='BILINEAR')

    translations = tf.random_uniform((1, 2), minval=-3, maxval=3)
    x_train = tf.contrib.image.translate(x_train, translations, interpolation='BILINEAR')

    # image distortions:
    # x_train = tf.image.random_brightness(x_train, max_delta=0.05)
    # x_train = tf.image.random_contrast(x_train, lower=0.9, upper=1.1)

    # add noise as regularizer
    std = 0.05  # assuming images standardized as pre-processing
    #           # TODO: we can compute std of the current image using tf.nn.moments
    #           # TODO: (i.e. mean, var = tf.nn.moments(x, axes=[1]))
    noise = tf.random_normal(shape=tf.shape(x_train), mean=0.0, stddev=std)
    x_train = x_train + noise

    return x_train

