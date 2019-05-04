import tensorflow as tf
from math import pi


def data_augmentation_ops(x_train, y_train):
    """ Data augmentation pipeline (to be applied on training samples)
    """
    # shape = x_train.shape  # == y_train.shape
    # x_train = tf.reshape(x_train, shape=[-1, 28, 28, 1])
    # y_train = tf.reshape(y_train, shape=[-1, 28, 28, 1])

    angles = tf.random_uniform((1, 1), minval=-pi / 6, maxval=pi / 6)
    x_train = tf.contrib.image.rotate(x_train, angles[0], interpolation='BILINEAR')
    y_train = tf.contrib.image.rotate(y_train, angles[0], interpolation='BILINEAR')

    translations = tf.random_uniform((1, 2), minval=-3, maxval=3)
    x_train = tf.contrib.image.translate(x_train, translations, interpolation='BILINEAR')
    y_train = tf.contrib.image.translate(y_train, translations, interpolation='BILINEAR')

    # add noise only to inpout to avoid AE to simply learn copying (it has to learn meaningful representation of the
    # input data)
    std = 1  # mnist dataset std value = 0.3081, but here we made image standardization as pre-processing
    noise = tf.random_normal(shape=tf.shape(x_train), mean=0.0, stddev=0.1 * std)
    x_train = x_train + noise

    # x_train = tf.reshape(x_train, shape=shape)
    # y_train = tf.reshape(y_train, shape=shape)

    return x_train, y_train