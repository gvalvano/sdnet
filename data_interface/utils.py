"""
utility functions for the data interfaces
"""
import tensorflow as tf


def one_hot_encode(y, nb_classes):
    # N.B. it would be preferable to embed this method into the computation graph, if needed
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


def rescale(incoming, w, h, method=tf.image.ResizeMethod.BILINEAR):
    """
    Rescales to a common size all the data using a bi-linear interpolation.
    :param incoming: incoming tensor
    :param w: output width
    :param h: output height
    :param method: interpolation method
    :return: scaled output
    """
    return tf.image.resize_images(incoming, size=[w, h], method=method)
