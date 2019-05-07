import tensorflow as tf
from idas.metrics.tf_metrics import dice_coe, jaccard_coe


def l2_weights_regularization_loss(exclude_bias=False):
    """ l2 regularization loss on all variables """
    vars = tf.trainable_variables()
    if exclude_bias:
        loss = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])
    else:
        loss = tf.add_n([tf.nn.l2_loss(v) for v in vars])
    return loss


def dice_loss(output, target, axis=(1, 2, 3), smooth=1e-12):
    """ Returns Soft Sørensen–Dice loss """
    return 1.0 - dice_coe(output, target, axis=axis, smooth=smooth)


def jaccard_loss(output, target, axis=(1, 2, 3), smooth=1e-12):
    """ Returns Soft Jaccard (also known as Intersection over Union) loss.
    Refs. https://arxiv.org/pdf/1608.01471.pdf"""
    return 1.0 - jaccard_coe(output, target, axis=axis, smooth=smooth)


def iou_loss(output, target, axis=(1, 2, 3), smooth=1e-12):
    """ Wrapper to Soft Jaccard (also known as Intersection over Union) loss.
    Refs. https://arxiv.org/pdf/1608.01471.pdf
    """
    return 1.0 - jaccard_coe(output, target, axis=axis, smooth=smooth, _name='iou_coe')


def weighted_softmax_cross_entropy(y_pred, y_true, num_classes, eps=1e-12):
    """
    Define weighted cross-entropy function for classification tasks.
    :param y_pred: tensor [None, width, height, n_classes]
    :param y_true: tensor [None, width, height, n_classes]
    :param eps: (float) small value to avoid division by zero
    :param num_classes: (int) number of classes
    :return:
    """

    n = [tf.reduce_sum(tf.cast(y_true[..., c], tf.float32)) for c in range(num_classes)]
    n_tot = tf.reduce_sum(n)

    weights = [n_tot / (n[c] + eps) for c in range(num_classes)]

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))
    softmax = tf.nn.softmax(y_pred)

    w_cross_entropy = -tf.reduce_sum(tf.multiply(y_true * tf.log(softmax + eps), weights), reduction_indices=[1])
    loss = tf.reduce_mean(w_cross_entropy, name='weighted_softmax_cross_entropy')
    return loss


def segmentation_weighted_cross_entropy(y_pred, y_true, n_voxels, epsilon=1e-12):
    """
    Compute weighted cross-entropy on binary segmentation masks.
    :param y_pred: predicted output (i.e. with shape [None, N, M, num_classes]
    :param y_true: true output mask (i.e. with shape [None, N, M, num_classes]
    :param n_voxels: number of voxels (i.e. N * M)
    :param epsilon: constant to prevent division from 0
    :return: mean cross-entropy loss
    """

    a = tf.divide(1., tf.reduce_sum(tf.cast(y_true, tf.float32)))
    b = tf.divide(1., (n_voxels - a))
    weights = [b, a]  # [1/(number of zeros), 1/(number of ones)]

    num_classes = y_pred.get_shape().as_list()[-1]  # class on the last index
    assert (num_classes is not None)
    assert len(weights) == num_classes

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))
    softmax = tf.nn.softmax(y_pred) + epsilon

    w_cross_entropy = -tf.reduce_sum(tf.multiply(y_true * tf.log(softmax), weights), reduction_indices=[1])

    return tf.reduce_mean(w_cross_entropy, name='weighted_cross_entropy')


def vae_loss(z_mean, z_logstd, y_true, y_pred, n_outputs):
    """
    Computes classical loss for variational auto-encoders, as:
      Loss = Reconstruction_loss + KL_divergence_loss
           = MSE - ELBO
    :param z_mean: encoded z_mean vector
    :param z_logstd: encoded z_logstd vector
    :param n_outputs: (i.e. in mnist we have 28x28 = 784)
    :return:
    """
    x_true = tf.reshape(y_true, shape=(-1, n_outputs))
    x_reconstructed = tf.reshape(y_pred, shape=(-1, n_outputs))

    # _______
    # Reconstruction loss:
    with tf.variable_scope('Reconstruction_loss'):
        generator_loss = tf.losses.mean_squared_error(x_reconstructed, x_true)

    # _______
    # KL Divergence loss:
    with tf.variable_scope('KL_divergence_loss'):
        kl_div_loss = 1.0 + z_logstd - tf.square(z_mean) - tf.exp(z_logstd)
        kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)

    total_loss = tf.reduce_mean(generator_loss + kl_div_loss)

    return total_loss, generator_loss, kl_div_loss


def gradient_loss(y_true, y_pred):
    """
    Computes gradient loss as the mean square error on the sobel filtered image
    # TODO: update implementation to make it work for any input dimension
    """
    pred_grad = tf.reduce_sum(tf.image.sobel_edges(y_pred[:, :, :, 0]), -1)
    input_grad = tf.reduce_sum(tf.image.sobel_edges(y_true[:, :, :, 0]), -1)
    
    x_reconstructed_grad = tf.reshape(pred_grad, shape=(-1, 128*128))
    x_true_grad = tf.reshape(input_grad, shape=(-1, 128*128))

    gradient_loss = tf.reduce_mean(tf.squared_difference(x_reconstructed_grad, x_true_grad), 1)
    return gradient_loss
