import tensorflow as tf


def film_layer(incoming, gamma, beta, name='film'):
    """
    FiLM layer
    :param incoming: incoming tensor
    :param gamma: incoming gamma
    :param beta: incoming beta
    :param name: (string) name scope
    :return:
    """
    with tf.name_scope(name):
        # get shape of incoming tensors:
        in_shape = tf.shape(incoming)
        gamma_shape = tf.shape(gamma)
        beta_shape = tf.shape(beta)

        # tile gamma and beta:
        gamma = tf.tile(tf.reshape(gamma, (gamma_shape[0], 1, 1, gamma_shape[-1])),
                        (1, in_shape[1], in_shape[2], 1))
        beta = tf.tile(tf.reshape(beta, (beta_shape[0], 1, 1, beta_shape[-1])),
                       (1, in_shape[1], in_shape[2], 1))

        # compute output:
        output = incoming * gamma + beta

    return output
