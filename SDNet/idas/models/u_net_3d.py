"""
Implementation of an U-Net to deal with 3D inputs (5D tensors in). The implementation is just an extension to the 3D
case of the original 2D U-Net from the 2015 paper by Ronneberger et al., at:
  https://arxiv.org/pdf/1505.04597.pdf
"""

import tensorflow as tf
from tensorflow.layers import conv3d, max_pooling3d, batch_normalization, conv3d_transpose


def _encode_brick(incoming, nb_filters, is_training, scope):
    k_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)  # He init.
    b_init = tf.zeros_initializer()
    with tf.variable_scope(scope):
        conv1 = conv3d(incoming, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                       kernel_initializer=k_init, bias_initializer=b_init)
        conv1_act = tf.nn.relu(conv1)
        conv1_bn = batch_normalization(conv1_act, training=is_training)

        conv2 = conv3d(conv1_bn, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                       kernel_initializer=k_init, bias_initializer=b_init)
        conv2_act = tf.nn.relu(conv2)
        conv2_bn = batch_normalization(conv2_act, training=is_training)

        pool = max_pooling3d(conv2_bn, pool_size=2, strides=2, padding='same')  # TODO: stride 2?

        concat_layer_out = conv2_bn
    return pool, concat_layer_out


def _decode_brick(incoming, concat_layer_in, nb_filters, is_training, scope):
    k_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)  # He init.
    b_init = tf.zeros_initializer()
    with tf.variable_scope(scope):
        conv1t = conv3d_transpose(incoming, filters=nb_filters, kernel_size=2, strides=2, padding='same',
                                  kernel_initializer=k_init, bias_initializer=b_init)
        conv1t_act = tf.nn.relu(conv1t)
        conv1t_bn = batch_normalization(conv1t_act, training=is_training)

        concat = tf.concat([conv1t_bn, concat_layer_in], axis=-1)

        conv2 = conv3d(concat, filters=nb_filters, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init,
                       bias_initializer=b_init)
        conv2_act = tf.nn.relu(conv2)
        conv2_bn = batch_normalization(conv2_act, training=is_training)

    return conv2_bn


def _core_brick(incoming, nb_filters, is_training, scope):
    k_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)  # He init.
    b_init = tf.zeros_initializer()
    with tf.variable_scope(scope):
        core = conv3d(incoming, filters=nb_filters, kernel_size=3, strides=1, padding='same', kernel_initializer=k_init,
                      bias_initializer=b_init)
        core_act = tf.nn.relu(core)
        core_bn = batch_normalization(core_act, training=is_training)
    return core_bn


def _ouput_layer(incoming, nb_classes, scope):
    k_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)  # He init.
    b_init = tf.zeros_initializer()

    with tf.variable_scope(scope):
        output = conv3d(incoming, filters=nb_classes, kernel_size=1, strides=1, padding='same',
                        kernel_initializer=k_init, bias_initializer=b_init)
        # activation = linear
    return output


class UNet3d:

    def __init__(self, input_data):
        super().__init__()
        self.input_data = input_data
        self.prediction = None
        self.is_training = True

    def u_net_3d(self):

        with tf.variable_scope('U-Net_3D'):
            # Encoder
            en_brick_0, concat_0 = _encode_brick(self.input_data, 64, is_training=self.is_training, scope='encode_brick_1')
            en_brick_1, concat_1 = _encode_brick(en_brick_0, 128, is_training=self.is_training, scope='encode_brick_2')
            en_brick_2, concat_2 = _encode_brick(en_brick_1, 256, is_training=self.is_training, scope='encode_brick_3')
            en_brick_3, concat_3 = _encode_brick(en_brick_2, 512, is_training=self.is_training, scope='encode_brick_4')

            # Core
            core = _core_brick(en_brick_3, 1024, is_training=self.is_training, scope='core')

            # Decoder
            dec_brick_1 = _decode_brick(core, concat_layer_in=concat_3, nb_filters=512, is_training=self.is_training,
                                        scope='decode_brick_1')
            dec_brick_2 = _decode_brick(dec_brick_1, concat_layer_in=concat_2, nb_filters=256, is_training=self.is_training,
                                        scope='decode_brick_2')
            dec_brick_3 = _decode_brick(dec_brick_2, concat_layer_in=concat_1, nb_filters=128, is_training=self.is_training,
                                        scope='decode_brick_3')
            dec_brick_4 = _decode_brick(dec_brick_3, concat_layer_in=concat_0, nb_filters=64, is_training=self.is_training,
                                        scope='decode_brick_4')

            # Output
            output = _ouput_layer(dec_brick_4, nb_classes=2, scope='ouput_layer')

            # self.prediction = output
            return output

