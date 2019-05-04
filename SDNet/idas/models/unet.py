import tensorflow as tf
from tensorflow import layers

# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class UNet(object):

    def __init__(self, incoming, n_out, is_training, name='U-Net_2D'):
        """
        Class for UNet architecture. This is a 2D version (hence the vanilla UNet), which means it only employs
        bi-dimensional convolution and strides. This implementation also uses batch normalization after each conv layer.
        :param incoming: (tensor) incoming tensor
        :param n_out: (int) number of channels for the network output. For instance, to predict a binary mask use
                        n_out=2 (one-hot encoding); to predict a grayscale image use n_out=1
        :param is_training: (tf.placeholder(dtype=tf.bool) or bool) variable to define training or test mode; it is
                        needed for the behaviour of dropout, batch normalization, ecc. (which behave differently
                        at train and test time)

        - - - - - - - - - - - - - - - -
        Notice that:
          - this implementation works for incoming tensors with shape [None, N, M, K], where N and M must be divisible
            by 16 without any rest (in fact, there are 4 pooling layers with kernels 2x2 --> input reduced to:
            [None, N/16, M/16, K'])
          - the output of the network has activation linear
        - - - - - - - - - - - - - - - -

        Example of usage:

            # build the entire unet model:
            unet = UNet(incoming, n_out, is_training).build()

            # build the unet with access to the internal code:
            unet = UNet(incoming, n_out, is_training)
            encoder = unet.build_encoder()
            code = unet.build_code(encoder)
            decoder = unet.build_decoder(code)
            output = unet.build_output(decoder)

        """
        self.incoming = incoming
        self.n_out = n_out
        self.is_training = is_training
        self.name = name

    def build(self):
        with tf.variable_scope(self.name):
            encoder = self.build_encoder()
            code = self.build_code(encoder)
            decoder = self.build_decoder(code)
            output = self.build_output(decoder)
        return output

    def build_encoder(self):
        with tf.variable_scope('Encoder'):
            en_brick_0, concat_0 = self._encode_brick(self.incoming, 64, self.is_training, scope='encode_brick_0')
            en_brick_1, concat_1 = self._encode_brick(en_brick_0, 128, self.is_training, scope='encode_brick_1')
            en_brick_2, concat_2 = self._encode_brick(en_brick_1, 256, self.is_training, scope='encode_brick_2')
            en_brick_3, concat_3 = self._encode_brick(en_brick_2, 512, self.is_training, scope='encode_brick_3')

        return en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2, en_brick_3, concat_3

    def build_code(self, encoder):
        en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2, en_brick_3, concat_3 = encoder

        with tf.variable_scope('Code'):
            code = self._code_brick(en_brick_3, 1024, self.is_training, scope='code')

        return en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2, en_brick_3, concat_3, code

    def build_decoder(self, code):
        en_brick_0, concat_0, en_brick_1, concat_1, en_brick_2, concat_2, en_brick_3, concat_3, code = code

        with tf.variable_scope('Decoder'):
            dec_brick_0 = self._decode_brick(code, concat_3, 512, self.is_training, scope='decode_brick_0')
            dec_brick_1 = self._decode_brick(dec_brick_0, concat_2, 256, self.is_training, scope='decode_brick_1')
            dec_brick_2 = self._decode_brick(dec_brick_1, concat_1, 128, self.is_training, scope='decode_brick_2')
            dec_brick_3 = self._decode_brick(dec_brick_2, concat_0, 64, self.is_training, scope='decode_brick_3')

        return dec_brick_3

    def build_output(self, decoder):
        # output linear
        return self._output_layer(decoder, n_channels_out=self.n_out, scope='output')

    @staticmethod
    def _encode_brick(incoming, nb_filters, is_training, scope, trainable=True):
        """ Encoding brick: conv --> conv --> max pool.
        """
        with tf.variable_scope(scope):
            conv1 = layers.conv2d(incoming, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init, trainable=trainable)
            conv1_act = tf.nn.relu(conv1)
            conv1_bn = layers.batch_normalization(conv1_act, training=is_training, trainable=trainable)

            conv2 = layers.conv2d(conv1_bn, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init, trainable=trainable)
            conv2_act = tf.nn.relu(conv2)
            conv2_bn = layers.batch_normalization(conv2_act, training=is_training, trainable=trainable)

            pool = layers.max_pooling2d(conv2_bn, pool_size=2, strides=2, padding='same')

            with tf.variable_scope('concat_layer_out'):
                concat_layer_out = conv2_bn
        return pool, concat_layer_out

    @staticmethod
    def _decode_brick(incoming, concat_layer_in, nb_filters, is_training, scope):
        """ Decoding brick: deconv (up-pool) --> conv --> conv.
        """
        with tf.variable_scope(scope):
            conv1t = layers.conv2d_transpose(incoming, filters=nb_filters, kernel_size=2, strides=2, padding='same',
                                             kernel_initializer=he_init, bias_initializer=b_init)
            conv1t_act = tf.nn.relu(conv1t)
            conv1t_bn = layers.batch_normalization(conv1t_act, training=is_training)

            concat = tf.concat([conv1t_bn, concat_layer_in], axis=-1)

            conv2 = layers.conv2d(concat, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init,
                                  bias_initializer=b_init)
            conv2_act = tf.nn.relu(conv2)
            conv2_bn = layers.batch_normalization(conv2_act, training=is_training)

            conv3 = layers.conv2d(conv2_bn, filters=nb_filters, kernel_size=3, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            conv3_act = tf.nn.relu(conv3)
            conv3_bn = layers.batch_normalization(conv3_act, training=is_training)

        return conv3_bn

    @staticmethod
    def _code_brick(incoming, nb_filters, is_training, scope, trainable=True):
        """ Code brick: conv --> conv .
        """
        with tf.variable_scope(scope):
            code1 = layers.conv2d(incoming, filters=nb_filters, kernel_size=1, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            code1_act = tf.nn.relu(code1)
            code1_bn = layers.batch_normalization(code1_act, training=is_training, trainable=trainable)

            code2 = layers.conv2d(code1_bn, filters=nb_filters, kernel_size=1, strides=1, padding='same',
                                  kernel_initializer=he_init, bias_initializer=b_init)
            code2_act = tf.nn.relu(code2)
            code2_bn = layers.batch_normalization(code2_act, training=is_training, trainable=trainable)

        return code2_bn

    @staticmethod
    def _output_layer(incoming, n_channels_out, scope):
        """ Output layer: conv .
        """
        with tf.variable_scope(scope):
            output = layers.conv2d(incoming, filters=n_channels_out, kernel_size=1, strides=1, padding='same',
                                   kernel_initializer=he_init, bias_initializer=b_init)
            # activation = linear
        return output
