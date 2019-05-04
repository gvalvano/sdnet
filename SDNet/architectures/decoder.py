import tensorflow as tf
from tensorflow import layers
from .layers.film_layer import film_layer

# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class Decoder(object):

    def __init__(self, z_factors, encoded_anatomy, n_channels, is_training, name='Decoder'):
        """
        Decoder that generates an image by combining an anatomical and a modality representation.
        :param z_factors: (tensor) incoming tensor with the modality representation
        :param encoded_anatomy: (tensor) incoming tensor with input anatomy information
        :param n_channels: (int) number of anatomical channels
        :param is_training: (tf.placeholder(dtype=tf.bool) or bool) variable to define training or test mode; it is
                        needed for the behaviour of dropout, batch normalization, ecc. (which behave differently
                        at train and test time)
        :param name: (string) name scope for the unet

        - - - - - - - - - - - - - - - -
        Notice that:
          - the network output is linear (regression task)

        - - - - - - - - - - - - - - - -

        Example of usage:

            # build the entire model:
            model = Decoder(z_factors, encoded_anatomy, is_training).build()

        """
        self.z_factors = z_factors
        self.encoded_anatomy = encoded_anatomy
        self.n_channels = n_channels
        self.is_training = is_training
        self.name = name

        self.reconstruction = None

    def build(self):
        """
        Build the model.
        """
        with tf.variable_scope(self.name):

            with tf.variable_scope('film_fusion'):
                film1 = self._film_layer(self.encoded_anatomy, self.z_factors, scope='_film_layer_0')
                film2 = self._film_layer(film1, self.z_factors, scope='_film_layer_1')
                film3 = self._film_layer(film2, self.z_factors, scope='_film_layer_2')
                film4 = self._film_layer(film3, self.z_factors, scope='_film_layer_3')

            self.reconstruction = layers.conv2d(film4, filters=1, kernel_size=3, strides=1, padding='same')

        return self

    def get_reconstruction(self):
        return self.reconstruction

    def _film_layer(self, spatial_input, resd_input, scope='_film_layer'):
        with tf.variable_scope(scope):

            conv1 = layers.conv2d(spatial_input, filters=self.n_channels, kernel_size=3, strides=1, padding='same')
            conv1_act = tf.nn.leaky_relu(conv1)

            conv2 = layers.conv2d(conv1_act, filters=self.n_channels, kernel_size=3, strides=1, padding='same')
            conv2_act = tf.nn.leaky_relu(conv2)

            gamma_l2, beta_l2 = self._film_pred(resd_input, 2 * self.n_channels)

            film = film_layer(conv2_act, gamma_l2, beta_l2)
            film_act = tf.nn.leaky_relu(film)

            film_sum = conv1_act + film_act

        return film_sum

    @staticmethod
    def _film_pred(incoming, num_chn):
        with tf.variable_scope('_film_pred'):

            fc = layers.dense(incoming, units=num_chn)
            fc_act = tf.nn.leaky_relu(fc)
            film_pred = layers.dense(fc_act, units=num_chn)

            gamma = film_pred[:, :int(num_chn / 2)]
            beta = film_pred[:, int(num_chn / 2):]

        return gamma, beta

