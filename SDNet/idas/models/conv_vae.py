"""
Implementation of a Convolutional Variational Autoencoder (Convolutional VAE).
"""

import tensorflow as tf
from tensorflow.layers import conv2d, dense, batch_normalization, conv2d_transpose, max_pooling2d


class ConvVAE:

    def __init__(self, input_data):
        super().__init__()
        self.input_data = input_data
        self.prediction = None
        self.is_training = True

        self.batch_size = 8
        self.n_latent = 128

    def _recognition_model(self):
        """ This function generates a probabilistic encoder (recognition network), mapping inputs to a
        normal distribution in latent space. The transformation is parametrized and can be learned.
        """
        k_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)  # He init.
        b_init = tf.zeros_initializer()

        with tf.variable_scope('Recognition_model'):
            conv1 = conv2d(self.input_data, filters=64, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=k_init, bias_initializer=b_init)
            conv1_act = tf.nn.relu(conv1)
            conv1_bn = batch_normalization(conv1_act, training=self.is_training)

            pool = max_pooling2d(conv1_bn, pool_size=2, strides=2, padding='same')

            conv2 = conv2d(pool, filters=128, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=k_init, bias_initializer=b_init)
            conv2_act = tf.nn.relu(conv2)
            conv2_bn = batch_normalization(conv2_act, training=self.is_training)

            features = tf.contrib.layers.flatten(conv2_bn)

            z_mean = dense(features, units=self.n_latent, name='z_mean_dense')
            z_logvar = dense(features, units=self.n_latent, name='z_stddev_dense')

        return z_mean, z_logvar

    def _generative_model(self, z_mean, z_logvar):
        """ Generate probabilistic decoder (decoder network), mapping points from the latent space into a distribution
        in data space. The transformation is parametrized and can be learned.
        """
        k_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)  # He init.
        b_init = tf.zeros_initializer()

        latent_shape = z_mean.get_shape().as_list()
        latent_shape[0] = self.batch_size

        with tf.variable_scope('Generative_model', reuse=True):

            # sample from normal distribution:
            eps = tf.random_normal(latent_shape, dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
            z = z_mean + eps * tf.exp(0.5 * z_logvar)

            # convolutional layers:
            conv1t = conv2d_transpose(z, filters=128, kernel_size=2, strides=2, padding='same',
                                      kernel_initializer=k_init, bias_initializer=b_init, reuse=True)
            conv1t_act = tf.nn.relu(conv1t)
            conv1t_bn = batch_normalization(conv1t_act, training=self.is_training)

            # output
            conv = conv2d(conv1t_bn, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=k_init, bias_initializer=b_init)

            generated_sample = conv2d(conv, filters=1, kernel_size=3, strides=1, padding='same',
                                      kernel_initializer=k_init, bias_initializer=b_init)
        return generated_sample

    def vae(self):
        with tf.variable_scope('Variational_AE'):

            # Recognition model: map to latent space distribution
            self.z_mean, self.z_logvar = self._recognition_model()

            # Generative model: map from latent space to pixel space
            prediction = self._generative_model(self.z_mean, self.z_logvar)

            self.prediction = prediction

    def loss(self):

        in_shape = self.input_data.get_shape().as_list()
        n_values = in_shape[1] * in_shape[2] * in_shape[3]

        x_true = tf.reshape(self.input_data, [-1, n_values])
        x_reconstructed = tf.reshape(self.prediction, [-1, n_values])

        # _______
        # Reconstruction loss:
        # generator_loss = x_true * tf.log(1e-10 + x_reconstructed) + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
        # generator_loss = -tf.reduce_sum(generator_loss, 1)
        self.generator_loss = tf.losses.mean_squared_error(x_reconstructed, x_true)

        # _______
        # KL Divergence loss:
        kl_div_loss = 1.0 + self.z_logvar - tf.square(self.z_mean) - tf.exp(self.z_logvar)
        kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
        self.latent_loss = kl_div_loss

        self.loss = tf.reduce_mean(self.generator_loss + self.latent_loss)

