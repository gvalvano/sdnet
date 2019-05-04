import tensorflow as tf
from tensorflow import layers

# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class MLP(object):

    def __init__(self, incoming, n_in, n_hidden, n_out, is_training, k_prob=1.0, name='MLP'):
        """
        Class for 3-layered multilayer perceptron (MLP).
        :param incoming: (tensor) incoming tensor
        :param n_in: (int) number of input units
        :param n_hidden: (int) number of hidden units
        :param n_out: (int) number of output units
        :param is_training: (tf.placeholder(dtype=tf.bool) or bool) variable to define training or test mode; it is
                        needed for the behaviour of dropout (which behaves differently at train and test time)
        :param k_prob: (float) keep probability for dropout layer. Default = 1, i.e. no dropout applied. A common value
                        for keep probability is 0.8 (e.g. 80% of active units at training time)
        """

        assert 0.0 <= k_prob <= 1.0
        self.incoming = incoming
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.is_training = is_training
        self.k_prob = k_prob
        self.name = name

    def build(self):
        keep_prob = tf.cond(tf.equal(self.is_training, tf.constant(True)), lambda: self.k_prob, lambda: 1.0)

        with tf.variable_scope(self.name):
            incoming = layers.flatten(self.incoming)

            input_layer = layers.dense(incoming, units=self.n_in, kernel_initializer=he_init,
                                       bias_initializer=b_init, activation=tf.nn.relu)
            input_layer = tf.nn.dropout(input_layer, keep_prob=keep_prob)

            hidden_layer = layers.dense(input_layer, units=self.n_hidden, kernel_initializer=he_init,
                                        bias_initializer=b_init, activation=tf.nn.relu)
            hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=keep_prob)

            output_layer = layers.dense(hidden_layer, units=self.n_out, bias_initializer=b_init)

            # final activation: linear
        return output_layer
