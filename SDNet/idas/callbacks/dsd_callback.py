"""
Callback for DSD training [Han et al. 2017].
"""
import tensorflow as tf
from idas.callbacks.callbacks import Callback
import idas.logger.json_logger as jlogger
import os


def run_sparse_step(sess, sparsity=0.30):
    """ 
    Refer to Dense-Sparse-Dense training [Han et al. 2017].
    Here weights < threshold _lambda are set to 0 accordingly to the desired sparsity (i.e. S=30%).
    The _lambda value is chosen to leave the layer weight matrix with the desired sparsity.
    """
    print("Thresholding elements below chosen value (Sparsity {0}%)".format(sparsity * 100))
    layers = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('kernel:0')]
    for lyr in layers:
        layer = sess.graph.get_tensor_by_name(lyr)
        flat_layer = tf.reshape(layer, [-1])
        N = tf.to_float(tf.shape(flat_layer)[0])
        k = tf.to_int32(N * (1. - tf.constant(sparsity, dtype=tf.float32)))
        values, indices = tf.nn.top_k(tf.abs(flat_layer), k=k)
        _lambda = tf.reduce_min(values)

        # TODO: define outside tf.assign() operation to avoid adding a new node to the graph every time we call it
        sess.run(tf.assign(layer, tf.multiply(layer, tf.to_float(tf.abs(layer) >= _lambda))))


class DSDCallback(Callback):
    def __init__(self):
        super().__init__()
        # Define variables here because the callback __init__() is called before the initialization of all variables
        # in the graph.
        self.history_log_file = None

    def on_train_begin(self, training_state, **kwargs):

        cnn = kwargs['cnn']
        if cnn is None:
            raise Exception

        try:
            sparsity = kwargs['sparsity']
        except KeyError:
            sparsity = 0.30
        try:
            beta = kwargs['beta']
        except KeyError:
            beta = 0.10

        self.history_log_file = kwargs['history_log_dir'] + os.sep + 'train_history.json'

        vals = {'done_before': True, 'sparsity': sparsity, 'beta': beta}
        jlogger.add_new_node('SPARSE_TRAINING', vals, fname=self.history_log_file)

        # set learning rate to 1/10 (beta) of its initial value
        cnn.lr = beta * cnn.lr
        print("\nRunning training with sparse constrain: SPARSITY = \033[94m{0}%\033[0m.".format(sparsity * 100))
        print("Reducing original learning rate to \033[94m{0}%\033[0m of its value.".format(beta * 100))

    def on_epoch_begin(self, training_state, **kwargs):

        sess = kwargs['sess']
        try:
            sparsity = kwargs['sparsity']
        except KeyError:
            sparsity = 0.30

        run_sparse_step(sess, sparsity=sparsity)
