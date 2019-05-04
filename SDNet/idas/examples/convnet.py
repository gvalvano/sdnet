"""
Example of generic structure to write neural networks in tensorflow.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import errno
from idas.callbacks import callbacks as tf_callbacks
from idas.callbacks.dsd_callback import DSDCallback
from idas.callbacks.routine_callback import RoutineCallback
import numpy as np


FLAGS = None  # config_file.define_flags()


class ConvNet:  # (DatasetInterface):
    def __init__(self, run_id=None):
        super().__init__()

        # check weather to take a specific RUN_ID
        if run_id is not None:
            FLAGS.RUN_ID = run_id
        self.run_id = FLAGS.RUN_ID

        self.lr = tf.Variable(FLAGS.lr, dtype=tf.float32, trainable=False, name='learning_rate')  # learning rate
        self.batch_size = FLAGS.b_size
        self.global_epoch = 0

        # path to save checkpoints and graph
        self.checkpoint_dir = './results/checkpoints/' + FLAGS.RUN_ID
        self.graph_dir = './results/graphs/' + FLAGS.RUN_ID + '/convnet'
        self.history_log_dir = './mnist_exp/results_mnist/history_logs/' + FLAGS.RUN_ID

        # list of path for the training and validation files:
        self.list_of_files_train = FLAGS.list_of_files_train
        self.list_of_files_valid = FLAGS.list_of_files_valid

        # init the list of callbacks to be called and relative arguments
        self.callbacks = []
        self.callbacks_kwargs = {'history_log_dir': self.history_log_dir}

        # routine callback always runs:
        self.callbacks.append(RoutineCallback())

        # eventually add the callback to perform the sparse training with given the sparsity constraint:
        self.perform_sparse_training = FLAGS.perform_sparse_training
        if self.perform_sparse_training:
            self.callbacks.append(DSDCallback())
            self.callbacks_kwargs['sparsity'] = FLAGS.sparsity
            self.callbacks_kwargs['beta'] = FLAGS.beta

        # eventually add the callback to perform learning rate annealing:
        self.perform_lr_annealing = FLAGS.perform_lr_annealing
        if self.perform_lr_annealing:
            self.callbacks.append(LrAnnealingCallback())
            self.callbacks_kwargs['annealing_strategy'] = FLAGS.annealing_strategy
            self.callbacks_kwargs['annealing_epoch_delay'] = FLAGS.annealing_epoch_delay
            self.callbacks_kwargs['annealing_parameters'] = FLAGS.annealing_parameters

        # Define global step for training e validation and counter for global epoch:
        self.g_train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_train_step')
        self.g_valid_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_validation_step')
        self.g_epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_epoch')

        # other variables:
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')  # training or test mode (needed for the behaviour of dropout, BN, ecc.)
        self.skip_step = FLAGS.skip_step  # frequency of batch report
        self.tensorboard_verbose = FLAGS.tensorboard_verbose  # if True: save also layers weights at the end of epoch

    def get_data(self, num_threads, *kwargs):
        """ 
        Use this function to wrap to super(ConvNet, self).get_data(...) and define:
         self.train_init, self.valid_init, self.input_data, self.labels
        """
        # augm = True  # perform data augmentation
        # stdz = True  # perform data standardization
        # print("Data augmentation: \033[94m{0}\033[0m, Data standardization: \033[94m{1}\033[0m.".format(augm, stdz))
        # self.train_init, self.valid_init, self.input_data, self.labels = \
        #     super(ConvNet, self).get_data(self.list_of_files_train, self.list_of_files_valid, b_size=self.batch_size,
        #                                   augment=augm, standardize=stdz, num_threads=num_threads)
        self.train_init, self.valid_init, self.input_data, self.labels = None, None, None, None
        raise NotImplementedError

    def inference(self):
        """ 
        Use this function to define the network architecture and define: 
         self.output_data = output layer
        """
        self.output_data = None
        raise NotImplementedError

    def loss(self):
        """
        define loss function.
        1. In binary classification, each output channel corresponds to a binary (soft) decision. Therefore, the 
        weighting needs to happen within the computation of the loss --> 'weighted_cross_entropy_with_logits'.
        2. In mutually exclusive multilabel classification each output channel corresponds to the score of a class 
        candidate. The decision comes after and then --> 'softmax_cross_entropy_with_logits'
        """
        # with tf.name_scope('weighted_cross_entropy'):
        #     weights = np.array([0.085, 0.915])
        #     epsilon = tf.constant(1e-12, dtype=self.output_data.dtype)
        #     print("Loss function: 'weighted cross_entropy', weights = \033[94m{0}\033[0m".format(weights))
        #
        #     num_classes = self.output_data.get_shape().as_list()[-1]  # class on the last index
        #     assert (num_classes is not None)
        #     assert len(weights) == num_classes
        #
        #     y_pred = tf.reshape(self.output_data, (-1, num_classes))
        #     y_true = tf.to_float(tf.reshape(self.labels, (-1, num_classes)))
        #     softmax = tf.nn.softmax(y_pred) + epsilon
        #
        #     w_cross_entropy = -tf.reduce_sum(tf.multiply(y_true * tf.log(softmax), weights), reduction_indices=[1])
        #     self.loss = tf.reduce_mean(w_cross_entropy, name='weighted_cross_entropy')
        self.loss = None
        raise NotImplementedError

    def optimize(self):
        """
        Define training op
        using Adam Gradient Descent to minimize cost
        """
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.g_train_step)

    def summary(self):
        """
        Create summaries to write on TensorBoard
        """
        with tf.name_scope('Loss'):
            l_tr = tf.summary.scalar('train_set', self.loss)
            l_val = tf.summary.scalar('validation_set', self.loss)
        with tf.name_scope('Accuracy'):
            a_tr = tf.summary.scalar('train_set', self.accuracy)
            a_val = tf.summary.scalar('validation_set', self.accuracy)
        with tf.name_scope('Dice'):
            d_val = tf.summary.scalar('validation_set', self.dice_metric)
        self.train_summary_op = tf.summary.merge([l_tr, a_tr])
        self.valid_summary_op = tf.summary.merge([l_val, a_val, d_val])

        # ---- #
        if self.tensorboard_verbose:
            vars = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if
                    "_brick" in v.name and 'kernel' in v.name]
            weights_summary = [tf.summary.histogram(v, tf.get_default_graph().get_tensor_by_name(v)) for v in vars]
            self.weights_summary = tf.summary.merge(weights_summary)

    def eval(self):
        """
        Count the number of right predictions in a batch
        """
        with tf.name_scope('predict'):
            # Accuracy metric:
            preds = tf.nn.softmax(self.output_data)
            correct_preds = tf.equal(tf.argmax(preds, -1), tf.argmax(self.labels, -1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

            # Dice metric:
            true_positives = tf.logical_and(tf.equal(tf.argmax(self.labels, -1), True), tf.equal(tf.argmax(preds, -1), True))
            false_positives = tf.logical_and(tf.equal(tf.argmax(self.labels, -1), False), tf.equal(tf.argmax(preds, -1), True))
            false_negatives = tf.logical_and(tf.equal(tf.argmax(self.labels, -1), True), tf.equal(tf.argmax(preds, -1), False))
            true_positives_sum = tf.reduce_sum(tf.to_float(true_positives))
            false_positives_sum = tf.reduce_sum(tf.to_float(false_positives))
            false_negatives_sum = tf.reduce_sum(tf.to_float(false_negatives))
            eps = tf.constant(1e-16)
            self.dice_metric = (2.*true_positives_sum + eps) / (2.*true_positives_sum + false_positives_sum + false_negatives_sum + eps)

    def build(self):
        """ Build the computation graph """
        print('Building the computation graph...\nRUN_ID = \033[94m{0}\033[0m'.format(FLAGS.RUN_ID))
        self.get_data(num_threads=FLAGS.num_threads)
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, init, writer, step, caller):
        """ train the model for one epoch. """
        start_time = time.time()
        sess.run(init)
        total_loss = 0
        total_correct_preds = 0
        n_batches = 0
        try:
            while True:
                caller.on_batch_begin(training_state=True, **self.callbacks_kwargs)

                _, l, a, summaries = sess.run([self.train_op, self.loss, self.accuracy, self.train_summary_op],
                                           feed_dict={self.is_training: True})  # 'is_training:0':
                writer.add_summary(summaries, global_step=step)
                step += 1
                total_loss += l
                total_correct_preds += a
                n_batches += 1
                if (n_batches % self.skip_step) == 0:
                    print('\r  ...training over batch {1}: {0} batch_loss = {2:.4f} {0} batch_accuracy = {3:.4f}'
                          .format(' '*3, n_batches, l, a), end='\n')

                caller.on_batch_end(training_state=True, **self.callbacks_kwargs)
        except tf.errors.OutOfRangeError:
            # Fine dell'epoca. Qui valutare eventuali statistiche, fare log, ecc..
            avg_loss = total_loss/n_batches
            avg_acc = total_correct_preds/n_batches
            delta_t = time.time() - start_time
            pass

        # update global epoch counter:
        sess.run(self.g_epoch.assign_add(1))

        print('\033[31m  TRAIN\033[0m:{0}{0} average loss = {1:.4f} {0} average accuracy = {2:.4f} {0} Took: {3:.3f} '
              'seconds'.format(' '*3, avg_loss, avg_acc, delta_t))
        return step

    def eval_layer_activation(self, input_data, layer_name):
        """ Test the model on input_data
        layer_name = name_scope of the layer to evaluate
        i.e. layer_name = 'network/layer_conv1/Relu:0'  (notice ':0' at the end)
        """
        if self.stdz:
            mu, sigma = np.mean(input_data), np.std(input_data)
            input_data = (input_data - mu) / sigma

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                layer = tf.get_default_graph().get_tensor_by_name(layer_name)
                output = sess.run(layer, feed_dict={self.input_data: input_data, self.is_training: False})
                return output
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        self.checkpoint_dir + ' (checkpoint_dir)')

    def eval_once(self, sess, init, writer, step, caller):
        """ Eval the model once """
        start_time = time.time()
        sess.run(init)
        total_correct_preds = 0
        total_loss = 0
        n_batches = 0
        try:
            while True:
                caller.on_batch_begin(training_state=False, **self.callbacks_kwargs)

                loss_batch, accuracy_batch, summaries = sess.run([self.loss, self.accuracy, self.valid_summary_op],
                                                 feed_dict={self.is_training: False})
                writer.add_summary(summaries, global_step=step)
                step += 1
                total_loss += loss_batch
                total_correct_preds += accuracy_batch
                n_batches += 1

                caller.on_batch_end(training_state=False, **self.callbacks_kwargs)
        except tf.errors.OutOfRangeError:
            # Fine del validation_set set. Qui valutare eventuali statistiche, fare log, ecc..
            avg_loss = total_loss/n_batches
            avg_acc = total_correct_preds/n_batches
            delta_t = time.time() - start_time
            pass

        # update global epoch counter:
        sess.run(self.g_valid_step.assign(step))

        print('\033[31m  VALIDATION\033[0m:  average loss = {1:.4f} {0} average accuracy = {2:.4f} {0} Took: {3:.3f} '
              'seconds'.format(' '*3, avg_loss, avg_acc, delta_t))
        return step

    def test(self, input_data):
        """ Test the model on input_data """

        if self.stdz:
            mu, sigma = np.mean(input_data), np.std(input_data)
            input_data = (input_data - mu) / sigma

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(self.valid_init)

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                output = sess.run(tf.argmax(self.output_data, -1), feed_dict={self.input_data: input_data,
                                                                              self.is_training: False})
                # output = sess.run(self.output_data, feed_dict={self.input_data: input_data, self.is_training: False})

                if self.stdz:
                    mu, sigma = np.mean(input_data), np.std(input_data)
                    output = output * sigma + mu
                return output
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        self.checkpoint_dir + ' (checkpoint_dir)')

    def train(self, n_epochs):
        """ The train function alternates between training one epoch and evaluating """
        print("\nStarting network training... Number of epochs to train: \033[94m{0}\033[0m".format(n_epochs))
        print("Tensorboard verbose mode: \033[94m{0}\033[0m".format(self.tensorboard_verbose))
        print("Tensorboard dir: \033[94m{0}\033[0m".format(self.graph_dir))
        utils.safe_mkdir(self.checkpoint_dir)
        utils.safe_mkdir(self.history_log_dir)
        writer = tf.summary.FileWriter(self.graph_dir, tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver()  # keep_checkpoint_every_n_hours=2
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            print("Model already trained for \033[94m{0}\033[0m epochs.".format(self.g_epoch.eval()))
            t_step = self.g_train_step.eval()  # global step for train
            v_step = self.g_valid_step.eval()  # global step for validation

            # Define a caller to call the callbacks
            self.callbacks_kwargs.update({'sess': sess, 'cnn': self})
            caller = tf_callbacks.ChainCallback(callbacks=self.callbacks)
            caller.on_train_begin(training_state=True, **self.callbacks_kwargs)

            for epoch in range(n_epochs):
                print('_'*40 + '\n\033[1;33mEPOCH {0}:\033[0m'.format(epoch))
                caller.on_epoch_begin(training_state=True, **self.callbacks_kwargs)

                t_step = self.train_one_epoch(sess, self.train_init, writer, t_step, caller)
                caller.on_epoch_end(training_state=True, **self.callbacks_kwargs)

                v_step = self.eval_once(sess, self.valid_init, writer, v_step, caller)

                # save updated variables and weights
                saver.save(sess, self.checkpoint_dir + '/checkpoint', t_step)

                if self.tensorboard_verbose and (epoch % 50 == 0):
                    # writing summary for the weights:
                    summary = sess.run(self.weights_summary)
                    writer.add_summary(summary, global_step=t_step)

            caller.on_train_end(training_state=True, **self.callbacks_kwargs)
        writer.close()


if __name__ == '__main__':
    print('\n' + '-'*3)
    model = ConvNet()
    model.build()
    model.train(n_epochs=2)
