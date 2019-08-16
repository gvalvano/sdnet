#  Copyright 2019 Gabriele Valvano
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from data_interface.dataset_wrapper import DatasetInterfaceWrapper
import config_file
from architectures.sdnet import SDNet
import errno


class Model(DatasetInterfaceWrapper):
    def __init__(self, run_id=None):
        """
        General model.
        This is a simplified version, used for a faster building of the model for test.

        :param run_id: (str) used when we want to load a specific pre-trained model. Default run_id is taken from
                config_file.py
        """

        FLAGS = config_file.define_flags()

        self.run_id = FLAGS.RUN_ID if (run_id is None) else run_id
        self.num_threads = FLAGS.num_threads
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.CUDA_VISIBLE_DEVICE)

        # -----------------------------
        # Model hyper-parameters:
        self.lr = tf.Variable(FLAGS.lr, dtype=tf.float32, trainable=False, name='learning_rate')
        self.batch_size = FLAGS.b_size
        self.nz_latent = FLAGS.nz_latent
        self.n_anatomical_masks = FLAGS.n_anatomical_masks
        self.n_frame_composing_masks = FLAGS.n_frame_composing_masks

        # -----------------------------
        # Data

        # data specifics
        self.input_size = FLAGS.input_size
        self.n_classes = FLAGS.n_classes

        # ACDC data set
        self.acdc_data_path = FLAGS.acdc_data_path  # list of path for the training and validation files:

        # data pre-processing
        self.augment = FLAGS.augment  # perform data augmentation
        self.standardize = FLAGS.standardize  # perform data standardization

        # -----------------------------
        # Report

        # path to save checkpoints and graph
        self.checkpoint_dir = './results/checkpoints/' + FLAGS.RUN_ID
        self.graph_dir = './results/graphs/' + FLAGS.RUN_ID + '/convnet'
        self.history_log_dir = './results/history_logs/' + FLAGS.RUN_ID
        # verbosity
        self.skip_step = FLAGS.skip_step  # frequency of batch report
        self.tensorboard_verbose = FLAGS.tensorboard_verbose  # (bool) save also layers weights at the end of epoch

        # training or test mode (needed for the behaviour of dropout, BN, ecc.)
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        # -----------------------------
        # initialize wrapper to the data set
        super().__init__(augment=self.augment,
                         standardize=self.standardize,
                         batch_size=self.batch_size,
                         input_size=self.input_size,
                         num_threads=self.num_threads)

    def build(self):
        """ Build the computation graph """
        print('Building the computation graph...\nRUN_ID = \033[94m{0}\033[0m'.format(self.run_id))
        self.get_data()
        self.define_model()
        self.define_losses()
        self.define_optimizers()
        self.define_eval_metrics()
        self.define_summaries()

    def get_data(self):
        pass

    def define_model(self):

        # Define the placeholders to be used in define_model().
        self.global_seed = tf.placeholder(tf.int64, shape=())

        # Repeat indefinitely all the iterators, exception made for the one iterating over the biggest dataset. This
        # ensures that every data is used during training.
        self.sup_train_init, self.sup_valid_init, self.sup_test_init, self.sup_input_data, self.sup_output_data = \
            super(Model, self).get_acdc_sup_data(data_path=self.acdc_data_path, repeat=False, seed=self.global_seed)

        # ----------------------------------------------- 
        # define the model architecture:    

        sdnet_sup = SDNet(self.n_anatomical_masks, self.nz_latent, self.n_classes, self.is_training, name='Model')
        sdnet_sup = sdnet_sup.build(self.sup_input_data)

        self.sup_soft_anatomy = sdnet_sup.get_soft_anatomy()
        self.sup_hard_anatomy = sdnet_sup.get_hard_anatomy()
        self.sup_pred_mask_oh = sdnet_sup.get_pred_mask(one_hot=True)
        self.sup_reconstruction = sdnet_sup.get_input_reconstruction()

    def define_losses(self):
        """
        Define loss function.
        """
        pass

    def define_optimizers(self):
        """
        Define training op
        using Adam Gradient Descent to minimize cost
        """
        pass

    def define_eval_metrics(self):
        """
        Evaluate the model on the current batch
        """
        pass

    def define_summaries(self):
        """
        Create summaries to write on TensorBoard
        """
        pass

    def _train_all_op(self, sess, writer, step):
        pass

    def train_one_epoch(self, sess, iterator_init_list, writer, step, caller, seed):
        pass

    def _eval_all_op(self, sess, writer, step):
        pass

    def eval_once(self, sess, iterator_init_list, writer, step, caller):
        pass

    def test_once(self, sess, sup_test_init, writer, step, caller):
        pass

    def test(self, input_data, checkpoint_dir=None):
        """ Test the model on input_data """
        if self.standardize:
            print('Remember to standardize your data!')

        if checkpoint_dir is not None:
            self.checkpoint_dir = checkpoint_dir

        # config for the session: allow growth for GPU to avoid OOM when other processes are running
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                print('Returning: (soft anatomy, hard anatomy, predicted mask, reconstruction)')
                output = sess.run([self.sup_soft_anatomy,
                                   self.sup_hard_anatomy,
                                   self.sup_pred_mask_oh,
                                   self.sup_reconstruction],
                                  feed_dict={self.sup_input_data: input_data,
                                             self.is_training: False})
                return output
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        self.checkpoint_dir + ' (checkpoint_dir)')

    def train(self, n_epochs):
        pass


if __name__ == '__main__':
    print('\n' + '-' * 3)
    model = Model()
    model.build()
