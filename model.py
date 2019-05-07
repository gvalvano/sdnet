import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from idas import utils
import tensorflow as tf
from data_interface.dataset_wrapper import DatasetInterfaceWrapper
from idas.callbacks import callbacks as tf_callbacks
from idas.callbacks.routine_callback import RoutineCallback
from idas.callbacks.early_stopping_callback import EarlyStoppingCallback, EarlyStoppingException
import config_file
from architectures.mask_discriminator import MaskDiscriminator
from architectures.sdnet import SDNet
from idas.metrics.tf_metrics import dice_coe
from idas.losses.tf_losses import weighted_softmax_cross_entropy
from tensorflow.core.framework import summary_pb2
import errno


class Model(DatasetInterfaceWrapper):
    def __init__(self, run_id=None):
        """
        Class used to extend the SDNet framework leveraging temporal information.
        :param run_id: (str) used when we want to load a specific pre-trained model. Default run_id is taken from
                config_file
        """

        FLAGS = config_file.define_flags()

        self.run_id = FLAGS.RUN_ID if (run_id is None) else run_id
        self.num_threads = FLAGS.num_threads

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

        # -----------------------------
        # Callbacks
        # init the list of callbacks to be called and relative arguments
        self.callbacks = []
        self.callbacks_kwargs = {'history_log_dir': self.history_log_dir}
        self.callbacks.append(RoutineCallback())  # routine callback always runs
        # Early stopping callback:
        self.callbacks_kwargs['es_loss'] = None
        self.callbacks.append(EarlyStoppingCallback(min_delta=0.01, patience=100))

        # -----------------------------
        # Other settings

        # Define global step for training e validation and counter for global epoch:
        self.g_train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_train_step')
        self.g_valid_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_validation_step')
        self.g_epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_epoch')

        # define their update operations
        up_value = tf.placeholder(tf.int32, None, name='update_value')
        self.update_g_train_step = self.g_train_step.assign(up_value, name='update_g_train_step')
        self.update_g_valid_step = self.g_valid_step.assign(up_value, name='update_g_valid_step')
        self.increase_g_epoch = self.g_epoch.assign_add(1, name='increase_g_epoch')

        # training or test mode (needed for the behaviour of dropout, BN, ecc.)
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        # -----------------------------
        # initialize wrapper to the data set
        super().__init__(augment=self.augment,
                         standardize=self.standardize,
                         batch_size=self.batch_size,
                         input_size=self.input_size,
                         num_threads=self.num_threads)

    def get_data(self):  # *kwargs

        self.sup_train_init, self.sup_valid_init, self.sup_input_data, self.sup_output_data = \
            super(Model, self).get_acdc_sup_data(data_path=self.acdc_data_path)

        self.unsup_train_init, self.unsup_valid_init, self.unsup_input_data, self.unsup_output_data = \
            super(Model, self).get_acdc_unsup_data(data_path=self.acdc_data_path)

    def define_model(self):

        # - - - - - - -
        # define the model for supervised, unsupervised and temporal frame prediction data:
        sdnet = SDNet(self.n_anatomical_masks, self.nz_latent, self.n_classes, self.is_training, name='Model')

        sdnet_sup = sdnet.build(self.sup_input_data)
        sdnet_unsup = sdnet.build(self.unsup_input_data, reuse=True)

        # - - - - - - -
        # define tensors for the losses:

        # sup pathway
        self.pred_mask = sdnet_sup.get_pred_mask(one_hot=False)
        self.pred_mask_oh = sdnet_sup.get_pred_mask(one_hot=True)
        self.soft_anatomy = sdnet_sup.get_soft_anatomy()
        self.hard_anatomy = sdnet_sup.get_hard_anatomy()

        # unsup pathway
        self.unsup_reconstruction = sdnet_unsup.get_input_reconstruction()
        self.z_mean, self.z_logvar, self.sampled_z = sdnet_unsup.get_z_distribution()
        self.z_regress = sdnet_unsup.get_z_sample_estimate()

        # - - - - - - -
        # build Mask Discriminator (Least Square GAN)
        with tf.variable_scope('MaskDiscriminator'):
            model_real = MaskDiscriminator(self.sup_output_data, self.is_training, n_filters=64).build()
            model_fake = MaskDiscriminator(self.pred_mask_oh, self.is_training, n_filters=64).build(reuse=True)
            self.disc_real = model_real.get_prediction()
            self.disc_fake = model_fake.get_prediction()

    def define_losses(self):
        """
        Define loss function.
        """
        # _______
        # Reconstruction loss:
        with tf.variable_scope('Reconstruction_loss'):
            self.rec_loss = tf.reduce_mean(tf.abs(self.unsup_reconstruction - self.unsup_output_data))
            self.z_regress_loss = tf.reduce_mean(tf.abs(self.z_regress - self.sampled_z))

        # _______
        # Dice loss:
        # with tf.variable_scope('3Chs_Dice_loss'):
        #     soft_pred_mask = tf.nn.softmax(self.pred_mask)
        #     dice_3chs = dice_coe(output=soft_pred_mask[..., 1:], target=self.sup_output_data[..., 1:])
        #     dice = dice_coe(output=soft_pred_mask, target=self.sup_output_data)
        #     self.dice_loss = 1.0 - dice  # dice_3chs

        # _______
        # Weighted Cross Entropy loss:
        with tf.variable_scope('WXEntropy_loss'):
            self.wxentropy_loss = weighted_softmax_cross_entropy(y_pred=self.pred_mask, y_true=self.sup_output_data, num_classes=4)

        # _______
        # KL Divergence loss:
        with tf.variable_scope('KL_divergence_loss'):
            kl_i = 1.0 + self.z_logvar - tf.square(self.z_mean) - tf.exp(self.z_logvar)
            kl_div_loss = -0.5 * tf.reduce_sum(kl_i, 1)
            self.kl_div_loss = tf.reduce_mean(kl_div_loss)

        # _______
        # Mask Discriminator loss:
        # this is a LeastSquare GAN: use MSE as loss
        with tf.variable_scope('MaskDiscriminator_loss'):
            self.adv_disc_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.disc_real, 1.0)) + \
                                 0.5 * tf.reduce_mean(tf.squared_difference(self.disc_fake, 0.0))
            self.adv_gen_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.disc_fake, 1.0))

        # _______
        # L2 regularization loss:
        # with tf.variable_scope('L2_regularization_loss'):
        #     self.l2_reg_loss = idas_losses.l2_weights_regularization_loss()

        # - - - - - - - - - - - -

        # define weights for the cost contributes:
        w_kl = 0.01
        w_rec = 1.0
        w_zrec = 1.0
        w_adv = 0.0

        # define losses for supervised, unsupervised and frame prediction steps:
        self.sup_loss = self.wxentropy_loss + \
                        w_adv * self.adv_gen_loss      # you can try both: wxentropy_loss / dice_loss

        self.unsup_loss = w_kl * self.kl_div_loss + \
                          w_rec * self.rec_loss + \
                          w_zrec * self.z_regress_loss

        # add regularization:
        # self.sup_loss += 0.01 * self.l2_reg_loss
        # self.unsup_loss += 0.01 * self.l2_reg_loss

    def define_optimizers(self):
        """
        Define training op
        using Adam Gradient Descent to minimize cost
        """

        def _train_op_wrapper(loss_function, lr, clip_grads=False, clip_value=5.0):
            """ define optimizer and train op with gradient clipping. """
            # define optimizer:
            optimizer = tf.train.AdamOptimizer(lr)
            # define update_ops to update batch normalization population statistics
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                # gradient clipping for stability:
                gradients, variables = zip(*optimizer.compute_gradients(loss_function))
                if clip_grads:
                    gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
                # train op:
                train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.g_train_step)

            return train_op

        clip = True
        self.train_op_sup = _train_op_wrapper(self.sup_loss, self.lr, clip)
        self.train_op_unsup = _train_op_wrapper(self.unsup_loss, self.lr, clip)
        self.train_op_disc = _train_op_wrapper(self.adv_disc_loss, self.lr, clip)

    def define_eval_metrics(self):
        """
        Evaluate the model on the current batch
        """
        # Dice
        with tf.variable_scope('Dice'):
            self.dice = dice_coe(output=self.pred_mask_oh, target=self.sup_output_data)

        with tf.variable_scope('Dice_3channels'):
            self.dice_3chs = dice_coe(output=self.pred_mask_oh[..., 1:], target=self.sup_output_data[..., 1:])

    def define_summaries(self):
        """
        Create summaries to write on TensorBoard
        """
        # Scalar summaries:
        with tf.name_scope('Reconstruction'):
            tr_rec = tf.summary.scalar('train/rec_loss', self.rec_loss)
            val_rec = tf.summary.scalar('validation/rec_loss', self.rec_loss)

        with tf.name_scope('Z_Reconstruction'):
            val_zrec = tf.summary.scalar('validation/loss', self.z_regress_loss)

        with tf.name_scope('KL_Divergence'):
            tr_kl = tf.summary.scalar('train/loss', self.kl_div_loss)

        # with tf.name_scope('L2_Weight'):
        #     tr_l2 = tf.summary.scalar('train/loss', self.l2_reg_loss)

        with tf.name_scope('Dice_1'):
            val_dice = tf.summary.scalar('validation/dice', self.dice)
            val_dice_3chs = tf.summary.scalar('validation/dice_3channels', self.dice_3chs)

        with tf.name_scope('WXEntropy_loss'):
            tr_wxe = tf.summary.scalar('train/loss', self.wxentropy_loss)
            val_wxe = tf.summary.scalar('validation/loss', self.wxentropy_loss)

        with tf.name_scope('Adversarial_loss'):
            tr_adv_d = tf.summary.scalar('train/disc_loss', self.adv_disc_loss)
            tr_adv_g = tf.summary.scalar('train/gen_loss', self.adv_gen_loss)
            val_adv_d = tf.summary.scalar('validation/disc_loss', self.adv_disc_loss)
            val_adv_g = tf.summary.scalar('validation/gen_loss', self.adv_gen_loss)

        # Image summaries:
        with tf.name_scope('0_Input'):
            img_inp_s = tf.summary.image('input_sup', self.sup_input_data, max_outputs=3)
            img_inp_us = tf.summary.image('input_unsup', self.unsup_input_data, max_outputs=3)
        with tf.name_scope('1_Reconstruction'):
            img_rec_us = tf.summary.image('unsup_rec', self.unsup_reconstruction, max_outputs=3)
        with tf.name_scope('2_Segmentation'):
            img_pred_mask = tf.summary.image('pred_mask', self.pred_mask_oh[..., 1:], max_outputs=3)
        with tf.name_scope('3_Segmentation'):
            img_mask = tf.summary.image('gt_mask', self.sup_output_data[..., 1:], max_outputs=3)

        def get_slice(incoming, idx):
            return tf.expand_dims(incoming[..., idx], -1)

        with tf.name_scope('4_SoftAnatomy'):
            img_s_an_lst = [tf.summary.image('soft_{0}'.format(i), get_slice(self.soft_anatomy, i), max_outputs=1)
                            for i in range(8)]
        with tf.name_scope('5_HardAnatomy'):
            img_h_an_lst = [tf.summary.image('hard_{0}'.format(i), get_slice(self.hard_anatomy, i), max_outputs=1)
                            for i in range(8)]

        # _______________________________
        # merging all scalar summaries:
        sup_train_scalar_summaries = [tr_wxe]
        sup_valid_scalar_summaries = [val_wxe, val_dice, val_dice_3chs]
        unsup_train_scalar_summaries = [tr_rec, tr_kl]  # , tr_l2]
        unsup_valid_scalar_summaries = [val_rec, val_zrec]
        disc_train_summaries = [tr_adv_d, tr_adv_g]
        disc_valid_summaries = [val_adv_d, val_adv_g]

        self.sup_train_scalar_summary_op = tf.summary.merge(sup_train_scalar_summaries)
        self.sup_valid_scalar_summary_op = tf.summary.merge(sup_valid_scalar_summaries)
        self.unsup_train_scalar_summary_op = tf.summary.merge(unsup_train_scalar_summaries)
        self.unsup_valid_scalar_summary_op = tf.summary.merge(unsup_valid_scalar_summaries)
        self.disc_train_summary_op = tf.summary.merge(disc_train_summaries)
        self.disc_valid_summary_op = tf.summary.merge(disc_valid_summaries)

        # _______________________________
        # merging all images summaries:
        sup_valid_images_summaries = [img_inp_s, img_mask, img_pred_mask]
        unsup_valid_images_summaries = [img_inp_us, img_rec_us]
        tframe_valid_images_summaries = []
        tframe_valid_images_summaries.extend(img_s_an_lst)
        tframe_valid_images_summaries.extend(img_h_an_lst)

        self.sup_valid_images_summary_op = tf.summary.merge(sup_valid_images_summaries)
        self.unsup_valid_images_summary_op = tf.summary.merge(unsup_valid_images_summaries)
        self.tframe_valid_images_summary_op = tf.summary.merge(tframe_valid_images_summaries)

        # ---- #
        if self.tensorboard_verbose:
            _vars = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'kernel' in v.name]
            weights_summary = [tf.summary.histogram(v, tf.get_default_graph().get_tensor_by_name(v)) for v in _vars]
            self.weights_summary = tf.summary.merge(weights_summary)

    def build(self):
        """ Build the computation graph """
        print('Building the computation graph...\nRUN_ID = \033[94m{0}\033[0m'.format(self.run_id))
        self.get_data()
        self.define_model()
        self.define_losses()
        self.define_optimizers()
        self.define_eval_metrics()
        self.define_summaries()

    def _train_sup_step(self, sess, writer, step):
        """ train the model with a supervised step. """
        _, l, scalar_summaries = sess.run([self.train_op_sup, self.sup_loss, self.sup_train_scalar_summary_op],
                                          feed_dict={self.is_training: True})
        writer.add_summary(scalar_summaries, global_step=step)
        return l

    def _train_disc_step(self, sess, writer, step):
        """ train the model with a supervised step. """
        _, l, scalar_summaries = sess.run([self.train_op_disc, self.adv_disc_loss, self.disc_train_summary_op],
                                          feed_dict={self.is_training: True})
        writer.add_summary(scalar_summaries, global_step=step)
        return l

    def _train_unsup_step(self, sess, writer, step):
        """ train the model with an unsupervised step. """
        _, l, scalar_summaries = sess.run([self.train_op_unsup, self.unsup_loss, self.unsup_train_scalar_summary_op],
                                          feed_dict={self.is_training: True})
        writer.add_summary(scalar_summaries, global_step=step)
        return l

    def train_one_epoch(self, sess, sup_init, unsup_init, writer, step, caller):
        """ train the model for one epoch. """
        start_time = time.time()

        # initialize data set iterators:
        sess.run(sup_init)
        sess.run(unsup_init)

        total_sup_loss = 0
        total_unsup_loss = 0
        n_batches = 0

        try:
            while True:
                caller.on_batch_begin(training_state=True, **self.callbacks_kwargs)

                total_sup_loss += self._train_sup_step(sess, writer, step)
                step += 1

                total_unsup_loss += self._train_disc_step(sess, writer, step)
                step += 1

                # This is the only tensorflow dataset with repeat=False, so it finishes and launch an exception:
                total_unsup_loss += self._train_unsup_step(sess, writer, step)
                step += 1

                n_batches += 1
                if (n_batches % self.skip_step) == 0:
                    print('\r  ...training over batch {1}: {0} batch_sup_loss = {2:.4f}\tbatch_unsup_loss = {3:.4f} {0}'
                          .format(' ' * 3, n_batches, total_sup_loss, total_unsup_loss), end='\n')

                caller.on_batch_end(training_state=True, **self.callbacks_kwargs)

        except tf.errors.OutOfRangeError:
            # End of the epoch. Compute statistics here:
            total_loss = total_sup_loss + total_unsup_loss
            avg_loss = total_loss / n_batches
            delta_t = time.time() - start_time
            pass

        # update global epoch counter:
        sess.run(self.increase_g_epoch)
        sess.run(self.update_g_train_step, feed_dict={'update_value:0': step})

        print('\033[31m  TRAIN\033[0m:{0}{0} average loss = {1:.4f} {0} Took: {2:.3f} seconds'
              .format(' ' * 3, avg_loss, delta_t))
        return step

    def _eval_sup_step(self, sess, writer, step):
        """ evaluate the model on supervised task. """
        l, dice_3chs, scalar_summaries, images_summaries = sess.run([self.sup_loss, self.dice_3chs, self.sup_valid_scalar_summary_op,
                                                          self.sup_valid_images_summary_op], feed_dict={self.is_training: False})
        writer.add_summary(scalar_summaries, global_step=step)
        writer.add_summary(images_summaries, global_step=step)
        return l, dice_3chs

    def _eval_unsup_step(self, sess, writer, step):
        """ evaluate the model on unsupervised task. """
        l, scalar_summaries, images_summaries = sess.run([self.unsup_loss, self.unsup_valid_scalar_summary_op,
                                                          self.unsup_valid_images_summary_op], feed_dict={self.is_training: False})
        writer.add_summary(scalar_summaries, global_step=step)
        writer.add_summary(images_summaries, global_step=step)
        return l

    def _eval_disc_step(self, sess, writer, step):
        """ evaluate the model on unsupervised task. """
        l, scalar_summaries = sess.run([self.adv_disc_loss, self.disc_valid_summary_op], feed_dict={self.is_training: False})
        writer.add_summary(scalar_summaries, global_step=step)
        return l

    def eval_once(self, sess, sup_init, unsup_init, writer, step, caller):
        """ Eval the model once """
        start_time = time.time()

        # initialize data set iterators:
        sess.run(sup_init)
        sess.run(unsup_init)

        total_sup_loss = 0
        total_dice_score = 0
        total_unsup_loss = 0
        n_batches = 0
        try:
            while True:
                caller.on_batch_begin(training_state=False, **self.callbacks_kwargs)

                loss, score = self._eval_sup_step(sess, writer, step)
                total_sup_loss += loss
                total_dice_score += score
                step += 1

                total_sup_loss += self._eval_disc_step(sess, writer, step)
                step += 1

                total_unsup_loss += self._eval_unsup_step(sess, writer, step)
                step += 1

                n_batches += 1
                caller.on_batch_end(training_state=False, **self.callbacks_kwargs)

        except tf.errors.OutOfRangeError:
            # End of the validation set. Compute statistics here:
            total_loss = total_sup_loss + total_unsup_loss
            avg_loss = total_loss / n_batches
            avg_dice = total_dice_score / n_batches
            dice_loss = 1.0 - avg_dice
            delta_t = time.time() - start_time

            value = summary_pb2.Summary.Value(tag="Dice_1/validation/Dice_3channels_avg", simple_value=avg_dice)
            summary = summary_pb2.Summary(value=[value])
            writer.add_summary(summary, global_step=step)

            pass

        # update global epoch counter:
        sess.run(self.update_g_valid_step, feed_dict={'update_value:0': step})

        print('\033[31m  VALIDATION\033[0m:  average loss = {1:.4f} {0} Took: {2:.3f} seconds'
              .format(' ' * 3, avg_loss, delta_t))
        return step, dice_loss

    def test(self, input_data):
        """ Test the model on input_data """
        if self.standardize:
            print('Remember to standardize your data!')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Returning: (soft anatomy, hard anatomy, predicted mask)')
                output = sess.run([self.soft_anatomy, self.hard_anatomy, self.pred_mask],
                                  feed_dict={self.sup_input_data: input_data, self.is_training: False})
                return output
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        self.checkpoint_dir + ' (checkpoint_dir)')

    def train(self, n_epochs):
        """ The train function alternates between training one epoch and evaluating """
        print("\nStarting network training... Number of epochs to train: \033[94m{0}\033[0m".format(n_epochs))
        print("Tensorboard verbose mode: \033[94m{0}\033[0m".format(self.tensorboard_verbose))
        print("Tensorboard dir: \033[94m{0}\033[0m".format(self.graph_dir))
        print("Data augmentation: \033[94m{0}\033[0m, Data standardization: \033[94m{1}\033[0m."
              .format(self.augment, self.standardize))
        utils.safe_mkdir(self.checkpoint_dir)
        utils.safe_mkdir(self.history_log_dir)
        writer = tf.summary.FileWriter(self.graph_dir, tf.get_default_graph())

        # config for the session: allow growth for GPU to avoid OOM when other processes are running
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver()  # keep_checkpoint_every_n_hours=2
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_dir + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            trained_epochs = self.g_epoch.eval()
            print("Model already trained for \033[94m{0}\033[0m epochs.".format(trained_epochs))
            t_step = self.g_train_step.eval()  # global step for train
            v_step = self.g_valid_step.eval()  # global step for validation

            # Define a caller to call the callbacks
            self.callbacks_kwargs.update({'sess': sess, 'cnn': self})
            caller = tf_callbacks.ChainCallback(callbacks=self.callbacks)
            caller.on_train_begin(training_state=True, **self.callbacks_kwargs)

            # trick to find performance bugs: this will raise an exception if any new node is inadvertently added to the
            # graph. This will ensure that I don't add many times the same node to the graph (which could be expensive):
            tf.get_default_graph().finalize()

            for epoch in range(n_epochs):
                ep_str = str(epoch + 1) if (trained_epochs == 0) else '({0}+) '.format(trained_epochs) + str(epoch + 1)
                print('_' * 40 + '\n\033[1;33mEPOCH {0}:\033[0m'.format(ep_str))
                caller.on_epoch_begin(training_state=True, **self.callbacks_kwargs)

                t_step = self.train_one_epoch(sess, self.sup_train_init, self.unsup_train_init, writer, t_step, caller)

                curr_ep = sess.run(self.g_epoch)

                if curr_ep >= 0:  # and not (curr_ep % 5):  # when to evaluate the model
                    v_step, val_loss = self.eval_once(sess, self.sup_valid_init, self.unsup_valid_init, writer, v_step, caller)
                    self.callbacks_kwargs['es_loss'] = val_loss

                # save updated variables and weights
                saver.save(sess, self.checkpoint_dir + '/checkpoint', t_step)

                if self.tensorboard_verbose and (epoch % 50 == 0):
                    # writing summary for the weights:
                    summary = sess.run(self.weights_summary)
                    writer.add_summary(summary, global_step=t_step)

                try:
                    caller.on_epoch_end(training_state=True, **self.callbacks_kwargs)
                except EarlyStoppingException:
                    print('Early stopping...')
                    break

            caller.on_train_end(training_state=True, **self.callbacks_kwargs)
        writer.close()


if __name__ == '__main__':
    print('\n' + '-' * 3)
    model = Model()
    model.build()
    model.train(n_epochs=2)
