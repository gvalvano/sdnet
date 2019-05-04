"""
Automatic Cardiac Diagnostic Challenge 2017 database. In total there are images of 100 patients, for which manual
segmentations of the heart cavity, myocardium and right ventricle are provided.
Database at: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
Atlas of the heart in each projection at: http://tuttops.altervista.org/ecocardiografia_base.html
"""
import tensorflow as tf
import numpy as np
from idas.utils import get_available_gpus
import os
from glob import glob
import cv2


class DatasetInterface(object):

    def __init__(self, root_dir, input_size):
        """
        Interface to the ADCD data set
        :param root_dir: (string) path to directory containing ACDC training data
        :param input_size: (int, int) input size for the neural network. It should be the same across all data sets
        """
        self.input_size = input_size

        path_dict = dict()
        for d_set in ['train', 'validation']:
            path_list = []

            data_dir = root_dir + '/{0}'.format(d_set)
            suffix = '*/' if root_dir.endswith('/') else '/*/'
            subdir_list = [d[:-1] for d in glob(data_dir + suffix)]

            for subdir in subdir_list:
                folder_name = subdir.rsplit('/')[-1]
                if folder_name.startswith('patient'):
                    prefix = os.path.join(data_dir, folder_name)
                    pt_number = folder_name.split('patient')[1]
                    pt_full_path = os.path.join(prefix, 'patient' + pt_number + '_4d_preproc.npy')
                    path_list.append(pt_full_path)

            path_dict[d_set] = path_list

        self.x_train_paths = path_dict['train']
        self.x_validation_paths = path_dict['validation']

    @staticmethod
    def _py_data_augmentation_ops(x_batch):

        n_samples, rows, cols, channels = x_batch.shape
        center = (cols // 2, rows // 2)  # open cv requires swapped rows and cols

        # create and apply transformation matrix for each element
        x_batch_augmented = []
        for i in range(n_samples):
            curr_slice = x_batch[i, ..., 0]

            # sample transformation parameters for the current element of the batch
            tx, ty = np.random.randint(low=-10, high=10, size=2)
            scale = np.random.uniform(low=0.98, high=1.02)
            angle = np.random.uniform(low=-90.0, high=90.0)

            # transformation matrices:
            m1 = np.float32([[1, 0, tx], [0, 1, ty]])
            m2 = cv2.getRotationMatrix2D(center, angle, scale)

            # apply transformation
            transform_slice = cv2.warpAffine(curr_slice, m1, (cols, rows))
            transform_slice = cv2.warpAffine(transform_slice, m2, (cols, rows))

            x_batch_augmented.append(np.expand_dims(transform_slice, axis=-1))

        return np.array(x_batch_augmented)

    # @staticmethod
    # def _py_resize_2d_slices(batch, new_size):
    #     """
    #     Resize the batch frames on the first two axis.
    #     :param batch: (np.array) input batch of images, with shape [n_samples, width, height]
    #     :param new_size: (int, int) output size, with shape (N, M)
    #     :return: resized batch
    #     """
    #     n_samples, x, y = batch.shape
    #     output = []
    #     for k in range(n_samples):
    #         output.append(cv2.resize(batch[k, ...], (new_size[1], new_size[0])))
    #     return np.array(output)

    @staticmethod
    def _data_augmentation_ops(x_train):
        """ Data augmentation pipeline (to be applied on training samples)
        """
        # image distortions:
        x_train = tf.image.random_brightness(x_train, max_delta=0.025)
        x_train = tf.image.random_contrast(x_train, lower=0.95, upper=1.05)

        return tf.cast(x_train, tf.float32)

    def data_parser(self, filename, standardize=False, augment=True):
        """
        Given a subject, returns the sequence of frames for a random z coordinate
        :param filename: (str) path to the patient mri sequence
        :param standardize: (bool) if True, standardize input data
        :param augment: (bool) if True, perform data augmentation operations
        :return: (array) = numpy array with the frames on the first dimension, s.t.: [None, width, height]
        """
        # batch = slice_pre_processing_pipeline(filename.decode('utf-8'))
        # batch = np.expand_dims(batch, -1)

        batch = np.load(filename.decode('utf-8')).astype(np.float32)

        if standardize:
            print("Data won't be standardized, as they already have been pre-processed.")

        # # Undersample data:
        # new_size = [self.input_size[0] // 4, self.input_size[1] // 4]
        # batch = self._py_resize_2d_slices(np.squeeze(batch, axis=-1), new_size)
        # batch = np.expand_dims(batch, axis=-1)

        if augment:
            batch = self._py_data_augmentation_ops(batch).astype(np.float32)

        assert not np.any(np.isnan(batch))

        # batch = self._undersample(batch)
        return batch

    def get_data(self, b_size, augment=False, standardize=False, num_threads=4):
        """ Returns iterators on the dataset along with their initializers.
        :param b_size: batch size
        :param augment: if to perform data augmentation
        :param standardize: if to standardize the input data
        :param num_threads: for parallel computing
        :return: train_init, valid_init, input_data, label
        """
        with tf.name_scope('acdc_data'):

            _train_data = tf.constant(self.x_train_paths)
            _valid_data = tf.constant(self.x_validation_paths)

            train_data = tf.data.Dataset.from_tensor_slices(_train_data)
            train_data = train_data.map(
                lambda filename: tf.py_func(  # Parse the record into tensors
                    self.data_parser,
                    [filename, standardize, augment],
                    [tf.float32]), num_parallel_calls=num_threads)

            valid_data = tf.data.Dataset.from_tensor_slices(_valid_data)
            valid_data = valid_data.map(
                lambda filename: tf.py_func(  # Parse the record into tensors
                    self.data_parser,
                    [filename, standardize, False],
                    [tf.float32]), num_parallel_calls=num_threads)

            # - - - - - - - - - - - - - - - - - - - -

            if augment:
                train_data = train_data.map(self._data_augmentation_ops, num_parallel_calls=num_threads)
                valid_data = valid_data.map(lambda v: tf.cast(v, dtype=tf.float32), num_parallel_calls=num_threads)

            train_data = train_data.shuffle(buffer_size=len(self.x_train_paths))
            valid_data = valid_data.shuffle(buffer_size=len(self.x_validation_paths))
            # train_data = train_data.repeat()  # Repeat the input indefinitely

            # un-batch first, then batch the data
            train_data = train_data.apply(tf.data.experimental.unbatch())
            valid_data = valid_data.apply(tf.data.experimental.unbatch())

            train_data = train_data.batch(b_size, drop_remainder=True)  # TODO be aware of this
            valid_data = valid_data.batch(b_size, drop_remainder=True)

            if len(get_available_gpus()) > 0:
                if tf.__version__ < '1.7.0':
                    train_data = train_data.prefetch(buffer_size=2)  # buffer_size depends on the machine
                else:
                    train_data = train_data.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))
                    # train_data = train_data.apply(prefetching_ops.copy_to_device("/gpu:0")).prefetch(100 * b_size)  # 2)

            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

            _input_data = iterator.get_next()
            train_init = iterator.make_initializer(train_data)  # initializer for train_data
            valid_init = iterator.make_initializer(valid_data)  # initializer for test_data

            with tf.name_scope('input_unsup'):
                input_data = tf.reshape(_input_data, shape=[-1, self.input_size[0], self.input_size[1], 1])
                input_data += tf.random.normal(mean=0.0, stddev=0.02, shape=tf.shape(input_data))

            with tf.name_scope('output_unsup'):
                output_data = tf.reshape(_input_data, shape=[-1, self.input_size[0], self.input_size[1], 1])

            return train_init, valid_init, input_data, output_data
