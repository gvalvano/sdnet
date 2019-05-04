"""
TODO: this file is intended to be modified depending on the application
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import nibabel as nib
import numpy as np
from utils import get_available_gpus


class DatasetInterface(object):
    @staticmethod
    def to_categorical(y, nb_classes):
        w, h = y.shape[0], y.shape[1]
        categ_array = np.zeros((w, h, nb_classes), dtype=bool)
        for c in range(nb_classes):
            _slice = np.zeros((w, h), dtype=bool)
            _slice[np.where(y == c)] = 1
            categ_array[:, :, c] = _slice
        return categ_array

    def _data_augmentation_ops(self, x_train, y_train):
        """ Data augmentation pipeline (to be applied on training samples)
        :param x_train: input matrix, dimension [-1, N, M, C]
        :param y_train: output matrix, dimension [-1, N, M, C]
        :return: data augmented samples
        """
        raise NotImplementedError
        # return x_train, y_train

    def _parse_nifti_data(self, path, augment=False, standardize=False):
        """ python function to parse nifti data to tf.data.Dataset object.
            augment: if True apply data augmentation
        """
        path_str = path.decode('utf-8')
        fname = path_str.split('.')[0]
        ext = '.nii.gz'
        volume = nib.load(fname + ext).get_data().astype(np.int16)
        mask = nib.load(fname + '_brainmask' + ext).get_data().astype(np.bool)

        dimens = volume.shape
        assert dimens == mask.shape

        if augment:  # data augmentation
            volume, mask = self._data_augmentation_ops(volume, mask)

        if standardize:
            volume = volume.astype(np.float32)
            volume -= np.mean(volume)
            volume /= np.std(volume)

        mask = mask.astype(np.bool)
        return volume, mask

    def get_data(self, list_of_files_train, list_of_files_valid, b_size, augment=False, standardize=False,
                 num_threads=4):
        """ Returns iterators on the dataset along with their initializers.
        :param list_of_files_train: list of strings, path for the train_set files
        :param list_of_files_valid: list of strings, path for the validation_set files
        :param b_size: batch size
        :param augment: if to perform data augmentation
        :param standardize: if to standardize the input data
        :param num_threads: for parallelization
        :return: train_init, valid_init, input_data, label
        """
        with tf.name_scope('data'):
            train_filenames = tf.constant(list_of_files_train)
            valid_filenames = tf.constant(list_of_files_valid)

            train_data = tf.data.Dataset.from_tensor_slices(train_filenames)
            train_data = train_data.map(lambda filename: tf.py_func(  # provare Dataset.from_generator
                self._parse_nifti_data,
                [filename, augment, standardize],
                [tf.float32, tf.bool]), num_parallel_calls=num_threads)

            valid_data = tf.data.Dataset.from_tensor_slices(valid_filenames)
            valid_data = valid_data.map(lambda filename: tf.py_func(
                self._parse_nifti_data,
                [filename, False, standardize],
                [tf.float32, tf.bool]), num_parallel_calls=num_threads)

            train_data.shuffle(buffer_size=len(list_of_files_train))

            train_data.batch(b_size)

            if len(get_available_gpus()) > 0:
                if tf.__version__ < '1.7.0':
                    train_data.prefetch(buffer_size=2),  # buffer_size dipende dal pc
                else:
                    train_data.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))  # sceglie automaticamente il buffer_size (tf>=1.7)

            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

            input_data, label = iterator.get_next()
            train_init = iterator.make_initializer(train_data)  # initializer for train_data
            valid_init = iterator.make_initializer(valid_data)  # initializer for test_data

            with tf.name_scope('input'):
                input_data = tf.reshape(input_data, shape=[-1, 256, 256, 1])
            with tf.name_scope('label'):
                label = tf.cast(tf.reshape(label, shape=[-1, 256, 256, 2]), tf.int8)

            return train_init, valid_init, input_data, label

