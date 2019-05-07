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
from math import pi
from idas.utils import print_yellow_text


class DatasetInterface(object):

    def __init__(self, root_dir, input_size):
        """
        Interface to the ADCD data set
        :param root_dir: (string) path to directory containing ACDC training data
        :param input_size: (int, int) input size for the neural network. It should be the same across all data sets
        """
        self.input_size = input_size

        self.x_train = np.load(os.path.join(root_dir, 'preprocessed/{0}/sup_{0}.npy'.format('train')))
        self.y_train = np.load(os.path.join(root_dir, 'preprocessed/{0}/sup_mask_{0}.npy'.format('train')))
        self.x_validation = np.load(os.path.join(root_dir, 'preprocessed/{0}/sup_{0}.npy'.format('validation')))
        self.y_validation = np.load(os.path.join(root_dir, 'preprocessed/{0}/sup_mask_{0}.npy'.format('validation')))

        # self.x_train, self.y_train = self._undersample(self.x_train, self.y_train)
        # self.x_validation, self.y_validation = self._undersample(self.x_validation, self.y_validation)

    @staticmethod
    def _data_augmentation_ops(x_train, y_train):
        """ Data augmentation pipeline (to be applied on training samples)
        """
        angles = tf.random_uniform((1, 1), minval=-pi / 2, maxval=pi / 2)
        x_train = tf.contrib.image.rotate(x_train, angles[0], interpolation='BILINEAR')
        y_train = tf.contrib.image.rotate(y_train, angles[0], interpolation='NEAREST')

        translations = tf.random_uniform((1, 2), minval=-10, maxval=10)
        x_train = tf.contrib.image.translate(x_train, translations, interpolation='BILINEAR')
        y_train = tf.contrib.image.translate(y_train, translations, interpolation='NEAREST')

        # image distortions:
        x_train = tf.image.random_brightness(x_train, max_delta=0.025)
        x_train = tf.image.random_contrast(x_train, lower=0.95, upper=1.05)

        x_train = tf.cast(x_train, tf.float32)
        y_train = tf.cast(y_train, tf.float32)

        # add noise as regularizer
        std = 0.02  # data are standardized
        noise = tf.random_normal(shape=tf.shape(x_train), mean=0.0, stddev=std)
        x_train = x_train + noise

        return x_train, y_train

    def get_data(self, b_size, augment=False, standardize=False, num_threads=4):
        """ Returns iterators on the dataset along with their initializers.
        :param b_size: batch size
        :param augment: if to perform data augmentation
        :param standardize: if to standardize the input data
        :param num_threads: for parallel computing
        :return: train_init, valid_init, input_data, label
        """
        with tf.name_scope('acdc_data'):

            _train_images = tf.constant(self.x_train, dtype=tf.float32)
            _train_masks = tf.constant(self.y_train, dtype=tf.float32)
            _valid_images = tf.constant(self.x_validation, dtype=tf.float32)
            _valid_masks = tf.constant(self.y_validation, dtype=tf.float32)

            train_data = tf.data.Dataset.from_tensor_slices((_train_images, _train_masks))
            valid_data = tf.data.Dataset.from_tensor_slices((_valid_images, _valid_masks))

            if standardize:
                print("Data won't be standardized, as they already have been pre-processed.")

            if augment:
                train_data = train_data.map(self._data_augmentation_ops, num_parallel_calls=num_threads)

            train_data = train_data.shuffle(buffer_size=len(self.x_train))

            print_yellow_text(' --> Repeat the input indefinitely  = True', sep=False)
            train_data = train_data.repeat()  # Repeat the input indefinitely

            train_data = train_data.batch(b_size, drop_remainder=True)
            valid_data = valid_data.batch(b_size, drop_remainder=True)

            if len(get_available_gpus()) > 0:
                if tf.__version__ < '1.7.0':
                    train_data = train_data.prefetch(buffer_size=2)  # buffer_size depends on the machine
                else:
                    train_data = train_data.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))

            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

            _input_data, _output_data = iterator.get_next()
            train_init = iterator.make_initializer(train_data)  # initializer for train_data
            valid_init = iterator.make_initializer(valid_data)  # initializer for test_data

            with tf.name_scope('input_sup'):
                input_data = tf.reshape(_input_data, shape=[-1, self.input_size[0], self.input_size[1], 1])
                input_data = tf.cast(input_data, tf.float32)

            with tf.name_scope('output_sup'):
                output_data = tf.reshape(_output_data, shape=[-1, self.input_size[0], self.input_size[1], 4])
                output_data = tf.cast(output_data, tf.float32)

            return train_init, valid_init, input_data, output_data
