"""
Wrapper to the dataset interfaces
"""
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

from data_interface.interfaces.acdc_sup_interface import DatasetInterface as ACDCSupInterface
from data_interface.interfaces.acdc_unsup_interface import DatasetInterface as ACDCUnsupInterface


class DatasetInterfaceWrapper(object):

    def __init__(self, augment, standardize, batch_size, input_size, num_threads):
        """
        Wrapper to the data set interfaces.
        :param augment: (bool) if True, perform data augmentation
        :param standardize: (bool) if True, standardize data as x_new = (x - mean(x))/std(x)
        :param batch_size: (int) batch size
        :param input_size: (int, int) tuple containing (image width, image height)
        :param num_threads: (int) number of parallel threads to run for CPU data pre-processing
        """
        # class variables
        self.augment = augment
        self.standardize = standardize
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_threads = num_threads

    def get_acdc_sup_data(self, data_path, repeat=False, seed=None):
        """
        wrapper to ACDC data set. Gets input images and annotated masks.
        :param data_path: (str) path to data directory
        :param repeat: (bool) whether to repeat the input indefinitely
        :param seed: (int or placeholder) seed for the random operations
        :return: iterator initializer for train valid, and test data; input and output frame; time and delta time.
        """
        print('Define input pipeline for supervised data...')

        # initialize data set interfaces
        acdc_itf = ACDCSupInterface(data_path, self.input_size)

        train_init, valid_init, test_init, input_data, output_data = acdc_itf.get_data(
            b_size=self.batch_size,
            augment=self.augment,
            standardize=self.standardize,
            repeat=repeat,
            num_threads=self.num_threads,
            seed=seed
        )

        return train_init, valid_init, test_init, input_data, output_data

    def get_acdc_unsup_data(self, data_path, repeat=False, seed=None):
        """
        wrapper to ACDC data set. Gets input images without ground truth mask. Notice that output_data is just an alias
        for input_data
        :param data_path: (str) path to data directory
        :param repeat: (bool) whether to repeat the input indefinitely
        :param seed: (int or placeholder) seed for the random operations
        :return: iterator initializer for train and valid data; input and output frame; time and delta time.
        """
        print('Define input pipeline for unsupervised data...')

        # initialize data set interfaces
        acdc_itf = ACDCUnsupInterface(data_path, self.input_size)

        train_init, valid_init, input_data, output_data = acdc_itf.get_data(
            b_size=self.batch_size,
            augment=self.augment,
            standardize=self.standardize,
            repeat=repeat,
            num_threads=self.num_threads,
            seed=seed
        )

        return train_init, valid_init, input_data, output_data
