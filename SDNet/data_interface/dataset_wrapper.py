"""
Wrapper to the dataset interfaces
"""
from data_interface.interfaces.acdc_temporal_frames_interface import DatasetInterface as ACDCTemporalInterface
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

    def get_acdc_sup_data(self, data_path):
        """
        wrapper to ACDC data set. Gets input images and annotated masks.
        :param data_path: (str) path to data directory
        :return: iterator initializer for train and valid data; input and output frame; time and delta time.
        """
        print('Define input pipeline for supervised data...')

        # initialize data set interfaces
        acdc_itf = ACDCSupInterface(data_path, self.input_size)

        train_init, valid_init, input_data, output_data = acdc_itf.get_data(
            b_size=self.batch_size,
            augment=self.augment,
            standardize=self.standardize,
            num_threads=self.num_threads)

        return train_init, valid_init, input_data, output_data

    def get_acdc_unsup_data(self, data_path):
        """
        wrapper to ACDC data set. Gets input images without ground truth mask. Notice that output_data is just an alias
        for input_data
        :param data_path: (str) path to data directory
        :return: iterator initializer for train and valid data; input and output frame; time and delta time.
        """
        print('Define input pipeline for unsupervised data...')

        # initialize data set interfaces
        acdc_itf = ACDCUnsupInterface(data_path, self.input_size)

        train_init, valid_init, input_data, output_data = acdc_itf.get_data(
            b_size=self.batch_size,
            augment=self.augment,
            standardize=self.standardize,
            num_threads=self.num_threads)

        return train_init, valid_init, input_data, output_data
