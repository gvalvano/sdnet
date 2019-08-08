"""
Running this file will split the volumes under './data/acdc_data/training/' in train, validation and test sets.
Splits are random.
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

import os
from glob import glob
import random
from idas.utils import safe_mkdir, print_yellow_text

#  - - - - - - - - - - - - - - - - - - - - - - - - - - -
SPLIT_NUMBER = 0  # <---- change this to obtain a different split of the data set
assert SPLIT_NUMBER in [0, 1, 2, 3, 4]
root = './data/acdc_data/'
n_train = 60
n_valid = 20
n_test = 20
#  - - - - - - - - - - - - - - - - - - - - - - - - - - -


def get_splits():
    """""
    Returns an array of splits into validation, test and train indices. Indices are the same of the original paper ones.
    For details, please refer to:
      "Factorised Representation Learning in Cardiac Image Analysis" (2019), arXiv preprint arXiv:1903.09467
      Chartsias, A., Joyce, T., Papanastasiou, G., Williams, M., Newby, D., Dharmakumar, R., & Tsaftaris, S. A.
    """

    splits = [
        {'validation': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
         'test': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
         'train': [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                      47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                      64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                      81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
         },
        {'validation': [85, 13, 9, 74, 73, 68, 59, 79, 47, 80, 14, 95, 25, 92, 87],
         'test': [54, 55, 99, 63, 91, 24, 51, 3, 64, 43, 61, 66, 96, 27, 76],
         'train': [46, 57, 49, 34, 17, 8, 19, 28, 97, 1, 90, 22, 88, 45, 12, 4, 5,
                      75, 53, 94, 62, 86, 35, 58, 82, 37, 84, 93, 6, 33, 15, 81, 23, 48,
                      71, 70, 11, 77, 36, 60, 31, 65, 32, 78, 98, 52, 100, 42, 38, 2, 20,
                      69, 26, 18, 40, 50, 16, 7, 41, 10, 83, 21, 39, 72, 56, 67, 44, 30, 89, 29]
         },
        {'validation': [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
         'test': [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78],
         'train': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                      100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 62, 63,
                      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 79, 80,
                      81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
         },
        {'validation': [20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 37, 33, 34, 35, 36],
         'test': [38, 39, 40, 41, 43, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55],
         'train': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 32, 42, 44, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                      71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                      91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
         },
        {'validation': [11, 12, 13, 14, 15],
         'test': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
         'train': [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                      47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                      64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                      81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                      1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
         }
    ]
    return splits


def random_split(root_dir, n_tr, n_val, n_tst):
    """
    Create a new random split.
    :param root_dir: root with all the data
    :param n_tr: number of training samples
    :param n_val: number of validation samples
    :param n_tst: number of test samples
    :return:
    """
    subdir_list = [d for d in glob(root_dir)]
    random.shuffle(subdir_list)

    for _dir in subdir_list:
        if n_tr > 0:
            os.system('mv {0} {1}'.format(_dir, 'train'))
            n_tr -= 1
        else:
            if n_val > 0:
                os.system('mv {0} {1}'.format(_dir, 'validation'))
                n_val -= 1
            else:
                if n_tst > 0:
                    os.system('mv {0} {1}'.format(_dir, 'test'))
                    n_tst -= 1


if __name__ == '__main__':

    # # create a random split:
    # random_split(root, n_train, n_valid, n_test)

    # - - - - - - - - - - - - - - - - - - - - - - - -
    default_data_folder = 'training'

    print_yellow_text('Splitting the data in: train, validation, test ...', sep=False)

    if SPLIT_NUMBER > 0:  # then move again the files under the folder "training"
        safe_mkdir(root + default_data_folder)
        for source_dir in ['train', 'validation', 'test']:
            os.system('mv {0}{1}/* {2}'.format(root, source_dir, '{0}{1}'.format(root, default_data_folder)))

    # create splits according to the original paper:
    splits_ids = get_splits()[SPLIT_NUMBER]

    subdir_list = [d for d in glob(root + default_data_folder + '/*')]
    for dset in ['train', 'validation', 'test']:
        safe_mkdir(root + dset)
        ids = splits_ids[dset]
        ids_list = [str(el).zfill(3) for el in ids]
        path_list = [el for el in subdir_list if el.rsplit('/patient')[-1] in ids_list]
        for _dir in path_list:
            os.system('mv {0} {1}'.format(_dir, '{0}{1}'.format(root, dset)))

    print_yellow_text('Done.', sep=False)
