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
from idas.utils import print_yellow_text

#  - - - - - - - - - - - - - - - - - - - - - - - - - - -
root = './data/acdc_data/'
#  - - - - - - - - - - - - - - - - - - - - - - - - - - -


def get_splits():
    """""
    Returns an array of splits into validation, test and train indices. Indices are the same of the original paper ones.
    For details, please refer to:
      "Factorised Representation Learning in Cardiac Image Analysis" (2019), arXiv preprint arXiv:1903.09467
      Chartsias, A., Joyce, T., Papanastasiou, G., Williams, M., Newby, D., Dharmakumar, R., & Tsaftaris, S. A.
    """

    splits = [
        # ------------------------------------------------------------------------------------------------------------
        # 100% of the training data (70 samples):

        {'validation': [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95],
         'test': [5, 39, 77, 82, 78, 10, 64, 24, 30, 73, 80, 41, 36, 60, 72],
         'train_unsup': [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89, 71, 6, 52, 43,
                         45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40, 7, 13, 47,
                         55, 12, 58, 87, 9, 65, 62, 33, 42, 23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15],
         'train_tframe': [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89, 71, 6, 52, 43,
                          45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40, 7, 13, 47,
                          55, 12, 58, 87, 9, 65, 62, 33, 42, 23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15],
         'train_disc': [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89, 71, 6, 52, 43,
                        45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40, 7, 13, 47,
                        55, 12, 58, 87, 9, 65, 62, 33, 42, 23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15],
         'train_sup': [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89, 71, 6, 52, 43,
                       45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40, 7, 13, 47,
                       55, 12, 58, 87, 9, 65, 62, 33, 42, 23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]
         }
    ]
    return splits


if __name__ == '__main__':

    default_data_folder = 'training'

    print_yellow_text('\nSplitting the data in: train, validation, test ...', sep=False)

    # create split
    splits_ids = get_splits()[0]
    subdir_list = [d for d in glob(root + default_data_folder + '/*')]

    dset_list = ['train_sup', 'train_disc', 'train_unsup', 'validation', 'test']

    for dset in dset_list:
        try:
            os.makedirs(root + dset)
        except FileExistsError:
            os.system('rm -rf {0}'.format(root + dset))
            os.makedirs(root + dset)
        ids = splits_ids[dset]
        ids_list = [str(el).zfill(3) for el in ids]
        path_list = [el for el in subdir_list if el.rsplit('/patient')[-1] in ids_list]
        for _dir in path_list:
            print('cp -r {0} {1}'.format(_dir, '{0}{1}/'.format(root, dset)))
            os.system('cp -r {0} {1}'.format(_dir, '{0}{1}/'.format(root, dset)))

    print_yellow_text('Done.\n', sep=False)
