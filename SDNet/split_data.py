"""
Running this file will split the volumes under './data/acdc_data/training/' in train, validation and test sets.
Splits are random.
"""
import os
from glob import glob
import random

# - - - - - - - - - - - - - - - - - - - - - - - - - - -
root = './data/acdc_data/training/*'
n_train = 60
n_valid = 20
n_test = 20
# - - - - - - - - - - - - - - - - - - - - - - - - - - -

subdir_list = [d for d in glob(root)]
random.shuffle(subdir_list)

for _dir in subdir_list:
    if n_train > 0:
        os.system('mv {0} {1}'.format(_dir, 'train'))
        n_train -= 1
    else:
        if n_valid > 0:
            os.system('mv {0} {1}'.format(_dir, 'validation'))
            n_valid -= 1
        else:
            if n_test > 0:
                os.system('mv {0} {1}'.format(_dir, 'test'))
                n_test -= 1
