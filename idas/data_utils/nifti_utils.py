"""
Utilities for nifti data
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

import nibabel as nib
import numpy as np


def get_nifti_matrix(filename, dtype=np.int16):
    """ Returns array from nifti filename and affine matrix. """
    array = nib.load(filename).get_data().astype(dtype)  # array
    affine = nib.load(filename).affine  # affine matrix
    return array, affine


def save_nifti_matrix(array, affine, filename, dtype=np.int16):
    """ Saves nifti array with a given affine matrix.
    Notice that the nifti file will be saved in the given dtype (default int16)"""
    nimage = nib.Nifti1Image(array.astype(dtype), affine)
    nib.save(nimage, filename=filename)
