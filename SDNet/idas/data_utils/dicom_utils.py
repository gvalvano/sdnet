"""
Utilities for dicom data
"""

import dicom
import numpy as np


def get_dicom_matrix(filename, dtype=np.float):
    """ gets dicom matrix from a given filename. """
    mdicom = dicom.read_file(filename)
    return mdicom.pixel_array.astype(dtype)
