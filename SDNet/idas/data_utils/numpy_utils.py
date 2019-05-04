"""
Utilities for numpy data
"""

import numpy as np


def get_numpy_matrix(path, shape, dtype=np.float32):
    np.fromfile(path, dtype).reshape(shape)

