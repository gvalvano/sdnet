"""
Utilities for .jpg and .jpeg data
"""

from PIL import Image
import numpy as np


def get_jpg_image(filename):
    """ Loads JPEG image into 3D Numpy array of shape (width, height, channels)."""
    with Image.open(filename) as image:
        im_arr = np.array(image)
    return im_arr


def save_jpg_image(array, filename):
    """ Saves JPEG image from array 3D Numpy array of shape (width, height, channels)."""
    img = Image.fromarray(array)
    img.save(filename)
