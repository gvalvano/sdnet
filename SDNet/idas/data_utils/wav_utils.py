"""
Utilities for audio .wav data
"""

from scipy.io import wavfile


def get_nifti_matrix(filename):
    """ Returns sampling frequency and 1D array data from .wav filename. """
    freq, data = wavfile.read(filename)
    return data, freq
