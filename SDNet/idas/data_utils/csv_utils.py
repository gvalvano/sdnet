"""
Utilities for .csv data
"""
import pandas as pd


def get_csv_data(filename, sep=','):
    """ Loads CSV into Numpy array."""
    data_frame = pd.read_csv(filename, sep=sep)
    return data_frame

#
# def save_csv_data(array, filename, sep=','):
#     """ Saves CSV data from Numpy array."""
#     data_frame = pd.DataFrame(data=, index=, columns=)
#     data_frame.to_csv(filename, sep=sep)
