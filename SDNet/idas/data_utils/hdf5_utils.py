"""
Utilities for hdf5 data
"""
# TODO: file to be completed

import h5py
import os


def create_hdf5_db(x_train, y_train, x_validation, y_validation, x_test, y_test, db_name='data.h5'):
    """ Creates hdf5 database. """
    print("Building database: " + db_name)

    # Create a hdf5 dataset
    h5f = h5py.File(db_name, 'w')
    h5f.create_dataset('x_train', data=x_train)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('x_validation', data=x_validation)
    h5f.create_dataset('y_validation', data=y_validation)
    h5f.create_dataset('x_test', data=x_test)
    h5f.create_dataset('y_test', data=y_test)
    h5f.close()
    print("Done.")


def get_data(db_name, key):
    """ Returns what is behind key node on the HDF5 db named db_name. """
    # Load hdf5 dataset
    hdf5 = h5py.File(db_name, 'r')
    data = hdf5[key]  # i.e. xt = h5f['x_train']
    # TODO: note that the database remains open
    return data


def add_node(db_name, key, shape):
    """ Add node with name key to hdf5 database. """
    if not os.path.isfile(db_name):
        h5f = h5py.File(db_name, 'w')
    else:
        h5f = h5py.File(db_name, 'r+')

    h5f.create_dataset(key, shape, maxshape=(None, 1))
    h5f.close()


def update_node(db_name, key):
    """ Change the content of a node. """
    pass


def add_elements_to_existing_node(db_name, key,):
    """  Add elements below the node. """
    #h5f = h5py.File(db_name, 'r+')
    #h5f[key].resize((curr_num_samples, dimPatches, dimPatches, n_channel))
    #h5f[key][curr_num_samples - 1, :, :, :] = imgMatrix
    #h5f.close()
    pass
