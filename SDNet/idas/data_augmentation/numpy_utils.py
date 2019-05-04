import numpy as np


def np_zero_pad(array, size=None, reference=None, offset=None):
    """
    Zero-pads numpy array to the given size.
    :param array: array to be padded
    :param size: desired shape
    :param reference: if size is None, then the desired size is evaluated as the shape of the reference array
    :param offset: list of offsets (number of elements must be equal to the dimension of the array)
    :return:
    """
    # Create an array of zeros with the desired shape
    if size is None:
        output_shape = reference.shape
    else:
        output_shape = size

    result = np.zeros(output_shape)

    # if it is None, fill 'offset' variable with zeros along each dimension:
    if offset is None:
        offset = np.zeros(array.ndim)

    # Create a list of slices from offset to offset + shape in each dimension
    insert_here = [slice(offset[dim], offset[dim] + array.shape[dim]) for dim in range(array.ndim)]

    # Insert the array in the result at the specified offsets
    result[insert_here] = array

    return result
