import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# arrays with shape [None, width, height, channels]
# the mean width and height and the mean spatial resolutions for the ACDC data set are (rounded to the closest int):
mean_width = 227.03  # values computed after resizing the data set to the same resolution
mean_height = 254.88  # values computed after resizing the data set to the same resolution
mean_dx = 1.5117105
mean_dy = 1.5117105

# unet input dimensions must be fully divisible for 16, so find the multiple of 16 which is closest to mean width
# and variance and the data:
img_width = int(np.round(mean_width / 16) * 16)
img_height = int(np.round(mean_height / 16) * 16)
img_dx = mean_dx
img_dy = mean_dy

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def get_independent_slices(incoming):
    """
    Reshape the input array so that has every frame on axis 0. Temporal frames are treated as different spatial frames.
    :param incoming: np.array of shape [width, height, depth, time]
    :return: [depth * time, width, height]
    """
    try:
        x, y, z, t = incoming.shape
        incoming = np.transpose(incoming, (2, 3, 0, 1))
        incoming = np.reshape(incoming, (z * t, x, y))
    except:
        x, y, z = incoming.shape
        incoming = np.transpose(incoming, (2, 0, 1))
    return incoming


def resize_2d_slices(batch, new_size, interpolation):
    """
    Resize the frames
    :param batch: [np.array] input batch of images, with shape [n_batches, width, height]
    :param new_size: [int, int] output size, with shape (N, M)
    :param interpolation: interpolation type
    :return: resized batch, with shape (n_batches, N, M)
    """
    n_batches, x, y = batch.shape
    output = []
    for k in range(n_batches):
        output.append(cv2.resize(batch[k], (new_size[1], new_size[0]), interpolation=interpolation))
    return np.array(output)


def crop_or_pad_slice_center(batch, new_size):
    """
    For every image in the batch, crop the image in the center so that it has a final size = new_size
    :param batch: [np.array] input batch of images, with shape [n_batches, width, height]
    :param new_size: [int, int] output size, with shape (N, M)
    :return: cropped batch, with shape (n_batches, N, M)
    """
    # pad always and then crop to the correct size:
    n_batches, x, y = batch.shape
    pad_0 = (0, 0)
    pad_1 = (int(np.ceil(max(0, img_width - x)/2)), int(np.floor(max(0, img_width - x)/2)))
    pad_2 = (int(np.ceil(max(0, img_height - y)/2)), int(np.floor(max(0, img_height - y)/2)))
    batch = np.pad(batch, (pad_0, pad_1, pad_2), mode='constant', constant_values=0)

    # delta along axis and central coordinates
    n_batches, x, y = batch.shape
    delta_x = new_size[0] // 2
    delta_y = new_size[1] // 2
    x0 = x // 2
    y0 = y // 2

    output = []
    for k in range(n_batches):
        output.append(batch[k,
                      x0 - delta_x: x0 + delta_x,
                      y0 - delta_y: y0 + delta_y])
    return np.array(output)


def standardize_and_clip(batch):
    """
    Standardize the input batch as x = (x - m)/s, where m = median, s = inter-quartile distance.
    The batch images are also clipped to be in the interval 2nd - 98th percentile.
    :param batch: (np.array) batch with images stacked on axis 0. Has shape [None, width, height].
    :return: transformed batch.
    """
    m = np.percentile(batch, 50)
    s = np.percentile(batch, 75) - np.percentile(batch, 25)
    lower_limit = np.percentile(batch, 2)
    upper_limit = np.percentile(batch, 98)

    batch = np.clip(batch, a_min=lower_limit, a_max=upper_limit)
    batch = (batch - m) / (s + 1e-12)

    assert not np.any(np.isnan(batch))
    return batch


def parse_info_cfg(filename):
    """
    Extracts information contained in the Info.cfg file given as input.
    :param filename: path/to/patient/folder/Info.cfg
    :return: values for: ed, es, group, h, nf, w
    """
    ed, es, group, h, nf, w = None, None, None, None, None, None
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('ED: '):
                ed = int(line.split('ED: ')[1])

            elif line.startswith('ES: '):
                es = int(line.split('ES: ')[1])

            elif line.startswith('Group: '):
                group = line.split('Group: ')[1]

            elif line.startswith('Height: '):
                h = float(line.split('Height: ')[1])

            elif line.startswith('NbFrame: '):
                nf = int(line.split('NbFrame: ')[1])

            elif line.startswith('Weight: '):
                w = float(line.split('Weight: ')[1])

    assert all(v is not None for v in [ed, es, group, h, nf, w])

    return ed, es, group, h, nf, w


def one_hot_encode(y, nb_classes):
    y_shape = list(y.shape)
    y_shape.append(nb_classes)
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            res = sess.run(tf.one_hot(indices=y, depth=nb_classes))
    return res.reshape(y_shape)


def slice_pre_processing_pipeline(filename):
    """ Pre-processing pipeline.
     With respect to mask_pre_processing_pipeline():
            point 7 uses bi-cubic interpolation and point 9 is performed
    """
    # 1. load nifti file
    img = nib.load(filename)

    # 2. get image resolution on the slice axis x and y
    header = img.header
    try:
        dx, dy, dz, dt = header.get_zooms()
    except:
        dx, dy, dz = header.get_zooms()

    # 3. evaluate scaling factors to get to that resolution
    scale_x = dx / img_dx
    scale_y = dy / img_dy

    # 4. evaluate output shape after rescaling
    shape = img.shape
    x_max_scaled = int(scale_x * shape[0])
    y_max_scaled = int(scale_y * shape[1])

    # 5. get array
    img_array = img.get_data()

    # 6. put all the slices on the first axis
    img_array = get_independent_slices(img_array)

    # 7. interpolate to obtain output shapes computed at 4.
    img_array = resize_2d_slices(img_array, new_size=(x_max_scaled, y_max_scaled), interpolation=cv2.INTER_CUBIC)

    # 8. crop to maximum size
    img_array = crop_or_pad_slice_center(img_array, new_size=(img_width, img_height))

    # 9. standardize and clip values out of +- 3 standard deviations
    img_array = standardize_and_clip(img_array)

    return img_array


def mask_pre_processing_pipeline(filename, one_hot):
    """ Pre-processing pipeline.
     With respect to slice_pre_processing_pipeline():
            point 7 uses nearest neighbour interpolation and point 9 is substituted with a one-hot encoding operation
    one_hot: (bool) if True, one-hot encode the output mask
    """
    # 1. load nifti file
    img = nib.load(filename)

    # 2. get image resolution on the slice axis x and y
    header = img.header
    try:
        dx, dy, dz, dt = header.get_zooms()
    except:
        dx, dy, dz = header.get_zooms()

    # 3. evaluate scaling factors to get to that resolution
    scale_x = dx / img_dx
    scale_y = dy / img_dy

    # 4. evaluate output shape after rescaling
    shape = img.shape
    x_max_scaled = int(scale_x * shape[0])
    y_max_scaled = int(scale_y * shape[1])

    # 5. get array
    img_array = img.get_data()

    # 6. put all the slices on the first axis
    img_array = get_independent_slices(img_array)

    # 7. interpolate to obtain output shapes computed at 4.
    img_array = resize_2d_slices(img_array, new_size=(x_max_scaled, y_max_scaled), interpolation=cv2.INTER_NEAREST)

    # 8. crop to maximum size
    img_array = crop_or_pad_slice_center(img_array, new_size=(img_width, img_height))

    # 9. one-hot encode: 4 classes (background + heart structures)
    if one_hot:
        img_array = one_hot_encode(img_array, 4)

    return img_array

