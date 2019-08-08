"""
_______________________________________________________________________________________________________________________

N.B. this will load the ACDC data set in your RAM memory. Despite it is a small data set, be aware that you need enough
    free memory.
_______________________________________________________________________________________________________________________

Running this file you will create pre-processed volumes to use to train the SDNet (useful to avoid to overload the CPU
during training). In this way the pre-processing will be entirely off-line. Data augmentation is instead performed at
run time.

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

import numpy as np
import nibabel as nib
from glob import glob
from idas.utils import safe_mkdir
import os
import cv2
import tensorflow as tf
from sklearn.utils import shuffle


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# data set dirs:

source_dir = './data/acdc_data/'  # + '_tmp/'
dest_dir = './data/acdc_data/preprocessed/'  # + '_tmp/'

for subdir in ['train', 'validation', 'test']:
    safe_mkdir(os.path.join(dest_dir, subdir))

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

final_shape = (128, 128)  # (128, 128)

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

    return np.array(incoming)


def remove_empty_slices(image, mask=None):
    """ remove empty slices """
    image_no_empty = []
    if mask is not None:
        mask_no_empty = []

    n_frames = len(image)
    for k in range(n_frames):
        img = image[k, ...]
        if img.max() == img.min():
            print('Skipping blank images')
            continue
        else:
            image_no_empty.append(img)
            if mask is not None:
                mask_no_empty.append(mask[k, ...])

    if mask is not None:
        return np.array(image_no_empty), np.array(mask_no_empty)
    else:
        return np.array(image_no_empty)


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
        img = cv2.resize(batch[k], (new_size[1], new_size[0]), interpolation=interpolation)
        output.append(img)
    return np.array(output)


def crop_or_pad_slice_center(batch, new_size, value):
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

    if value is 'mean':
        batch = np.pad(batch, (pad_0, pad_1, pad_2), mode='mean')
    elif value == 'min':
        batch = np.pad(batch, (pad_0, pad_1, pad_2), mode='minimum')
    else:
        c_value = value
        batch = np.pad(batch, (pad_0, pad_1, pad_2), mode='constant', constant_values=c_value)

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
    The batch images are also clipped to be in the interval 5th - 95th percentile.
    :param batch: (np.array) batch with images stacked on axis 0. Has shape [None, width, height].
    :return: transformed batch.
    """
    m = np.percentile(batch, 50)
    s = np.percentile(batch, 75) - np.percentile(batch, 25)
    lower_limit = np.percentile(batch, 10)
    upper_limit = np.percentile(batch, 90)

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
    size = (x_max_scaled, y_max_scaled)
    img_array = resize_2d_slices(img_array, new_size=size, interpolation=cv2.INTER_CUBIC)

    # 8. crop to maximum size
    size = (img_width, img_height)
    img_array = crop_or_pad_slice_center(img_array, new_size=size, value='min')

    # 8b. undersample and make 64x64
    img_array = resize_2d_slices(img_array, new_size=final_shape, interpolation=cv2.INTER_CUBIC)
    img_array = crop_or_pad_slice_center(img_array, new_size=final_shape, value='min')

    # 9. standardize and clip values out of +- 3 standard deviations
    img_array = standardize_and_clip(img_array)

    return img_array


def tframe_pre_processing_pipeline(filename):
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

    # 6. put all the slices on the first axis, times on the last one
    img_array = np.transpose(img_array, (2, 0, 1, 3))
    _img_array = []

    for t in range(img_array.shape[-1]):
        img = img_array[..., t]

        # 7. interpolate to obtain output shapes computed at 4.
        size = (x_max_scaled, y_max_scaled)
        img = resize_2d_slices(img, new_size=size, interpolation=cv2.INTER_CUBIC)

        # 8. crop or pad to maximum size
        size = (img_width, img_height)
        img = crop_or_pad_slice_center(img, new_size=size, value='min')

        # 8b. undersample and make final_shape
        img = resize_2d_slices(img, new_size=final_shape, interpolation=cv2.INTER_CUBIC)
        img = crop_or_pad_slice_center(img, new_size=final_shape, value='min')

        # 9. standardize and clip values out of +- 3 standard deviations
        img = standardize_and_clip(img)

        _img_array.append(img)
    img_array = np.array(_img_array)

    # put time on last index again
    img_array = np.transpose(img_array, (1, 2, 3, 0))

    return img_array


def mask_pre_processing_pipeline(filename):
    """ Pre-processing pipeline.
     With respect to slice_pre_processing_pipeline():
            point 7 uses nearest neighbour interpolation and point 9 is substituted with a one-hot encoding operation
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
    size = (x_max_scaled, y_max_scaled)
    img_array = resize_2d_slices(img_array, new_size=size, interpolation=cv2.INTER_NEAREST)

    # 8. crop to maximum size
    size = (img_width, img_height)
    img_array = crop_or_pad_slice_center(img_array, new_size=size, value=0)

    # 8b. undersample and make 64x64
    img_array = resize_2d_slices(img_array, new_size=final_shape, interpolation=cv2.INTER_NEAREST)
    img_array = crop_or_pad_slice_center(img_array, new_size=final_shape, value=0)

    # 9. one-hot encode: 4 classes (background + heart structures)
    img_array = one_hot_encode(img_array, 4)

    return img_array


def build_unsup_sets():
    for set in ['train', 'validation', 'test']:
        print('  | Processing {0} data...'.format(set))

        root_dir = source_dir + set

        suffix = '*/' if root_dir.endswith('/') else '/*/'
        subdir_list = [d[:-1] for d in glob(root_dir + suffix)]

        stack = []
        for subdir in subdir_list:
            folder_name = subdir.rsplit('/')[-1]
            if folder_name.startswith('patient'):
                prefix = os.path.join(root_dir, folder_name)
                pt_number = folder_name.split('patient')[1]
                pt_full_path = os.path.join(prefix, 'patient' + pt_number + '_4d.nii.gz')

                # Pre-process image and add to the stack
                img_array = slice_pre_processing_pipeline(pt_full_path)
                img_array = np.expand_dims(img_array, -1)

                np.save(os.path.join(prefix, 'patient' + pt_number + '_4d_preproc.npy'), img_array)
                stack.extend(img_array)

        # define array
        stack_array = np.array(stack)

        # remove empty slices:
        stack_array = remove_empty_slices(stack_array)

        # shuffle training set and save
        stack_array = shuffle(stack_array)

        np.save(os.path.join(dest_dir, '{0}/unsup_{0}.npy'.format(set)), stack_array)


def build_sup_sets():

    for set in ['train', 'validation', 'test']:
        print('  | Processing {0} data...'.format(set))

        root_dir = source_dir + set

        suffix = '*/' if root_dir.endswith('/') else '/*/'
        subdir_list = [d[:-1] for d in glob(root_dir + suffix)]

        stack = []
        stack_masks = []
        for subdir in subdir_list:
            folder_name = subdir.rsplit('/')[-1]
            if folder_name.startswith('patient'):
                prefix = os.path.join(root_dir, folder_name)
                pt_number = folder_name.split('patient')[1]

                ed, es, _, _, _, _ = parse_info_cfg(prefix + '/Info.cfg')
                pt_ed_full_path = os.path.join(prefix, 'patient' + pt_number + '_frame{0}.nii.gz'.format(str(ed).zfill(2)))
                pt_es_full_path = os.path.join(prefix, 'patient' + pt_number + '_frame{0}.nii.gz'.format(str(es).zfill(2)))
                pt_ed_mask_full_path = os.path.join(prefix, 'patient' + pt_number + '_frame{0}_gt.nii.gz'.format(str(ed).zfill(2)))
                pt_es_mask_full_path = os.path.join(prefix, 'patient' + pt_number + '_frame{0}_gt.nii.gz'.format(str(es).zfill(2)))

                # Pre-process image and add to the stack
                img_array = slice_pre_processing_pipeline(pt_ed_full_path)
                img_array = np.expand_dims(img_array, -1)
                stack.extend(img_array)
                mask = mask_pre_processing_pipeline(pt_ed_mask_full_path)
                stack_masks.extend(mask)

                np.save(os.path.join(prefix, 'patient' + pt_number + '_frame{0}_preproc.npy'.format(str(ed).zfill(2))), img_array)
                np.save(os.path.join(prefix, 'patient' + pt_number + '_frame{0}_gt_preproc.npy'.format(str(ed).zfill(2))), mask)

                # Pre-process image and add to the stack
                img_array = slice_pre_processing_pipeline(pt_es_full_path)
                img_array = np.expand_dims(img_array, -1)
                stack.extend(img_array)
                mask = mask_pre_processing_pipeline(pt_es_mask_full_path)
                stack_masks.extend(mask)

                np.save(os.path.join(prefix, 'patient' + pt_number + '_frame{0}_preproc.npy'.format(str(es).zfill(2))), img_array)
                np.save(os.path.join(prefix, 'patient' + pt_number + '_frame{0}_gt_preproc.npy'.format(str(es).zfill(2))), mask)

        # define array
        stack_array = np.array(stack)
        stack_mask_array = np.array(stack_masks)

        # remove empty slices:
        stack_array, stack_mask_array = remove_empty_slices(stack_array, stack_mask_array)

        # shuffle training set and save
        stack_array, stack_mask_array = shuffle(stack_array, stack_mask_array)

        np.save(os.path.join(dest_dir, '{0}/sup_{0}.npy'.format(set)), stack_array)
        np.save(os.path.join(dest_dir, '{0}/sup_mask_{0}.npy'.format(set)), stack_mask_array)


def build_tframe_sets():
    # output array: [slice, rows, cols, 1, time]

    for set in ['train', 'validation', 'test']:
        print('  | Processing {0} data...'.format(set))

        root_dir = source_dir + set

        suffix = '*/' if root_dir.endswith('/') else '/*/'
        subdir_list = [d[:-1] for d in glob(root_dir + suffix)]

        stack = []
        for subdir in subdir_list:
            folder_name = subdir.rsplit('/')[-1]
            if folder_name.startswith('patient'):
                prefix = os.path.join(root_dir, folder_name)
                pt_number = folder_name.split('patient')[1]
                pt_full_path = os.path.join(prefix, 'patient' + pt_number + '_4d.nii.gz')

                # Pre-process image and add to the stack
                img_array = tframe_pre_processing_pipeline(pt_full_path)

                # remove empty slices:
                img_array = remove_empty_slices(img_array)

                img_array = np.expand_dims(img_array, axis=-2)

                np.save(os.path.join(prefix, 'patient' + pt_number + '_4d_preproc_tframe.npy'), img_array)
                stack.extend(img_array)


def main():
    print('\nBuilding SUPERVISED sets.')
    build_sup_sets()
    print('\nBuilding UNSUPERVISED sets.')
    build_unsup_sets()
    print('\nBuilding T-FRAME sets.')
    build_tframe_sets()
    print('\nEnd.')


if __name__ == '__main__':
    main()
