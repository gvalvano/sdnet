import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from idas.data_augmentation.transformations import *
import nibabel as nib
import tensorflow as tf


def one_hot_encode(y, nb_classes):
    y_shape = list(y.shape)
    y_shape.append(nb_classes)
    with tf.Session() as sess:
        res = sess.run(tf.one_hot(indices=y, depth=nb_classes))
    return res.reshape(y_shape)


if __name__ == '__main__':

    fnames = ['NFBS_Dataset/volumes/train/A00028185/sub-A00028185_ses-NFB3_T1w']

    x_batch = []
    y_batch = []
    for name in fnames:
        x_batch.append(nib.load(name + '.nii.gz').get_data().astype(np.float32))
        y_batch.append(nib.load(name + '_brainmask.nii.gz').get_data().astype(np.int16))
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)

    n_samples, dim0, dim1, dim2 = x_batch.shape

    print('n_samples = \t', n_samples)
    print('shapes = \t', dim0, dim1, dim2)

    # angle_z = np.random.uniform(0, 360)
    # angle_y = np.random.uniform(0, 360)
    # angle_x = np.random.uniform(0, 360)
    #
    # print('angles = \t', angle_x, angle_y, angle_z)
    for i in range(len(fnames)):
        vol = x_batch[i, :, :, :]
        mask = y_batch[i, :, :, :]

        coords = np.meshgrid(np.arange(dim0), np.arange(dim1), np.arange(dim2))

        # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
        xyz = np.vstack([coords[0].reshape(-1)-float(dim0)/2,         # x coordinate, centered
                         coords[1].reshape(-1)-float(dim1)/2,         # y coordinate, centered
                         coords[2].reshape(-1)-float(dim2)/2,         # z coordinate, centered
                         np.ones((dim0, dim1, dim2)).reshape(-1)])    # 1 for homogeneous coordinates

        # create transformation matrix
        origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]

        scale_factor = np.random.uniform(low=0.95, high=1.05)
        trasl_factor = np.random.uniform(low=-5, high=5, size=3)
        shear_factor = np.random.uniform(low=0, high=np.pi / 45)  # 0°, +4°
        alpha = np.random.uniform(low=-np.pi / 18, high=np.pi / 18)  # -10°, +10°
        beta = np.random.uniform(low=-np.pi / 18, high=np.pi / 18)  # -10°, +10°
        gamma = np.random.uniform(low=-np.pi / 18, high=np.pi / 18)  # -10°, +10°

        S = scale_matrix(scale_factor, origin)
        T = translation_matrix(trasl_factor)
        Z = shear_matrix(shear_factor, xaxis, origin, yaxis)

        R = concatenate_matrices(
            rotation_matrix(alpha, xaxis),
            rotation_matrix(beta, yaxis),
            rotation_matrix(gamma, zaxis))

        refl_axis = np.random.permutation(np.arange(6))
        refl = [reflection_matrix(origin, xaxis),
                reflection_matrix(origin, yaxis),
                reflection_matrix(origin, zaxis),
                identity_matrix(),
                identity_matrix(),
                identity_matrix()]
        Refl = concatenate_matrices(refl[refl_axis[0]], refl[refl_axis[1]], refl[refl_axis[2]])

        M = concatenate_matrices(Refl, T, R, Z, S)

        # apply transformation
        transformed_xyz = np.dot(M, xyz)

        # extract coordinates, don't use transformed_xyz[3,:] that's the homogeneous coordinate, always 1
        x = transformed_xyz[0, :] + float(dim0) / 2
        y = transformed_xyz[1, :] + float(dim1) / 2
        z = transformed_xyz[2, :] + float(dim2) / 2

        x = x.reshape((dim0, dim1, dim2))
        y = y.reshape((dim0, dim1, dim2))
        z = z.reshape((dim0, dim1, dim2))

        # the coordinate system seems to be strange, it has to be ordered like this
        new_xyz = [y, x, z]

        # sample
        new_vol = scipy.ndimage.map_coordinates(vol, new_xyz, order=0)
        new_mask = scipy.ndimage.map_coordinates(mask, new_xyz, order=0)
        new_mask = one_hot_encode(new_mask, nb_classes=2)

    fig, a = plt.subplots(2, 2)
    a[0, 0].set_title('vol')
    a[0, 1].set_title('vol + mask')
    a[1, 0].set_title('new(vol)')
    a[1, 1].set_title('new(vol + mask)')
    for k in range(0, dim2, 5):
        a[0, 0].imshow(vol[:, :, k], cmap='gray')
        a[0, 1].imshow(vol[:, :, k], cmap='gray')
        a[0, 1].imshow(mask[:, :, k], alpha=0.2)

        a[1, 0].imshow(new_vol[:, :, k], cmap='gray')
        a[1, 1].imshow(new_vol[:, :, k], cmap='gray')
        a[1, 1].imshow(new_mask[:, :, k, 0], alpha=0.2)
        plt.pause(1e-16)

