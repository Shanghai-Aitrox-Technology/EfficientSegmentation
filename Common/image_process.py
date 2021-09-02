import numpy as np


def voxel_coord_2_world(voxel_coord, origin, spacing, directions=None):
    if directions is None:
        directions = np.array([1]*len(voxel_coord))
    stretched_voxel_coord = np.array(voxel_coord) * np.array(spacing).astype(float)
    world_coord = np.array(origin) + stretched_voxel_coord * np.array(directions)

    return world_coord


def world_2_voxel_coord(world_coord, origin, spacing):
    stretched_voxel_coord = np.array(world_coord) - np.array(origin)
    voxel_coord = np.absolute(stretched_voxel_coord / np.array(spacing))

    return voxel_coord


def change_axes_of_image(npy_image, orientation):
    '''default orientation=[1, -1, -1]'''
    if orientation[0] < 0:
        npy_image = np.flip(npy_image, axis=0)
    if orientation[1] > 0:
        npy_image = np.flip(npy_image, axis=1)
    if orientation[2] > 0:
        npy_image = np.flip(npy_image, axis=2)
    return npy_image


def clip_image(image, min_window=-1200, max_window=600):
    return np.clip(image, min_window, max_window)


def normalize_min_max_and_clip(image, min_window=-1200.0, max_window=600.0):
    """
    Normalize image HU value to [-1, 1] using window of [min_window, max_window].
    """
    image = (image - min_window) / (max_window - min_window)
    image = image * 2 - 1.0
    image = image.clip(-1, 1)
    return image


def normalize_mean_std(image, global_mean=None, global_std=None):
    """
    Normalize image by (voxel - mean) / std, the operate should be local or global normalization.
    """
    if not global_mean or not global_std:
        mean = np.mean(image)
        std = np.std(image)
    else:
        mean, std = global_mean, global_std

    image = (image - mean) / (std + 1e-5)
    return image


def clip_and_normalize_mean_std(image, min_window=-1200, max_window=600):
    """
    Clip image in a range of [min_window, max_window] in HU values.
    """
    image = np.clip(image, min_window, max_window)
    mean = np.mean(image)
    std = np.std(image)

    image = (image - mean) / (std + 1e-5)
    return image