"""Augmentation functions for 3D microscopy images used in StarDist3D training."""

import numpy as np


def random_fliprot(img, mask, axis=None):
    """Random flips and/or rotations of image and mask.

    Originally from StarDist 3D training example:
    https://github.com/stardist/stardist/blob/master/examples/3D/2_training.ipynb
    """
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)

    assert img.ndim >= mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(transpose_axis)
    for ax in axis:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    """Random intensity change of image.

    Originally from StarDist 3D training example:
    https://github.com/stardist/stardist/blob/master/examples/3D/2_training.ipynb
    """
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def default_augmenter(x, y, axis):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image

    Note that we usually only use fliprot along axis=(1,2), i.e. the yx axis
    as 3D microscopy acquisitions are usually not axially symmetric

    Originally from StarDist 3D training example:
    https://github.com/stardist/stardist/blob/master/examples/3D/2_training.ipynb
    """
    x, y = random_fliprot(x, y, axis=axis)
    x = random_intensity_change(x)
    return x, y


def augmenter3d_anisotropic(x, y):
    return default_augmenter(x, y, axis=(1, 2))


def augmenter3d_isotropic(x, y):
    return default_augmenter(x, y, axis=(0, 1, 2))
