"""HDF5 IO functions adapted from PlantSeg

https://github.com/hci-unihd/plant-seg/blob/master/plantseg/io/h5.py
"""

import warnings

import h5py

# allowed h5 keys
H5_EXTENSIONS = [".hdf", ".h5", ".hd5", "hdf5"]
H5_KEYS = ["raw", "predictions", "segmentation"]


def read_h5_voxel_size(f, h5key):
    """
    :returns the voxels size stored in a h5 dataset (if absent returns [1, 1, 1])
    """
    ds = f[h5key]

    # parse voxel_size
    if 'element_size_um' in ds.attrs:
        voxel_size = ds.attrs['element_size_um']
    else:
        warnings.warn('Voxel size not found, returning default [1.0, 1.0. 1.0]', RuntimeWarning)
        voxel_size = [1.0, 1.0, 1.0]

    return voxel_size


def _find_input_key(h5_file):
    """
    returns the first matching key in H5_KEYS or only one dataset is found the key to that dataset
    """
    found_datasets = []

    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            found_datasets.append(name)

    h5_file.visititems(visitor_func)

    if not found_datasets:
        raise RuntimeError(f"No datasets found in '{h5_file.filename}'")

    if len(found_datasets) == 1:
        return found_datasets[0]
    else:
        for h5_key in H5_KEYS:
            if h5_key in found_datasets:
                return h5_key

        raise RuntimeError(
            f"Ambiguous datasets '{found_datasets}' in {h5_file.filename}. "
            f"plantseg expects only one dataset to be present in input H5."
        )


def load_h5(path, key, slices=None, info_only=False):
    """
    Load a dataset from a h5 file and returns some meta info about it.
    Args:
        path (str): Path to the h5file
        key (str): internal key of the desired dataset
        slices (Optional[slice], optional): Optional, slice to load. Defaults to None.
        info_only (bool, optional): if true will return a tuple with infos such as voxel resolution, units and shape. \
        Defaults to False.

    Returns:
        Union[tuple, tuple[np.array, tuple]]: dataset as numpy array and infos
    """
    with h5py.File(path, 'r') as f:
        if key is None:
            key = _find_input_key(f)

        voxel_size = read_h5_voxel_size(f, key)
        file_shape = f[key].shape

        infos = (voxel_size, file_shape, key, 'um')
        if info_only:
            return infos

        file = f[key][...] if slices is None else f[key][slices]

    return file, infos


def create_h5(path, stack, key, voxel_size=(1.0, 1.0, 1.0), mode='a'):
    """
    Helper function to create a dataset inside a h5 file
    Args:
        path: file path
        stack: numpy array to save as dataset in the h5 file
        key: key of the dataset in the h5 file
        voxel_size: voxel size in micrometers
        mode: mode to open the h5 file ['w', 'a']

    Returns:
        None
    """

    with h5py.File(path, mode) as f:
        f.create_dataset(key, data=stack, compression='gzip')
        # save voxel_size
        f[key].attrs['element_size_um'] = voxel_size


def list_keys(path):
    """
    returns all datasets in a h5 file
    """
    with h5py.File(path, 'r') as f:
        return [key for key in f.keys() if isinstance(f[key], h5py.Dataset)]


def del_h5_key(path, key, mode='a'):
    """
    helper function to delete a dataset from a h5file
    """
    with h5py.File(path, mode) as f:
        if key in f:
            del f[key]
            f.close()


def rename_h5_key(path, old_key, new_key, mode='r+'):
    """Rename the 'old_key' dataset to 'new_key'"""
    with h5py.File(path, mode) as f:
        if old_key in f:
            f[new_key] = f[old_key]
            del f[old_key]
            f.close()
