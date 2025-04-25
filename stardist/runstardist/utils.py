"""Utility functions for StarDist pipeline."""

import argparse
import json
import logging
import os
import platform
from pathlib import Path
from typing import List, Union

import numpy as np
import requests
import yaml

from runstardist import path_dir_models, repo_global_path
from runstardist.dataio.hdf5 import load_h5
from runstardist.dataio.tiff import load_tiff

EXT_HDF5 = ('hdf5', 'h5', '.hdf5', '.h5')
EXT_TIFF = ('tiff', 'tif', '.tiff', '.tif')

logger = logging.getLogger(__name__)


def log_gpu_info():
    """Print environment/GPU info to logger"""
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    node = platform.node()
    env = os.environ.get("CONDA_DEFAULT_ENV", "not set")
    prefix = os.environ.get("CONDA_PREFIX", "not set")
    logger.info(f"Using GPU: {gpu} on {node}")
    logger.info(f"Using CONDA environment '{env}' at {prefix}")


def dir_path(path):
    """If the input is a valid directory, return the path, otherwise raise error.

    Used as a type for argparse arguments.
    """
    path = Path(path)
    if path.is_dir():
        return path
    else:
        raise NotADirectoryError(path)


def file_path(path):
    """If the input is a valid file, return the path, otherwise raise error.

    Used as a type for argparse arguments.
    """
    path = Path(path)
    if path.is_file():
        return path
    else:
        raise FileNotFoundError(path)


def hdf5_path(path):
    """If the input is a valid HDF5 file, return the path, otherwise raise error.

    Used as a type for argparse arguments.
    """
    path = Path(path)
    if path.suffix[1:] in EXT_HDF5:
        return True
    else:
        raise FileNotFoundError(f"{path} is not a HDF5 file.")


def tiff_path(path):
    """If the input is a valid TIFF file, return the path, otherwise raise error.

    Used as a type for argparse arguments.
    """
    path = Path(path)
    if path.suffix[1:] in EXT_TIFF:
        return True
    else:
        raise FileNotFoundError(f"{path} is not a TIFF file.")


def parse_arguments():
    """Parse arguments from command line.

    For now, only the config file is required and parsed.
    """
    logger.info("Parsing arguments")
    parser = argparse.ArgumentParser(description="Parse Arguments!")
    parser.add_argument('--config', dest='config_file', required=True, type=file_path)
    args = parser.parse_args()
    logger.info(f"Arguments are: {args}")
    return args


def load_config(path_config):
    """Load config file and return as a dictionary."""

    logger.info("Loading config file")
    config = yaml.safe_load(open(path_config, 'r', encoding='utf-8'))
    config_str = '    ' + yaml.dump(config, indent=4).replace('\n', '\n    ')
    logger.info(f"Config contains: \n{config_str}")
    return config


def set_tf_op_parallelism_threads(n_inter: int, n_intra: int):
    """Set TensorFlow's threading settings to avoid certain errors. (Optional)

    Args:
        n_inter (int): inter_op_parallelism_threads
        n_intra (int): intra_op_parallelism_threads
    """
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    logger.debug(f"Before: inter_op_parallelism_threads = {tf.config.threading.get_inter_op_parallelism_threads()}")
    logger.debug(f"Before: intra_op_parallelism_threads = {tf.config.threading.get_intra_op_parallelism_threads()}")
    tf.config.threading.set_inter_op_parallelism_threads(n_inter)
    tf.config.threading.set_intra_op_parallelism_threads(n_intra)
    logger.debug(f"After: inter_op_parallelism_threads = {tf.config.threading.get_inter_op_parallelism_threads()}")
    logger.debug(f"After: intra_op_parallelism_threads = {tf.config.threading.get_intra_op_parallelism_threads()}")


def configure_tensorflow(config_tf):
    try:
        config_threading = config_tf['threading']
        n_inter = config_threading['inter_op_parallelism_threads']
        n_intra = config_threading['intra_op_parallelism_threads']
        set_tf_op_parallelism_threads(n_inter, n_intra)
    except KeyError:
        logger.error("Erronous TensorFlow config: No change to TensorFlow's threading settings.")


def find_files_with_ext(paths: Union[List[str], List[Path]], file_format: str = 'hdf5') -> List[Path]:
    """
    Return paths as a list if it is a list of file, otherwise if it is a list of directory then return a list of files in the directories.
    """

    paths = [Path(path) for path in paths]

    extensions = _identify_format(file_format)
    logger.info(f"Looking for files with extensions {extensions} in {paths}.")

    files = []
    for path in paths:
        if path.is_dir():
            files += [file for file in path.glob('*') if file.suffix[1:].lower() in extensions]
        elif path.is_file():
            if path.suffix[1:].lower() in extensions:
                files.append(path)
        else:
            raise FileNotFoundError(f"{path} is not a file or directory.")

    logger.info(f"Found {len(files)} files:")
    for file in files:
        logger.info(f"    Found file {file}.")

    return files


def _identify_format(file_format):
    extensions = EXT_HDF5 if file_format.lower() in EXT_HDF5 else EXT_TIFF
    return extensions


def load_dataset(filepath, dset_format=None, dset_name=None):
    """Check format from config file and load dataset accordingly."""

    # check filepath is a file
    if not Path(filepath).is_file():
        raise FileNotFoundError(f"File {filepath} does not exist.")

    # get file extension from filepath
    ext = Path(filepath).suffix[1:].lower()
    if dset_format is not None:
        if ext not in _identify_format(dset_format):
            raise ValueError(
                f"File extension {ext} does not match the format {dset_format} specified in the config file."
            )

    # load dataset according to file extension
    if ext in EXT_HDF5:
        if dset_name is None:
            raise ValueError("Please specify the HDF5 dataset name in the config file.")
        image, infos = load_h5(filepath, dset_name)  # pylint: disable=unbalanced-tuple-unpacking
    elif ext in EXT_TIFF:
        image, infos = load_tiff(filepath)  # pylint: disable=unbalanced-tuple-unpacking
    else:
        raise ValueError(f"File extension {ext} is not supported.")

    return image, infos  # image, (voxel_size, file_shape, key/None, voxel_size_unit)


def get_model_parameters(model, trainable=True):
    """
    Count all the trainable parameters or non-trainable parameters of your Tensorflow models.
    `keras.utils.layer_utils.count_params()` uses `id`s but `np.prod(layer.get_shape())` doesn't.

    However, `sum(count_params(layer) for layer in model.keras_model.trainable_weights)`
    causes `TypeError: 'Variable' object is not iterable`. Use `np.prod(v.get_shape())` instead!

    https://wandb.ai/wandb_fc/tips/reports/How-to-Calculate-Number-of-Model-Parameters-for-PyTorch-and-Tensorflow-Models--VmlldzoyMDYyNzIx
    """

    if trainable:
        trainable_params = sum(np.prod(layer.get_shape()) for layer in model.keras_model.trainable_weights)
        return trainable_params
    else:
        non_trainable_params = sum(np.prod(layer.get_shape()) for layer in model.keras_model.non_trainable_weights)
        return non_trainable_params


def check_models(model_name: str, update_files: bool = False, config_only: bool = False) -> bool:
    """
    Simple script to check and download trained modules
    :param model_name: name of the model in the model zoo
    :param update_files: if true force the re-download of the model
    :param config_only: if true only downloads the config file and skips the model file
    """

    # check if model is already downloaded
    path_model = path_dir_models / model_name
    if path_model.is_dir():
        logger.info(f"Model {model_name} already downloaded.")
        return True
    else:
        path_model.mkdir(exist_ok=False, parents=True)

    # download model
    logger.info(f"Downloading model {model_name}.")
    zoo = load_config(repo_global_path / "resources/zoo.yaml")
    url_config_stardist = zoo[model_name]['url_rdf']
    url_model_weights = zoo[model_name]['url_weights']
    with requests.get(url_config_stardist, allow_redirects=True) as r:
        with open(path_model / 'rdf.yaml', 'wb') as f:
            f.write(r.content)
    with requests.get(url_model_weights, allow_redirects=True) as r:
        with open(path_model / 'weights_bioimageio.h5', 'wb') as f:
            f.write(r.content)

    # save config file
    config_stardist = load_config(path_model / 'rdf.yaml')
    with open(path_model / 'config.json', 'w') as json_file:
        json.dump(config_stardist['config']['stardist']['config'], json_file)
    with open(path_model / 'thresholds.json', 'w') as json_file:
        json.dump(config_stardist['config']['stardist']['thresholds'], json_file)

    return True
