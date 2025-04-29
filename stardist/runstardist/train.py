import argparse
import logging
import sys
from pprint import pformat

import numpy as np
import wandb
from csbdeep.utils import normalize
from skimage.transform import rescale
from stardist import Rays_GoldenSpiral, calculate_extents
from stardist.models import Config3D, StarDist3D

from runstardist import utils
from runstardist.augment import augmenter3d_anisotropic, augmenter3d_isotropic
from runstardist.config import ConfigConfig3D, ConfigTrain

logger = logging.getLogger(__name__)


def configure_model(config_config3d: ConfigConfig3D, anisotropy):
    logger.info(
        f"Predict on sub-sampled grid for increased efficiency and larger field of view: {config_config3d.grid}"
    )
    logger.info(
        f"At the mean time, {tuple(1 if a > 1.5 else 2 for a in anisotropy)} is the recommended grid"
    )  # Not optimal, better let user input
    logger.info(
        f"{'NOT ' if not config_config3d.use_gpu else ''}Using OpenCL for training data generator (requires 'gputools')"
    )

    rays = Rays_GoldenSpiral(
        config_config3d.n_rays, anisotropy=anisotropy
    )  # Rays on a Fibonacci lattice adjusted for anisotropy
    config3d = Config3D(
        rays=rays,
        anisotropy=anisotropy,
        **config_config3d.dict(),
    )
    logger.info(f"Config3D: {pformat(vars(config3d))}")
    return config3d


def _load_training_data(list_path_raw, list_path_lab, name_raw=None, name_lab=None, factor=None):
    """
    TODO: make this lazy.
    """
    images_raw_all = []
    images_lab_all = []
    assert len(list_path_raw) == len(list_path_lab)
    for path_raw, path_lab in zip(list_path_raw, list_path_lab):
        logger.info(f"Loading raw image from {path_raw}")
        logger.info(f"Loading label image from {path_lab}")
        image_lab, (voxel_size_lab, _, _, _) = utils.load_dataset(path_lab, None, name_lab)
        image_raw, (voxel_size_raw, _, _, _) = utils.load_dataset(path_raw, None, name_raw)

        if factor is not None:
            image_lab = rescale(image_lab, factor, order=0, preserve_range=True, anti_aliasing=False, channel_axis=None)
            image_raw = rescale(image_raw, factor, order=1, preserve_range=True, anti_aliasing=False, channel_axis=None)

        logger.info("Normalizing raw image")
        image_raw = normalize(image_raw, 1, 99.8)

        if image_raw.shape != image_lab.shape:
            raise IndexError(f"'{name_raw}' has shape {image_raw.shape} but '{name_lab}' has shape {image_lab.shape}")
        # if two voxel size array are not the same, raise error
        if any(voxel_size_raw != voxel_size_lab):
            raise IndexError(
                f"'{name_raw}' has voxel size {voxel_size_raw} but '{name_lab}' has voxel size {voxel_size_lab}"
            )
        logger.info(
            f"Loaded raw: shape {image_raw.shape}, dtype {image_raw.dtype}, and range {image_raw.min()} to {image_raw.max()}"
        )
        logger.info(
            f"Loaded lab: shape {image_lab.shape}, dtype {image_lab.dtype}, and range {image_lab.min()} to {image_lab.max()}"
        )

        get_extent_and_anisotropy(image_lab)

        images_raw_all.append(image_raw)
        images_lab_all.append(image_lab)
    return images_raw_all, images_lab_all


def load_training_data(paths_dataset_train, paths_dataset_val, raw_name, label_name, rescale_factor):
    assert isinstance(paths_dataset_train, list)
    paths_file_dataset_train = utils.find_files_with_ext(paths_dataset_train)
    logger.info(f"Found the following files: \n\t{paths_file_dataset_train}")
    if not paths_file_dataset_train:
        raise ValueError("Check your path for training dataset, no dataset found.")

    logger.info("Scanning validation datasets, please make sure file extensions are in lower case.")
    assert isinstance(paths_dataset_val, list)
    paths_file_dataset_val = utils.find_files_with_ext(paths_dataset_val)
    logger.info(f"Found the following files: \n\t{paths_file_dataset_val}")
    if not paths_file_dataset_val:
        raise ValueError("Check your path for validation dataset, no dataset found.")

    X, Y = _load_training_data(paths_file_dataset_train, paths_file_dataset_train, raw_name, label_name, rescale_factor)
    X_val, Y_val = _load_training_data(
        paths_file_dataset_val, paths_file_dataset_val, raw_name, label_name, rescale_factor
    )
    if len(X) < 1:
        raise ValueError("Not enough training data.")
    if len(X_val) < 1:
        raise ValueError("Not enough validation data.")

    return X, Y, X_val, Y_val


def create_model(model_dir, model_name, config3d, extents):
    model = StarDist3D(config=config3d, name=model_name, basedir=model_dir)
    try:
        n_trainable = utils.get_model_parameters(model)
        logger.info(f"In this StarDist3D model, there are {n_trainable} trainable parameters.")
    except Exception as e:
        logger.error(f"Error while getting model parameters: {e}")

    # Check if the median object size is within the field of view of the neural network
    logger.info("Checking field of view (FoV)")
    fov = np.array(model._axes_tile_overlap("ZYX"))
    logger.info(f"\tmedian object size:      {extents}")
    logger.info(f"\tnetwork field of view :  {fov}")
    if any(extents > fov):
        logger.warning("WARNING: median object size larger than field of view of the neural network.")
    else:
        logger.info("Median object size smaller than field of view of the neural network.")

    return model


def get_extent_and_anisotropy(labels):
    extents = calculate_extents(labels)
    anisotropy = tuple(np.max(extents) / extents)
    logger.info(f"\tMedian extents of labeled objects: {extents}")
    logger.info(f"\tEmpirical anisotropy of labeled objects: {anisotropy}")
    return extents, anisotropy


def configure_augmenter(config_augmenter):
    if config_augmenter.name == "augmenter_isotropic_3d":
        augmenter = augmenter3d_isotropic  # rotate in all three dimensions
        logger.warning("Rotating training data in all three dimension.")
    else:
        augmenter = augmenter3d_anisotropic  # rotate only in xy-plane
        logger.info("Rotating training data in xy-plane only.")
    return augmenter


def main():
    # Parse Arguments:
    parser = argparse.ArgumentParser(description="Parse Arguments!")
    parser.add_argument('--log_level', dest='log_level', required=False, type=str, default="INFO")
    parser.add_argument('--config', dest='config_file', required=True, type=utils.file_path)
    parser.add_argument('--dry_run', dest='dry_run', required=False, action='store_true')
    args = parser.parse_args()

    # Setup Logging:
    logger.setLevel(args.log_level.upper())
    logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
    logger.info("Running train.py")
    utils.log_gpu_info()

    # Load Config:
    logger.info("Loading configs")
    config = utils.load_config(args.config_file)
    config_all = ConfigTrain(**config)

    # Initialise W&B:
    logger.info("Initialising W&B (if configured)")
    if config_all.wandb is not None:
        logger.info("W&B configured, initialising...")
        wandb.init(config=config, sync_tensorboard=True, **config_all.wandb.dict())

    # Configure Tensorflow:
    logger.info("Configuring Tensorflow (optional)")
    if config_all.tensorflow is not None:
        utils.configure_tensorflow(config_all.tensorflow.dict())

    # Load Data:
    logger.info("Scanning training datasets, please make sure file extensions are in lower case.")
    X, Y, X_val, Y_val = load_training_data(*config_all.data.dict().values())

    # Data Stats:
    logger.info("Computing average anisotropy")
    extents, anisotropy = get_extent_and_anisotropy(Y + Y_val)

    # Configure Model:
    logger.info("Creating Config3D")
    config3d = configure_model(config_all.stardist.model_config, anisotropy)

    # Create Model:
    logger.info("Creating StarDist3D")
    model = create_model(config_all.stardist.model_dir, config_all.stardist.model_name, config3d, extents)

    # Choose Augmenter:
    logger.info("Choosing augmenter")
    augmenter = configure_augmenter(config_all.augmenter)

    # Train:
    if args.dry_run:
        logger.info("Dry run, exiting...")
        sys.exit(0)
    logger.info("TRAIN STARTING!")
    model.train(X, Y, validation_data=(X_val, Y_val), augmenter=augmenter)
    logger.info("TRAIN SUCCESS!")
    model.optimize_thresholds(X_val, Y_val)
    logger.info("OPTIMISE SUCCESS!")


if __name__ == "__main__":
    main()
