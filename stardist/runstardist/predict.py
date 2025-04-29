import argparse
import logging
from pathlib import Path
from pprint import pformat

from csbdeep.utils import normalize
from skimage.transform import rescale, resize
from stardist.models import StarDist2D, StarDist3D

from runstardist import utils
from runstardist.config import ConfigPred
from runstardist.dataio.hdf5 import create_h5
from runstardist.dataio.tiff import create_tiff

logger = logging.getLogger(__name__)


def load_model(model_dir: Path, model_name: str, model_type: str):
    """Load StarDist model from config file"""
    if model_type == 'StarDist3D':
        Model = StarDist3D
    elif model_type == 'StarDist2D':
        Model = StarDist2D
    else:
        raise ValueError(f"'{model_type}' is not a valid StarDist model type!")
    return Model(None, name=model_name, basedir=model_dir)


def load_dataset(filepath, format=None, name=None, **kwargs):
    """Check format from config file and load dataset accordingly."""
    image, infos = utils.load_dataset(filepath, format, name)  # pylint: disable=unbalanced-tuple-unpacking
    image = normalize(image, 1, 99.8)
    return image, infos


def save_image(dset_format, output_dir, suffix, filepath, image, output_dtype, voxel_size, voxel_size_unit='um'):
    """Save prediction to file in the specified format. Still save to TIFF if an unsupported format is used."""
    output_dir = Path(output_dir)
    filepath = Path(filepath)

    image = image.astype(output_dtype)
    logger.info(f"Saving {suffix} to {output_dir}")
    if dset_format == 'hdf5':
        path_out_file = output_dir / (filepath.stem + '_' + suffix + '.h5')
        create_h5(path_out_file, image, suffix, voxel_size=voxel_size)
    else:
        path_out_file = output_dir / (filepath.stem + '_' + suffix + '.tif')
        create_tiff(path_out_file, image, voxel_size=voxel_size, voxel_size_unit=voxel_size_unit)

        if dset_format != 'tiff':
            logger.warning(f"Format {dset_format} is not supported! But the prediction is saved as tiff.")
    logger.info(f"Saved {suffix} as {path_out_file}")


def predict(config_data, model, filepath, image, voxel_size, original_shape, voxel_size_unit):
    results = model.predict_instances(
        image,
        n_tiles=model._guess_n_tiles(image),  # pylint: disable=protected-access
        return_predict=config_data.save_probability_map,
        scale=config_data.rescale_factor,
        show_tile_progress=True,
        verbose=True,
    )
    if config_data.save_probability_map:  # get probability map: (segmentation, details), (prob, dist) = results
        (segmentation, _), (image_prob, _) = results
    else:
        segmentation, _ = results

    # Resize back to original shape if necessary:
    if config_data.resize_to_original and segmentation.shape != original_shape:
        logger.info(f"Resizing segmentation from shape {segmentation.shape} to {original_shape}")
        segmentation = resize(segmentation, original_shape, order=0, preserve_range=True, anti_aliasing=False).astype(
            segmentation.dtype
        )
        if config_data.save_probability_map:
            image_prob = resize(image_prob, original_shape, order=1, preserve_range=True, anti_aliasing=False).astype(
                image_prob.dtype
            )

    save_image(
        config_data.format,
        config_data.output_dir,
        "segmentation",
        filepath,
        segmentation,
        config_data.output_dtype,
        voxel_size=voxel_size,
        voxel_size_unit=voxel_size_unit,
    )
    if config_data.save_probability_map:
        save_image(
            config_data.format,
            config_data.output_dir,
            "probability",
            filepath,
            image_prob,
            'float32',
            voxel_size=voxel_size,
            voxel_size_unit=voxel_size_unit,
        )


def main():
    print("updated")
    # Parse Arguments:
    parser = argparse.ArgumentParser(description="Parse Arguments!")
    parser.add_argument('--log_level', dest='log_level', required=False, type=str, default="INFO")
    parser.add_argument('--config', dest='config_file', required=True, type=utils.file_path)
    args = parser.parse_args()

    # Setup Logging:
    logger.setLevel(args.log_level.upper())
    logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
    logger.info("Running predict.py")
    utils.log_gpu_info()

    # Load Config:
    logger.info("Loading configs")
    config = utils.load_config(args.config_file)
    config_all = ConfigPred(**config)
    logger.info(f"Config:\n{pformat(config_all.dict())}")

    # Configure Tensorflow:
    logger.info("Configuring Tensorflow (optional)")
    if config_all.tensorflow is not None:
        utils.configure_tensorflow(config_all.tensorflow.dict())

    # Find Data:
    logger.info("Scanning datasets to predict")
    paths_file_dataset = utils.find_files_with_ext(config_all.data.path_dataset_pred, config_all.data.format)
    logger.info(f"Found the following files: \n\t{paths_file_dataset}")

    # Load Model:
    logger.info("Loading model")
    model = load_model(*config_all.stardist.dict().values())

    # Predict:
    logger.info("Predicting")
    for dataset_idx, filepath in enumerate(paths_file_dataset):
        logger.info(f"Predicting dataset No. {dataset_idx}, {len(paths_file_dataset)} datasets in total")
        logger.info(f"Predicting {filepath}")
        # Load and rescale image:
        image, (voxel_size, original_shape, _, voxel_size_unit) = load_dataset(filepath, **config_all.data.dict())
        # Predict and save image:
        predict(config_all.data, model, filepath, image, voxel_size, original_shape, voxel_size_unit)


if __name__ == "__main__":
    main()
