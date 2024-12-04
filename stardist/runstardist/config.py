import logging
from pathlib import Path

from pydantic.v1 import BaseModel, Field, validator  # pylint: disable=no-name-in-module
from stardist import gputools_available

from runstardist.utils import check_models, load_config
from runstardist import path_dir_models, repo_global_path

logger = logging.getLogger(__name__)


class ConfigWB(BaseModel):
    project: str
    name: str


class ConfigData(BaseModel):
    path_dataset_train: list[Path] = Field(alias='training')
    path_dataset_val: list[Path] = Field(alias='validation')
    raw_name: str
    label_name: str
    rescale_factor: tuple[float, float, float] | None


class ConfigPredData(BaseModel):
    path_dataset_pred: list[Path] = Field(alias='prediction')
    format: str
    name: str | None
    output_dir: Path
    output_dtype: str = 'uint16'
    resize_to_original: bool = True
    target_voxel_size: tuple[float, float, float] | None
    rescale_factor: tuple[float, float, float] | None
    save_probability_map: bool = False

    @validator('format')
    def format_is_valid(cls, v):  # pylint: disable=no-self-argument,no-self-use
        if v not in ['hdf5', 'tiff']:
            raise ValueError("Format must be either 'hdf5' or 'tiff'.")
        return v

    @validator('name')
    def name_not_none_if_format_is_hdf5(cls, v, values):  # pylint: disable=no-self-argument,no-self-use
        if values['format'] == 'hdf5' and v is None:
            raise ValueError("Please specify the HDF5 dataset name in the config file.")
        return v

    @validator('output_dir')
    def output_dir_is_valid(cls, v):  # pylint: disable=no-self-argument,no-self-use
        if not v.is_dir():
            logger.warning(f"Output directory {v} does not exist. Creating it now.")
            v.mkdir(parents=True, exist_ok=True)
        return v

    @validator('output_dtype')
    def output_dtype_is_valid(cls, v):  # pylint: disable=no-self-argument,no-self-use
        if v not in ['uint8', 'uint16', 'float32']:
            raise ValueError("Output dtype must be either 'uint8', 'uint16' or 'float32'.")
        return v

    @validator('target_voxel_size', 'rescale_factor')
    def either_fit_to_voxel_size_or_rescale_manually(cls, v, values):  # pylint: disable=no-self-argument,no-self-use
        non_none_fields = [field for field in ['target_voxel_size', 'rescale_factor'] if values.get(field) is not None]
        if len(non_none_fields) > 1:
            raise ValueError("Either specify 'target_voxel_size' or 'rescale_factor', not both.")
        return v

    @validator('target_voxel_size')
    def target_voxel_size_is_valid(cls, v):  # pylint: disable=no-self-argument,no-self-use
        if v is not None:
            raise NotImplementedError("Please find a `rescale_factor` manually to rescale your data because only nucleus size in absolute pixel matters. Matching voxel size may help PlantSeg but not StarDist.")


class ConfigConfig3D(BaseModel):
    backbone: str
    n_rays: int = 96  # v1.10 doesn't support __post_init__ yet
    n_channel_in: int
    grid: tuple[int, int, int]
    train_patch_size: tuple[int, int, int] = Field(alias='patch_size')
    train_batch_size: int
    train_n_val_patches: int
    train_steps_per_epoch: int = Field(alias='steps_per_epoch')
    train_epochs: int = Field(alias='epochs')
    use_gpu: bool = False

    @validator('backbone')
    def backbone_either_resnet_or_unet(cls, v):  # pylint: disable=no-self-argument,no-self-use
        if v not in ['resnet', 'unet']:
            raise ValueError("backbone must be either 'resnet' or 'unet'")
        return v

    @validator('use_gpu')
    def use_gpu_or_not(cls, v):  # pylint: disable=no-self-argument,no-self-use
        if v and gputools_available():
            return True
        else:
            return False


class ConfigStarDist3D(BaseModel):
    model_dir: Path
    model_name: str
    model_config: ConfigConfig3D


class ConfigPredStarDist3D(BaseModel):
    model_dir: Path | None = None
    model_name: str
    model_type: str = 'StarDist3D'

    @validator('model_type')
    def model_type_is_valid(cls, v):  # pylint: disable=no-self-argument,no-self-use
        if v not in ['StarDist3D', 'StarDist2D']:
            raise ValueError("Model type must be either 'StarDist3D' or 'StarDist2D'.")
        return v

    @validator('model_dir')
    def model_dir_is_valid(cls, v, values):  # pylint: disable=no-self-argument,no-self-use
        if v is None:
            v = path_dir_models
        elif not v.is_dir():
            raise NotADirectoryError(f"Model directory {v} does not exist.")
        return v

    @validator('model_name')
    def model_name_is_valid(cls, v, values, config):  # pylint: disable=no-self-argument,no-self-use
        try:  # use try block because v may be anything
            is_dir = values['model_dir'].is_dir()
        except:
            is_dir = False
        if not is_dir or values['model_dir'] == path_dir_models:
            zoo = load_config(repo_global_path / "resources/zoo.yaml")
            if v not in zoo.keys():
                raise NotADirectoryError(f"Model directory {v} does not exist.")
            elif check_models(v, update_files=False, config_only=False):
                logger.warning(f"Model is in {path_dir_models}")
            else:
                raise NotADirectoryError(f"Model directory {v} does not exist, encounterd error when downloading model.")
        elif not (values['model_dir'] / v).is_dir():
            raise NotADirectoryError(f"Model {v} does not exist.")
        return v


class ConfigAugmenter(BaseModel):
    name: str


class ConfigTFThreading(BaseModel):
    inter: int = 2
    intra: int = 16


class ConfigTF(BaseModel):
    threading: ConfigTFThreading | None


class ConfigTrain(BaseModel):
    wandb: ConfigWB | None
    data: ConfigData
    stardist: ConfigStarDist3D
    augmenter: ConfigAugmenter
    tensorflow: ConfigTF | None


class ConfigPred(BaseModel):
    data: ConfigPredData
    stardist: ConfigPredStarDist3D
    tensorflow: ConfigTF | None
