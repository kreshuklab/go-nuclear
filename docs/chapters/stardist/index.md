# Run Stardist: A Guide and A Pipeline <!-- omit in toc -->

![version](https://anaconda.org/qin-yu/run-stardist/badges/version.svg)
![latest_release_date](https://anaconda.org/qin-yu/run-stardist/badges/latest_release_date.svg)
![license](https://anaconda.org/qin-yu/run-stardist/badges/license.svg)
![downloads](https://anaconda.org/qin-yu/run-stardist/badges/downloads.svg)

A complete training and inference pipeline for 3D StarDist with an example on 3D biological (ovules) datasets. Please submit an issue if you encountered errors or if you have any questions or suggestions.

## Models and Data

A 3D nucleus segmentation model is available for download from Bioimage.IO and ready to be used directly for segmenting your nuclei. The model is trained on a 3D confocal ovule dataset from *Arabidopsis thaliana*. The StarDist version v0.8.3 was used for the paper.

### Use Pre-trained Model

Model weights and related files can be found at [DOI 10.5281/zenodo.8421755](https://zenodo.org/doi/10.5281/zenodo.8421755). The programme downloads the model automatically for you to make inference on your images as long as you specify `generic_plant_nuclei_3D` as the `model_name` in the configuration file.

This is the only 3D StarDist model available on Bioimage Model Zoo at the moment. If you have another model, put its folder in your `PATH_TO_MODEL_DIR` and specify the folder name as `MY_MODEL_NAME` in the configuration file (see below). Then you can run `predict-stardist` to use the model for inference. For more information on inference, see [Prediction](#prediction) section below.

### Training data statistics and links

The training data is publicly available on Zenodo at [BioImage Archive S-BIAD1026](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD1026). Some key information about the training data is listed below:

```python
original_voxel_size = {  # z, y, x
    1135: [0.28371836501901143, 0.12678642066720086, 0.12678642066720086],  # validation
    1136: [0.2837183895131086,  0.12756971653115998, 0.12756971653115998],  # training
    1137: [0.2837183895131086,  0.1266211463645486,  0.1266211463645486 ],  # training
    1139: [0.2799036917562724,  0.12674335484590543, 0.12674335484590543],  # training
    1170: [0.27799632231404964, 0.12698523961670266, 0.12698522349145364],  # training
}  # [0.2837, 0.1268, 0.1268] is taken as the median

original_median_extents = {  # z, y, x
    1135: [16, 32, 33],  # validation
    1136: [16, 32, 32],  # training
    1137: [16, 32, 32],  # training
    1139: [16, 32, 33],  # training
    1170: [16, 29, 30],  # training
    'average':
}  # [16, 32, 32] is taken as the median
```

## Installation

It is recommended to install this package with `mamba` (see below). If you don't have `mamba` installed, you can install it with `conda`. We start the guide by installing Mini-`conda`.

### Install Miniconda

First step required to use the pipeline is installing Miniconda. If you already have a working Anaconda setup you can go directly to the next step. Anaconda can be downloaded for all platforms from [here](https://www.anaconda.com/products/distribution). We suggest to use Miniconda, because it is lighter and install fewer unnecessary packages.

To download Miniconda, open a terminal and type:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Then install by typing:

```bash
bash ./Miniconda3-latest-Linux-x86_64.sh
```

and follow the installation instructions. The Miniconda3-latest-Linux-x86_64.sh file can be safely deleted.

### Install `run-stardist` using `mamba`

Fist step is to install mamba, which is an alternative to conda:

```bash
conda install -c conda-forge mamba
```

If you have a nvidia gpu, install `run-stardist` using:

```bash
mamba create -n run-stardist -c qin-yu -c conda-forge "python>=3.10" tensorflow stardist wandb "pydantic<2" run-stardist
```

or if you don't have a nvidia gpu, install `run-stardist` using:

```bash
mamba create -n run-stardist -c qin-yu -c conda-forge "python>=3.10" tensorflow-cpu stardist wandb "pydantic<2" run-stardist
```

## Usage

### Example configuration file for both training and inference

The original configuration file used for training the final ResNet StarDist model published on Bioimage.IO for wide applicability can be found at `stardist/configs/final_resnet_model_config.yml`, which can used for both training and inference (note that the inference output is only used for illustration in this repository because it's segmenting the training data).

The generic template is shown below. A configuration template with more guidelines can be found at `stardist/configs/train_and_infer.yml`.

```yaml
wandb: # optional, remove this part if not using W&B
  project: ovules-instance-segmentation
  name: final-stardist-model

data:
  # Rescale outside StarDist
  rescale_factor: Null

  # Training (ignored in inference config file)
  training:
    - PATH_TO_INPUT_DIR_1 or PATH_TO_INPUT_FILE_1
    - PATH_TO_INPUT_DIR_2 or PATH_TO_INPUT_FILE_2
  validation:
    - PATH_TO_INPUT_DIR_3 or PATH_TO_INPUT_FILE_3
  raw_name: raw/noisy      # only required if HDF5
  label_name: label/gold   # only required if HDF5

  # Inference (ignored in training config file)
  prediction:
    - PATH_TO_INPUT_DIR_4 or PATH_TO_INPUT_FILE_4
    - PATH_TO_INPUT_DIR_5 or PATH_TO_INPUT_FILE_5
  format: tiff             # only 'hdf5' or 'tiff'
  name: raw/nuclei         # dataset name of the raw image in HDF5 files, only required if format is `hdf5`
  output_dir: MY_OUTPUT_DIR
  output_dtype: uint16     # `uint8`, `uint16`, or `float32` are recommended
  resize_to_original: True # output should be of he same shape as input
  target_voxel_size: Null  # the desired voxel size to rescale to during inference, null if rescale factor is set
  save_probability_map: True

stardist:
  model_dir: PATH_TO_MODEL_DIR  # set to `null` if model name is `generic_plant_nuclei_3D`
  model_name: MY_MODEL_NAME     # set to `generic_plant_nuclei_3D` to use the builtin model
  model_type: StarDist3D
  model_config:  # model configuration should stay identical for training and inference
    backbone: resnet
    n_rays: 96
    grid: [2, 4, 4]
    use_gpu: False
    n_channel_in: 1
    patch_size: [96, 96, 96]  # multiple of 16 prefered
    train_batch_size: 8
    train_n_val_patches: 16
    steps_per_epoch: 400
    epochs: 1000

augmenter:
  name: default
```

### Training

```shell
train-stardist --config CONFIG_PATH
```

where CONFIG_PATH is the path to the YAML configuration file. For example, if you want to train the model with the example configuration file `configs/train_and_infer.yml`:

```shell
cd ovules-instance-segmentation/stardist/
CUDA_VISIBLE_DEVICES=0 train-stardist --config configs/train_and_infer.yml
```

### Prediction

```shell
predict-stardist --config CONFIG_PATH
```

where CONFIG_PATH is the path to the YAML configuration file. For example, if you want to use the model with the example configuration file `configs/train_and_infer.yml`:

```shell
cd ovules-instance-segmentation/stardist/
CUDA_VISIBLE_DEVICES=0 predict-stardist --config configs/train_and_infer.yml
```

**Preprocessing:** For the published [StarDist Plant Nuclei 3D ResNet](https://zenodo.org/doi/10.5281/zenodo.8421755) the median size of nuclei in training data is `[16, 32, 32]`. To achieve the best segmentation result, the input 3D images should be rescaled so that your nucleus size in ZYX matches the training data. For example, if the median nucleus size of your data is `[32, 32, 32]`, then `rescale_factor` should be `[0.5, 1., 1.]`; if it's `[15, 33, 31]`, then it does not have to be rescaled. You may also choose to leave `rescale_factor` as `Null` and rescale your images with Fiji or other tools before running the pipeline. If `resize_to_original` is `True` then the output will have the original size of the input image.

### Specifying a Graphic Card (GPU)

If you need to specify a graphic card, for example to use the No. 7 card (the eighth), do:

```shell
CUDA_VISIBLE_DEVICES=7 predict-stardist --config CONFIG_PATH
```

If you have only one graphic card, use `CUDA_VISIBLE_DEVICES=0` to select the first card (No. 0).

## Cite

If you find the code/models/datasets useful, please cite our paper and StarDist:

```bibtex
@article{vijayan2024deep,
  title={A deep learning-based toolkit for 3D nuclei segmentation and quantitative analysis in cellular and tissue context},
  author={Vijayan, Athul and Mody, Tejasvinee Atul and Yu, Qin and Wolny, Adrian and Cerrone, Lorenzo and Strauss, Soeren and Tsiantis, Miltos and Smith, Richard S and Hamprecht, Fred A and Kreshuk, Anna and others},
  journal={Development},
  volume={151},
  number={14},
  year={2024},
  publisher={The Company of Biologists}
}

@inproceedings{weigert2020star,
  title={Star-convex polyhedra for 3D object detection and segmentation in microscopy},
  author={Weigert, Martin and Schmidt, Uwe and Haase, Robert and Sugawara, Ko and Myers, Gene},
  booktitle={Proceedings of the IEEE/CVF winter conference on applications of computer vision},
  pages={3666--3673},
  year={2020}
}
```
