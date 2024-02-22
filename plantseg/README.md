# Run PlantSeg: A Guide <!-- omit in toc -->

- [Installation](#installation)
  - [Install Miniconda](#install-miniconda)
  - [Install `plant-seg` using `mamba`](#install-plant-seg-using-mamba)
- [Inference](#inference)
  - [Example configuration file for both training and inference](#example-configuration-file-for-both-training-and-inference)
  - [Prediction](#prediction)
  - [Specifying a Graphic Card (GPU)](#specifying-a-graphic-card-gpu)
- [Cite](#cite)
- [PlantSeg Version and Code](#plantseg-version-and-code)


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

### Install `plant-seg` using `mamba`

Fist step is to install mamba, which is an alternative to conda:

```bash
conda install -c conda-forge mamba
```

PlantSeg version >= v1.6.2 is required. If you have a nvidia gpu, install `plant-seg` using:

```bash
mamba create -n plant-seg -c pytorch -c nvidia -c conda-forge pytorch pytorch-cuda=12.1 pyqt lcerrone::plantseg
```

or if you don't have a nvidia gpu, install `plant-seg` using:

```bash
mamba create -n plant-seg -c pytorch -c nvidia -c conda-forge pytorch cpuonly pyqt lcerrone::plantseg
```

## Inference

### Example configuration file for both training and inference

The original configuration file used for training the final UNet PlantSeg model published on Bioimage.IO for wide applicability can be found at `plantseg/configs/config_train_final.yml`, which is a configuration file for [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet), the core network of PlantSeg.

An example config file for segmentation can be found at `plantseg/configs/config_pred_wide_applicability.yaml`. To modify it and use it for your own data, you need to change the `path` parameters:

- `path`: path to the folder containing the images to be segmented or to the image to be segmented

You may also need to change these parameters:

- `preprocessing:factor`: a rescale factor to match the nucleus size of your data to the training data, not necessary but may help in specific cases
- `cnn_prediction:patch`: patch size should be smaller than the dimension of your image, and smaller than the GPU memory

The full configuration file is shown below:

```yaml
# Contains the path to the directory or file to process
path: PATH_TO_YOUR_DATA

preprocessing:
  # enable/disable preprocessing
  state: True
  # key for H5 or ZARR, can be set to null if only one key exists in each file
  key: Null
  # channel to use if input image has shape CZYX or CYX, otherwise set to null
  channel: Null
  # create a new sub folder where all results will be stored
  save_directory: 'PreProcessing'
  # rescaling the volume is essential for the generalization of the networks. The rescaling factor can be computed as the resolution
  # of the volume at hand divided by the resolution of the dataset used in training. Be careful, if the difference is too large check for a different model.
  factor: [1.0, 1.0, 1.0]
  # the order of the spline interpolation
  order: 2
  # cropping out areas of little interest can drastically improve the performance of plantseg.
  # crop volume has to be input using the numpy slicing convention [b_z:e_z, b_x:e_x, b_y:e_y], where b_zxy is the
  # first point of a bounding box and e_zxy is the second. eg: [:, 100:500, 400:900]
  crop_volume: '[:,:,:]'
  # optional: perform Gaussian smoothing or median filtering on the input.
  filter:
    # enable/disable filtering
    state: False
    # Accepted values: 'gaussian'/'median'
    type: gaussian
    # sigma (gaussian) or disc radius (median)
    filter_param: 1.0

cnn_prediction:
  # enable/disable UNet prediction
  state: True
  # key for H5 or ZARR, can be set to null if only one key exists in each file; null is recommended if the previous steps has state True
  key: Null
  # channel to use if input image has shape CZYX or CYX, otherwise set to null; null is recommended if the previous steps has state True
  channel: Null
  # Trained model name, more info on available models and custom models in the README
  model_name: 'generic_plant_nuclei_3D'
  # If a CUDA capable gpu is available and corrected setup use "cuda", if not you can use "cpu" for cpu only inference (slower)
  device: 'cuda'
  # (int or tuple) padding to be removed from each axis in a given patch in order to avoid checkerboard artifacts
  patch_halo: [64, 64, 64]
  # how many subprocesses to use for data loading
  num_workers: 8
  # patch size given to the network (adapt to fit in your GPU mem)
  patch: [192, 256, 256]
  # stride between patches will be computed as `stride_ratio * patch`
  # recommended values are in range `[0.5, 0.75]` to make sure the patches have enough overlap to get smooth prediction maps
  stride_ratio: 0.50
  # If "True" forces downloading networks from the online repos
  model_update: False

cnn_postprocessing:
  # enable/disable cnn post processing
  state: True
  # key for H5 or ZARR, can be set to null if only one key exists in each file; null is recommended if the previous steps has state True
  key: Null
  # channel to use if input image has shape CZYX or CYX, otherwise set to null; null is recommended if the previous steps has state True
  channel: 1
  # if True convert to result to tiff
  tiff: True
  # rescaling factor
  factor: [1, 1, 1]
  # spline order for rescaling
  order: 2

segmentation:
  # enable/disable segmentation
  state: True
  # key for H5 or ZARR, can be set to null if only one key exists in each file; null is recommended if the previous steps has state True
  key: 'predictions'
  # channel to use if prediction has shape CZYX or CYX, otherwise set to null; null is recommended if the previous steps has state True
  channel: 1
  # Name of the algorithm to use for inferences. Options: MultiCut, MutexWS, GASP, DtWatershed
  name: 'GASP'
  # Segmentation specific parameters here
  # balance under-/over-segmentation; 0 - aim for undersegmentation, 1 - aim for oversegmentation. (Not active for DtWatershed)
  beta: 0.5
  # directory where to save the results
  save_directory: 'GASP'
  # enable/disable watershed
  run_ws: True
  # use 2D instead of 3D watershed
  ws_2D: False
  # probability maps threshold
  ws_threshold: 0.4
  # set the minimum superpixels size
  ws_minsize: 50
  # sigma for the gaussian smoothing of the distance transform
  ws_sigma: 2.0
  # sigma for the gaussian smoothing of boundary
  ws_w_sigma: 0
  # set the minimum segment size in the final segmentation. (Not active for DtWatershed)
  post_minsize: 100

segmentation_postprocessing:
  # enable/disable segmentation post processing
  state: True
  # key for H5 or ZARR, can be set to null if only one key exists in each file; null is recommended if the previous steps has state True
  key: Null
  # channel to use if input image has shape CZYX or CYX, otherwise set to null; null is recommended if the previous steps has state True
  channel: Null
  # if True convert to result to tiff
  tiff: True
  # rescaling factor
  factor: [1, 1, 1]
  # spline order for rescaling (keep 0 for segmentation post processing
  order: 0
  # save raw input in the output segmentation file h5 file
  save_raw: False

```

### Prediction

```shell
plantseg --config CONFIG_PATH
```

where CONFIG_PATH is the path to the YAML configuration file. For example, if you want to use the model with the example configuration file `configs/config_pred_wide_applicability.yaml`:

```shell
cd ovules-instance-segmentation/plantseg/
CUDA_VISIBLE_DEVICES=0 plantseg --config configs/train_and_infer.yml
```

### Specifying a Graphic Card (GPU)

If you need to specify a graphic card, for example to use the No. 7 card (the eighth), do:

```shell
CUDA_VISIBLE_DEVICES=7 plantseg --config CONFIG_PATH
```

If you have only one graphic card, use `CUDA_VISIBLE_DEVICES=0` to select the first card (No. 0).

## Cite

If you find this work useful, please cite both papers:

```bibtex
@article {Vijayan2024.02.19.580954,
  author = {Athul Vijayan and Tejasvinee Atul Mody and Qin Yu and Adrian Wolny and Lorenzo Cerrone and Soeren Strauss and Miltos Tsiantis and Richard S. Smith and Fred Hamprecht and Anna Kreshuk and Kay Schneitz},
  title = {A deep learning-based toolkit for 3D nuclei segmentation and quantitative analysis in cellular and tissue context},
  elocation-id = {2024.02.19.580954},
  year = {2024},
  doi = {10.1101/2024.02.19.580954},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2024/02/21/2024.02.19.580954},
  eprint = {https://www.biorxiv.org/content/early/2024/02/21/2024.02.19.580954.full.pdf},
  journal = {bioRxiv}
}

@article{wolny2020accurate,
  title={Accurate and versatile 3D segmentation of plant tissues at cellular resolution},
  author={Wolny, Adrian and Cerrone, Lorenzo and Vijayan, Athul and Tofanelli, Rachele and Barro, Amaya Vilches and Louveaux, Marion and Wenzl, Christian and Strauss, S{\"o}ren and Wilson-S{\'a}nchez, David and Lymbouridou, Rena and others},
  journal={Elife},
  volume={9},
  pages={e57613},
  year={2020},
  publisher={eLife Sciences Publications Limited}
}
```

## PlantSeg Version and Code

See [PlantSeg's website](https://github.com/hci-unihd/plant-seg) for more details. The PlantSeg version v1.4.3 was used for testing, and PlantSeg v1.6.2 was released for this paper.
