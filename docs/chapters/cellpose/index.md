# Use Cellpose: A Guide <!-- omit in toc -->

This part of the repo concisely shows how to install, train and segment with Cellpose. In other word, it is a record of how Cellpose is used in this paper. Since my experiments show StarDist and PlantSeg have better 3D segmentation performance than Cellpose, this section is complete yet not extensive.

- [Installation](#installation)
  - [Install Miniconda](#install-miniconda)
  - [Install `cellpose` using `pip`](#install-cellpose-using-pip)
- [Segmentation](#segmentation)
  - [Data Preparation](#data-preparation)
  - [Segmentation Command](#segmentation-command)
- [Training](#training)
  - [Data Preparation](#data-preparation-1)
  - [Training Command](#training-command)
- [Cellpose Version and Code](#cellpose-version-and-code)


## Installation

It is recommended to install this package in an environment managed by `conda`. We start the guide by installing Mini-`conda`.

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

### Install `cellpose` using `pip`
To create and activate an `conda` environment for `cellpose`, then install `cellpose` in it, run the following commands in the terminal:
```bash
conda create --name cellpose python=3.8
conda activate cellpose
pip install cellpose
```

If you have a nvidia gpu, follow these steps to make use of it:
```bash
pip uninstall torch
conda install pytorch==1.12.0 cudatoolkit=11.3 -c pytorch
```

If you encounter error or need more explanation, go to [Cellpose's official instruction](https://github.com/MouseLand/cellpose#instructions).

## Segmentation

### Data Preparation
Cellpose inference only segmenet TIFF images, not HDF5. However, it can take 3D volumes as input.

### Segmentation Command

There are two ways of segmenting 3D images with Cellpose:
- Segment 3D images slice by slice then stitch 2D segmentation results into 3D segmentation results. With this approach, the images doesn't have to be isotropic, as long as the XY planes have similar properties as the training data.
    ```bash
    cellpose \
        --pretrained_model PATH_TO_MODEL \
        --savedir PATH_TO_OUTPUT_DIR \
        --dir PATH_TO_3D_TIFF_FOLDER \
        --diameter 26.5 \
        --verbose \
        --use_gpu \
        --stitch_threshold 0.9 \
        --chan 0 \
        --no_npy \
        --save_tif
    ```
- Compute spatial flow of 3D images in all dimensions then segment the images in 3D directly. You may choose to rescale the images to be isotropic before segmentation, or specify the anisotropy to let Cellpose deal with the rescaling. Here I show the later.
    ```bash
    cellpose \
        --pretrained_model PATH_TO_MODEL \
        --savedir PATH_TO_OUTPUT_DIR \
        --dir PATH_TO_3D_TIFF_FOLDER \
        --diameter 26.5 \
        --anisotropy 2.24 \
        --verbose \
        --use_gpu \
        --do_3D \
        --chan 0 \
        --no_npy \
        --save_tif
    ```


## Training

### Data Preparation
Cellpose training only takes 2D images as input. To train on 3D images, we first need to split the 3D images into 2D images. Note that 3D images are better to be rescaled for isotropy in the resulting 2D training data.

### Training Command
An example training command is shown below, which is used in the paper. The parameters `--learning_rate 0.1` and `--weight_decay 0.0001` are recommended by the [Cellpose official documentation](https://cellpose.readthedocs.io/en/latest/train.html).

```bash
cellpose --train --use_gpu \
    --dir PATH_TO_TRAINING_DATA \
    --pretrained_model nuclei \
    --learning_rate 0.1 \
    --weight_decay 0.0001 \
    --mask_filter _masks \
    --verbose
```

## Cellpose Version and Code
See [Cellpose's GitHub page](https://github.com/MouseLand/cellpose) for the code. Cellpose v2.0.5 was used for training and inference in this paper.
