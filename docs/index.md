# Nuclear Segmentation Pipelines <!-- omit in toc -->

The GoNuclear repository hosts the code and guides for the pipelines used in the paper [_A deep learning-based toolkit for 3D nuclei segmentation and quantitative analysis in cellular and tissue context_](https://doi.org/10.1242/dev.202800). It is structured in to four folders:

- **stardist/** contains a 3D StarDist training and inference pipeline, `run-stardist`.
- **plantseg/** contains configuration files for training and inference with PlantSeg.
- **cellpose/** contains scripts for training and inference with Cellpose.
- **evaluation/** contains modules for evaluating the segmentation results.

and are described in this documentation.

## Tools and Workflows

### StarDist

_See [GoNuclear Documentation - `run-stardist`](chapters/stardist/index.md) for more details._

This is one of the most important contribution of this repository. If your nuclei are more or less uniform in shape, please consider using the `run-stardist` pipeline in this repository. It generate separate and round instance segmentation masks for your nuclei images.

- The code and tutorial for running StarDist inference is in the `stardist/` folder
- The pretrained model is automatically downloaded during inference (also available at [BioImage.IO: StarDist Plant Nuclei 3D ResNet](https://bioimage.io/#/?id=10.5281%2Fzenodo.8421755))
- An example of segmentation results is shown below.

![stardist_raw_and_segmentation](https://zenodo.org/records/8432366/files/stardist_raw_and_segmentation.jpg)

### PlantSeg

_See [GoNuclear Documentation - PlantSeg](chapters/plantseg/index.md) for more details._

If your nuclei have irregular shapes, please consider using the PlantSeg pipeline. It generates instance masks for your nuclei images regardless of their nucleus size and shape.

- The code and tutorial for running PlantSeg inference is in the `plantseg/` folder
- The pretrained model is automatically downloaded during inference (also available at [BioImage.IO: PlantSeg Plant Nuclei 3D UNet](https://bioimage.io/#/?id=10.5281%2Fzenodo.8401064))
- An example of segmentation results is shown below.

![plantseg_raw_and_gasp_segmentation](https://zenodo.org/records/10070349/files/plantseg_raw_and_gasp_segmentation.jpg)

### Cellpose

_See [GoNuclear Documentation - Cellpose](chapters/cellpose/index.md) for more details._

- The guide for running Cellpose inference and training is in the `cellpose/` folder

## Data and Models

### Training Data and Trained Models

The training data is publicly available on [BioImage Archive S-BIAD1026](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD1026). I organised them in the following structure:

```bash
Training data
├── 2d/
│   ├── isotropic/
│   │   ├── gold/
│   │   └── initial/
│   └── original/
│       ├── gold/
│       └── README.txt
└── 3d_all_in_one/
    ├── 1135.h5
    ├── 1136.h5
    ├── 1137.h5
    ├── 1139.h5
    └── 1170.h5

Models
├── cellpose/
│   ├── cyto2_finetune/
│   │   └── gold/
│   ├── nuclei_finetune/
│   │   ├── gold/
│   │   └── initial/
│   └── scratch_trained/
│       └── gold/
├── plantseg/
│   └── 3dunet/
│       ├── gold/
│       ├── initial/
│       ├── platinum/
│       └── train_example.yml
└── stardist/
    ├── resnet/
    │   ├── gold/
    │   ├── initial/
    │   └── platinum/
    ├── train_example.yml
    └── unet/
        └── gold/
```

An example of the raw image:

![raw](https://zenodo.org/records/10070349/files/plantseg_raw.jpg)

Some key information about the training data is listed below:

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

**Note for training Cellpose:** The best image form for training StarDist and PlantSeg models are the original forms, i.e. the linked dataset is the one that provide the best results. However, to train Cellpose which only takes 2D training data, the images are prepared to be 2D slices of the rescaled isotropic 3D images. The 2D slices includes all XY, XZ and YZ slices ordered randomly by a random prefix in the file name. The 2D slices are saved as TIFF files and are provided along with the 3D images in the same [BioImage Archive S-BIAD1026](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD1026) repository.

### Preparing Data for Inference

Both HDF5 files and TIFF files can be directly used for both `run-stardist` and `plant-seg` inference. Go to the respective GoNuclear documentation for more details.

## Cite

If you find this work useful, please cite our paper and the respective tools' papers:

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
```
