# Nuclear Segmentation Pipelines <!-- omit in toc -->

![stardist_raw_and_segmentation](https://zenodo.org/records/8432366/files/stardist_raw_and_segmentation.jpg)

The GoNuclear repository hosts the code and guides for the pipelines used in the paper [_A deep learning-based toolkit for 3D nuclei segmentation and quantitative analysis in cellular and tissue context_](https://doi.org/10.1242/dev.202800). It is structured into four folders:

- **stardist/** contains a 3D StarDist training and inference pipeline, `run-stardist`.
  - The StarDist model is automatically downloaded and can be used directly by this package.
- **plantseg/** contains configuration files for training and inference with PlantSeg.
  - The PlantSeg model is included in the latest version of PlantSeg.
- **cellpose/** contains scripts for training and inference with Cellpose.
  - The Cellpose model can be directly added to the Cellpose GUI or used via the command line.
- **evaluation/** contains modules for evaluating segmentation results.

These are described in detail in the [**GoNuclear documentation** :book:](https://kreshuklab.github.io/go-nuclear/).

## Data and Models

The median size of nuclei in the training data is `[16, 32, 32]` in `ZYX` order. The manuscript and documentation provide guidance on how to best segment your data with each pipeline. All three final models are available on BioImage Model Zoo:

- [StarDist model `modest-octopus`](https://bioimage.io/#/artifacts/modest-octopus)
- [PlantSeg model `efficient-chipmunk`](https://bioimage.io/#/artifacts/efficient-chipmunk)
- [Cellpose model `philosophical-panda`](https://bioimage.io/#/artifacts/philosophical-panda)

All training data and models are available on [BioImage Archive S-BIAD1026](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD1026), organised in the following structure:

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

## Citation

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
