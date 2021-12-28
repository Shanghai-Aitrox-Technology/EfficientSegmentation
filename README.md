# EfficientSegmentation
## Introduction
- EfficientSegmentation is an open source, PyTorch-based segmentation method for 3D medical image. 
- For more information about efficientSegmentation, please read the following paper:
[Efficient Context-Aware Network for Abdominal Multi-organ Segmentation](https://arxiv.org/abs/2109.10601). Please also cite this paper if you are using the method for your research!

## Features
- A whole-volume-based coarse-to-fine segmentation framework. The segmentation network is decomposed into different components, including basic encoder, slim decoder and efficient context blocks.
  Anisotropic convolution block and anisotropic context block are designed for efficient and effective segmentation.
- Pre-process data in multi-process. Distributed and Apex training support. In the inference phase, preprocess and postprocess are computed in GPU.
- This method won the 1st place on the [2021-MICCAI-FLARE](https://flare.grand-challenge.org/Awards/) challenge. Where participants were required to effectively and efficiently segment multi-organ in abdominal CT.
## Benchmark
| Task | Architecture | Parameters(MB) | Flops(GB) | DSC | NSC | Inference time(s) | GPU memory(MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[FLARE21](https://flare.grand-challenge.org/FLARE21/)| BaseUNet | 11 | 812 | 0.908 | 0.837 | 0.92 | 3183 |
|[FLARE21](https://flare.grand-challenge.org/FLARE21/)| EfficientSegNet | 9 | 333 | 0.919 | 0.848 | 0.46 | 2269 |


## Installation
#### Environment
- Ubuntu 16.04.12
- Python 3.6+
- Pytorch 1.5.0+
- CUDA 10.0+ 

1.Git clone
```
git clone https://github.com/Shanghai-Aitrox-Technology/EfficientSegmentation.git
```

2.Install Nvidia Apex
- Perform the following command:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```

3.Install dependencies
```
pip install -r requirements.txt
```

## Get Started
### preprocessing
1. Download [FLARE21](https://flare.grand-challenge.org/Data/), resulting in 361 training images and masks, 50 validation images.
2. Copy image and mask to 'FlareSeg/dataset/' folder.
3. Edit the 'FlareSeg/data_prepare/config.yaml'. 
   'DATA_BASE_DIR'(Default: FlareSeg/dataset/) is the base dir of databases.
   If set the 'IS_SPLIT_5FOLD'(Default: False) to true, 5-fold cross-validation datasets will be generated.
4. Run the data preprocess with the following command:
```bash
cd FlareSeg/data_prepare
python run.py
```
The image data and lmdb file are stored in the following structure:
```wiki
DATA_BASE_DIR directory structure：
├── train_images
   ├── train_000_0000.nii.gz
   ├── train_001_0000.nii.gz
   ├── train_002_0000.nii.gz
   ├── ...
├── train_mask
   ├── train_000.nii.gz
   ├── train_001.nii.gz
   ├── train_002.nii.gz
   ├── ...
└── val_images
    ├── validation_001_0000.nii.gz
    ├── validation_002_0000.nii.gz
    ├── validation_003_0000.nii.gz
    ├── ...
├── file_list
    ├──'train_series_uids.txt', 
    ├──'val_series_uids.txt',
    ├──'lesion_case.txt',
├── db
    ├──seg_raw_train         # The 361 training data information.
    ├──seg_raw_test          # The 50 validation images information.
    ├──seg_train_database    # The default training database.
    ├──seg_val_database      # The default validation database.
    ├──seg_pre-process_database # Temporary database.
    ├──seg_train_fold_1
    ├──seg_val_fold_1
├── coarse_image
    ├──160_160_160
          ├── train_000.npy
          ├── train_001.npy
          ├── ...
├── coarse_mask
    ├──160_160_160
          ├── train_000.npy
          ├── train_001.npy
          ├── ...
├── fine_image
    ├──192_192_192
          ├── train_000.npy
          ├── train_001.npy
          ├──  ...
├── fine_mask
    ├──192_192_192
          ├── train_000.npy
          ├── train_001.npy
          ├── ...
```
The data information is stored in the lmdb file with the following format:
```wiki
{
    series_id = {
        'image_path': data.image_path,
        'mask_path': data.mask_path,
        'smooth_mask_path': data.smooth_mask_path,
        'coarse_image_path': data.coarse_image_path,
        'coarse_mask_path': data.coarse_mask_path,
        'fine_image_path': data.fine_image_path,
        'fine_mask_path': data.fine_mask_path
    }
}
```

### Models
- Models can be downloaded through [Baidu Netdisk](https://pan.baidu.com/s/1rRWKDdUTTBNl-uwwSEbxbg), password: vjy5 
- Put the models in the "FlareSeg/model_weights/" folder.

### AbdomenCT-1K models
- We also trained models in [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K) dataset. 
- You can download the models through [Baidu Netdisk](https://pan.baidu.com/s/1SiP6LzCS-3Py4W2MeM5vzQ), password:5k77

### Training
Remark: Coarse segmentation is trained on Nvidia GeForce 2080Ti(Number:8), while fine segmentation on Nvidia A100(Number:4). If you use different hardware, please set the "ENVIRONMENT.NUM_GPU", "DATA_LOADER.NUM_WORKER" and "DATA_LOADER.BATCH_SIZE" in 'FlareSeg/coarse_base_seg/config.yaml' and 'FlareSeg/fine_efficient_seg/config.yaml' files. You also need to set the 'nproc_per_node' in 'FlareSeg/coarse_base_seg/run.sh' file.
#### Coarse segmentation:
- Edit the 'FlareSeg/coarse_base_seg/config.yaml' and 'FlareSeg/coarse_base_seg/run.sh'
- Train coarse segmentation with the following command:
```bash
cd FlareSeg/coarse_base_seg
sh run.sh
```

#### Fine segmentation:
- Put the trained coarse model in the 'FlareSeg/model_weights/base_coarse_model/' folder.
- Edit the 'FlareSeg/fine_efficient_seg/config.yaml'.
- Edit the 'FlareSeg/fine_efficient_seg/run.py', set the 'tune_params' for different experiments.
- Train fine segmentation with the following command:
```bash
cd  FlareSeg/fine_efficient_seg
sh run.sh
```

### Inference:
- Put the trained models in the 'FlareSeg/model_weights/' folder.
- Run the inference with the following command:
```bash
sh predict.sh
```

### Evaluation:
Refer to [FLARE2021 Evaluation](https://github.com/JunMa11/FLARE2021/tree/main/Evaluation).

## Contact
This repository is currently maintained by Fan Zhang (zf2016@mail.ustc.edu.cn) and Yu Wang (wangyu@fosun.com)

## References
[1] Z. e. a. Zhu, “A 3d coarse-to-fine framework for volumetric medical image segmentation.” 2018 International Conference on 3D Vision (3DV), 2018.

[2] Q. e. a. Hou, “Strip pooling: Rethinking spatial pooling for scene parsing.” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

## Acknowledgement
Thanks for FLARE organizers with the donation of the dataset.
