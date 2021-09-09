# EfficientSegmentation
## Introduction
EfficientSegmentation is an open source, PyTorch-based segmentation framework for 3D medical image. 
## Features
- A whole-volume-based coarse-to-fine segmentation framework. The segmentation network is decomposed into different components, including encoder, decoder and context module.
  Anisotropic convolution block and anisotropic context block are designed for efficient and effective segmentation.
- Pre-process data in multi-process. Distributed and Apex training support. Postprocess is performed asynchronously in inference stage.
## Benchmark
| Task | Architecture | Parameters(MB) | Flops(GB) | DSC | NSC | Inference time(s) | GPU memory(MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[FLARE21](https://flare.grand-challenge.org/FLARE21/)| BaseUNet | 11 | 812 | 0.908 | 0.837 | 0.92 | 3183 |
|[FLARE21](https://flare.grand-challenge.org/FLARE21/)| EfficientSegNet | 9 | 333 | 0.919 | 0.848 | 0.46 | 2269 |

## Installation
### Installation by docker image
* Download the docker image.
```angular2html
  link: https://pan.baidu.com/s/1UkMwdntwAc5paCWHoZHj9w 
  password：9m3z
```
* Put the abdomen CT image in current folder $PWD/inputs/.
* Run the testing cases with the following code:
```bash
docker image load < fosun_aitrox.tgz
nvidia-docker container run --name fosun_aitrox --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ fosun_aitrox:latest /bin/bash -c "sh predict.sh"'
```

### Installation step by step
#### Environment
- Ubuntu 16.04.10
- Python 3.6+
- Pytorch 1.5.0+
- CUDA 11.0+

1.Git clone
```
git clone https://github.com/Shanghai-Aitrox-Technology/EfficientSegmentation.git
```

2.Install Nvidia Apex
- Perform the following command:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
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
python FlareSeg/data_prepare/run.py
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
### Training
#### Coarse segmentation:
- Edit the 'FlareSeg/coarse_base_seg/config.yaml'
- Train coarse segmentation with the following command:
```bash
cd FlareSeg/coarse_base_seg
sh run.sh
```

#### Fine segmentation:
- Edit the 'FlareSeg/fine_efficient_seg/config.yaml'. 
- Train fine segmentation with the following command:
```bash
cd  FlareSeg/fine_efficient_seg
sh run.sh
```

### Inference:
- The model weights are stored in 'FlareSeg/model_weights/'. 
- Run the inference with the following command:
```bash
sh predict.sh
```

## Contact
This repository is currently maintained by Fan Zhang (zf2016@mail.ustc.edu.cn) and Yu Wang (wangyu@fosun.com)

## Citation

## Acknowledgement
