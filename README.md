# ONCE Benchmark 
## Using the Flex_Match method



This is a reproduced benchmark for 3D object detection on the [ONCE](https://once-for-auto-driving.github.io/index.html) (One Million Scenes) dataset. 

The code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Introduction
We provide the dataset API and some reproduced models on the ONCE dataset. 

## Installation
The repo is based on OpenPCDet. If you have already installed OpenPCDet (version >= v0.3.0), you can skip this part and use the existing environment, but remember to re-compile CUDA operators by
```shell
pip install -r requirements.txt 
python setup.py develop
```
If you haven't installed OpenPCDet, please refer to [INSTALL.md](docs/INSTALL.md) for the installation.

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.

1. Preparation the dataset
   
  * Flow these instructure to organize the data
  ```
  ONCE_Benchmark
├── data
│   ├── once
│   │   │── ImageSets
|   |   |   ├──train.txt
|   |   |   ├──val.txt
|   |   |   ├──test.txt
|   |   |   ├──raw_small.txt (100k unlabeled)
|   |   |   ├──raw_medium.txt (500k unlabeled)
|   |   |   ├──raw_large.txt (1M unlabeled)
│   │   │── data
│   │   │   ├──000000
|   |   |   |   |──000000.json (infos)
|   |   |   |   |──lidar_roof (point clouds)
|   |   |   |   |   |──frame_timestamp_1.bin
|   |   |   |   |  ...
|   |   |   |   |──cam0[1-9] (images)
|   |   |   |   |   |──frame_timestamp_1.jpg
|   |   |   |   |  ...
|   |   |   |  ...
├── pcdet
├── tools
```
  * Using once_dataset to generate the data

```
python -m pcdet.datasets.once.once_dataset --func create_once_infos --cfg_file tools/cfgs/dataset_configs/once_dataset.yaml

```
2. Traing with signal GPU
```
 python semi_train.py --cfg_file ./cfgs/once_models/semi_learning_models/ioumatch3d_second_small.yaml
```
3. Traing with mutil GPUs
```
bash scripts/dist_train.sh 2 --cfg_file ./cfgs/once_models/semi_learning_models/ioumatch3d_second_small.yaml
```