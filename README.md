# ONCE Benchmark 
## Using the Flex_Match method



This is a reproduced benchmark for 3D object detection on the [ONCE](https://once-for-auto-driving.github.io/index.html) (One Million Scenes) dataset. 

The code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Introduction
We provide the dataset API and some reproduced models on the ONCE dataset. 

## Installation
The repo is based on OpenPCDet. If you have already installed OpenPCDet (version >= v0.3.0), you can skip this part and use the existing environment, but remember to re-compile CUDA operators by
```
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv
git checkout -f 7342772
cd third_party
git clone https://github.com/pybind/pybind11.git
cd pybind11
git checkout -f 085a29436a8c472caaaf7157aa644b571079bcaa
cd /spconv
python setup.py bdist_wheel
cd dist
pip install *
```
```
pip install -r requirements.txt 
python setup.py develop
```

Init Conda

```
conda create --name Once python=3.6
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 

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

# Decripation of methods
## 获得比例系数的方法：

```
  Mask_acc=(classwise_acc+iouwise_acc)/2
```
Method1：
1. 使用Mask动态调整每次过滤pseduo_label的threshold的系数：
当前方法：
  * 当前不调正threshold，之前做过每个iteration调整threshold的方法，对于个别类的效果AP效果有提升，但是loss前期振荡严重，AP也不稳定
  * 当前不加入Mask对threshold进行调整
  
可以尝试的方法：
  * 可以尝试在使用累加的方法统计在每个epoch中通过threshold的比例用于调加threshold，但是需要在每个epoch进行调整，在这里可能需要新增加一个mask，不同于之前的Mask_acc


可能有的问题：
  * 如果每个iteration都调整系数的话可能造成无法筛选好初期的标签，造成loss开始振荡很严重
  
2. 使用Mask给予loss一个系数，让训练不好的类有更大的loss
当前方法：
   * 当前方法将loss从0-1之间映射到了1.5到1之间，也就是之前训练结果较好的模型在之后的训练过程中的mask=1，训练越不好的模型乘的系数越大，但是结果都在1-1.5之间
  

可以尝试的方法：
  * 将loss在每个iteartion中调整更改为每个epoch调整，但是这样的话需要用到累加的方法，可能存在对于本身就学习不好的类的比重过低，这个方法需要考虑

可能存在的问题：
  * 感觉loss这个参数范围不好界定，目前只能放在1.5-1之间

3. 当前模型没有使用MEA对teacher_model进行参数调整，teacher的参数还是强烈依赖鱼pre_train
  * 在dev_mea中经行了MEA的尝试，结果远超不是用ema的结果
  * 很奇怪的一点，作者说使用MEA效果不好，等我试一试


# Contrast of methods
## Not using EMA
```
Epoch=30
Prtrain=1
```

| Method                | Car   | Bus   | Truck  | Pedestrain | Cyclist | mAP   | Description |
| :-------------:       | :---: | :---: | :---:  | :---:      | :---:   | :---: | :---:       |
| Baseline_30epoch_Out  | 45.02 | 8.82  | 1.72   | 5.56       | 15.80   | 15.38 |
| Flex_Loss_30epoch_Out | 45.09 | 13.84 | 1.36   | 4.04       | 12.88   | 15.44 |

## Using EMA(Student_Model)
```
Epoch=30
Prtrain=1
```
| Method                | Car   | Bus   | Truck  | Pedestrain | Cyclist | mAP   | Description |
| :-------------:       | :---: | :---: | :---:  | :---:      | :---:   | :---: | :---:       |
| Baseline_Loss_Ema_out | 57.12 | 15.89 | 3.49   | 14.50      | 16.30   | 21.46 |
| Flex_Loss_Ema_Out     | 59.52 | 27.06 | 4.09   | 12.74      | 16.11   | 23.91 | 每个iteration都更新loss的参数
## Adjust method(Student_Model)
```
Epoch=25
Prtrain=20
```
| Method                | Vehicle   | Pedestrian   | Cyclist  | mAP     | Description |
| :-------------:       | :---:     | :---:        | :---:    | :---:   | :---:       |
| Baseline              | 51.07     | 19.26        | 38.22    | 42.14   | 这里的Vehicle是使用均加的，可能不对，因为本身有5个类|
| Plan_A                | 64.96     | 19.48        | 43.89    | 42.78   | 给threshold增加T       |
| Plan_B                | 63.73     | 19.50        | 41.54    | 41.59   | 给Loss增加了T |