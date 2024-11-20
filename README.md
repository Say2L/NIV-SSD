## ["NIV-SSD: Neighbor IoU-Voting Single-Stage Object Detector From Point Cloud"](https://arxiv.org/abs/2401.12447)

Thanks for the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), this implementation of the NIV-SSD is mainly based on the pcdet v0.6.

Abstract: Previous single-stage detectors typically suffer the misalignment between localization accuracy and classification confidence. To solve the misalignment problem, we introduce a novel rectification method named neighbor IoU-voting (NIV) strategy. Typically, classification and regression are treated as separate branches, making it challenging to establish a connection between them. Consequently, the classification confidence cannot accurately reflect the regression quality. NIV strategy can serve as a bridge between classification and regression branches by calculating two types of statistical data from the regression output to correct the classification confidence. Furthermore, to alleviate the imbalance of detection accuracy for complete objects with dense points (easy objects) and incomplete objects with sparse points (difficult objects), we propose a new data augmentation scheme named object resampling. It undersamples easy objects and oversamples difficult objects by randomly transforming part of easy objects into difficult objects. Finally, combining the NIV strategy and object resampling augmentation, we design an efficient single-stage detector termed NIV-SSD. Extensive experiments on several datasets indicate the effectiveness of the NIV strategy and the competitive performance of the NIV-SSD detector.

### 1. Recommended Environment

- Linux (tested on Ubuntu 20.04)
- Python 3.6+
- PyTorch 1.1 or higher (tested on PyTorch 1.13)
- CUDA 9.0 or higher (tested on 11.6)

### 2. Set the Environment

```shell
pip install -r requirement.txt
python setup.py develop
```

### 3. Data Preparation

- Prepare [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and [road planes](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing)

```shell
# Download KITTI and organize it into the following form:
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2

# Generatedata infos:
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

### 4. Train

- Train with a single GPU

```shell
python tools/train.py --cfg_file ${CONFIG_FILE}

# e.g.,
python tools/train.py --cfg_file tools/cfgs/kitti_models/niv-ssd.yaml
```

- Train with multiple GPUs or multiple machines

```shell
bash tools/scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
# or 
bash tools/scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# e.g.,
bash tools/scripts/dist_train.sh 8 --cfg_file tools/cfgs/kitti_models/niv-ssd.yaml
```

### 5. Test

- Test with a pretrained model:

```shell
python tools/test.py --cfg_file ${CONFIG_FILE} --ckpt ${CKPT}

# e.g., 
python tools/test.py --cfg_file tools/cfgs/kitti_models/niv-ssd.yaml --ckpt {path}
```
## Paper

Please cite our paper if you find our work useful for your research:

```
@article{liu2024niv,
  title={NIV-SSD: Neighbor IoU-voting single-stage object detector from point cloud},
  author={Liu, Shuai and Wang, Di and Wang, Quan and Huang, Kai},
  journal={Neurocomputing},
  pages={127987},
  year={2024}
}
```
