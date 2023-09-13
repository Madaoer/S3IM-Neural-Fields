# S3IM
### [Paper](https://arxiv.org/abs/2308.07032) | [Project Page](https://madaoer.github.io/s3im_nerf/)

This repository contains the official pytorch implementation of our paper: [S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields]. 

The implementation of S3IM is quite simple. In this repo, we provide usage examples of S3IM and present some video demos. 

[SDFStudio](https://github.com/autonomousvision/sdfstudio) has supported our S3IM method.

# Abstract

Recently, Neural Radiance Field (NeRF) has shown great success in rendering novel-view images of a given scene by learning an implicit representation with only posed RGB images. NeRF and relevant neural field methods (e.g., neural surface representation) typically optimize a point-wise loss and make point-wise predictions, where one data point corresponds to one pixel. Unfortunately, this line of research failed to use the collective supervision of distant pixels, although it is known that pixels in an image or scene can provide rich structural information. To the best of our knowledge, we are the first to design a nonlocal multiplex training paradigm for NeRF and relevant neural field methods via a novel Stochastic Structural SIMilarity (S3IM) loss that processes multiple data points as a whole set instead of process multiple inputs independently. Our extensive experiments demonstrate the unreasonable effectiveness of S3IM in improving NeRF and neural surface representation for nearly free. The improvements of quality metrics can be particularly significant for those relatively difficult tasks: e.g., the test MSE loss unexpectedly drops by more than **90%** for TensoRF and DVGO over eight novel view synthesis tasks; a **198%** F-score gain and a **64%** Chamfer L1 distance reduction for NeuS over eight surface reconstruction tasks. Moreover, S3IM is consistently robust even with sparse inputs, corrupted images, and dynamic scenes.


# Video Demo


## TensoRF RGB results w/o and with S3IM (refer to paper table 2) 

https://github.com/Madaoer/S3IM-Neural-Fields/assets/111342277/54d7bc4f-a03f-4087-972b-a9e82f01a258
<div align="center">Left: Standard Training (baseline);Right: Multiplex Training via S3IM (ours).</div>

## TensoRF Depth results w/o and with S3IM (refer to paper table 2) 

https://github.com/Madaoer/S3IM-Neural-Fields/assets/111342277/efd22307-aebe-496b-8085-a7e06353d3a8
<div align="center">Left: Standard Training (baseline);Right: Multiplex Training via S3IM (ours).</div>


## DVGO RGB results w/o and with S3IM (refer to paper figure 2)

https://github.com/Madaoer/S3IM-Neural-Fields/assets/111342277/02634f9c-ad8d-41c0-a21c-601cf7bfc970
<div align="center">Left: Standard Training (baseline);Right: Multiplex Training via S3IM (ours).</div>


## DVGO Depth results w/o and with S3IM (refer to paper figure 2)

https://github.com/Madaoer/S3IM-Neural-Fields/assets/111342277/200042dd-81e0-4af9-9914-22aee4a9f4b5
<div align="center">Left: Standard Training (baseline);Right: Multiplex Training via S3IM (ours).</div>


## Installation

#### Tested on Ubuntu 20.04 + Pytorch 1.10.0 + cu113

Install environment:
```
pip install -r requirements.txt
```


## Dataset
* [Replica](https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/Replica.tar) 

You can try other dataset as well. S3IM is powerful and robust.


## Hyperparameters
The recommended setting for S3IM is 
```python
s3im_kernel=4
s3im_stride=4
s3im_repeat_time=10 # repeat time of s3im
s3im_patch_height=64 # height of random mini-patch in s3im 
s3im_patch_width=64 # width of random mini-patch in s3im 
```

## Quick Start
You can prepare the dataset using the following script:
```
sh scripts/preprocess_data/prepare_data.sh
```

You can train the TensoRF/DVGO model with s3im using the following script:
```
#for TensoRF
sh scripts/TensoRF/train_replica.sh
#for DVGO
sh scripts/DVGO/train_replica.sh
```

You can eval the TensoRF/DVGO model with s3im using the following script:
```
#for TensoRF
sh scripts/TensoRF/eval_replica.sh
#for DVGO
sh scripts/DVGO/eval_replica.sh
```

You can render a video based on TensoRF/DVGO with s3im using the following script:
```
#for TensoRF
sh scripts/TensoRF/render_path_replica.sh
#for DVGO
sh scripts/DVGO/render_path_replica.sh
```

If you want to try other setting in S3IM, you can modify the config file in
```
#for TensoRF
models/TensoRF/configs/replica_exp/replica_scan1_s3im_1.0.txt
#for DVGO
models/DVGO/configs/replica_exp/replica_scan1_s3im_1.0.txt
```



## Performance
Here we report our results in Replica Dataset using TensoRF. Please refer to our paper for more quantitative results.


<img width="651" alt="results in Replica" src="https://github.com/Madaoer/S3IM/assets/111342277/44051a11-f5af-47aa-899a-47043416c02f">



## Citation
If you find our code or paper helps, please consider citing:
```
@inproceedings{xie2023s3im,
  title = {S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields},
  author = {Xie, Zeke and Yang, Xindi and Yang, Yujie and Sun, Qi and Jiang, Yixiang and Wang, Haoran and Cai, Yunfeng and Sun, Mingming},
  booktitle = {International Conference on Computer Vision},
  year = {2023}
}
```

## Acknowledgement
The code base is adapted from [DVGO](https://github.com/sunset1995/DirectVoxGO) and [TensoRF](https://github.com/apchenstu/TensoRF), thanks for their great work!
