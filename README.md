# MsIFT : Multi-source Image Fusion Transformer
This is the official implementation of ***MsIFT*** (Information Fusion), a transformer-based image fusion method for classification and segmantation. For more details, please refer to:

**MsIFT : Multi-source Image Fusion Transformer[[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9508842)**  <br />
Xin Zhang , Hangzhi Jiang, Nuo Xu, Lei Ni, Chunlei Huo , Chunhong Pan<br />

![intro](framework.jpg)

Our code is based on mmsegmentation and mmclassification.

## Usage
### Requirements
**For classification:**

- Python 3.6+
- PyTorch 1.3+
- mmcv 1.1.4+
- torchvision
- timm
- mmcv-full==1.3.17
- mmcls==0.8.0


## Getting Started
## Preparation
Clone the code
```
git clone git@github.com:XinZhangNLPR/MsIFT.git
```


Download the model weight used in the paper:

#### VAIS dataset
|                                             |Accuracy|download | 
|---------------------------------------------|:-------:|:---------:|
|[MsIFT](classification/work_dirs/VAIS/SOTA_VAIS.py)| 92.3|[Google](https://drive.google.com/file/d/1zUT3dc_swMoL5w8s5DGCloj65aR0Er1W/view?usp=sharing)

Put the model to ***classification/work_dirs/VAIS/***
#### DFC2013 dataset
|                                             |Accuracy|download | 
|---------------------------------------------|:-------:|:---------:|
| [MsIFT](classification/work_dirs/DFC2013/finetune_9285.py)| 93.02 |[Google](https://drive.google.com/file/d/13eJiJymZYaZjMxMCuqHE0FEfmD_Q4uAQ/view?usp=sharing)

Put the model to ***classification/work_dirs/DFC2013/***

#### SpaceNet6 dataset
|                                             |Seg Method|mIoU|Accuarcy | download | 
|---------------------------------------------|:-------:|:-------:|:---------:|:---------:|:---------:|
| [MsIFT](segmentation/work_dir/PSPNet/pspnet_r50-d8.py) |PSPNet|67.51|70.49|[Google](https://drive.google.com/file/d/1S_LFVtEoE_L6hJpu8FGzah4DszFFe6ma/view?usp=sharing)
| [MsIFT](segmentation/work_dir/DANet/danet.py) |DANet|67.94|70.82|[Google](https://drive.google.com/file/d/1r-IHv73nZda4EEdGSZ3N7gYWLaeATd_y/view?usp=sharing)

Put the PSPNet model to ***segmentation/work_dir/PSPNet/***
Put the DANet model to ***segmentation/work_dir/DANet/***

## Evaluate
1.Multi-GPUs Test
```shell
./tools/dist_test.sh work_dirs/HTL_1x_faster/HTL_ins_faster_rcnn_r50_fpn_1x_hrsid.py work_dirs/HTL_1x_faster/epoch_11.pth 8 --eval mAP
```
2.Single-GPU Test
```shell
python tools/test.py work_dirs/HTL_1x_faster/HTL_ins_faster_rcnn_r50_fpn_1x_hrsid.py work_dirs/HTL_1x_faster/epoch_11.pth --eval mAP
```

## Citation

**Coming soon**
