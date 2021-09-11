# LFPNet

Official repository for the paper [**Long-Range Feature Propagating for Natural Image Matting.**](https://arxiv.org/)

## Description

LFPNet is a matting network that can process high-resolution images on GPUs. It leverages long-range context features for alpha matte estimation. With the context features, LFPNet can estimate alpha mattes well for some difficult data (e.g. the foreground colors are similar to the background colors in a local area, the foreground regions in the trimap are annotated as unknown regions).

LFPNet是一个可以在GPU上进行高分辨率图像抠图的网络。它利用长距离上下文特征帮助预测前景的透明度遮罩。算法对一些困难数据也能很好地预测透明度遮罩，比如局部前景背景颜色相近、三分图中前景标为未知区域等。

## Requirements
#### Hardware:

A GPU with 11GB memory for any resolution images.

#### Packages:

- torch >= 1.6
- numpy >= 1.16
- opencv-python >= 4.0
- imgaug >= 0.3.0

## Models
**The model can only be used and distributed for noncommercial purposes.** 

| Model Name  |   Size   | MSE | SAD | Grad | Conn |
| :------------: |:-----------:| :----:|:---:|:---:|:---:|
| [LFPNet](https://1drv.ms/u/s!AtvVjmAsYCTQdbNFLnArtH4LZDk?e=3BCBjS) | 449MiB | 3.60266 | 22.3688 | 7.59364 | 17.1374 |

Results on Adobe Composite 1K (with TTA): https://1drv.ms/u/s!AtvVjmAsYCTQce52EKbuHDmvPWg?e=qZrw5I

## Evaluation
We provide the script `eval.py`  for evaluation (patch based with TTA).

## Citation

```
@inproceedings{liu2021lfpnet,
  title   = {Long-Range Feature Propagating for Natural Image Matting},
  author  = {Qinglin Liu, Haozhe Xie, Shengping Zhang, Bineng Zhong and Rongrong Ji},
  booktitle={ACM Multimedia},
  year={2021}
}
```


