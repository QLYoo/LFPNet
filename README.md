# LFPNet

Official repository for the paper **Long-Range Feature Propagating for Natural Image Matting.**

## Requirements
#### Hardware:

A GPU with 11GB memory for any resolution images.

#### Packages:

- torch >= 1.6
- numpy >= 1.16
- opencv-python >= 4.0
- imgaug >= 0.3.0

## Models
**The model trained on Adobe Image Matting Dataset can only be used and distributed for noncommercial purposes.** 

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


