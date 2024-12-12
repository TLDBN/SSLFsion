# SSLFusion: Scale and Space Aligned Latent Fusion Model for Multimodal 3D Object Detection

This is an official code release of [SSLFusion](https://github.com/TLDBN/SSLFsion/edit/main/README.md)(Scale and Space Aligned Latent Fusion Model for Multimodal 3D Object Detection). This code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

![image](https://github.com/TLDBN/SSLFusion/blob/main/tools/images/Overview.png)
## Dataset Preparation & Installation

Please refer to OpenPCDet's original document for [dataset preparation](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) and [installation](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md).

### Training

``` python
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# or 

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

For example(Train SSLFusion on KITTI dataset with 4 RTX 3090):

``` python
sh scripts/dist_train.sh 4 --cfg_file tools/cfgs/kitti/voxel_rcnn_3class_SSLFusion.yaml
```

## Acknowledgement

[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

[MMDet3D](https://github.com/open-mmlab/mmdetection3d)

## Citation 
If you find this project useful in your research, please consider cite:


```
@inproceedings{
  sslfusion,
  title={SSLFusion: Scale and Space Aligned Latent Fusion Model for Multimodal 3D Object Detection},
  booktitle={The 39th Annual AAAI Conference on Artificial Intelligence},
  year={2024},
  url={https://openreview.net/forum?id=Zz7a1xEWrf}
}
```



