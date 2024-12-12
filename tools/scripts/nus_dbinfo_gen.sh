#!/usr/bin/env bash
set -x

source ~/anaconda3/etc/profile.d/conda.sh
conda activate OpenPCD

python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file /nfs/volume-382-179/perception-detection/dingbonan/OpenPCDet-master/tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval --with_cam --with_cam_gt --share_memory

