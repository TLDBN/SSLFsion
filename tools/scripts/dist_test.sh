#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate OpenPCD
if [ -z $LUBAN_AVAILBLE_PORT_0 ]; then
    export LUBAN_AVAILBLE_PORT_0=29500
fi
echo "LUBAN_AVAILBLE_PORT_0: ${LUBAN_AVAILBLE_PORT_0}"

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=${LUBAN_AVAILBLE_PORT_0} /nfs/volume-382-179/perception-detection/dingbonan/OpenPCDet-master/tools/test.py --launcher pytorch ${PY_ARGS}
