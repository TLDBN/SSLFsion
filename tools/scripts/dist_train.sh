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

# while true
# do
#     PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#     status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#     if [ "${status}" != "0" ]; then
#         break;
#     fi
# done
# echo $PORT
# python /nfs/volume-382-179/perception-detection/dingbonan/OpenPCDet-master/setup.py develop
python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=${LUBAN_AVAILBLE_PORT_0} /nfs/volume-382-179/perception-detection/dingbonan/OpenPCDet-master/tools/train.py --launcher pytorch ${PY_ARGS}
# python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} /nfs/volume-382-179/perception-detection/dingbonan/OpenPCDet-master/tools/train.py --launcher pytorch ${PY_ARGS}

