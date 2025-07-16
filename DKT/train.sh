set -e

GPUS=$1
NB_COMMA=$(echo ${GPUS} | tr -cd , | wc -c)
NB_GPUS=$((${NB_COMMA} + 1))
PORT=$((9000 + RANDOM % 1000))
shift  # 移除第一个参数（GPUS），保留其他参数

echo "Launching exp on $GPUS..."
export CUDA_VISIBLE_DEVICES=${GPUS}
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
# 使用 torchrun 启动分布式训练
torchrun \
    --nproc_per_node=${NB_GPUS} \
    --master_port=${PORT} \
    main.py "$@"