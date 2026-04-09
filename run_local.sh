#!/bin/bash
set -euo pipefail

mkdir -p logs

# ── GPU selection ─────────────────────────────────────────────────────────────
# Usage: GPUS=0,1,2,3 ./run_local.sh
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1}

# Count selected GPUs
NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

# ── Rendezvous info ───────────────────────────────────────────────────────────
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

echo "Master: $MASTER_ADDR:$MASTER_PORT  |  Nodes: $NNODES  |  GPUs: $CUDA_VISIBLE_DEVICES ($NPROC_PER_NODE)  |  Rank: $NODE_RANK"

source .venv/bin/activate

# ── Launch torchrun directly ──────────────────────────────────────────────────
python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train.py \
        --data_dir      /home/data/ \
        --checkpoint_path checkpoint.pt \
        --seq_len       1024 \
        --batch_size    8 \
        --grad_accum_steps 4 \
        --max_steps        5000 \
        --time_limit_min   10
