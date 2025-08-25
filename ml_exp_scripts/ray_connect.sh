#!/bin/bash

HOSTS_FILE="./ml_exp_scripts/hosts.txt"
MASTER_ADDR="28.12.130.153"

while IFS= read -r host; do
    # 跳过空行和注释
    [[ -z "$host" || "$host" =~ ^# ]] && continue

    echo "===== $host ====="
    ssh -o ConnectTimeout=5 "$host" "cd /apdcephfs_fsgm/share_303809740/ml/long2short && \
        source /apdcephfs_fsgm/share_303809740/ml/miniconda3/bin/activate && \
        source ./ml_exp_scripts/env.sh && \
        export MASTER_ADDR=$MASTER_ADDR && \
        conda activate long2short && \
        export NUMEXPR_MAX_THREADS=50 && \
        ray stop && \
        ray start --address=$MASTER_ADDR:6379 --num-cpus=50" &
done < "$HOSTS_FILE"
