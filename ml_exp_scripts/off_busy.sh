#!/bin/bash

HOSTS_FILE="./exp_scripts/hosts.txt"

while IFS= read -r host; do
    # 跳过空行和注释
    [[ -z "$host" || "$host" =~ ^# ]] && continue

    echo "===== $host ====="
    ssh -o ConnectTimeout=5 "$host" "pkill -f /usr/bin/python" &
    ssh -o ConnectTimeout=5 "$host" "pkill -f /jizhicfs/hymiezhao/miniconda3/envs/over_estimation/bin/python" &
done < "$HOSTS_FILE"