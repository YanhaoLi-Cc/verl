#!/bin/bash

HOSTS_FILE="./exp_scripts/hosts.txt"
MASTER_ADDR="28.12.130.153"

while IFS= read -r host; do
    # 跳过空行和注释
    [[ -z "$host" || "$host" =~ ^# ]] && continue

    echo "===== $host ====="
    ssh -o ConnectTimeout=5 "$host" "python  /jizhicfs/hymiezhao/ml/busy.py"
done < "$HOSTS_FILE"