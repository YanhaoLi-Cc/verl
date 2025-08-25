export NUMEXPR_MAX_THREADS=50
export MASTER_ADDR="28.12.130.153"
ray stop
ray start --head --num-cpus=50 --node-ip-address=$MASTER_ADDR --port=6379