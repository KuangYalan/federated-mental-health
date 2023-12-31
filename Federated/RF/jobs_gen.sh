#!/usr/bin/env bash

# change to "gpu_hist" for gpu training
TREE_METHOD="hist"
DATA_SPLIT_ROOT="Fed-XGBoost/random_forest/data_splits"

prepare_job_config() {
    python3 utils/prepare_job_config.py --site_num "$1" --num_local_parallel_tree "$2" --local_subsample "$3" \
    --split_method "$4" --lr_mode "$5" --nthread 16 --tree_method "$6" --data_split_root "$7"
}

echo "Generating job configs"
prepare_job_config 20 5 0.8 uniform uniform $TREE_METHOD $DATA_SPLIT_ROOT


echo "Job configs generated"
