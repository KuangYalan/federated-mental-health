#!/usr/bin/env bash
# change to "gpu_hist" for gpu training
TREE_METHOD="hist"

prepare_job_config() {
    python3 utils/prepare_job_config.py --site_num "$1" --training_mode "$2" --split_method "$3" \
    --lr_mode "$4" --nthread 16 --tree_method "$5"
}

echo "Generating job configs"

prepare_job_config 20 bagging square uniform $TREE_METHOD

echo "Job configs generated"
