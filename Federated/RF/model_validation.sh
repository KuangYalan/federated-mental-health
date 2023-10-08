#!/usr/bin/env bash
#DATASET_PATH="Fed-XGBoost/data/test_all.csv"

###section
echo $1

DATASET_PATH="Fed-XGBoostdata/$1/test.csv"
WORKSPACE_ROOT="Fed-XGBoost/random_forest/workspaces"

n=20
for subsample in 0.8
do
    for study in uniform_split_uniform_lr 
    do
        echo ${n}_clients_${study}_split_${subsample}_subsample
        python3 utils/model_validation.py --data_path $DATASET_PATH --model_path $WORKSPACE_ROOT/workspace_${n}_${subsample}_${study}/simulate_job/app_server/xgboost_model.json --size_valid 100000 --num_trees 100 --sec $1
    done
done
