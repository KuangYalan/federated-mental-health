#!/usr/bin/env bash

WORKSPACE_ROOT="/public/home/kuangyl/Fed-XGBoost/nvflare/NVFlare-main/examples/advanced/random_forest/workspaces"

n=20
for subsample in 0.8
do
    for study in uniform_split_uniform_lr
    do
        nvflare simulator jobs/higgs_${n}_${subsample}_${study} -w $WORKSPACE_ROOT/workspace_${n}_${subsample}_${study} -n ${n} -t ${n}
    done
done
