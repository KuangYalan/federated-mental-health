#!/usr/bin/env bash
DATASET_PATH="Fed-XGBoostdata/train_all.csv"
OUTPUT_PATH="Fed-XGBoost/nvflare/xgboost/xgboost_higgs_dataset"

if [ ! -f "${DATASET_PATH}" ]
then
    echo "Please check if you saved HIGGS dataset in ${DATASET_PATH}"
fi

echo "Generated HIGGS data splits, reading from ${DATASET_PATH}"
for site_num in 20;
do
    for split_mode in uniform exponential square;
    do
        python3 utils/prepare_data_split.py \
        --data_path "${DATASET_PATH}" \
        --site_num ${site_num} \
        --size_total 466713 \
        --size_valid 93342 \
        --split_method ${split_mode} \
        --out_path "${OUTPUT_PATH}/${site_num}_${split_mode}"
    done
done
echo "Data splits are generated in ${OUTPUT_PATH}"
