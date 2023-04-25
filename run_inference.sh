#!/bin/sh

PEFT=$1
MODEL_PATH=$2
DATASET_PATH=$3
LABEL_COLUMN=$4
TEXT_COLUMN=$5
ADAPTER_PATH=$6

if [[ $PEFT == "sft" ]]; then
    PATCH_PATH=$MODEL_PATH/sft
    DEBIAS_CONFIGURATION="none"
elif [[ $PEFT == "pfeiffer" ]]; then
    PATCH_PATH=$ADAPTER_PATH
    DEBIAS_CONFIGURATION="before"
elif [[ $PEFT == "prefix_tuning_flat" ]]; then
    PATCH_PATH=$ADAPTER_PATH
    DEBIAS_CONFIGURATION="before" 
fi

echo $PEFT $MODEL_PATH $DATASET_PATH $LABEL_COLUMN $TEXT_COLUMN
echo $PATCH_PATH $DEBIAS_CONFIGURATION

python run_extrinsic.py \
  --peft $PEFT \
  --debias_configuration $DEBIAS_CONFIGURATION \
  --patch_path $PATCH_PATH \
  --model_name_or_path $MODEL_PATH \
  --output_dir $MODEL_PATH \
  --test_file $DATASET_PATH \
  --label_column $LABEL_COLUMN \
  --text_column $TEXT_COLUMN \
  --do_predict \
  --per_device_eval_batch_size 1024 \
  --metric_for_best_model eval_accuracy