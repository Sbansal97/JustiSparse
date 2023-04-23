#!/bin/bash
# conda activate finetune
export CUDA_VISIBLE_DEVICES=4,5

DEBIAS=$1
AXIS=$2
DATASET=$3

corpora=/usr1/datasets/sumita/comp-ethics/project/JustiSparse/corpora
train_path=${corpora}/${DEBIAS}/${DATASET}_train.txt
val_path=${corpora}/${DEBIAS}/${DATASET}_validation.txt

nohup python run_mlm.py \
  --model_name_or_path bert-base-uncased \
  --train_file $train_path \
  --persistent_dir $corpora \
  --validation_file $val_path \
  --output_dir ../../../models_compacter/${DEBIAS}/${DATASET}/${AXIS} \
  --do_train \
  --do_eval \
  --log_level 'info' \
  --preprocessing_num_workers 4 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 512 \
  --save_steps 500 \
  --overwrite_output_dir \
  --learning_rate 5e-5 \
  --evaluation_strategy steps \
  --max_steps 2000 \
  --eval_steps 500 \
  --train_adapter \
  --counterfactual_augmentation ${AXIS} \
  --adapter_config "compacter" \
  --load_best_model_at_end > ../../../models_compacter/${DEBIAS}/${DATASET}/${AXIS}/training.log
