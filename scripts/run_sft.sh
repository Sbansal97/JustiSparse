#!/bin/bash

GPU_ID=$1

DEBIAS=cda
AXIS=gender
DATASET=bias_bios

corpora=corpora
train_path=data/${DATASET}/mlm/train.txt
val_path=data/${DATASET}/mlm/validation.txt


train_path=${corpora}/${DEBIAS}/${DATASET}_train.txt
val_path=${corpora}/${DEBIAS}/${DATASET}_validation.txt

mkdir -p composable-sft/models/${DEBIAS}/${DATASET}/${AXIS}

CUDA_VISIBLE_DEVICES=$GPU_ID python composable-sft/examples/language-modeling/run_mlm.py \
  --model_name_or_path bert-base-uncased \
  --train_file $train_path \
  --validation_file $val_path \
  --output_dir composable-sft/models/${DEBIAS}/${DATASET}/${AXIS} \
  --do_train \
  --do_eval \
  --log_level 'info' \
  --preprocessing_num_workers 4 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 256 \
  --save_steps 1000 \
  --overwrite_output_dir \
  --freeze_layer_norm \
  --freeze_decoder \
  --full_l1_reg 0.1 \
  --sparse_l1_reg 0.1 \
  --counterfactual_augmentation ${AXIS} \
  --learning_rate 5e-5 \
  --full_ft_min_steps_per_iteration 10000 \
  --sparse_ft_min_steps_per_iteration 10000 \
  --full_ft_max_steps_per_iteration 10000 \
  --sparse_ft_max_steps_per_iteration 10000 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --cache_dir /usr1/datasets/sumita/comp-ethics/project/JustiSparse/~/.cache \
  --load_best_model_at_end > composable-sft/models/${DEBIAS}/${DATASET}/${AXIS}/training.log