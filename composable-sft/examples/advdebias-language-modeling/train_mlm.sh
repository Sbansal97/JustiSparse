#!/bin/bash
DEBIAS=$1
AXIS=$2
DATASET=$3

export CUDA_VISIBLE_DEVICES=3

DATA_FOLDER=/usr1/datasets/sumita/comp-ethics/project/JustiSparse/data

mkdir -p /usr1/datasets/sumita/comp-ethics/project/JustiSparse/composable-sft/models/${DEBIAS}/${AXIS}/${DATASET}

nohup python run_mlm.py \
  --model_name_or_path bert-base-uncased \
  --train_file ${DATA_FOLDER}/${DATASET}/train.json \
  --validation_file ${DATA_FOLDER}/${DATASET}/validation.json \
  --output_dir ../../models/${DEBIAS}/${AXIS}/${DATASET} \
  --do_train \
  --do_eval \
  --log_level 'info' \
  --preprocessing_num_workers 4 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 256 \
  --save_steps 500 \
  --overwrite_output_dir \
  --freeze_layer_norm \
  --freeze_decoder \
  --full_l1_reg 0.1 \
  --sparse_l1_reg 0.1 \
  --learning_rate 5e-5 \
  --full_ft_min_steps_per_iteration 10000 \
  --sparse_ft_min_steps_per_iteration 10000 \
  --full_ft_max_steps_per_iteration 10000 \
  --sparse_ft_max_steps_per_iteration 10000 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --patience 1000 \
  --adv_debias \
  --cache_dir /usr1/datasets/sumita/comp-ethics/project/JustiSparse/~/.cache \
  --protected_attribute_column 'g' \
  --load_best_model_at_end > ../../models/${DEBIAS}/${AXIS}/${DATASET}/training.log
