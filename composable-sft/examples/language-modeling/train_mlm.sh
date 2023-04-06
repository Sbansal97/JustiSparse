#!/bin/bash
DEBIAS=$1
AXIS=$2
DATASET=$3

nohup python run_mlm.py \
  --model_name_or_path bert-base-uncased \
  --train_file ../../corpora/${DEBIAS}/${DATASET}_train.txt \
  --validation_file ../../corpora/${DEBIAS}/${DATASET}_validation.txt \
  --output_dir ../../models/${DEBIAS}/${DATASET}/${AXIS} \
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
  --eval_steps 500 \
  --patience 3 \
  --load_best_model_at_end > ../../models/${DEBIAS}/${DATASET}/${AXIS}/training.log
