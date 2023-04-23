#!/bin/bash

python run_sst.py \
  --model_name_or_path roberta-base \
  --dataset_name sst2 \
  --output_dir models/ \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 5 \
  --sparse_ft_max_epochs_per_iteration 5 \
  --save_steps 1000000 \
  --ft_params_num 6155776 \
  --evaluation_strategy steps \
  --eval_steps 625 \
  --freeze_layer_norm \
  --learning_rate 2e-5 \
  --metric_for_best_model eval_accuracy \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2 \