#!/bin/bash
DEBIAS=$1
AXIS=$2

nohup python run_mlm.py \
  --model_name_or_path bert-base-uncased \
  --train_file ../../../corpora/${DEBIAS}/wikipedia-10.txt \
  --output_dir ../../../models/${DEBIAS}/${AXIS} \
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
  --adapter_config "pfeiffer" \
  --adapter_reduction_factor 48 \
  --load_best_model_at_end > ../../../models/${DEBIAS}/${AXIS}/training.log
