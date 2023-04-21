#!/bin/sh
#SBATCH --job-name=no_debias
#SBATCH --mail-user=adityasv@cs.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH --mem 32G # memory
#SBATCH -t 0 # unlimited time for executing
#SBATCH --gres=gpu:A6000:1

python run_biosbias.py \
  --debias_configuration none \
  --diffs_path /projects/tir6/general/adityasv/JustiSparse/composable-sft/examples/text-classification-debias/biosbias/models/base-model/sft/pytorch_diff.bin \
  --model_name_or_path /projects/tir6/general/adityasv/JustiSparse/composable-sft/examples/text-classification-debias/biosbias/models/base-model/sft/ \
  --train_file /projects/tir6/general/adityasv/JustiSparse/composable-sft/examples/text-classification-debias/biosbias/data/train.json \
  --validation_file /projects/tir6/general/adityasv/JustiSparse/composable-sft/examples/text-classification-debias/biosbias/data/validation.json \
  --test_file /projects/tir6/general/adityasv/JustiSparse/composable-sft/examples/text-classification-debias/biosbias/data/test.json \
  --label_file /projects/tir6/general/adityasv/JustiSparse/composable-sft/examples/text-classification-debias/biosbias/data/labels.txt \
  --output_dir models/debias-none \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 5 \
  --sparse_ft_max_epochs_per_iteration 5 \
  --eval_steps 5000 \
  --save_steps 5000 \
  --evaluation_strategy steps \
  --freeze_layer_norm \
  --learning_rate 2e-5 \
  --metric_for_best_model eval_accuracy \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2 \
  --overwrite_cache