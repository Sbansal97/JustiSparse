#!/bin/sh
#SBATCH --job-name=debias_none
#SBATCH --mail-user=adityasv@cs.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH --mem 32G # memory
#SBATCH -t 0 # unlimited time for executing
#SBATCH --gres=gpu:A6000:1

PEFT=$1
DATASET=$2
PATCH_PATH=$3
LABEL_COLUMN=$4
DEBIAS_METHOD=$5

if [[ $DATASET == "gab" ]]; then
  N_EPOCHS=10
  BS=32
  LR=2e-5
  METRIC="eval_f1"
  CLASS_WEIGHTS="{\"0\":1,\"1\":10}"
  EVAL_STEPS=1000
else
  N_EPOCHS=5
  BS=32
  LR=2e-5
  METRIC="eval_accuracy"
  EVAL_STEPS=5000
  CLASS_WEIGHTS="fail"
fi

mkdir -p models/$DATASET/$PEFT/$DEBIAS_METHOD-none

python run_extrinsic.py \
  --peft $PEFT \
  --debias_configuration none \
  --model_name_or_path bert-base-uncased \
  --train_file data/$DATASET/train.jsonl \
  --validation_file data/$DATASET/validation.jsonl \
  --test_file data/$DATASET/test.jsonl \
  --label_column $LABEL_COLUMN \
  --output_dir models/$DATASET/$PEFT/$DEBIAS_METHOD-none \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size $BS \
  --per_device_eval_batch_size 128 \
  --overwrite_output_dir \
  --overwrite_cache \
  --full_ft_max_epochs_per_iteration $N_EPOCHS \
  --sparse_ft_max_epochs_per_iteration $N_EPOCHS \
  --num_train_epochs $N_EPOCHS \
  --eval_steps $EVAL_STEPS \
  --save_steps $EVAL_STEPS \
  --evaluation_strategy steps \
  --freeze_layer_norm \
  --learning_rate $LR \
  --metric_for_best_model $METRIC \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2 \
  --log_level debug \
  --cls_weights $CLASS_WEIGHTS > models/$DATASET/$PEFT/$DEBIAS_METHOD-none/training.log