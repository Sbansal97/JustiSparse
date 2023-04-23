# !/bin/sh

# # SBATCH --job-name=peft-debias
# # SBATCH --mem 32G
# # SBATCH -t 0
# # SBATCH --gres=gpu:A6000:1
# # SBATCH --exclude=tir-0-32
# # SBATCH --ntasks 1
# # SBATCH --output /projects/tir6/general/srijanb/Spr23/JustiSparse/slurm_logs/log-%x-%J.txt

PEFT=$1
GPU_ID=$2

DEBIAS=cda
AXIS=gender
DATASET=bias-bios

corpora=corpora
train_path=data/${DATASET}/mlm/train.txt
val_path=data/${DATASET}/mlm/validation.txt


mkdir -p adapter-transformers/models/${PEFT}/${DEBIAS}/${DATASET}/${AXIS}

CUDA_VISIBLE_DEVICES=$GPU_ID python adapter-transformers/examples/pytorch/language-modeling/run_mlm.py \
--model_name_or_path bert-base-uncased \
--train_file $train_path \
--persistent_dir $corpora \
--validation_file $val_path \
--output_dir adapter-transformers/models/${PEFT}/${DEBIAS}/${DATASET}/${AXIS} \
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
--learning_rate 5e-5 \
--evaluation_strategy steps \
--max_steps 10000 \
--eval_steps 1000 \
--train_adapter \
--cache_dir /usr1/datasets/sumita/comp-ethics/project/JustiSparse/~/.cache \
--counterfactual_augmentation ${AXIS} \
--adapter_config ${PEFT} \
--load_best_model_at_end > adapter-transformers/models/${PEFT}/${DEBIAS}/${DATASET}/${AXIS}/training.log