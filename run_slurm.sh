#!/bin/sh

#SBATCH --job-name=peft-debias
#SBATCH --mem 32G
#SBATCH -t 0
#SBATCH --gres=gpu:A6000:1
#SBATCH --exclude=tir-0-32
#SBATCH --ntasks 1
#SBATCH --output slurm_logs/log-%x-%J.txt

bash run_intrinsic.sh $1 0 $2 group gab

# bash run_intrinsic.sh pfeiffer 0 cda group gab
# bash run_intrinsic.sh prefix_tuning_flat 0 cda group gab
# bash run_intrinsic.sh sft 0 cda group gab
