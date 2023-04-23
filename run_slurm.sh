# !/bin/sh

# SBATCH --job-name=peft-debias
# SBATCH --mem 32G
# SBATCH -t 0
# SBATCH --gres=gpu:A6000:1
# SBATCH --exclude=tir-0-32
# SBATCH --ntasks 1
# SBATCH --output /projects/tir6/general/srijanb/Spr23/JustiSparse/slurm_logs/log-%x-%J.txt

bash run_intrinsic.sh pfeiffer 0 adv gender bias-bios