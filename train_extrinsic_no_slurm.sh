CUDA_VISIBLE_DEVICES=3 bash train_extrinsic-debias_after.sh sft gab models/sft/cda/group/gab/sft/ label cda
CUDA_VISIBLE_DEVICES=5 bash train_extrinsic-debias_after.sh pfeiffer gab models/pfeiffer/cda/group/gab/mlm/ label cda
CUDA_VISIBLE_DEVICES=5 bash train_extrinsic-debias_after.sh prefix_tuning_flat gab models/prefix_tuning_flat/cda/group/gab/mlm/ label cda


CUDA_VISIBLE_DEVICES=6 bash train_extrinsic-debias_after.sh pfeiffer bias-bios models/pfeiffer/adv/gender/bias-bios/only-adv/class p adv
CUDA_VISIBLE_DEVICES=7 bash train_extrinsic-debias_before.sh pfeiffer bias-bios models/pfeiffer/adv/gender/bias-bios/only-adv/class p adv


CUDA_VISIBLE_DEVICES=6 bash train_extrinsic-debias_before.sh sft gab models/sft/cda/group/gab/sft/ label cda
CUDA_VISIBLE_DEVICES=3 bash train_extrinsic-debias_before.sh pfeiffer gab models/pfeiffer/cda/group/gab/mlm/ label cda
# CUDA_VISIBLE_DEVICES=5 bash train_extrinsic-debias_before.sh prefix_tuning_flat gab models/prefix_tuning_flat/cda/group/gab/mlm/ label cda

# CUDA_VISIBLE_DEVICES=0 bash train_extrinsic-no_debias.sh sft gab models/sft/cda/group/gab/sft/ label cda
# CUDA_VISIBLE_DEVICES=2 bash train_extrinsic-no_debias.sh pfeiffer gab models/pfeiffer/cda/group/gab/mlm/ label cda
# CUDA_VISIBLE_DEVICES=2 bash train_extrinsic-no_debias.sh prefix_tuning_flat gab models/prefix_tuning_flat/cda/group/gab/mlm/ label cda

# CUDA_VISIBLE_DEVICES=3 bash train_extrinsic-debias_after.sh sft bias-bios models/sft/cda/gender/bias-bios/sft/ p cda
# CUDA_VISIBLE_DEVICES=4 bash train_extrinsic-debias_after.sh pfeiffer bias-bios models/pfeiffer/cda/gender/bias-bios/mlm/ p cda
# CUDA_VISIBLE_DEVICES=5 bash train_extrinsic-debias_after.sh prefix_tuning_flat bias-bios models/prefix_tuning_flat/cda/gender/bias-bios/mlm/ p cda

# CUDA_VISIBLE_DEVICES=6 bash train_extrinsic-debias_before.sh sft bias-bios models/sft/cda/gender/bias-bios/sft/ p cda 
# CUDA_VISIBLE_DEVICES=7 bash train_extrinsic-debias_before.sh pfeiffer bias-bios models/pfeiffer/cda/gender/bias-bios/mlm/ p cda
# CUDA_VISIBLE_DEVICES=2 bash train_extrinsic-debias_before.sh prefix_tuning_flat bias-bios models/prefix_tuning_flat/cda/gender/bias-bios/mlm/ p cda

# CUDA_VISIBLE_DEVICES=3 bash train_extrinsic-no_debias.sh sft bias-bios models/sft/cda/gender/bias-bios/sft/ p cda
# CUDA_VISIBLE_DEVICES=4 bash train_extrinsic-no_debias.sh pfeiffer bias-bios models/pfeiffer/cda/gender/bias-bios/mlm/ p cda
# CUDA_VISIBLE_DEVICES=5 bash train_extrinsic-no_debias.sh prefix_tuning_flat bias-bios models/prefix_tuning_flat/cda/gender/bias-bios/mlm/ p cda