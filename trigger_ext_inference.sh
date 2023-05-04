
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh pfeiffer models/gab/pfeiffer/cda-before/ "data/iptts77k/test.csv" Label Text models/pfeiffer/cda/group/gab/mlm before iptts &
CUDA_VISIBLE_DEVICES=1 bash run_inference.sh pfeiffer models/gab/pfeiffer/cda-after/ "data/iptts77k/test.csv" Label Text models/pfeiffer/cda/group/gab/mlm after iptts &
# CUDA_VISIBLE_DEVICES=2 bash run_inference.sh pfeiffer models/gab/pfeiffer/cda-none/ "data/iptts77k/test.csv" Label Text models/pfeiffer/cda/group/gab/mlm none iptts &

CUDA_VISIBLE_DEVICES=2 bash run_inference.sh prefix_tuning_flat models/gab/prefix_tuning_flat/cda-before/ "data/iptts77k/test.csv" Label Text models/prefix_tuning_flat/cda/group/gab/mlm  before iptts &
CUDA_VISIBLE_DEVICES=3 bash run_inference.sh prefix_tuning_flat models/gab/prefix_tuning_flat/cda-after/ "data/iptts77k/test.csv" Label Text models/prefix_tuning_flat/cda/group/gab/mlm  after iptts &
# CUDA_VISIBLE_DEVICES=0 bash run_inference.sh prefix_tuning_flat models/gab/prefix_tuning_flat/-none/ "data/iptts77k/test.csv" Label Text models/prefix_tuning_flat/cda/group/gab/ none mlm

CUDA_VISIBLE_DEVICES=4 bash run_inference.sh sft models/gab/sft/cda-before/ "data/iptts77k/test.csv" Label Text random before iptts &
CUDA_VISIBLE_DEVICES=5 bash run_inference.sh sft models/gab/sft/cda-after/ "data/iptts77k/test.csv" Label Text random after iptts &
CUDA_VISIBLE_DEVICES=6 bash run_inference.sh sft models/gab/sft/-none/ "data/iptts77k/test.csv" Label none Text

# CUDA_VISIBLE_DEVICES=6 bash run_inference.sh sft /usr1/datasets/sumita/comp-ethics/project/JustiSparse/models/bias-bios/sft/cda-after/checkpoint-25000/ "data/bias-bios/test.jsonl" p text random before temp &
# CUDA_VISIBLE_DEVICES=5 bash run_inference.sh sft models/bias-bios/sft/cda-before/ "data/bias-bios/test.jsonl" p text random before temp &
# CUDA_VISIBLE_DEVICES=4 bash run_inference.sh sft models/bias-bios/prefix_tuning_flat/cda-none/ "data/bias-bios/test.jsonl" p text random after temp &


# CUDA_VISIBLE_DEVICES=6 bash run_inference.sh sft models/gab/sft/cda-before/ "data/gab/test.jsonl" label text random before temp &