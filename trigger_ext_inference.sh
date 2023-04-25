
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh pfeiffer models/gab/pfeiffer/cda-before/ "data/iptts77k/test.csv" Label Text models/pfeiffer/cda/group/gab/mlm
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh pfeiffer models/gab/pfeiffer/cda-after/ "data/iptts77k/test.csv" Label Text models/pfeiffer/cda/group/gab/mlm
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh pfeiffer models/gab/pfeiffer/cda-none/ "data/iptts77k/test.csv" Label Text models/pfeiffer/cda/group/gab/mlm

CUDA_VISIBLE_DEVICES=0 bash run_inference.sh prefix_tuning_flat models/gab/prefix_tuning_flat/cda-before/ "data/iptts77k/test.csv" Label Text models/prefix_tuning_flat/cda/group/gab/mlm
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh prefix_tuning_flat models/gab/prefix_tuning_flat/cda-after/ "data/iptts77k/test.csv" Label Text models/prefix_tuning_flat/cda/group/gab/mlm
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh prefix_tuning_flat models/gab/prefix_tuning_flat/-none/ "data/iptts77k/test.csv" Label Text models/prefix_tuning_flat/cda/group/gab/mlm

CUDA_VISIBLE_DEVICES=0 bash run_inference.sh sft models/gab/sft/cda-before/ "data/iptts77k/test.csv" Label Text 
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh sft models/gab/sft/cda-after/ "data/iptts77k/test.csv" Label Text
CUDA_VISIBLE_DEVICES=0 bash run_inference.sh sft models/gab/sft/-none/ "data/iptts77k/test.csv" Label Text

