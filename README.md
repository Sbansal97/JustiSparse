# JustiSparse

## Environment Setup
```
conda create -n justisparse python=3.9
conda activate justisparse
pip install -r requirements.txt


```

## Train CDA (PEFT)
```
# adapter_config, gpu_id, debias, axis, dataset
bash run_intrinsic.sh pfeiffer 0 cda gender bias-bios
```
