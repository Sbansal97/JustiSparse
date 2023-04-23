#!/bin/bash

SETTING=CDA # "orig", "CDA", ""
BIAS_TYPE=gender # "gender", "race", "religion"
METRIC=crows # "stereo", "crows", "perplexity"
MODEL_CLASS=CDABertForMaskedLM
MODEL_NAME_OR_PATH=bert-base-uncased


#compacter ia3 

for id in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000
do
    for PEFT in pfeiffer houlsby parallel scaled_parallel pfeiffer+inv prefix_tuning_flat lora     
    do
        python bias-bench/experiments/crows_debias.py \
            --bias_type $BIAS_TYPE \
            --model $MODEL_CLASS \
            --model_name_or_path $MODEL_NAME_OR_PATH \
            --adapter_path adapter-transformers/models/${PEFT}/cda/bias_bios/gender/checkpoint-${id}/mlm \
            --adapter_config $PEFT
        echo ${PEFT}
    done
done

# pfeiffer 0
# houlsby 1
# parallel 2
# scaled_parallel 3
# pfeiffer+inv 4
# compacter 5
# prefix_tuning_flat 6
# lora 7
# ia3 8