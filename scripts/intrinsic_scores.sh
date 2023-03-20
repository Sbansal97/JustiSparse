#!/bin/bash







SETTING=$1 # "orig", "CDA", ""
BIAS_TYPE=$2 # "gender", "race", "religion"
METRIC=$3 # "stereo", "crows", "perplexity"

CHECKPOINTS=/projects/tir6/general/adityasv/JustiSparse/bias-bench/debias-eval/checkpoints



if [[ "$METRIC" = "stereo" && "$SETTING" = "orig" ]]; then
    MODEL_CLASS=BertForMaskedLM
    MODEL_NAME_OR_PATH=bert-base-uncased
    python bias-bench/experiments/stereoset.py \
        --model $MODEL_CLASS \
        --model_name_or_path $MODEL_NAME_OR_PATH

    python bias-bench/experiments/stereoset_evaluation.py \
        --persistent_dir bias-bench \
        --predictions_file bias-bench/results/stereoset/stereoset-m-$MODEL_CLASS-c-$MODEL_NAME_OR_PATH.json \
        --output_file eval_results/$MODEL_CLASS-$MODEL_NAME_OR_PATH.json

elif [[ "$METRIC" = "stereo" && "$SETTING" = "CDA" ]]; then
    MODEL_CLASS=CDABertForMaskedLM
    MODEL_NAME_OR_PATH=bert-base-uncased
    
    python bias-bench/experiments/stereoset_debias.py \
        --model $MODEL_CLASS \
        --bias_type $BIAS_TYPE \
        --load_path $CHECKPOINTS/cda_c-bert-base-uncased_t-"$BIAS_TYPE"_s-0/
            
    python bias-bench/experiments/stereoset_evaluation.py \
        --persistent_dir bias-bench \
        --predictions_file bias-bench/results/stereoset/stereoset-m-$MODEL_CLASS-c-$MODEL_NAME_OR_PATH-t-$BIAS_TYPE.json \
        --output_file eval_results/$MODEL_CLASS-$MODEL_NAME_OR_PATH.json


elif [[ "$METRIC" = "crows" && "$SETTING" = "orig" ]]; then
    MODEL_CLASS=BertForMaskedLM
    MODEL_NAME_OR_PATH=bert-base-uncased
    python bias-bench/experiments/crows.py \
        --bias_type $BIAS_TYPE \
        --model $MODEL_CLASS \
        --model_name_or_path $MODEL_NAME_OR_PATH




fi




# ======================================================================================================

    

# Perplexity (Without Debias)
# python bias-bench/experiments/perplexity.py \
#     --bias_type $BIAS_TYPE \
#     --model $MODEL_CLASS \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --output_dir eval_results