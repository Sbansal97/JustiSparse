#!/bin/bash

DEBIAS=$1       # "orig", "cda" , "adv"
BIAS_TYPE=$2    # "gender", "race"
METRIC=$3       # "stereo", "crows", "perplexity"

CHECKPOINTS=models/checkpoints/sft-ckpts/



if [[ $DEBIAS == "orig" ]];then
    MODEL_CLASS=BertForMaskedLM
    MODEL_NAME_OR_PATH=bert-base-uncased
    if  [[ $METRIC == "crows" ]];then
        python bias-bench/experiments/crows.py \
            --bias_type $BIAS_TYPE \
            --model $MODEL_CLASS \
            --model_name_or_path $MODEL_NAME_OR_PATH
    elif [[ $METRIC == "stereo" ]];then
        python bias-bench/experiments/stereoset.py \
            --model $MODEL_CLASS \
            --model_name_or_path $MODEL_NAME_OR_PATH

        python bias-bench/experiments/stereoset_evaluation.py \
            --persistent_dir bias-bench \
            --predictions_file bias-bench/results/stereoset/stereoset-m-$MODEL_CLASS-c-$MODEL_NAME_OR_PATH.json \
            --output_file eval_results/$MODEL_CLASS-$MODEL_NAME_OR_PATH.json
    else 
        echo "Metric Not Impemented"
    fi
elif [[ $DEBIAS == "cda" ]];then
    if  [[$METRIC == "crows"]];then
        MODEL_CLASS=CDABertForMaskedLM
        MODEL_NAME_OR_PATH=bert-base-uncased
        python bias-bench/experiments/crows_debias.py \
            --bias_type $BIAS_TYPE \
            --model $MODEL_CLASS \
            --model_name_or_path $MODEL_NAME_OR_PATH \
            --adapter_path /projects/tir6/general/srijanb/Spr23/JustiSparse/adapter-transformers/models/cda/mlm
            --adapter_path /projects/tir6/general/srijanb/Spr23/JustiSparse/adapter-transformers/models/cda/mlm
            --load_path composable-sft/models/cda/religion/sft

    elif  [[$METRIC == "stereo"]];then
        python bias-bench/experiments/stereoset_debias.py \
            --model $MODEL_CLASS \
            --bias_type $BIAS_TYPE \
            --load_path composable-sft/models/cda/gender/sft
                
        python bias-bench/experiments/stereoset_evaluation.py \
            --persistent_dir bias-bench \
            --predictions_file bias-bench/results/stereoset/stereoset-m-$MODEL_CLASS-c-$MODEL_NAME_OR_PATH-t-$BIAS_TYPE.json \
            --output_file eval_results/$MODEL_CLASS-$MODEL_NAME_OR_PATH.json


    else 
        echo "Metric Not Impemented"
    fi
fi


# Perplexity (Without Debias)
# python bias-bench/experiments/perplexity.py \
#     --bias_type $BIAS_TYPE \
#     --model $MODEL_CLASS \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --output_dir eval_results