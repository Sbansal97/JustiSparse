#!/bin/bash

DEBIAS=$1       # "orig", "cda" , "adv"
BIAS_TYPE=$2    # "gender", "race"
METRIC=$3       # "stereo", "crows", "perplexity"
PEFT=$4         # "ft", "sft", etc. 

if [[ $BIAS_TYPE == "gender" ]];then
    DATA=bias-bios
elif [[ $BIAS_TYPE == "race" ]];then
    DATA=gab
fi

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
            --model_name_or_path $MODEL_NAME_OR_PATH \
            --batch_size 128

        python bias-bench/experiments/stereoset_evaluation.py \
            --persistent_dir bias-bench \
            --predictions_file bias-bench/results/stereoset/stereoset_m-${MODEL_CLASS}_c-${MODEL_NAME_OR_PATH}.json
        
        

    else 
        echo "Metric Not Impemented"
    fi
elif [[ $DEBIAS == "cda" ]];then    
    MODEL_NAME_OR_PATH=bert-base-uncased
    MODEL_CLASS=CDABertForMaskedLM
    if  [[ $METRIC == "crows" ]];then        
        if  [[ $PEFT == "ft" ]];then
            python bias-bench/experiments/crows_debias.py \
                --bias_type $BIAS_TYPE \
                --model $MODEL_CLASS \
                --model_name_or_path $MODEL_NAME_OR_PATH \
                --load_path models/sft/$DEBIAS/$BIAS_TYPE/$DATA
        elif  [[ $PEFT == "sft" ]];then
            python bias-bench/experiments/crows_debias.py \
                --bias_type $BIAS_TYPE \
                --model $MODEL_CLASS \
                --model_name_or_path $MODEL_NAME_OR_PATH \
                --load_path models/$PEFT/$DEBIAS/$BIAS_TYPE/$DATA/$PEFT
        else
            python bias-bench/experiments/crows_debias.py \
                --bias_type $BIAS_TYPE \
                --model $MODEL_CLASS \
                --model_name_or_path $MODEL_NAME_OR_PATH \
                --adapter_path models/$PEFT/$DEBIAS/$BIAS_TYPE/$DATA/mlm
        fi
    elif  [[ $METRIC == "stereo" ]];then
        if  [[ $PEFT == "ft" ]];then
            python bias-bench/experiments/stereoset_debias.py \
                --model $MODEL_CLASS \
                --model_name_or_path $MODEL_NAME_OR_PATH \
                --load_path models/sft/$DEBIAS/$BIAS_TYPE/$DATA \
                --batch_size 128
        elif  [[ $PEFT == "sft" ]];then
        
        else
        
        fi   
        python bias-bench/experiments/stereoset_evaluation.py \
            --persistent_dir bias-bench \
            --predictions_file bias-bench/results/stereoset/stereoset_m-${MODEL_CLASS}_c-${MODEL_NAME_OR_PATH}.json            
                     
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