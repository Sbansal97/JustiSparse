MODEL_CLASS=BertForMaskedLM
MODEL_NAME_OR_PATH=bert-base-uncased
BIAS_TYPE=gender

# ======================================================================================================

# # Stereo-Set (Without Debias)
# python bias-bench/experiments/stereoset.py \
#     --model $MODEL_CLASS \
#     --model_name_or_path $MODEL_NAME_OR_PATH \

# python bias-bench/experiments/stereoset_evaluation.py \
#     --persistent_dir bias-bench \
#     --predictions_file bias-bench/results/stereoset/stereoset-m-$MODEL_CLASS-c-$MODEL_NAME_OR_PATH.json \
#     --output_file eval_results/$MODEL_CLASS-$MODEL_CLASS.json

# ======================================================================================================

# CrowS (Without Debias)
python bias-bench/experiments/crows.py \
    --bias_type $BIAS_TYPE \
    --model $MODEL_CLASS \
    --model_name_or_path $MODEL_NAME_OR_PATH


    


#python bias-bench/experiments/perplexity.py --model_name_or_path bert-base-uncased
