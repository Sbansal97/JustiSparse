MODEL_CLASS=BertForMaskedLM
MODEL_NAME_OR_PATH=bert-base-uncased

MODEL_CLASS=GPT2LMHeadModel
MODEL_NAME_OR_PATH=gpt2

BIAS_TYPE=gender # "gender", "race", "religion"

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

# # CrowS (Without Debias)
# python bias-bench/experiments/crows.py \
#     --bias_type $BIAS_TYPE \
#     --model $MODEL_CLASS \
#     --model_name_or_path $MODEL_NAME_OR_PATH

# ======================================================================================================

    

# Perplexity (Without Debias)
python bias-bench/experiments/perplexity.py \
    --bias_type $BIAS_TYPE \
    --model $MODEL_CLASS \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir eval_results