
python bias-bench/experiments/inlp_projection_matrix.py --model BertModel --model_name_or_path bert-base-uncased --bias_type gender --seed 0

python bias-bench/experiments/crows_debias.py \
    --bias_type gender \
    --model INLPBertForMaskedLM \
    --model_name_or_path bert-base-uncased \
    --projection_matrix bias-bench/results/projection_matrix/projection_m-BertModel_c-bert-base-uncased_t-gender_s-0.pt

