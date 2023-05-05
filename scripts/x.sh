# for i in 1 2 4 6 8 10 12 14 16 18 20 
# do
#      bash intrinsic_scores.sh adv gender crows pfeiffer $i
# done





python bias-bench/experiments/inlp_projection_matrix.py --model BertModel --model_name_or_path bert-base-uncased --bias_type gender --seed 0


# python bias-bench/experiments/crows_debias.py \
#     --bias_type gender \
#     --model INLPBertForMaskedLM \
#     --model_name_or_path bert-base-uncased \
#     --projection_matrix bias-bench/results/projection_matrix/projection_m-BertModel_c-bert-base-uncased_t-gender_s-0.pt



CUDA_VISIBLE_DEVICES=1 bash 

