PEFT=$1
GPU_ID=$2
DEBIAS=$3
AXIS=$4 # gender, group
DATASET=$5 # bias-bios, gab

cache_dir=~/.cache
train_path=data/${DATASET}/train.jsonl
val_path=data/${DATASET}/validation.jsonl
export CUDA_VISIBLE_DEVICES=$GPU_ID

mkdir -p models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET}

if [[ $DEBIAS == "cda" ]];then
    if [[ $PEFT == "sft" ]];then
        python run_intrinsic.py \
            --model_name_or_path bert-base-uncased \
            --train_file $train_path \
            --validation_file $val_path \
            --output_dir models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET} \
            --do_train \
            --do_eval \
            --log_level 'info' \
            --preprocessing_num_workers 4 \
            --per_device_train_batch_size 128 \
            --per_device_eval_batch_size 128 \
            --gradient_accumulation_steps 1 \
            --max_seq_length 256 \
            --save_steps 1000 \
            --overwrite_output_dir \
            --learning_rate 5e-5 \
            --evaluation_strategy steps \
            --max_steps 10000 \
            --eval_steps 1000 \
            --train_adapter \
            --peft ${PEFT} \
            --cache_dir $cache_dir \
            --freeze_layer_norm \
            --freeze_decoder \
            --full_l1_reg 0.1 \
            --sparse_l1_reg 0.1 \
            --full_ft_min_steps_per_iteration 10000 \
            --sparse_ft_min_steps_per_iteration 10000 \
            --full_ft_max_steps_per_iteration 10000 \
            --sparse_ft_max_steps_per_iteration 10000 \
            --counterfactual_augmentation ${AXIS} \
            --adapter_config ${PEFT} \
            --load_best_model_at_end > models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET}/training.log
    else
        python run_intrinsic.py \
            --model_name_or_path bert-base-uncased \
            --train_file $train_path \
            --validation_file $val_path \
            --output_dir models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET} \
            --do_train \
            --do_eval \
            --log_level 'info' \
            --preprocessing_num_workers 4 \
            --per_device_train_batch_size 128 \
            --per_device_eval_batch_size 128 \
            --gradient_accumulation_steps 1 \
            --max_seq_length 256 \
            --save_steps 1000 \
            --overwrite_output_dir \
            --learning_rate 5e-5 \
            --evaluation_strategy steps \
            --max_steps 10000 \
            --eval_steps 1000 \
            --train_adapter \
            --peft ${PEFT} \
            --cache_dir $cache_dir \
            --counterfactual_augmentation ${AXIS} \
            --adapter_config ${PEFT} \
            --load_best_model_at_end > models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET}/training.log
    fi

elif [[ $DEBIAS == "adv" ]];then
    if [[ $PEFT == "sft" ]];then
        python run_intrinsic.py \
            --model_name_or_path bert-base-uncased \
            --protected_attribute_column 'g' \
            --train_file $train_path \
            --validation_file $val_path \
            --output_dir models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET} \
            --do_train \
            --do_eval \
            --log_level 'info' \
            --preprocessing_num_workers 4 \
            --per_device_train_batch_size 128 \
            --per_device_eval_batch_size 128 \
            --gradient_accumulation_steps 1 \
            --max_seq_length 256 \
            --save_steps 1000 \
            --overwrite_output_dir \
            --learning_rate 5e-5 \
            --evaluation_strategy steps \
            --max_steps 10000 \
            --eval_steps 1000 \
            --train_adapter \
            --peft ${PEFT} \
            --cache_dir $cache_dir \
            --freeze_layer_norm \
            --freeze_decoder \
            --full_l1_reg 0.1 \
            --sparse_l1_reg 0.1 \
            --full_ft_min_steps_per_iteration 10000 \
            --sparse_ft_min_steps_per_iteration 10000 \
            --full_ft_max_steps_per_iteration 10000 \
            --sparse_ft_max_steps_per_iteration 10000 \
            --adv_debias \
            --load_best_model_at_end > models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET}/training.log
    else
        python run_intrinsic.py \
            --model_name_or_path bert-base-uncased \
            --protected_attribute_column 'g' \
            --train_file $train_path \
            --validation_file $val_path \
            --output_dir models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET} \
            --do_train \
            --do_eval \
            --log_level 'info' \
            --preprocessing_num_workers 4 \
            --per_device_train_batch_size 128 \
            --per_device_eval_batch_size 128 \
            --gradient_accumulation_steps 1 \
            --max_seq_length 256 \
            --save_steps 1000 \
            --overwrite_output_dir \
            --learning_rate 5e-5 \
            --evaluation_strategy steps \
            --max_steps 10000 \
            --eval_steps 1000 \
            --train_adapter \
            --peft ${PEFT} \
            --adapter_config ${PEFT} \
            --cache_dir $cache_dir \
            --adv_debias \
            --load_best_model_at_end > models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET}/training.log
    fi
fi