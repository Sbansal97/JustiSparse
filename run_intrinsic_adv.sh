PEFT=$1 # pfeiffer, prefix_tuning_flat, lora, sft
GPU_ID=$2
DEBIAS=$3
AXIS=$4 # gender, group
DATASET=$5 # bias-bios, gab

cache_dir=~/.cache
train_path=data/${DATASET}/train.jsonl
val_path=data/${DATASET}/validation.jsonl
export CUDA_VISIBLE_DEVICES=$GPU_ID

if [[ $AXIS == "gender" ]];then
    attr='g'
elif [[ $AXIS == "group" ]];then
    attr='t'
elif [[ $AXIS == "dialect" ]];then
    attr='t'
fi

mkdir -p models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET}

if [[ $PEFT == "sft" ]];then
    python run_adversarial.py \
        --model_name_or_path bert-base-uncased \
        --protected_attribute_column $attr \
        --label_column $attr \
        --train_file $train_path \
        --validation_file $val_path \
        --output_dir models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET} \
        --do_train \
        --do_eval \
        --log_level 'info' \
        --per_device_train_batch_size 128 \
        --per_device_eval_batch_size 128 \
        --gradient_accumulation_steps 1 \
        --max_seq_length 256 \
        --save_steps 1000 \
        --overwrite_output_dir \
        --learning_rate 1e-4 \
        --adv_lr_scale 1 \
        --evaluation_strategy steps \
        --max_grad_norm 1.0 \
        --weight_decay 0.0 \
        --max_steps 10000 \
        --eval_steps 1000 \
        --peft $PEFT \
        --cache_dir $cache_dir \
        --freeze_layer_norm \
        --full_l1_reg 0.1 \
        --sparse_l1_reg 0.1 \
        --full_ft_min_steps_per_iteration 10000 \
        --sparse_ft_min_steps_per_iteration 10000 \
        --full_ft_max_steps_per_iteration 10000 \
        --sparse_ft_max_steps_per_iteration 10000 \
        --adv_debias \
        --log_level debug \
        --metric_for_best_model eval_accuracy \
        --greater_is_better False \
        --load_best_model_at_end > models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET}/training.log
else
    python run_adversarial.py \
        --model_name_or_path bert-base-uncased \
        --protected_attribute_column $attr \
        --label_column $attr \
        --train_file $train_path \
        --validation_file $val_path \
        --output_dir models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET} \
        --do_train \
        --do_eval \
        --log_level 'info' \
        --per_device_train_batch_size 128 \
        --per_device_eval_batch_size 128 \
        --gradient_accumulation_steps 1 \
        --max_seq_length 256 \
        --save_steps 1000 \
        --overwrite_output_dir \
        --learning_rate 1e-4 \
        --adv_lr_scale 1 \
        --evaluation_strategy steps \
        --max_grad_norm 1.0 \
        --weight_decay 0.0 \
        --max_steps 10000 \
        --eval_steps 1000 \
        --adapter_config $PEFT \
        --peft $PEFT \
        --cache_dir $cache_dir \
        --train_adapter \
        --log_level debug \
        --adv_debias \
        --metric_for_best_model eval_accuracy \
        --greater_is_better False \
        --load_best_model_at_end  > models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET}/training.log
fi
