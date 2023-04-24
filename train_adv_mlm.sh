PEFT=$1
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
fi

mkdir -p models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET}/adv-mlm

python run_intrinsic.py \
    --model_name_or_path bert-base-uncased \
    --protected_attribute_column $attr \
    --train_file $train_path \
    --validation_file $val_path \
    --output_dir models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET}/adv-mlm \
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
    --adv_lr_scale 10 \
    --evaluation_strategy steps \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --max_steps 20000 \
    --eval_steps 1000 \
    --train_adapter \
    --peft $PEFT \
    --adapter_config $PEFT \
    --cache_dir $cache_dir \
    --adv_debias \
    --load_best_model_at_end > models/${PEFT}/${DEBIAS}/${AXIS}/${DATASET}/adv-mlm/training.log