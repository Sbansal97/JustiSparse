CUDA_VISIBLE_DEVICES=1 python run_adversarial.py \
    --model_name_or_path models/pfeiffer/adv/gender/bias-bios/checkpoint-4000 \
    --protected_attribute_column g \
    --label_column g \
    --train_file data/bias-bios/train.jsonl \
    --validation_file data/bias-bios/validation.jsonl \
    --test_file data/bias-bios/test.jsonl \
    --output_dir models/pfeiffer/adv/gender/bias-bios/checkpoint-4000 \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 256 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --adv_lr_scale 1 \
    --evaluation_strategy steps \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --max_steps 10000 \
    --eval_steps 1000 \
    --adapter_config pfeiffer \
    --peft prefix_tuning_flat \
    --cache_dir ~/.cache \
    --train_adapter \
    --log_level debug \
    --adv_debias \
    --metric_for_best_model eval_accuracy \
    --greater_is_better False \
    --load_best_model_at_end

# CUDA_VISIBLE_DEVICES=1 python run_adversarial.py \
#     --model_name_or_path models/sft/adv/gender/bias-bios-old/sft/checkpoint-4000/ \
#     --protected_attribute_column g \
#     --label_column g \
#     --train_file data/bias-bios/train.jsonl \
#     --validation_file data/bias-bios/validation.jsonl \
#     --test_file data/bias-bios/validation.jsonl \
#     --output_dir models/sft/adv/gender/bias-bios-old/sft/checkpoint-4000/ \
#     --do_predict \
#     --per_device_train_batch_size 128 \
#     --per_device_eval_batch_size 128 \
#     --gradient_accumulation_steps 1 \
#     --max_seq_length 256 \
#     --save_steps 1000 \
#     --learning_rate 1e-4 \
#     --adv_lr_scale 1 \
#     --evaluation_strategy steps \
#     --max_grad_norm 1.0 \
#     --weight_decay 0.0 \
#     --max_steps 10000 \
#     --eval_steps 1000 \
#     --peft sft \
#     --cache_dir ~/.cache \
#     --freeze_layer_norm \
#     --full_l1_reg 0.1 \
#     --sparse_l1_reg 0.1 \
#     --full_ft_min_steps_per_iteration 10000 \
#     --sparse_ft_min_steps_per_iteration 10000 \
#     --full_ft_max_steps_per_iteration 10000 \
#     --sparse_ft_max_steps_per_iteration 10000 \
#     --adv_debias \
#     --log_level debug \
#     --metric_for_best_model eval_accuracy \
#     --greater_is_better False \
#     --load_best_model_at_end

