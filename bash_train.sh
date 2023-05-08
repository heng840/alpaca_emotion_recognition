export CUDA_VISIBLE_DEVICES=2,6,7,8,9
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'alpaca_data/GPT_emotion_data.json' \
    --output_dir './lora-alpaca_emotion' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length