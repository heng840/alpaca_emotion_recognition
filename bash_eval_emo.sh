export CUDA_VISIBLE_DEVICES=6,7,8,9
 python eval_emo.py \
     --load_8bit \
     --base_model 'decapoda-research/llama-7b-hf' \
     --lora_weights 'tloen/alpaca-lora-7b'
