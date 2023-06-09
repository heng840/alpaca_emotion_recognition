# Alpaca_emotion_recognition
This repository is based on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) 
and the [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf) 
and the[LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter)
To the best of our knowledge, this project is the first to use a large model for the vertical task of affective computing.


- Without hyperparameter tuning, the LoRA model produces outputs comparable to the Stanford Alpaca model. 
(Please see the outputs included below.) 
Further tuning might be able to achieve better performance; 
I invite interested users to give it a try and report their results.


- By inserting adapters into LLaMA's transformer, we turn a LLaMA into an instruction-following model. 
After fine-tuning, LLaMA-Adapter can generate high-quality instruction-following sentences, comparable to the fully fine-tuned Stanford Alpaca and Alpaca-Lora.
## Innovation:
This project is completely based on a large language model, without using external tools or datasets. Our results show that as the parameter size of the language model becomes larger, its performance is getting better and better; we also found that fine-tuning using only the prompts generated by the LLM itself can also enhance the performance of the LLM. The performance on the task of emotional computing shows that large language models have good prospects for this task.
## Local Setup
### Fine-Tune
1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

1. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

### Training (`finetune.py`)

TODO generate train data

Example usage:

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path ' \
    --output_dir './lora-alpaca'
```

We can also tweak our hyperparameters:

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path  \
    --output_dir './lora-alpaca' \
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
```

### Inference (`generate.py`)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `tloen/alpaca-lora-7b`, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

Example usage:

```bash
python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```

### Official weights

The most recent "official" Alpaca-LoRA adapter available at [`tloen/alpaca-lora-7b`](https://huggingface.co/tloen/alpaca-lora-7b) was trained on March 26 with the following command:

```bash
python finetune.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --num_epochs=10 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./lora-alpaca' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8
```

### Checkpoint export (`export_*_checkpoint.py`)

These files contain scripts that merge the LoRA weights back into the base model
for export to Hugging Face format and to PyTorch `state_dicts`.
They should help users
who want to run inference in projects like [llama.cpp](https://github.com/ggerganov/llama.cpp)
or [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp).

### Setup & Inference

1. Run the container (you can also use `finetune.py` and all of its parameters as shown above for training):

```bash
docker run --gpus=all --shm-size 64g -p 7860:7860 -v ${HOME}/.cache:/root/.cache --rm alpaca-lora generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```

2. Open `https://localhost:7860` in the browser

## Result
| Model Size | Direct Use Accuracy | Fine-tuned (100) | Fine-tuned (200) | Fine-tuned (300) | Fine-tuned (400) | Fine-tuned (500) |
|------------|---------------------|------------------|------------------|------------------|------------------|------------------|
| 7B         | 0.255               | 0.365            | 0.368            | 0.371            | 0.372            | 0.371            |
| 13B        | 0.465               | 0.533            | 0.564            | 0.566            | 0.565            | 0.567            |
| 30B        | 0.622               | 0.658            | 0.684            | 0.688            | 0.686            | 0.686            |



## Expansibility:
This project can be easily extended to multimodal models. See [this demo](https://huggingface.co/spaces/csuhan/LLaMA-Adapter)
However, due to the limitation of graphics card conditions, 
fine-tuning cannot be successfully performed. If you want to do your own experimentation, we have implemented the image input, please use the VisionModel in llama/model.py.

