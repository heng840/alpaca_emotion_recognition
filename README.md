# Alpaca_emotion_recognition
This repository contains code for reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) 
results using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).
To the best of our knowledge, this project is the first to use a large model for the vertical task of affective computing


Without hyperparameter tuning, the LoRA model produces outputs comparable to the Stanford Alpaca model. (Please see the outputs included below.) Further tuning might be able to achieve better performance; I invite interested users to give it a try and report their results.

### Local Setup

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

### Docker Setup & Inference

1. Build the container image:

```bash
docker build -t alpaca-lora .
```

2. Run the container (you can also use `finetune.py` and all of its parameters as shown above for training):

```bash
docker run --gpus=all --shm-size 64g -p 7860:7860 -v ${HOME}/.cache:/root/.cache --rm alpaca-lora generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```

3. Open `https://localhost:7860` in the browser

### Docker Compose Setup & Inference

1. (optional) Change desired model and weights under `environment` in the `docker-compose.yml`

2. Build and run the container

```bash
docker-compose up -d --build
```

3. Open `https://localhost:7860` in the browser

4. See logs:

```bash
docker-compose logs -f
```

5. Clean everything up:

```bash
docker-compose down --volumes --rmi all
```
