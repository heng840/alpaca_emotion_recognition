import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter


def train(
    # model/data params
    base_model: str = "decapoda-research/llama-7b-hf",  # the only required argument
    data_path: str = "alpaca_data/GPT_emotion_data.json",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 100,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    train_on_one_task: bool = True, # add
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params removed
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.distributed.init_process_group(
            "nccl", init_method="env://", world_size=world_size
        )
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.model_max_length = cutoff_len

    dataset = load_dataset("json", data_files=data_path, field="data")
    dataset = dataset["train"]
    if train_on_one_task:  # add
        dataset = dataset.filter(lambda x: x['task'] == 'emotion')  # add

    if group_by_length:
        dataset = dataset.sort("length")
        dataset = dataset.shuffle(256, seed=123)

    dataset = dataset.map(
        lambda x: tokenizer(x["prompt"], return_tensors="pt", padding="max_length"),
        batched=True,
        remove_columns=["prompt", "length"],
    )
    dataset.set_format("torch", columns=["input_ids", "attention_mask"])

    train_dataset = dataset.skip(val_set_size)
    val_dataset = dataset.take(val_set_size)

    base_model = LlamaForCausalLM.from_pretrained(base_model)
    base_model = base_model.to(device)
    if world_size > 1:
        base_model = torch.nn.parallel.DistributedDataParallel(
            base_model, device_ids=[device], output_device=device
        )

    if resume_from_checkpoint:
        set_peft_model_state_dict(
            base_model,
            torch.load(resume_from_checkpoint, map_location=torch.device(device)),
        )
    else:
        lora_config = LoraConfig(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        base_model = get_peft_model(base_model, lora_config)

    optimizer = torch.optim.AdamW(
        [
            {"params": [param for name, param in base_model.named_parameters() if name.startswith("lora")], "lr": learning_rate},
            {"params": [param for name, param in base_model.named_parameters() if not name.startswith("lora")], "lr": learning_rate / lora_alpha},
        ],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        shuffle=not group_by_length,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=micro_batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
    )

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")

        # Training loop
        base_model.train()
        torch.set_grad_enabled(True)
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = base_model(input_ids, attention_mask=attention_mask)
            loss = outputs.loss

            if gradient_accumulation_steps > 1:
                loss /= gradient_accumulation_steps
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * gradient_accumulation_steps

        print(f"Epoch {epoch + 1} training loss: {total_loss / len(train_dataloader)}")

        # Validation loop
        base_model.eval()
        torch.set_grad_enabled(False)
        total_loss = 0
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = base_model(input_ids, attention_mask=attention_mask)
            loss = outputs.loss

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} validation loss: {total_loss / len(val_dataloader)}")

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{epoch + 1}.pt")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(
                get_peft_model_state_dict(base_model),
                checkpoint_path,
            )
            print(f"Checkpoint saved at {checkpoint_path}")

    print("Training complete!")


if __name__ == "__main__":
    fire.Fire(train)
