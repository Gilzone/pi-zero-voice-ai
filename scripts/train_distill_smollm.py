#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for SmolLM2-360M-Instruct.
Distills Qwen-style reasoning and concise voice QA capabilities.
Optimized for multi-threaded CPU training on AMD Ryzen 9 6900HX (16 threads).
"""

import os
import sys
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

def main():
    # 1. Hardware Optimization for 16-thread CPU
    num_threads = 16
    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    print(f"=== Starting Training on {num_threads} CPU Threads ===")

    base_dir = os.path.dirname(__file__)
    data_file = os.path.join(base_dir, "data", "distillation_train.jsonl")
    output_merged_dir = os.path.join(base_dir, "models", "smollm2-360m-qwen-merged")
    os.makedirs(os.path.dirname(output_merged_dir), exist_ok=True)

    model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    print(f"Loading base model and tokenizer: {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load 360M model in float32 for maximum CPU math stability
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    # 2. Configure LoRA
    print("Configuring LoRA adapter (Rank=16, Alpha=32)...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Load and Tokenize Distillation Dataset
    print(f"Loading dataset from: {data_file}...")
    dataset = load_dataset("json", data_files=data_file, split="train")
    print(f"Total training examples: {len(dataset)}")

    max_seq_len = 256
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False
        )

    print("Tokenizing training data...")
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(base_dir, "checkpoints"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch size = 16
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        optim="adamw_torch",
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    print("=== Commencing LoRA Fine-Tuning ===")
    t_start = time.time()
    train_result = trainer.train()
    t_end = time.time()
    dur_min = (t_end - t_start) / 60.0
    print(f"Training finished in {dur_min:.2f} minutes!")
    print(f"Final Train Loss: {train_result.training_loss:.4f}")

    # 5. Merge LoRA directly into base model
    print(f"Merging LoRA weights into base model and saving to: {output_merged_dir}...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_merged_dir)
    tokenizer.save_pretrained(output_merged_dir)

    print("=== Model Successfully Saved and Ready for Quantization! ===")

if __name__ == "__main__":
    main()
