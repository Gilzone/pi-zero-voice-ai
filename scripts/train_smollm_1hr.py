#!/usr/bin/env python3
"""
1-Hour Knowledge & Reasoning Fine-Tuning Script for SmolLM2-360M.
Runs on AMD Ryzen 9 6900HX CPU with PyTorch OpenMP multi-threading.
Targets ~55-60 minutes runtime (240 update steps @ effective batch size 16).
"""

import os
import sys
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_file = os.path.join(base_dir, "data", "distillation_v2_1hr.jsonl")
    output_dir = os.path.join(base_dir, "models", "smollm2-360m-qwen-v2-lora")
    merged_output_dir = os.path.join(base_dir, "models", "smollm2-360m-qwen-v2-merged")
    base_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

    os.environ["OMP_NUM_THREADS"] = "16"
    os.environ["MKL_NUM_THREADS"] = "16"
    torch.set_num_threads(16)
    print(f"PyTorch using {torch.get_num_threads()} CPU threads.")

    # 1. Load Tokenizer & Base Model
    print(f"\n=== Step 1: Loading {base_model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    # 2. Configure High-Capacity LoRA (r=32, alpha=64)
    print("\n=== Step 2: Configuring High-Capacity LoRA (Rank 32, Alpha 64) ===")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Load & Tokenize Dataset
    print(f"\n=== Step 3: Loading Dataset from {data_file} ===")
    raw_dataset = load_dataset("json", data_files=data_file, split="train")
    print(f"Loaded {len(raw_dataset)} training examples.")

    max_seq_length = 256

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length"
        )
        tokens["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in label]
            for label in tokens["input_ids"]
        ]
        return tokens

    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 4. Training Arguments tuned for 1 Hour
    # 240 steps @ ~14s/step = ~56 minutes
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=240,
        learning_rate=2.5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=15,
        logging_steps=10,
        save_strategy="steps",
        save_steps=120,
        save_total_limit=2,
        report_to="none",
        use_cpu=True,
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    )

    print("\n========================================================")
    print("      LAUNCHING 1-HOUR DEEP KNOWLEDGE TRAINING          ")
    print(f"  Target Steps : 240 update steps                      ")
    print(f"  Batch Size   : 4 x 4 (Effective = 16)                ")
    print(f"  Estimated Time: ~55 - 60 minutes                     ")
    print("========================================================\n")

    t_start = time.time()
    train_result = trainer.train()
    t_end = time.time()
    elapsed_min = (t_end - t_start) / 60.0

    print(f"\nTraining completed in {elapsed_min:.2f} minutes!")
    print(f"Final Train Loss: {train_result.training_loss:.4f}")

    # 5. Save LoRA Adapter
    print(f"Saving LoRA adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 6. Merge LoRA into Base Model
    print("\n=== Step 6: Merging LoRA into FP16 Base Model ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    from peft import PeftModel
    merged_model = PeftModel.from_pretrained(base_model, output_dir)
    merged_model = merged_model.merge_and_unload()

    print(f"Saving merged model to: {merged_output_dir}")
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
    print("Successfully created merged model!")

if __name__ == "__main__":
    main()
