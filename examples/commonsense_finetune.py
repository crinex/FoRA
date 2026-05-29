"""FoRA fine-tuning on commonsense reasoning tasks (e.g., BoolQ, HellaSwag).

Usage:
    python examples/commonsense_finetune.py --model meta-llama/Llama-3.2-3B \
        --task boolq --output_dir ./fora-commonsense

This script mirrors the commonsense fine-tuning setup from the FoRA paper:
  - 3 epochs, batch size 16, lr 2e-4 (AdamW) / 1e-3 (CayleyAdam)
  - r=32, lora_alpha=64, layer_budget_fraction=0.5
  - Calibration on 128 training samples
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)

from fora_peft import FoRAConfig, FoRATrainer, apply_fora

TASK_DATASETS = {
    "boolq": ("super_glue", "boolq"),
    "hellaswag": ("hellaswag", None),
    "winogrande": ("winogrande", "winogrande_xl"),
    "arc_easy": ("ai2_arc", "ARC-Easy"),
    "arc_challenge": ("ai2_arc", "ARC-Challenge"),
}


def format_example(task: str, example: dict) -> str:
    if task == "boolq":
        return f"Passage: {example['passage']}\nQuestion: {example['question']}\nAnswer:"
    if task == "hellaswag":
        return f"{example['activity_label']}: {example['ctx']}"
    if task == "winogrande":
        return example["sentence"]
    if task in ("arc_easy", "arc_challenge"):
        choices = " | ".join(example["choices"]["text"])
        return f"Question: {example['question']}\nChoices: {choices}\nAnswer:"
    return str(example)


def main() -> None:
    parser = argparse.ArgumentParser(description="FoRA commonsense fine-tuning")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--task", default="boolq", choices=list(TASK_DATASETS))
    parser.add_argument("--output_dir", default="./fora-commonsense")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--lr_adamw", type=float, default=2e-4)
    parser.add_argument("--lr_stiefel", type=float, default=1e-3)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--calib_samples", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model} …")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)

    ds_name, ds_config = TASK_DATASETS[args.task]
    dataset = load_dataset(ds_name, ds_config)
    train_split = dataset["train"]

    def tokenize(example):
        text = format_example(args.task, example)
        return tokenizer(text, truncation=True, max_length=256, padding="max_length")

    tokenized = train_split.map(tokenize, remove_columns=train_split.column_names)

    calib_texts = [
        format_example(args.task, train_split[i])
        for i in range(min(args.calib_samples, len(train_split)))
    ]

    config = FoRAConfig(
        r=args.r,
        lora_alpha=args.r * 2,
        lora_dropout=0.05,
        layer_budget_fraction=args.fraction,
        fisher_calibration_samples=args.calib_samples,
        lr_stiefel=args.lr_stiefel,
    )

    print("Applying FoRA …")
    model = apply_fora(
        model, tokenizer, config,
        calibration_texts=calib_texts,
        device=device,
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
        learning_rate=args.lr_adamw,
        fp16=(device == "cuda"),
        logging_steps=20,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = FoRATrainer(
        model=model,
        args=training_args,
        fora_config=config,
        lr_adamw=args.lr_adamw,
        train_dataset=tokenized,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    print("Training …")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
