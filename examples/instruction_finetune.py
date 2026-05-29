"""FoRA instruction fine-tuning example (Alpaca-style).

Usage:
    python examples/instruction_finetune.py \
        --model meta-llama/Llama-3.2-3B \
        --output_dir ./fora-instruct

Follows the instruction-tuning setup from the FoRA paper:
  - Dataset: tatsu-lab/alpaca (52K examples)
  - 3 epochs, effective batch 128, lr 2e-4 / 1e-3
  - r=32, lora_alpha=64, fraction=0.5, 128 calibration samples
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)

from fora_peft import FoRAConfig, FoRATrainer, apply_fora

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_section}"
    "### Response:\n{output}"
)


def format_alpaca(example: dict) -> str:
    input_section = (
        f"### Input:\n{example['input']}\n\n" if example.get("input") else ""
    )
    return PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        input_section=input_section,
        output=example["output"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FoRA instruction fine-tuning")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--output_dir", default="./fora-instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
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

    print(f"Loading dataset: {args.dataset} …")
    dataset = load_dataset(args.dataset, split="train")

    def tokenize(example):
        text = format_alpaca(example)
        enc = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        num_proc=4,
    )

    calib_texts = [
        format_alpaca(dataset[i])
        for i in range(min(args.calib_samples, len(dataset)))
    ]

    config = FoRAConfig(
        r=args.r,
        lora_alpha=args.r * 2,
        lora_dropout=0.05,
        layer_budget_fraction=args.fraction,
        fisher_calibration_samples=args.calib_samples,
        lr_stiefel=args.lr_stiefel,
    )

    print("Applying FoRA (Fisher scoring + Stiefel init) …")
    model = apply_fora(
        model, tokenizer, config,
        calibration_texts=calib_texts,
        device=device,
    )
    model.print_trainable_parameters()
    print(f"Selected layers: {model.fora_selected_layers}")

    grad_accum = max(1, 128 // (args.batch_size * max(1, torch.cuda.device_count())))
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr_adamw,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=(device == "cuda"),
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    trainer = FoRATrainer(
        model=model,
        args=training_args,
        fora_config=config,
        lr_adamw=args.lr_adamw,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, pad_to_multiple_of=8
        ),
    )

    print("Training …")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
