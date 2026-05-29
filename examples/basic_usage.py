"""Minimal end-to-end FoRA example.

Usage:
    python examples/basic_usage.py

Requires:
    pip install peft-fora transformers datasets accelerate

Model access:
    meta-llama/Llama-3.2-3B requires a HuggingFace token with access granted.
    Set HF_TOKEN or log in with `huggingface-cli login` before running.
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset

from fora_peft import FoRAConfig, apply_fora, FoRATrainer


def main() -> None:
    model_name = "meta-llama/Llama-3.2-3B"

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # FoRA configuration — paper defaults
    config = FoRAConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        layer_budget_fraction=0.5,
        fisher_calibration_samples=128,
        lr_stiefel=1e-3,
        cayley_n_iter=5,
        qr_reset_period=100,
    )

    # Apply FoRA using a list of calibration strings.
    # Alternatively pass calibration_dataloader= for full control.
    print("Applying FoRA (Fisher scoring + Stiefel init) …")
    model = apply_fora(
        model,
        tokenizer,
        config,
        calibration_texts=["Hello world"] * 128,
        device="cuda",
    )

    print(f"Selected layers: {model.fora_selected_layers}")
    print(f"Stiefel modules: {model.fora_stiefel_info['n_modules']}")
    model.print_trainable_parameters()

    # Load a small dataset for demonstration
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:1%]")

    def tokenize(example):
        text = example["instruction"] + "\n" + example.get("input", "") + "\n" + example["output"]
        return tokenizer(text, truncation=True, max_length=512, padding="max_length")

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    tokenized = tokenized.rename_column("input_ids", "input_ids")

    training_args = TrainingArguments(
        output_dir="./fora-output",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = FoRATrainer(
        model=model,
        args=training_args,
        fora_config=config,
        lr_adamw=2e-4,
        train_dataset=tokenized,
    )

    print("Starting training …")
    trainer.train()
    print("Done.")


if __name__ == "__main__":
    main()
