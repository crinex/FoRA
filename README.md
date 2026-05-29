# peft-fora
This code is the official implementation of the following paper: [FoRA: Fisher-orthogonal Rank Adaptation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2605.29317)
We integrated into the HuggingFace PEFT ecosystem.

FoRA is a parameter-efficient fine-tuning method that combines:
1. **Fisher-based layer selection** — a one-time forward-backward pass identifies the top-K most task-relevant transformer layers
2. **Stiefel-constrained LoRA** — `lora_B` is trained on the Stiefel manifold via CayleyAdam, preserving the singular-value spectrum of the adapter update

## Installation


```bash
git clone https://github.com/crinex/FoRA
cd FoRA
pip install -e .
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from fora_peft import FoRAConfig, apply_fora, FoRATrainer

model_name = "meta-llama/Llama-3.2-3B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

config = FoRAConfig(
    r=32,
    lora_alpha=64,
    layer_budget_fraction=0.5,     # top 50% of layers by Fisher score
    fisher_calibration_samples=128, # batches used to score layers
    lr_stiefel=1e-3,               # CayleyAdam lr for lora_B
)

# calibration_texts can be a list of strings or pass calibration_dataloader=
model = apply_fora(
    model, tokenizer, config,
    calibration_texts=["Hello world"] * 128,
    device="cuda",
)

print(f"Selected layers: {model.fora_selected_layers}")
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./fora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    fp16=True,
)

trainer = FoRATrainer(
    model=model,
    args=training_args,
    fora_config=config,
    lr_adamw=2e-4,
    train_dataset=...,
)
trainer.train()
```

## How FoRA Works

### Step 1: Fisher Layer Scoring

A single forward-backward pass on `fisher_calibration_samples` batches computes the diagonal empirical Fisher score per transformer layer:

```
score(l) = E[ || grad_theta(l) loss ||^2 ]
```

Layers with higher scores are more sensitive to the task and are better candidates for adaptation.

### Step 2: Top-K Layer Selection

The top `K = int(n_layers * layer_budget_fraction)` layers are selected. For a 32-layer model with `fraction=0.5`, only 16 layers receive LoRA adapters.

### Step 3: Stiefel-Constrained LoRA

For selected layers:
- `lora_B` (shape `[out, r]`) is initialised column-orthonormal (`B^T B = I_r`) and optimised with **CayleyAdam** to stay on the Stiefel manifold
- `lora_A` is initialised with `N(0, 1/sqrt(r))` and optimised with standard **AdamW**
- A learnable scalar `cayley_gate` (init 0) ensures `delta_W = 0` at step 0

Because `B` remains column-orthonormal throughout training, the singular values of the adapter update `B @ A` equal those of `A` alone, giving FoRA precise control over the effective rank of the update.

## API Reference

| Symbol | Description |
|---|---|
| `FoRAConfig` | Configuration dataclass (extends `peft.LoraConfig`) |
| `apply_fora(model, tokenizer, config, ...)` | One-call FoRA setup: Fisher scoring, layer selection, PEFT wrapping, Stiefel init |
| `FoRATrainer` | `transformers.Trainer` subclass with dual-optimizer support |
| `CayleyAdam` | Stiefel manifold optimizer via iterative Cayley retraction |
| `make_fora_optimizer_groups(model, config, lr_adamw)` | Returns `(optimizer_adamw, optimizer_cayley)` for custom training loops |

### FoRAConfig Parameters

| Parameter | Default | Description |
|---|---|---|
| `r` | 32 | LoRA rank |
| `lora_alpha` | 64 | LoRA scaling alpha |
| `lora_dropout` | 0.05 | Dropout before lora_A |
| `target_modules` | `["q_proj","k_proj","v_proj","up_proj","down_proj"]` | Module name patterns to target |
| `layer_budget_fraction` | 0.5 | Fraction of layers to adapt (K = n_layers * fraction) |
| `fisher_calibration_samples` | 128 | Calibration batches for Fisher scoring |
| `lr_stiefel` | 1e-3 | CayleyAdam learning rate for lora_B |
| `cayley_n_iter` | 5 | Fixed-point iterations in Cayley retraction |
| `qr_reset_period` | 100 | QR re-projection period (0 = disabled) |

### Custom Training Loop

If you prefer not to use `FoRATrainer`:

```python
from fora_peft import make_fora_optimizer_groups

optimizer_adamw, optimizer_cayley = make_fora_optimizer_groups(
    model, config, lr_adamw=2e-4
)

for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()

    optimizer_adamw.step()
    optimizer_cayley.step()

    optimizer_adamw.zero_grad()
    optimizer_cayley.zero_grad()
```

## Comparison with Standard LoRA

| Feature | LoRA | FoRA |
|---|---|---|
| Layer selection | All target layers | Fisher top-K |
| `lora_B` constraint | None (zero init) | Stiefel manifold |
| `lora_B` optimizer | AdamW | CayleyAdam |
| `lora_A` optimizer | AdamW | AdamW |
| Effective rank control | Implicit | Explicit (B^T B = I) |
| Calibration cost | None | ~1% of training |

## Supported Architectures

`apply_fora` and the Fisher scorer support any model whose transformer layers are accessible as:

- `model.model.layers` — LLaMA, Qwen, Gemma, Mistral
- `model.transformer.h` — GPT-2, DistilGPT-2
- `model.gpt_neox.layers` — GPT-NeoX
- `model.encoder.layer` — BERT, RoBERTa

For other architectures, pass a custom `calibration_dataloader` and inspect `model.fora_selected_layers` after calling `apply_fora`.

## Examples

- `examples/basic_usage.py` — minimal end-to-end example
- `examples/commonsense_finetune.py` — BoolQ / HellaSwag / WinoGrande / ARC
- `examples/instruction_finetune.py` — Alpaca instruction tuning

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=fora_peft
```
