"""apply_fora() — one-call FoRA setup for any PEFT-compatible transformer.

Steps:
  1. Build a calibration DataLoader from texts or use a supplied one.
  2. Compute diagonal empirical Fisher scores per transformer layer.
  3. Select top-K layers (K = max(1, int(n_layers * fraction))).
  4. Build a LoraConfig restricted to target_modules in selected layers.
  5. Wrap the base model with peft.get_peft_model().
  6. Re-initialise lora_B (Stiefel), lora_A (N(0,1/sqrt(r))), and add gates.

The returned model is ready for dual-optimizer training with FoRATrainer or
a custom loop calling make_fora_optimizer_groups().
"""

from __future__ import annotations

import copy
from typing import List, Optional

import torch
import torch.nn as nn

from .config import FoRAConfig
from .fisher import compute_fisher_scores, select_top_k_layers, _get_transformer_layers
from .layer import stiefelize_lora


def _make_calibration_dataloader(
    texts: List[str],
    tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    device: str = "cuda",
) -> torch.utils.data.DataLoader:
    """Build a minimal DataLoader from a list of plain strings."""
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    dataset = torch.utils.data.TensorDataset(encodings["input_ids"])

    class _CollateFn:
        def __call__(self, batch):
            input_ids = torch.stack([item[0] for item in batch])
            return {"input_ids": input_ids}

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_CollateFn(),
    )


def _selected_target_modules(
    base_model: nn.Module,
    layer_indices: List[int],
    candidate_modules: List[str],
) -> List[str]:
    """Return target_modules patterns restricted to selected layers.

    For each candidate module name (e.g. 'q_proj') and each selected layer
    index, emit a unique qualified pattern so PEFT targets only those modules.

    Pattern format: ``layers.<idx>.<module_name>`` — PEFT matches the suffix
    against the full module path, so this is architecture-agnostic for models
    whose layers are indexed as ``.layers.<idx>.``.

    For models using ``.h.<idx>.`` or ``.layer.<idx>.`` the same suffix-matching
    logic still works, but we emit all three prefix variants to be safe.
    """
    patterns: list[str] = []
    for idx in layer_indices:
        for mod_name in candidate_modules:
            # Emit all common layer-naming conventions
            patterns.append(f"layers.{idx}.{mod_name}")
            patterns.append(f"h.{idx}.{mod_name}")
            patterns.append(f"layer.{idx}.{mod_name}")
    return patterns


def apply_fora(
    model: nn.Module,
    tokenizer,
    config: FoRAConfig,
    calibration_dataloader: Optional[torch.utils.data.DataLoader] = None,
    calibration_texts: Optional[List[str]] = None,
    device: str = "cuda",
) -> nn.Module:
    """Apply FoRA to a pretrained model in-place.

    At least one of ``calibration_dataloader`` or ``calibration_texts`` must
    be provided. If both are given, ``calibration_dataloader`` takes precedence.

    Args:
        model: A HuggingFace CausalLM (or compatible model). Must already be
            moved to ``device`` before calling this function.
        tokenizer: HuggingFace tokenizer matching ``model``.
        config: FoRAConfig instance.
        calibration_dataloader: Optional pre-built DataLoader yielding dicts
            with ``input_ids``. Used for Fisher scoring.
        calibration_texts: Optional list of plain strings used to build a
            minimal calibration DataLoader (batch_size=4, max_length=512).
        device: Device string used for Fisher computation.

    Returns:
        PEFT-wrapped model with Stiefel-constrained lora_B adapters on the
        top-K Fisher-scored layers. Ready for FoRATrainer or dual-optimizer
        training.

    Raises:
        ValueError: If neither calibration_dataloader nor calibration_texts
            is provided.
    """
    import peft

    if calibration_dataloader is None and calibration_texts is None:
        raise ValueError(
            "Provide either calibration_dataloader or calibration_texts."
        )

    # Build DataLoader from texts if not supplied
    if calibration_dataloader is None:
        calibration_dataloader = _make_calibration_dataloader(
            calibration_texts,  # type: ignore[arg-type]
            tokenizer,
            device=device,
        )

    # Step 1: compute Fisher scores on the base (non-PEFT) model
    model.eval()
    scores = compute_fisher_scores(
        model,
        calibration_dataloader,
        n_batches=config.fisher_calibration_samples,
        device=device,
    )

    # Step 2: select top-K layers
    layers = _get_transformer_layers(model)
    n_layers = len(layers)
    k = max(1, int(n_layers * config.layer_budget_fraction))
    selected_layers = select_top_k_layers(scores, k)

    # Step 3: build PEFT LoraConfig restricted to selected layers
    candidate_modules: List[str] = list(config.target_modules or [])
    restricted_targets = _selected_target_modules(
        model, selected_layers, candidate_modules
    )

    lora_config = peft.LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=restricted_targets,
        bias=config.bias,
    )

    # Step 4: wrap with PEFT
    peft_model = peft.get_peft_model(model, lora_config)

    # Step 5: re-initialise selected adapters on Stiefel and add gates
    info = stiefelize_lora(peft_model, selected_layers)

    # Attach metadata for downstream inspection
    peft_model.fora_selected_layers = selected_layers
    peft_model.fora_fisher_scores = scores
    peft_model.fora_stiefel_info = info
    peft_model.fora_config = config

    return peft_model
