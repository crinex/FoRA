"""Fisher score computation and top-K layer selection for FoRA.

FoRA uses a single forward-backward pass on a small calibration set
(128 samples by default) to compute the diagonal empirical Fisher
information score for each transformer layer. The top-K layers by
score are selected for LoRA+Stiefel adaptation; the remaining layers
are left frozen.

This incurs less than 1% of total training cost and requires no additional
hyperparameter tuning beyond K (set to L/2 by default).
"""

from __future__ import annotations

import gc
from typing import Dict, List

import torch
import torch.nn as nn


def _get_transformer_layers(model: nn.Module) -> list:
    """Return the list of transformer decoder layers (architecture-agnostic).

    Supports:
        - LLaMA / Qwen / Gemma / Mistral: model.model.layers
        - GPT-2 / DistilGPT-2:            model.transformer.h
        - GPT-NeoX:                        model.gpt_neox.layers
        - BERT / RoBERTa:                  model.encoder.layer

    For PEFT-wrapped models, unwraps one level to reach the base model first.

    Raises:
        NotImplementedError: If the architecture is not recognized.
    """
    unwrapped = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        unwrapped = model.base_model.model

    if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "layers"):
        return list(unwrapped.model.layers)
    if hasattr(unwrapped, "transformer") and hasattr(unwrapped.transformer, "h"):
        return list(unwrapped.transformer.h)
    if hasattr(unwrapped, "gpt_neox") and hasattr(unwrapped.gpt_neox, "layers"):
        return list(unwrapped.gpt_neox.layers)
    if hasattr(unwrapped, "encoder") and hasattr(unwrapped.encoder, "layer"):
        return list(unwrapped.encoder.layer)

    raise NotImplementedError(
        f"Unrecognized architecture: {type(model).__name__}. "
        "Expected model.model.layers, model.transformer.h, "
        "model.gpt_neox.layers, or model.encoder.layer."
    )


def compute_fisher_scores(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n_batches: int = 128,
    device: str = "cuda",
) -> Dict[int, float]:
    """Compute diagonal empirical Fisher score per transformer layer.

    Each layer's score is the sum of squared gradients of all parameters
    in that layer, averaged over `n_batches` forward-backward passes on
    the calibration set. Higher scores indicate layers where the model
    loss is more sensitive to parameter perturbation.

    The model is temporarily set to requires_grad=True for floating-point
    parameters during scoring, then restored. Gradients are cleared after.

    Args:
        model: A HuggingFace CausalLM (or PEFT-wrapped variant).
        dataloader: DataLoader yielding dicts with key ``input_ids`` of shape
            (batch_size, seq_len). Labels are set equal to input_ids
            (causal language modelling loss).
        n_batches: Number of batches to accumulate gradients over.
        device: Device to run computation on.

    Returns:
        Dict mapping layer index (int, 0-based) to Fisher score (float).
    """
    layers = _get_transformer_layers(model)
    n_layers = len(layers)
    layer_scores: Dict[int, float] = {i: 0.0 for i in range(n_layers)}

    # Enable gradients on floating-point parameters only.
    # Skips quantized (uint8 / Params4bit) weights in QLoRA-loaded models.
    orig_requires_grad: Dict[int, bool] = {}
    for pid, p in enumerate(model.parameters()):
        orig_requires_grad[pid] = p.requires_grad
        if p.dtype.is_floating_point:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    model.eval()
    n_processed = 0

    try:
        for batch in dataloader:
            if n_processed >= n_batches:
                break
            input_ids = batch["input_ids"].to(device)
            model.zero_grad()
            outputs = model(input_ids, labels=input_ids)
            outputs.loss.backward()

            for layer_idx, layer in enumerate(layers):
                layer_fisher = 0.0
                for param in layer.parameters():
                    if param.grad is not None:
                        layer_fisher += (param.grad.float() ** 2).sum().item()
                layer_scores[layer_idx] += layer_fisher

            n_processed += 1
    finally:
        # Always restore original requires_grad state and clear gradients
        model.zero_grad()
        for pid, p in enumerate(model.parameters()):
            p.requires_grad_(orig_requires_grad[pid])
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    denom = max(n_processed, 1)
    for idx in layer_scores:
        layer_scores[idx] /= denom

    return layer_scores


def select_top_k_layers(
    scores: Dict[int, float],
    k: int,
) -> List[int]:
    """Select the top-K layers by Fisher score.

    Args:
        scores: Dict mapping layer index to Fisher score, as returned by
            ``compute_fisher_scores``.
        k: Number of layers to select. For FoRA paper results, k = n_layers // 2.

    Returns:
        Sorted list of k layer indices with the highest Fisher scores
        (ascending order, 0-based).

    Raises:
        ValueError: If k > len(scores) or k < 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if k > len(scores):
        raise ValueError(
            f"Requested k={k} but only {len(scores)} layers available."
        )
    sorted_layers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted(idx for idx, _ in sorted_layers[:k])
