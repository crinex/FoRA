"""FoRAConfig — extends HuggingFace PEFT LoraConfig with FoRA-specific params."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from peft import LoraConfig


@dataclass
class FoRAConfig(LoraConfig):
    """Configuration for Fisher-orthogonal Rank Adaptation (FoRA).

    Extends LoraConfig with parameters controlling Fisher-based layer selection
    and Stiefel manifold optimization of lora_B.

    Fisher selection:
        A one-time forward-backward pass on `fisher_calibration_samples` batches
        is used to score each transformer layer. The top `layer_budget_fraction`
        fraction of layers receives LoRA+Stiefel adapters; the rest are frozen.

    Stiefel optimization:
        lora_B of selected layers is constrained to the Stiefel manifold and
        updated with CayleyAdam. lora_A is updated with standard AdamW.

    Args:
        layer_budget_fraction: Fraction of transformer layers to adapt.
            K = max(1, int(n_layers * layer_budget_fraction)).
        fisher_calibration_samples: Number of batches used to estimate the
            diagonal empirical Fisher score. More batches give more stable
            scores at higher compute cost.
        lr_stiefel: Learning rate for the CayleyAdam optimizer acting on lora_B.
        cayley_n_iter: Number of fixed-point iterations in the Cayley retraction.
            5 iterations suffice for r <= 64 to maintain || B^T B - I ||_F < 1e-5.
        qr_reset_period: Every N CayleyAdam steps, re-project lora_B to the
            Stiefel manifold via QR to absorb numerical drift. 0 disables.
        r: LoRA rank. Default follows FoRA paper (32).
        lora_alpha: LoRA scaling alpha. Default follows FoRA paper (64).
        lora_dropout: Dropout applied before lora_A.
        target_modules: Linear module name patterns to target with LoRA.
        bias: Whether to train bias parameters.
    """

    peft_type: str = "FORA"

    # Fisher layer selection
    layer_budget_fraction: float = 0.5
    fisher_calibration_samples: int = 128

    # Stiefel optimizer
    lr_stiefel: float = 1e-3
    cayley_n_iter: int = 5
    qr_reset_period: int = 100

    # LoRA defaults aligned with FoRA paper
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
