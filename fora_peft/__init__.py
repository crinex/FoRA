"""fora_peft — FoRA integration for the HuggingFace PEFT ecosystem.

Fisher-orthogonal Rank Adaptation (FoRA) is a parameter-efficient fine-tuning
method that combines Fisher-based layer selection with Stiefel-manifold
constrained LoRA adapters.

Public API::

    from fora_peft import FoRAConfig, apply_fora, CayleyAdam, make_fora_optimizer_groups, FoRATrainer
"""

from .config import FoRAConfig
from .model import apply_fora
from .optimizer import CayleyAdam, make_fora_optimizer_groups
from .trainer import FoRATrainer

__all__ = [
    "FoRAConfig",
    "apply_fora",
    "CayleyAdam",
    "make_fora_optimizer_groups",
    "FoRATrainer",
]

__version__ = "0.1.0"
