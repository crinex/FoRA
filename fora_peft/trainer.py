"""FoRATrainer — HuggingFace Trainer extended for dual-optimizer FoRA training.

FoRA requires two optimizers per training step:
  - AdamW for lora_A, cayley_gate, and all non-Stiefel parameters
  - CayleyAdam for lora_B of Stiefel-constrained layers

FoRATrainer overrides create_optimizer() and training_step() to handle both.
The AdamW instance is registered as self.optimizer so that the Trainer
scheduler and gradient-clipping machinery work without modification.
CayleyAdam is stored as self.optimizer_cayley and stepped after every
AdamW step.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments

from .config import FoRAConfig
from .optimizer import CayleyAdam, make_fora_optimizer_groups


class FoRATrainer(Trainer):
    """HuggingFace Trainer with dual-optimizer support for FoRA.

    Usage::

        trainer = FoRATrainer(
            model=fora_model,
            args=training_args,
            fora_config=config,
            lr_adamw=2e-4,
            train_dataset=...,
            ...
        )
        trainer.train()

    Args:
        fora_config: FoRAConfig instance providing Stiefel optimizer settings.
        lr_adamw: Learning rate for the AdamW optimizer (lora_A, gates, other
            trainable params). Defaults to 2e-4.
        All other keyword arguments are forwarded to ``transformers.Trainer``.
    """

    def __init__(
        self,
        *args,
        fora_config: Optional[FoRAConfig] = None,
        lr_adamw: float = 2e-4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fora_config = fora_config
        self.lr_adamw = lr_adamw
        self.optimizer_cayley: Optional[CayleyAdam] = None

    # ------------------------------------------------------------------
    # Optimizer creation
    # ------------------------------------------------------------------

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Override to build both AdamW and CayleyAdam."""
        if self.fora_config is None:
            return super().create_optimizer()

        optimizer_adamw, optimizer_cayley = make_fora_optimizer_groups(
            self.model,
            self.fora_config,
            lr_adamw=self.lr_adamw,
        )
        self.optimizer = optimizer_adamw
        self.optimizer_cayley = optimizer_cayley
        return self.optimizer

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(
        self, model: nn.Module, inputs: dict
    ) -> torch.Tensor:
        """Override to step CayleyAdam after the standard AdamW step.

        The parent class handles:
          - forward pass + loss computation
          - loss.backward()
          - gradient clipping
          - self.optimizer.step() + self.optimizer.zero_grad()

        We add an extra step for the Stiefel optimizer after the parent returns.
        """
        loss = super().training_step(model, inputs)

        if self.optimizer_cayley is not None:
            self.optimizer_cayley.step()
            self.optimizer_cayley.zero_grad()

        return loss

    # ------------------------------------------------------------------
    # Scheduler compatibility
    # ------------------------------------------------------------------

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """Use AdamW (self.optimizer) for the LR scheduler.

        CayleyAdam uses a fixed LR cap; no scheduler is applied to it.
        """
        return super().create_scheduler(
            num_training_steps=num_training_steps,
            optimizer=optimizer or self.optimizer,
        )
