"""CayleyAdam optimizer and dual-optimizer factory for FoRA.

CayleyAdam maintains the Stiefel constraint B^T B = I_r on lora_B weights
via iterative Cayley retraction. All Stiefel math runs in fp32 regardless
of the parameter's storage dtype.

make_fora_optimizer_groups() splits model parameters into:
  - optimizer_adamw:  AdamW for everything except lora_B of Stiefel layers
  - optimizer_cayley: CayleyAdam for lora_B of Stiefel layers only
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from ._stiefel_math import EPS, cayley_loop, matrix_norm_one, qr_retraction
from .config import FoRAConfig


class CayleyAdam(Optimizer):
    """Adam on Stiefel manifold via iterative Cayley transform.

    All parameters in this optimizer are assumed to require column-orthonormality
    (B^T B = I_r for tall (out, r) matrices). Tall parameters are auto-transposed
    internally so the optimizer math always works on row-orthonormal (p, n) views.

    Args:
        params: Iterable of parameters (typically lora_B weights only).
        lr: Learning rate cap. Actual step size is min(lr, 1/||W||_1).
        beta1: First-moment decay (default 0.9).
        beta2: Second-moment decay (default 0.99).
        eps: Numerical stability epsilon.
        qr_reset_period: Every N steps, re-project to manifold via QR to
            absorb numerical drift. 0 disables periodic re-projection.
        n_cayley_iter: Fixed-point iterations for the Cayley step.
        generator: Optional torch.Generator for reproducible QR resets.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        qr_reset_period: int = 100,
        n_cayley_iter: int = 5,
        generator: torch.Generator | None = None,
    ) -> None:
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            qr_reset_period=qr_reset_period,
            n_cayley_iter=n_cayley_iter,
        )
        super().__init__(params, defaults)
        self._gen = generator

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lr = group["lr"]
            qr_reset_period = group["qr_reset_period"]
            n_cayley_iter = group["n_cayley_iter"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise ValueError(
                        f"CayleyAdam expects 2-D parameters; got shape {tuple(p.shape)}"
                    )

                # Work on (p, n) with p <= n (row-orthonormal view).
                # Tall (out, r) matrices are transposed so B^T B = I_r becomes row-orth.
                native = p.data
                grad_native = p.grad.data
                tall = native.shape[0] > native.shape[1]
                if tall:
                    X_native = native.t()       # (r, out)
                    G_native = grad_native.t()
                else:
                    X_native = native           # (p, n), p <= n
                    G_native = grad_native

                # Force fp32 for all Stiefel math
                orig_dtype = X_native.dtype
                X = X_native.to(torch.float32)  # (p, n)
                G = G_native.to(torch.float32)  # (p, n)
                p_dim, n_dim = X.shape

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # m stored in (n, p) layout (matches Wen-Yin 2013 convention)
                    state["m_buffer"] = torch.zeros(
                        (n_dim, p_dim), dtype=torch.float32, device=X.device
                    )
                    state["v_buffer"] = torch.zeros(
                        (), dtype=torch.float32, device=X.device
                    )

                m = state["m_buffer"]
                v = state["v_buffer"]
                state["step"] += 1
                t_step = state["step"]

                # Periodic QR re-projection to fight numerical drift
                if qr_reset_period > 0 and (t_step % qr_reset_period == 0):
                    X = qr_retraction(X)

                # Adam-Frobenius update: scalar second moment via Frobenius norm
                m.mul_(beta1).add_(G.t(), alpha=1.0 - beta1)           # (n, p)
                v.mul_(beta2).add_(G.pow(2).sum(), alpha=1.0 - beta2)  # scalar

                bias_c1 = 1.0 - beta1 ** t_step
                bias_c2 = 1.0 - beta2 ** t_step
                m_hat = m / bias_c1     # (n, p)
                v_hat = v / bias_c2     # scalar

                # Riemannian gradient on Stiefel: skew-symmetric W from m_hat.
                # X is (p, n) with rows orthonormal. m_hat is (n, p).
                MX = m_hat @ X          # (n, n)
                XMX = X @ MX            # (p, n)
                XXMX = X.t() @ XMX     # (n, n)
                W_hat = MX - 0.5 * XXMX                         # (n, n)
                W = (W_hat - W_hat.t()) / (v_hat.sqrt() + eps)  # (n, n) skew-sym

                # Step size: cap by 1/||W||_1 to keep Cayley iteration contracting
                step_bound = (0.5 * 2.0) / (matrix_norm_one(W) + EPS)
                alpha = torch.minimum(
                    step_bound, torch.tensor(lr, device=X.device)
                )

                # Cayley iteration; cayley_loop expects X in (n, p)
                X_np = X.t().contiguous()                        # (n, p)
                W_X = W @ X_np                                   # (n, p) tangent
                X_new = cayley_loop(X_np, W, W_X, -alpha, n_iter=n_cayley_iter)
                # Returns (p, n)

                # Cast back and write into native storage
                X_new_native = X_new.to(orig_dtype)
                if tall:
                    native.copy_(X_new_native.t())
                else:
                    native.copy_(X_new_native)

                # Refresh m to be consistent with new X
                m.copy_((W @ X_new.t()) * (v_hat.sqrt() + eps) * bias_c1)

        return loss


def make_fora_optimizer_groups(
    model: nn.Module,
    config: FoRAConfig,
    lr_adamw: float = 2e-4,
) -> Tuple[AdamW, CayleyAdam]:
    """Split model parameters into two optimizer groups for FoRA training.

    Stiefel layers are identified by the presence of a ``cayley_gate`` attribute
    (injected by ``apply_fora``). For those layers:
        - lora_B weight  → CayleyAdam  (Stiefel constraint)
        - lora_A weight  → AdamW
        - cayley_gate    → AdamW

    All other trainable parameters also go to AdamW.

    Args:
        model: PEFT-wrapped model after ``apply_fora()``.
        config: FoRAConfig instance providing lr_stiefel, cayley_n_iter,
            and qr_reset_period.
        lr_adamw: Learning rate for the AdamW optimizer.

    Returns:
        Tuple (optimizer_adamw, optimizer_cayley).
        Both optimizers must be stepped every training iteration.
    """
    stiefel_params: list[nn.Parameter] = []
    euclidean_params: list[nn.Parameter] = []
    seen: set[int] = set()

    for module in model.modules():
        if not hasattr(module, "cayley_gate"):
            continue
        # Collect lora_B → Stiefel
        if "default" in getattr(module, "lora_B", {}):
            w_B = module.lora_B["default"].weight
            if id(w_B) not in seen and w_B.requires_grad:
                stiefel_params.append(w_B)
                seen.add(id(w_B))
        # Collect lora_A + gate → Euclidean
        if "default" in getattr(module, "lora_A", {}):
            w_A = module.lora_A["default"].weight
            if id(w_A) not in seen and w_A.requires_grad:
                euclidean_params.append(w_A)
                seen.add(id(w_A))
        gate = module.cayley_gate
        if id(gate) not in seen and gate.requires_grad:
            euclidean_params.append(gate)
            seen.add(id(gate))

    # Everything else trainable (non-stiefelized LoRA layers, etc.)
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            euclidean_params.append(p)
            seen.add(id(p))

    optimizer_adamw = AdamW(euclidean_params, lr=lr_adamw)
    optimizer_cayley = CayleyAdam(
        stiefel_params,
        lr=config.lr_stiefel,
        n_cayley_iter=config.cayley_n_iter,
        qr_reset_period=config.qr_reset_period,
    )
    return optimizer_adamw, optimizer_cayley
