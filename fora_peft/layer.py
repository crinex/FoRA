"""FoRA layer utilities.

After PEFT wraps the model with LoRA adapters, this module:
  1. Re-initialises lora_B to a column-orthonormal matrix (Stiefel init).
  2. Re-initialises lora_A with N(0, 1/sqrt(r)) — not zero.
  3. Adds a learnable scalar gate (cayley_gate) so delta-W = 0 at step 0.
  4. Patches the forward method of each selected LoraLayer to apply the gate.

These operations are performed in-place on the PEFT model returned by
peft.get_peft_model(), and must be called before any optimizer is created.
"""

from __future__ import annotations

import math
from types import MethodType
from typing import Iterable

import torch
import torch.nn as nn

from ._stiefel_math import qr_retraction


def _parse_layer_idx(name: str) -> int | None:
    """Extract the transformer layer index from a fully-qualified module name.

    Recognises:
      - LLaMA / Qwen / Gemma / Mistral: ``.model.layers.<idx>.``
      - GPT-2 / DistilGPT-2:            ``.transformer.h.<idx>.``
      - BERT / RoBERTa:                  ``.encoder.layer.<idx>.``
      - GPT-NeoX:                        ``.gpt_neox.layers.<idx>.``
    """
    parts = name.split(".")
    keys = ("layers", "h", "layer")
    for i, p in enumerate(parts):
        if p in keys and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return None


def _orth_init_b(weight: torch.Tensor) -> None:
    """In-place column-orthonormal initialisation for a (out_features, r) matrix.

    Generates a random Gaussian matrix then applies QR retraction to obtain a
    column-orthonormal result. All arithmetic is done in fp32 to avoid numerical
    issues; the result is cast back to the original dtype before writing.
    """
    out_dim, r = weight.shape
    # Build (r, out_dim) in fp32, QR-retract to row-orthonormal, then transpose
    g = torch.randn(r, out_dim, dtype=torch.float32, device=weight.device)
    q = qr_retraction(g)            # (r, out_dim), row-orthonormal
    weight.data.copy_(q.t().to(weight.dtype))   # (out_dim, r), column-orthonormal


def _normal_init_a(weight: torch.Tensor, r: int) -> None:
    """In-place N(0, 1/sqrt(r)) initialisation for lora_A.

    PEFT default initialises lora_A with Kaiming uniform and lora_B with zeros.
    FoRA requires lora_A != 0 at step 0 so that the delta-W = lora_B @ lora_A
    carries a meaningful signal from the start (gated by cayley_gate which
    opens gradually from zero).
    """
    std = 1.0 / math.sqrt(r)
    nn.init.normal_(weight, mean=0.0, std=std)


def _patched_lora_forward(self, x, *args, **kwargs):  # type: ignore[no-untyped-def]
    """LoRA forward with cayley_gate scaling the adapter branch.

    Mirrors peft.tuners.lora.Linear.forward but multiplies the LoRA delta by
    ``cayley_gate`` (a learnable scalar initialised to 0).  This ensures
    delta-W = 0 at step 0 regardless of lora_B / lora_A initialisations.
    """
    if hasattr(self, "_check_forward_args"):
        self._check_forward_args(x, *args, **kwargs)

    result = self.base_layer(x, *args, **kwargs)
    if getattr(self, "disable_adapters", False):
        return result

    torch_result_dtype = result.dtype
    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A:
            continue
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        gate = self.cayley_gate                     # scalar learnable parameter
        x_in = x.to(lora_A.weight.dtype)
        delta = lora_B(lora_A(dropout(x_in))) * (scaling * gate)
        result = result + delta.to(torch_result_dtype)
    return result


def stiefelize_lora(
    model: nn.Module,
    layer_indices: Iterable[int],
    gate_init: float = 0.0,
) -> dict:
    """Mutate a PEFT model so selected LoRA adapters are Stiefel-constrained.

    For each LoraLayer whose transformer-layer index is in ``layer_indices``:
      1. Re-initialise lora_B to column-orthonormal (Stiefel manifold).
      2. Re-initialise lora_A with N(0, 1/sqrt(r)).
      3. Attach a learnable scalar gate ``cayley_gate`` (init ``gate_init``).
      4. Patch the module's forward to apply the gate.

    Args:
        model: PEFT-wrapped model (output of ``peft.get_peft_model``).
        layer_indices: Iterable of 0-based transformer layer indices to adapt.
        gate_init: Initial value of cayley_gate. 0.0 ensures delta-W = 0
            at the start of training.

    Returns:
        Dict with keys:
            ``patched``    — list of (name, module) tuples that were modified.
            ``n_modules``  — total number of patched LoRA modules.
            ``n_params``   — total number of Stiefel parameters (lora_B elems).
    """
    try:
        from peft.tuners.lora.layer import LoraLayer
    except ImportError as exc:
        raise RuntimeError("peft >= 0.5 is required for FoRA.") from exc

    layer_set = set(layer_indices)
    patched: list[tuple[str, nn.Module]] = []

    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        idx = _parse_layer_idx(name)
        if idx is None or idx not in layer_set:
            continue
        if "default" not in module.lora_B:
            continue

        w_B = module.lora_B["default"].weight   # (out_features, r)
        w_A = module.lora_A["default"].weight   # (r, in_features)
        r = w_B.shape[1]

        _orth_init_b(w_B)
        _normal_init_a(w_A, r)

        gate = nn.Parameter(
            torch.tensor(
                gate_init,
                dtype=w_B.dtype,
                device=w_B.device,
            )
        )
        module.cayley_gate = gate
        module.forward = MethodType(_patched_lora_forward, module)
        patched.append((name, module))

    n_params = sum(m.lora_B["default"].weight.numel() for _, m in patched)
    return {"patched": patched, "n_modules": len(patched), "n_params": n_params}
