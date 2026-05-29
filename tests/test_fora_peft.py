"""Unit tests for fora_peft — no heavy model loading required.

Tests cover:
  - FoRAConfig dataclass
  - Stiefel math utilities (_stiefel_math)
  - CayleyAdam step maintains B^T B = I
  - Fisher layer selection logic
  - stiefelize_lora init and gate injection
  - make_fora_optimizer_groups parameter partitioning
  - _parse_layer_idx name parsing
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# FoRAConfig
# ---------------------------------------------------------------------------

class TestFoRAConfig:
    def test_defaults(self):
        from fora_peft import FoRAConfig
        cfg = FoRAConfig()
        assert cfg.r == 32
        assert cfg.lora_alpha == 64
        assert cfg.layer_budget_fraction == 0.5
        assert cfg.fisher_calibration_samples == 128
        assert cfg.lr_stiefel == 1e-3
        assert cfg.cayley_n_iter == 5
        assert cfg.qr_reset_period == 100

    def test_custom_values(self):
        from fora_peft import FoRAConfig
        cfg = FoRAConfig(r=8, lora_alpha=16, layer_budget_fraction=0.25)
        assert cfg.r == 8
        assert cfg.lora_alpha == 16
        assert cfg.layer_budget_fraction == 0.25


# ---------------------------------------------------------------------------
# Stiefel math
# ---------------------------------------------------------------------------

class TestStiefelMath:
    def test_qr_retraction_row_orthonormal(self):
        from fora_peft._stiefel_math import qr_retraction
        g = torch.randn(4, 16, dtype=torch.float32)
        q = qr_retraction(g)
        assert q.shape == (4, 16)
        err = (q @ q.t() - torch.eye(4)).norm().item()
        assert err < 1e-5, f"Row-orthonormality error: {err}"

    def test_orth_drift_tall(self):
        from fora_peft._stiefel_math import orth_drift, qr_retraction
        g = torch.randn(4, 64, dtype=torch.float32)
        q = qr_retraction(g)                    # (4, 64) row-orthonormal
        B = q.t()                               # (64, 4) column-orthonormal
        drift = orth_drift(B).item()
        assert drift < 1e-5, f"Drift too large: {drift}"

    def test_matrix_norm_one(self):
        from fora_peft._stiefel_math import matrix_norm_one
        W = torch.ones(4, 4)
        assert matrix_norm_one(W).item() == pytest.approx(4.0)

    def test_cayley_loop_returns_correct_shape(self):
        from fora_peft._stiefel_math import cayley_loop
        p, n = 4, 8
        X = torch.randn(n, p)
        W = torch.randn(n, n)
        W = W - W.t()   # skew-symmetric
        tan_vec = W @ X
        Y = cayley_loop(X, W, tan_vec, t=-0.01, n_iter=5)
        assert Y.shape == (p, n)


# ---------------------------------------------------------------------------
# CayleyAdam — Stiefel constraint
# ---------------------------------------------------------------------------

class TestCayleyAdam:
    def _make_stiefel_param(self, out: int, r: int) -> nn.Parameter:
        """Column-orthonormal (out, r) parameter."""
        from fora_peft._stiefel_math import qr_retraction
        g = torch.randn(r, out, dtype=torch.float32)
        q = qr_retraction(g)        # (r, out)
        return nn.Parameter(q.t().clone())  # (out, r)

    def test_step_preserves_orthonormality(self):
        from fora_peft import CayleyAdam
        p = self._make_stiefel_param(32, 8)
        opt = CayleyAdam([p], lr=1e-3, qr_reset_period=0)

        for _ in range(5):
            p.grad = torch.randn_like(p)
            opt.step()
            opt.zero_grad()

        # Check B^T B ≈ I_r
        BtB = p.data.t() @ p.data   # (r, r)
        err = (BtB - torch.eye(8)).norm().item()
        assert err < 1e-3, f"Orthonormality drift after 5 steps: {err}"

    def test_step_returns_none_without_closure(self):
        from fora_peft import CayleyAdam
        p = self._make_stiefel_param(16, 4)
        opt = CayleyAdam([p], lr=1e-3)
        p.grad = torch.randn_like(p)
        result = opt.step()
        assert result is None

    def test_step_with_closure(self):
        from fora_peft import CayleyAdam
        p = self._make_stiefel_param(16, 4)
        opt = CayleyAdam([p], lr=1e-3)
        p.grad = torch.randn_like(p)

        def closure():
            return torch.tensor(1.0)

        loss = opt.step(closure)
        assert loss is not None

    def test_rejects_non_2d_param(self):
        from fora_peft import CayleyAdam
        p = nn.Parameter(torch.randn(4, 4, 4))
        opt = CayleyAdam([p], lr=1e-3)
        p.grad = torch.randn_like(p)
        with pytest.raises(ValueError, match="2-D"):
            opt.step()

    def test_qr_reset(self):
        from fora_peft import CayleyAdam
        from fora_peft._stiefel_math import qr_retraction
        p = self._make_stiefel_param(32, 8)
        # Corrupt orthonormality slightly
        p.data += torch.randn_like(p.data) * 0.1
        opt = CayleyAdam([p], lr=1e-3, qr_reset_period=1)
        p.grad = torch.randn_like(p)
        opt.step()
        BtB = p.data.t() @ p.data
        err = (BtB - torch.eye(8)).norm().item()
        assert err < 5e-2


# ---------------------------------------------------------------------------
# Fisher layer selection
# ---------------------------------------------------------------------------

class TestFisherSelection:
    def test_select_top_k(self):
        from fora_peft.fisher import select_top_k_layers
        scores = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.3}
        top2 = select_top_k_layers(scores, k=2)
        assert top2 == [1, 2]   # sorted ascending

    def test_select_top_k_all(self):
        from fora_peft.fisher import select_top_k_layers
        scores = {i: float(i) for i in range(10)}
        result = select_top_k_layers(scores, k=10)
        assert result == list(range(10))

    def test_select_top_k_raises_if_k_too_large(self):
        from fora_peft.fisher import select_top_k_layers
        scores = {0: 1.0, 1: 2.0}
        with pytest.raises(ValueError):
            select_top_k_layers(scores, k=5)

    def test_select_top_k_raises_if_k_zero(self):
        from fora_peft.fisher import select_top_k_layers
        scores = {0: 1.0}
        with pytest.raises(ValueError):
            select_top_k_layers(scores, k=0)


# ---------------------------------------------------------------------------
# _parse_layer_idx
# ---------------------------------------------------------------------------

class TestParseLayerIdx:
    def test_llama_style(self):
        from fora_peft.layer import _parse_layer_idx
        assert _parse_layer_idx("base_model.model.model.layers.7.self_attn.q_proj") == 7

    def test_gpt2_style(self):
        from fora_peft.layer import _parse_layer_idx
        assert _parse_layer_idx("base_model.model.transformer.h.3.attn.c_attn") == 3

    def test_bert_style(self):
        from fora_peft.layer import _parse_layer_idx
        assert _parse_layer_idx("base_model.model.encoder.layer.11.attention.self.query") == 11

    def test_no_layer_idx_returns_none(self):
        from fora_peft.layer import _parse_layer_idx
        assert _parse_layer_idx("model.embed_tokens") is None

    def test_non_numeric_ignored(self):
        from fora_peft.layer import _parse_layer_idx
        assert _parse_layer_idx("model.layers.abc.q_proj") is None


# ---------------------------------------------------------------------------
# stiefelize_lora — stub LoRA model
# ---------------------------------------------------------------------------

def _make_stub_peft_model(n_layers: int = 4, r: int = 8, hidden: int = 32):
    """Build a minimal stub that mimics a PEFT-wrapped model for stiefelize_lora tests.

    Uses composition (LoraLayer wraps a plain nn.Linear as base_layer) to avoid
    the infinite recursion that occurs when LoraLayer and nn.Linear are combined
    via multiple inheritance (PEFT's weight property intercepts nn.Linear.__init__).
    """
    try:
        from peft.tuners.lora.layer import LoraLayer
    except ImportError:
        pytest.skip("peft not installed")

    class StubLoraLinear(nn.Module, LoraLayer):
        """Minimal stub that mirrors how PEFT's lora.Linear is built.

        Inherits nn.Module (so it registers sub-modules and parameters) and
        LoraLayer (so isinstance(module, LoraLayer) is True). The base_layer
        is a *separate* nn.Linear instance passed to LoraLayer.__init__ so
        LoraLayer's weight property doesn't recurse into self.
        """

        def __init__(self, in_f: int, out_f: int, r_val: int):
            nn.Module.__init__(self)
            base = nn.Linear(in_f, out_f, bias=False)
            LoraLayer.__init__(self, base_layer=base)
            # Register base as a sub-module so its parameters are visible
            self.base_linear = base
            self.r = {"default": r_val}
            self.lora_alpha = {"default": r_val}
            self.scaling = {"default": 1.0}
            self.lora_A = nn.ModuleDict(
                {"default": nn.Linear(in_f, r_val, bias=False)}
            )
            self.lora_B = nn.ModuleDict(
                {"default": nn.Linear(r_val, out_f, bias=False)}
            )
            self.lora_dropout = nn.ModuleDict({"default": nn.Identity()})
            # active_adapters is a property backed by _active_adapter in BaseTunerLayer
            self._active_adapter = "default"
            # PEFT default: lora_B = 0
            nn.init.zeros_(self.lora_B["default"].weight)

        def forward(self, x):
            return self.base_linear(x)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = StubLoraLinear(hidden, hidden, r)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer() for _ in range(n_layers)])

    model = FakeModel()
    return model, r


class TestStiefelize:
    def test_orth_init_applied(self):
        from fora_peft.layer import stiefelize_lora
        from fora_peft._stiefel_math import orth_drift

        model, r = _make_stub_peft_model(n_layers=4, r=8)
        info = stiefelize_lora(model, layer_indices=[0, 2])

        for _, mod in info["patched"]:
            w_B = mod.lora_B["default"].weight.float()
            drift = orth_drift(w_B).item()
            assert drift < 1e-4, f"lora_B not orthonormal: drift={drift}"

    def test_gate_attached(self):
        from fora_peft.layer import stiefelize_lora
        model, _ = _make_stub_peft_model(n_layers=4, r=8)
        info = stiefelize_lora(model, layer_indices=[1])
        for _, mod in info["patched"]:
            assert hasattr(mod, "cayley_gate")
            assert mod.cayley_gate.item() == pytest.approx(0.0)

    def test_lora_a_nonzero(self):
        from fora_peft.layer import stiefelize_lora
        model, _ = _make_stub_peft_model(n_layers=4, r=8)
        info = stiefelize_lora(model, layer_indices=[0])
        for _, mod in info["patched"]:
            w_A = mod.lora_A["default"].weight
            assert w_A.norm().item() > 0.0, "lora_A should be non-zero after init"

    def test_correct_modules_patched(self):
        from fora_peft.layer import stiefelize_lora
        model, _ = _make_stub_peft_model(n_layers=4, r=8)
        info = stiefelize_lora(model, layer_indices=[0, 3])
        patched_names = [name for name, _ in info["patched"]]
        # Layer 1 and 2 should not be patched
        for name in patched_names:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    idx = int(parts[i + 1])
                    assert idx in {0, 3}

    def test_n_modules_and_n_params(self):
        from fora_peft.layer import stiefelize_lora
        model, r = _make_stub_peft_model(n_layers=4, r=8, hidden=32)
        info = stiefelize_lora(model, layer_indices=[0, 1])
        assert info["n_modules"] == 2
        # Each lora_B is (32, 8) = 256 params
        assert info["n_params"] == 2 * 32 * 8


# ---------------------------------------------------------------------------
# make_fora_optimizer_groups
# ---------------------------------------------------------------------------

class TestOptimizerGroups:
    def test_partition_non_overlap(self):
        """No parameter should appear in both AdamW and CayleyAdam groups."""
        from fora_peft import CayleyAdam, FoRAConfig, make_fora_optimizer_groups
        from fora_peft.layer import stiefelize_lora

        model, _ = _make_stub_peft_model(n_layers=4, r=8, hidden=32)
        stiefelize_lora(model, layer_indices=[0, 2])

        cfg = FoRAConfig()
        adamw, cayley = make_fora_optimizer_groups(model, cfg, lr_adamw=2e-4)

        adamw_ids = {id(p) for g in adamw.param_groups for p in g["params"]}
        cayley_ids = {id(p) for g in cayley.param_groups for p in g["params"]}
        overlap = adamw_ids & cayley_ids
        assert len(overlap) == 0, f"Parameter overlap between optimizers: {overlap}"

    def test_cayley_has_stiefel_params(self):
        from fora_peft import CayleyAdam, FoRAConfig, make_fora_optimizer_groups
        from fora_peft.layer import stiefelize_lora

        model, _ = _make_stub_peft_model(n_layers=4, r=8, hidden=32)
        stiefelize_lora(model, layer_indices=[0])

        cfg = FoRAConfig()
        _, cayley = make_fora_optimizer_groups(model, cfg, lr_adamw=2e-4)
        total = sum(len(g["params"]) for g in cayley.param_groups)
        assert total > 0

    def test_adamw_has_euclidean_params(self):
        from fora_peft import FoRAConfig, make_fora_optimizer_groups
        from fora_peft.layer import stiefelize_lora

        model, _ = _make_stub_peft_model(n_layers=4, r=8, hidden=32)
        stiefelize_lora(model, layer_indices=[0])

        cfg = FoRAConfig()
        adamw, _ = make_fora_optimizer_groups(model, cfg, lr_adamw=2e-4)
        total = sum(len(g["params"]) for g in adamw.param_groups)
        assert total > 0
