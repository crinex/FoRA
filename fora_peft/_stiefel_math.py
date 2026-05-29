"""Stiefel manifold math utilities.

Convention: parameter is (p, n) with p <= n, row-orthonormal (A A^T = I_p).
Tall (out, r) matrices are auto-transposed by callers.
"""

from __future__ import annotations

import torch

EPS = 1e-8


def matrix_norm_one(W: torch.Tensor) -> torch.Tensor:
    """Induced 1-norm = max column sum of |W|."""
    return W.abs().sum(dim=0).max()


def qr_retraction(tan_vec: torch.Tensor) -> torch.Tensor:
    """Retract a (p, n) matrix back to Stiefel via QR. Returns (p, n)."""
    Y = tan_vec.t()                                   # (n, p)
    q, r = torch.linalg.qr(Y, mode="reduced")         # q: (n, p)
    d = torch.diagonal(r, 0)
    ph = d.sign()
    ph = torch.where(ph == 0, torch.ones_like(ph), ph)
    q = q * ph.unsqueeze(0)
    return q.t().contiguous()                         # (p, n)


def cayley_loop(
    X: torch.Tensor,
    W: torch.Tensor,
    tan_vec: torch.Tensor,
    t: float | torch.Tensor,
    n_iter: int = 5,
) -> torch.Tensor:
    """Iterative Cayley retraction.

    Solves Y = X + t * W * (X + Y) / 2 by fixed-point iteration.
    Equivalent to Y = (I - tW/2)^{-1} (I + tW/2) X without explicit inverse.

    Args:
        X: (n, p) starting point on Stiefel (column-orthonormal view).
        W: (n, n) skew-symmetric direction.
        tan_vec: (n, p) tangent direction (= W X is the standard choice).
        t: step size (scalar or 0-d tensor).
        n_iter: number of fixed-point iterations.

    Returns:
        Y: (p, n) updated point on Stiefel (transposed back to row-major).
    """
    Y = X + t * tan_vec
    for _ in range(n_iter):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))
    return Y.t().contiguous()


def orth_drift(B: torch.Tensor) -> torch.Tensor:
    """Frobenius orthogonality drift on the shorter dimension.

    For tall (d, r) with d >= r:  || B^T B - I_r ||_F  (column-orthonormal)
    For wide (r, d) with r <= d:  || B B^T - I_r ||_F  (row-orthonormal)
    """
    d0, d1 = B.shape
    if d0 >= d1:
        r = d1
        I = torch.eye(r, device=B.device, dtype=B.dtype)
        return (B.t() @ B - I).norm()
    else:
        r = d0
        I = torch.eye(r, device=B.device, dtype=B.dtype)
        return (B @ B.t() - I).norm()
