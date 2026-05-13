"""Expand base multimodal score vectors for learned fusion (logistic / linear head)."""

from __future__ import annotations

import numpy as np


def expansion_dim(base_dim: int, expansion: str) -> int:
    if expansion in ("", "none"):
        return int(base_dim)
    if expansion == "poly2":
        return int(base_dim + base_dim * (base_dim + 1) // 2)
    raise ValueError(f"Unknown feature expansion: {expansion!r}")


def expand_features(x: np.ndarray, expansion: str) -> np.ndarray:
    """
    x: (n_samples, n_base) or (n_base,) — single row ok as 1d length n_base.
    Order: [x_0..x_{n-1}, x_0*x_0, x_0*x_1, ..., x_{n-1}*x_{n-1}]
    """
    if expansion in ("", "none"):
        return np.asarray(x, dtype=np.float64)
    if expansion != "poly2":
        raise ValueError(f"Unknown feature expansion: {expansion!r}")
    v = np.asarray(x, dtype=np.float64)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    m, n = v.shape
    parts: list[np.ndarray] = [v]
    for i in range(n):
        for j in range(i, n):
            parts.append((v[:, i] * v[:, j]).reshape(m, 1))
    return np.hstack(parts)


def expand_single(x1d: np.ndarray, expansion: str) -> np.ndarray:
    """Return 1d expanded vector of shape (expansion_dim(n),)."""
    out = expand_features(x1d, expansion)
    return np.asarray(out, dtype=np.float64).reshape(-1)
