"""Ortak: metadata/cache okuma ve füzyon için tasarım matrisi."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.fusion_expand import expand_features
from src.fusion_features import cache_column, cache_load_keys


def read_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_cache(cache_csv: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    keys = cache_load_keys()
    with cache_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["video_path"]] = {k: float(row[k]) for k in keys}
    return out


def xy_aligned(
    rows: List[Dict[str, str]],
    cache: Dict[str, Dict[str, float]],
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    xs: List[List[float]] = []
    ys: List[float] = []
    paths: List[str] = []
    for row in rows:
        vp = row["video_path"]
        if vp not in cache:
            continue
        xs.append([cache[vp][cache_column(name)] for name in feature_names])
        ys.append(float(row["label"]))
        paths.append(vp)
    if not xs:
        raise RuntimeError("No matched samples between metadata and cache.")
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64), paths


def expand_std(
    x: np.ndarray,
    expansion: str,
    *,
    mu: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    use_std: bool = False,
) -> np.ndarray:
    x = expand_features(x, expansion)
    if use_std and mu is not None and sigma is not None and len(mu) == x.shape[1]:
        sigma_safe = np.where(sigma < 1e-9, 1.0, sigma)
        x = (x - mu) / sigma_safe
    return x


def train_mu_sigma(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    return mu, sigma
