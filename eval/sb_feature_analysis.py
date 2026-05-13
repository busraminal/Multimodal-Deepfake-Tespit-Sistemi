"""Sb ve diger modal skorlarin etiketle iliskisi (val/test uzerinden ozet)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.fusion_features import ALL_FUSION_FEATURES, cache_column, cache_load_keys

FEATURE_ORDER = list(ALL_FUSION_FEATURES)


def _read_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_cache(cache_csv: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    keys = cache_load_keys()
    with cache_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["video_path"]] = {k: float(row[k]) for k in keys}
    return out


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    mx = float(x.mean())
    my = float(y.mean())
    dx = x - mx
    dy = y - my
    denom = math.sqrt(float((dx * dx).sum()) * float((dy * dy).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((dx * dy).sum() / denom)


def _cohens_d(g0: np.ndarray, g1: np.ndarray) -> float:
    n0, n1 = len(g0), len(g1)
    if n0 < 2 or n1 < 2:
        return float("nan")
    v0 = float(g0.var(ddof=1))
    v1 = float(g1.var(ddof=1))
    sp = math.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2 + 1e-9))
    if sp < 1e-12:
        return float("nan")
    return float((float(g1.mean()) - float(g0.mean())) / sp)


def _split_xy(
    rows: List[Dict[str, str]],
    cache: Dict[str, Dict[str, float]],
    split: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    xs: List[np.ndarray] = []
    ys: List[float] = []
    for row in rows:
        if row["split"] != split:
            continue
        vp = row["video_path"]
        if vp not in cache:
            continue
        ys.append(float(row["label"]))
        xs.append(np.array([cache[vp][cache_column(name)] for name in FEATURE_ORDER], dtype=np.float64))
    if not xs:
        raise RuntimeError(f"No rows for split={split}")
    xmat = np.stack(xs, axis=0)
    y = np.array(ys, dtype=np.float64)
    cols = {FEATURE_ORDER[i]: xmat[:, i] for i in range(len(FEATURE_ORDER))}
    return y, cols


def main() -> None:
    p = argparse.ArgumentParser(description="Sb ve modal skorlarin etiketle korelasyonu / sinif ortalamalari.")
    p.add_argument("--metadata-csv", default="data/avlips_metadata.csv")
    p.add_argument("--cache-csv", default="data/feature_cache.csv")
    p.add_argument("--model-json", default="models/fusion_model.json", help="Secili ozellik listesi icin bilgi")
    p.add_argument("--splits", default="val,test", help="Virgulle ayrilmis split adlari")
    args = p.parse_args()

    metadata = _read_metadata(Path(args.metadata_csv))
    cache = _read_cache(Path(args.cache_csv))
    model_path = Path(args.model_json)
    selected: List[str] = []
    if model_path.exists():
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        selected = list(payload.get("feature_names") or [])

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print("=== Ozellik - etiket ozeti (Pearson r; etiket 0/1) ===\n")
    for sp in splits:
        y, cols = _split_xy(metadata, cache, sp)
        real = y < 0.5
        fake = ~real
        print(f"[{sp}] n={len(y)}  (real={(real).sum()} fake={fake.sum()})")
        for name in FEATURE_ORDER:
            xv = cols[name]
            r = _pearson_r(xv, y)
            m0 = float(xv[real].mean()) if real.any() else float("nan")
            m1 = float(xv[fake].mean()) if fake.any() else float("nan")
            d = _cohens_d(xv[real], xv[fake])
            tag = " *" if name in selected else ""
            print(f"  {name:3s}  r={r:+.4f}  mean_real={m0:.4f} mean_fake={m1:.4f}  Cohen_d={d:+.4f}{tag}")
        print()

    print("(* = su anki fusion_model.json icinde kullanilan ozellikler.)")
    print("\nNot: Sb secilmemis olsa da siniflar arasi ortalama fark (Cohen d) ve r,")
    print("    auto_select icindeki diger kombinasyonlarla birlikte degerlendirilir.")


if __name__ == "__main__":
    main()
