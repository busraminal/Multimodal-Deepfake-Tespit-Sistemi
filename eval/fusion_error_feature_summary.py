"""Hata CSV + feature_cache ile FN/FP/TP/TN grubuna gore modal skor ortalamalari."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
_eval_dir = ROOT / "eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import fusion_io  # noqa: E402

FEATURE_COLS = ["Sv", "Sl", "Sb", "Sh", "Sa", "Sf_pipeline"]


def _kind(label: int, pred_fake: int) -> str:
    if label == 1 and pred_fake == 0:
        return "FN_fake_as_real"
    if label == 0 and pred_fake == 1:
        return "FP_real_as_fake"
    if label == 1 and pred_fake == 1:
        return "TP"
    return "TN"


def main() -> None:
    p = argparse.ArgumentParser(description="FN/FP ozellik ortalamalari (cache ile birlestir).")
    p.add_argument("--errors-csv", default="results/fusion_errors/errors_val.csv")
    p.add_argument("--cache-csv", default="data/feature_cache.csv")
    p.add_argument("--out-json", default="", help="Opsiyonel ozet JSON")
    args = p.parse_args()

    err_path = Path(args.errors_csv)
    if not err_path.is_absolute():
        err_path = ROOT / err_path
    cache_path = Path(args.cache_csv)
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path

    cache = fusion_io.read_cache(cache_path)

    buckets: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)
    missing = 0

    with err_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            vp = row["video_path"].strip()
            label = int(row["label"])
            pred_fake = int(row["pred_fake"])
            k = _kind(label, pred_fake)
            counts[k] += 1
            if vp not in cache:
                missing += 1
                continue
            vec = np.array([cache[vp][c] for c in FEATURE_COLS], dtype=np.float64)
            buckets[k].append(vec)

    print(f"errors file: {err_path}")
    print(f"cache: {cache_path}")
    if missing:
        print(f"Uyari: cache'te olmayan satir: {missing}")

    summary: Dict[str, object] = {}
    for k in sorted(buckets.keys()):
        rows = buckets[k]
        if not rows:
            summary[k] = {"n_with_cache": 0}
            print(f"\n[{k}] n(rows)={counts.get(k, 0)} n(cache)={0}")
            continue
        mat = np.stack(rows, axis=0)
        means = mat.mean(axis=0).tolist()
        stds = mat.std(axis=0).tolist()
        summary[k] = {
            "n_rows_in_csv": counts.get(k, len(rows)),
            "n_with_cache": len(rows),
            "features": {FEATURE_COLS[i]: {"mean": means[i], "std": stds[i]} for i in range(len(FEATURE_COLS))},
        }
        print(f"\n[{k}] n(rows)={counts.get(k, len(rows))} n(cache)={len(rows)}")
        for i, name in enumerate(FEATURE_COLS):
            print(f"  {name:12s} mean={means[i]:.4f} std={stds[i]:.4f}")

    all_vecs = [v for lst in buckets.values() for v in lst]
    if all_vecs:
        mat = np.stack(all_vecs, axis=0)
        summary["_all_rows_with_cache_in_errors_file"] = {
            "n": len(all_vecs),
            "features": {
                FEATURE_COLS[i]: {"mean": float(mat[:, i].mean()), "std": float(mat[:, i].std())}
                for i in range(len(FEATURE_COLS))
            },
        }

    print("\n--- Yorum ---")
    print("FN_fake_as_real: gercekte fake ama model dusuk p_fake vermis (Sl/Sv dusuk mu?).")
    print("FP_real_as_fake: gercekte real ama model yuksek p_fake vermis.")

    if args.out_json:
        outp = Path(args.out_json)
        if not outp.is_absolute():
            outp = ROOT / outp
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {outp.resolve()}")


if __name__ == "__main__":
    main()
