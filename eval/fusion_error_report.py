"""Mevcut fusion_model ile val/test hatalarini CSV'ye yaz (zor ornek analizi)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
_eval_dir = ROOT / "eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import fusion_io  # noqa: E402


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _balanced_acc(y: np.ndarray, pred: np.ndarray) -> float:
    y_int = y.astype(np.int32)
    p_int = pred.astype(np.int32)
    tp = int(((p_int == 1) & (y_int == 1)).sum())
    fn = int(((p_int == 0) & (y_int == 1)).sum())
    tn = int(((p_int == 0) & (y_int == 0)).sum())
    fp = int(((p_int == 1) & (y_int == 0)).sum())
    return float(0.5 * (tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9)))


def main() -> None:
    p = argparse.ArgumentParser(description="Fusion hata raporu (CSV).")
    p.add_argument("--metadata-csv", default="data/avlips_metadata.csv")
    p.add_argument("--cache-csv", default="data/feature_cache.csv")
    p.add_argument("--model-json", default="models/fusion_model.json")
    p.add_argument("--out-dir", default="results/fusion_errors")
    p.add_argument("--splits", default="val,test")
    args = p.parse_args()

    metadata = fusion_io.read_metadata(Path(args.metadata_csv))
    cache = fusion_io.read_cache(Path(args.cache_csv))
    model = json.loads(Path(args.model_json).read_text(encoding="utf-8"))
    feature_names = list(model["feature_names"])
    expansion = str(model.get("feature_expansion") or "none").strip().lower()
    w = np.array(model["weights"], dtype=np.float64)
    b = float(model["bias"])
    thr = float(model.get("threshold", 0.5))
    use_std = bool(model.get("standardize", False))
    mu = np.array(model.get("scaler_mean") or [], dtype=np.float64)
    sigma = np.array(model.get("scaler_std") or [], dtype=np.float64)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        rows = [r for r in metadata if r["split"] == split]
        x, y, paths = fusion_io.xy_aligned(rows, cache, feature_names)
        x = fusion_io.expand_std(x, expansion, mu=mu, sigma=sigma, use_std=use_std)
        prob = _sigmoid(x @ w + b)
        pred = (prob >= thr).astype(np.int32)
        y_int = y.astype(np.int32)
        correct = (pred == y_int).astype(np.int32)

        out_csv = out_dir / f"errors_{split}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(
                [
                    "video_path",
                    "label",
                    "label_name",
                    "p_fake",
                    "pred_fake",
                    "correct",
                    "margin_to_threshold",
                ]
            )
            for i in range(len(paths)):
                wtr.writerow(
                    [
                        paths[i],
                        int(y_int[i]),
                        "fake" if y_int[i] == 1 else "real",
                        f"{float(prob[i]):.8f}",
                        int(pred[i]),
                        int(correct[i]),
                        f"{float(prob[i] - thr):.8f}",
                    ]
                )

        ba = _balanced_acc(y, pred)
        n_wrong = int((correct == 0).sum())
        print(f"[{split}] rows={len(paths)} wrong={n_wrong} balanced_acc={ba:.6f} -> {out_csv.resolve()}")

    print("\nFN (fake->real): label=1, pred_fake=0 | FP (real->fake): label=0, pred_fake=1")
    print("CSV'yi import edip suz veya Excel'de filtrele.")


if __name__ == "__main__":
    main()
