"""Olasilik kalibrasyonu (Platt / isotonic) ve ECE raporu.

Train uzerinde HistGB egitilir; val uzerinde calibrator (sigmoid/isotonic)
fit edilir; test uzerinde ham vs kalibre olasilik, ECE ve Brier skoru
hesaplanir. Sonuc: results/v2/fusion_calibration.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
_eval_dir = ROOT / "eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import fusion_io  # noqa: E402


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Expected Calibration Error (equal-width bins)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    bin_stats: List[Dict[str, float]] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            bin_stats.append({"bin": b, "n": 0, "p_mean": 0.0, "y_mean": 0.0, "gap": 0.0})
            continue
        p_mean = float(p[mask].mean())
        y_mean = float(y[mask].mean())
        gap = abs(p_mean - y_mean)
        ece += (n / len(y)) * gap
        bin_stats.append({"bin": b, "n": n, "p_mean": p_mean, "y_mean": y_mean, "gap": gap})
    return {"ece": float(ece), "bins": bin_stats}


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    ys = y[order]
    pos = float((ys == 1).sum())
    neg = float((ys == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5
    tpr = [0.0]
    fpr = [0.0]
    tp = fp = 0.0
    for yi in ys:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / pos)
        fpr.append(fp / neg)
    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) * 0.5
    return float(auc)


def _split_metric(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p >= thr).astype(np.int32)
    yi = y.astype(np.int32)
    tp = int(((pred == 1) & (yi == 1)).sum())
    fp = int(((pred == 1) & (yi == 0)).sum())
    tn = int(((pred == 0) & (yi == 0)).sum())
    fn = int(((pred == 0) & (yi == 1)).sum())
    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    return {"balanced_acc": float(0.5 * (tpr + tnr)), "thr": float(thr)}


def main() -> None:
    p = argparse.ArgumentParser(description="HistGB + Platt/isotonic kalibrasyon raporu.")
    p.add_argument("--metadata-csv", default="data/avlips_metadata.csv")
    p.add_argument("--cache-csv", default="data/feature_cache.csv")
    p.add_argument("--features", default="Sv,Sl,Sb,Sh,Sa,Sf")
    p.add_argument("--out-json", default="results/v2/fusion_calibration.json")
    args = p.parse_args()

    feature_names = [s.strip() for s in args.features.split(",") if s.strip()]

    metadata_path = Path(args.metadata_csv)
    if not metadata_path.is_absolute():
        metadata_path = ROOT / metadata_path
    cache_path = Path(args.cache_csv)
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path

    metadata = fusion_io.read_metadata(metadata_path)
    cache = fusion_io.read_cache(cache_path)

    train_rows = [r for r in metadata if r["split"] == "train"]
    val_rows = [r for r in metadata if r["split"] == "val"]
    test_rows = [r for r in metadata if r["split"] == "test"]

    x_tr, y_tr, _ = fusion_io.xy_aligned(train_rows, cache, feature_names)
    x_va, y_va, _ = fusion_io.xy_aligned(val_rows, cache, feature_names)
    x_te, y_te, _ = fusion_io.xy_aligned(test_rows, cache, feature_names)

    mu, sigma = fusion_io.train_mu_sigma(x_tr)
    x_tr_s = (x_tr - mu) / sigma
    x_va_s = (x_va - mu) / sigma
    x_te_s = (x_te - mu) / sigma

    base = HistGradientBoostingClassifier(
        max_depth=6, max_iter=250, learning_rate=0.06, random_state=42,
        class_weight="balanced", early_stopping=False,
    )
    base.fit(x_tr_s, y_tr)

    p_te_raw = base.predict_proba(x_te_s)[:, 1]
    ece_raw = _ece(y_te, p_te_raw)
    brier_raw = _brier(y_te, p_te_raw)
    auc_raw = _roc_auc(y_te, p_te_raw)

    out: Dict[str, object] = {
        "features": feature_names,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "raw_histgb": {
            "test_auc": auc_raw,
            "test_brier": brier_raw,
            "test_ece": ece_raw["ece"],
            "test_ece_bins": ece_raw["bins"],
        },
        "calibrators": {},
    }

    for method in ("sigmoid", "isotonic"):
        calibrated = CalibratedClassifierCV(estimator=base, method=method, cv="prefit")
        calibrated.fit(x_va_s, y_va)
        p_te_cal = calibrated.predict_proba(x_te_s)[:, 1]
        ece_cal = _ece(y_te, p_te_cal)
        brier_cal = _brier(y_te, p_te_cal)
        auc_cal = _roc_auc(y_te, p_te_cal)
        m_cal = _split_metric(y_te, p_te_cal, 0.5)
        out["calibrators"][method] = {
            "test_auc": auc_cal,
            "test_brier": brier_cal,
            "test_ece": ece_cal["ece"],
            "test_balanced_acc_thr0.5": m_cal["balanced_acc"],
            "test_ece_bins": ece_cal["bins"],
        }
        print(
            f"{method:8s}: AUC={auc_cal:.4f}  Brier={brier_cal:.4f}  ECE={ece_cal['ece']:.4f}  "
            f"BA@0.5={m_cal['balanced_acc']:.4f}"
        )

    print(f"raw     : AUC={auc_raw:.4f}  Brier={brier_raw:.4f}  ECE={ece_raw['ece']:.4f}")

    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
