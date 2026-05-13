"""
Asama 1b: Ayni cache ozellikleri uzerinde HistGradientBoosting (sklearn).
Lojistik basliga kiyasla hizli karsilastirma; iyiyse ayri JSON kaydedilir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
_eval_dir = ROOT / "eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import fusion_io  # noqa: E402


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    y_sorted = y[order]
    pos = float((y_sorted == 1).sum())
    neg = float((y_sorted == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5
    tpr = [0.0]
    fpr = [0.0]
    tp = fp = 0.0
    for yi in y_sorted:
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


def _metrics_at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(np.int32)
    y_int = y.astype(np.int32)
    tp = int(((pred == 1) & (y_int == 1)).sum())
    fp = int(((pred == 1) & (y_int == 0)).sum())
    tn = int(((pred == 0) & (y_int == 0)).sum())
    fn = int(((pred == 0) & (y_int == 1)).sum())
    acc = float((tp + tn) / max(len(y), 1))
    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    ba = float(0.5 * (tpr + tnr))
    if tp == 0:
        f1 = 0.0
    else:
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = float(2 * prec * rec / (prec + rec + 1e-9))
    return {"acc": acc, "balanced_acc": ba, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _best_threshold_balanced_acc(y: np.ndarray, p: np.ndarray) -> float:
    best_t = 0.5
    best_ba = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        ba = _metrics_at_threshold(y, p, float(t))["balanced_acc"]
        if ba > best_ba + 1e-12:
            best_ba = ba
            best_t = float(t)
        elif abs(ba - best_ba) <= 1e-12 and abs(float(t) - 0.5) < abs(best_t - 0.5):
            best_t = float(t)
    return best_t


def _prepare_split(
    x_raw: np.ndarray,
    expansion: str,
    mu: np.ndarray,
    sigma: np.ndarray,
    use_std: bool,
) -> np.ndarray:
    x_e = fusion_io.expand_std(x_raw, expansion, mu=None, sigma=None, use_std=False)
    if use_std:
        return fusion_io.expand_std(x_e, "none", mu=mu, sigma=sigma, use_std=True)
    return x_e


def main() -> None:
    parser = argparse.ArgumentParser(description="HistGradientBoosting fusion (cache tabanli).")
    parser.add_argument("--metadata-csv", default="data/avlips_metadata.csv")
    parser.add_argument("--cache-csv", default="data/feature_cache.csv")
    parser.add_argument("--from-model", default="models/fusion_model.json", help="feature_names + expansion + standardize")
    parser.add_argument("--features", default=None, help="Virgulle; bos ise from-model")
    parser.add_argument("--expansion", default=None, choices=["none", "poly2"], help="Yoksa model JSON")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-json", default="", help="Ornek: models/fusion_histgb.json")
    args = parser.parse_args()

    from_path = Path(args.from_model)
    if not from_path.is_absolute():
        from_path = ROOT / from_path
    payload = json.loads(from_path.read_text(encoding="utf-8"))

    if args.features:
        feature_names: List[str] = [x.strip() for x in args.features.split(",") if x.strip()]
    else:
        feature_names = list(payload["feature_names"])

    expansion = (
        str(payload.get("feature_expansion") or "none").strip().lower()
        if args.expansion is None
        else args.expansion
    )
    use_std = bool(payload.get("standardize", True))

    metadata = fusion_io.read_metadata(Path(args.metadata_csv))
    cache = fusion_io.read_cache(Path(args.cache_csv))

    train_rows = [r for r in metadata if r["split"] == "train"]
    val_rows = [r for r in metadata if r["split"] == "val"]
    test_rows = [r for r in metadata if r["split"] == "test"]

    x_tr, y_tr, _ = fusion_io.xy_aligned(train_rows, cache, feature_names)
    x_va, y_va, _ = fusion_io.xy_aligned(val_rows, cache, feature_names)
    x_te, y_te, _ = fusion_io.xy_aligned(test_rows, cache, feature_names)

    x_tr_e = fusion_io.expand_std(x_tr, expansion, mu=None, sigma=None, use_std=False)
    mu, sigma = fusion_io.train_mu_sigma(x_tr_e)
    x_tr_fit = fusion_io.expand_std(x_tr_e, "none", mu=mu, sigma=sigma, use_std=use_std)

    x_va_fit = _prepare_split(x_va, expansion, mu, sigma, use_std)
    x_te_fit = _prepare_split(x_te, expansion, mu, sigma, use_std)

    clf = HistGradientBoostingClassifier(
        max_depth=args.max_depth,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        class_weight="balanced",
        early_stopping=False,
    )
    clf.fit(x_tr_fit, y_tr)

    p_va = clf.predict_proba(x_va_fit)[:, 1]
    thr = _best_threshold_balanced_acc(y_va, p_va)
    p_te = clf.predict_proba(x_te_fit)[:, 1]

    val_m = _metrics_at_threshold(y_va, p_va, thr)
    val_m["auc"] = _roc_auc(y_va, p_va)
    test_m = _metrics_at_threshold(y_te, p_te, thr)
    test_m["auc"] = _roc_auc(y_te, p_te)

    out = {
        "classifier": "HistGradientBoostingClassifier",
        "feature_names": feature_names,
        "feature_expansion": expansion,
        "standardize": use_std,
        "threshold_val_tuned": thr,
        "train_config": {
            "max_depth": args.max_depth,
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "random_state": args.random_state,
            "class_weight": "balanced",
        },
        "metrics": {"val": val_m, "test": test_m},
    }

    print(json.dumps({"threshold": thr, "val": val_m, "test": test_m}, ensure_ascii=False, indent=2))

    if args.out_json:
        outp = Path(args.out_json)
        if not outp.is_absolute():
            outp = ROOT / outp
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {outp.resolve()}")


if __name__ == "__main__":
    main()
