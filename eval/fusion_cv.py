"""5-fold stratified cross-validation: lojistik vs HistGB.

Tum metadata (train+val+test) birlestirilir, 5 katmana ayrilir. Her kat icin
fold-train uzerinde standardize + model egitilir; fold-val uzerinde balanced_acc /
F1 / AUC olculur. Cikti: results/v2/fusion_cv_<model>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
_eval_dir = ROOT / "eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import fusion_io  # noqa: E402


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


def _metrics_at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p >= thr).astype(np.int32)
    yi = y.astype(np.int32)
    tp = int(((pred == 1) & (yi == 1)).sum())
    fp = int(((pred == 1) & (yi == 0)).sum())
    tn = int(((pred == 0) & (yi == 0)).sum())
    fn = int(((pred == 0) & (yi == 1)).sum())
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


def _best_thr_ba(y: np.ndarray, p: np.ndarray) -> float:
    best_t = 0.5
    best_ba = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        ba = _metrics_at_threshold(y, p, float(t))["balanced_acc"]
        if ba > best_ba + 1e-12:
            best_ba = ba
            best_t = float(t)
    return best_t


def _train_logreg(x_tr: np.ndarray, y_tr: np.ndarray, l2: float, max_iter: int) -> LogisticRegression:
    n_pos = float((y_tr == 1).sum())
    n_neg = float((y_tr == 0).sum())
    pos_w = n_neg / max(n_pos, 1.0)
    sample_weight = np.where(y_tr == 1, pos_w, 1.0)
    clf = LogisticRegression(C=1.0 / max(l2, 1e-6), max_iter=max_iter, solver="lbfgs")
    clf.fit(x_tr, y_tr, sample_weight=sample_weight)
    return clf


def _train_histgb(x_tr: np.ndarray, y_tr: np.ndarray, max_depth: int, max_iter: int, lr: float) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_depth=max_depth,
        max_iter=max_iter,
        learning_rate=lr,
        random_state=42,
        class_weight="balanced",
        early_stopping=False,
    )
    clf.fit(x_tr, y_tr)
    return clf


def main() -> None:
    p = argparse.ArgumentParser(description="5-fold CV: lojistik & HistGB on cached fusion features.")
    p.add_argument("--metadata-csv", default="data/avlips_metadata.csv")
    p.add_argument("--cache-csv", default="data/feature_cache.csv")
    p.add_argument("--features", default="Sv,Sl,Sb,Sh,Sa,Sf")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", default="results/v2/fusion_cv.json")
    p.add_argument("--no-standardize", action="store_true")
    args = p.parse_args()

    feature_names: List[str] = [s.strip() for s in args.features.split(",") if s.strip()]
    standardize = not args.no_standardize

    metadata_path = Path(args.metadata_csv)
    if not metadata_path.is_absolute():
        metadata_path = ROOT / metadata_path
    cache_path = Path(args.cache_csv)
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path

    metadata = fusion_io.read_metadata(metadata_path)
    cache = fusion_io.read_cache(cache_path)
    x_all, y_all, _ = fusion_io.xy_aligned(metadata, cache, feature_names)

    print(f"CV: features={feature_names} n_samples={len(y_all)} pos={(y_all==1).sum()} neg={(y_all==0).sum()}", flush=True)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    results: Dict[str, List[Dict[str, float]]] = {"logreg": [], "histgb": []}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(x_all, y_all), start=1):
        x_tr_raw, y_tr = x_all[tr_idx], y_all[tr_idx]
        x_te_raw, y_te = x_all[te_idx], y_all[te_idx]

        if standardize:
            mu, sigma = fusion_io.train_mu_sigma(x_tr_raw)
            x_tr = (x_tr_raw - mu) / sigma
            x_te = (x_te_raw - mu) / sigma
        else:
            x_tr, x_te = x_tr_raw, x_te_raw

        lr_clf = _train_logreg(x_tr, y_tr, l2=0.01, max_iter=1000)
        p_te_lr = lr_clf.predict_proba(x_te)[:, 1]
        thr_lr = _best_thr_ba(y_te, p_te_lr)
        m_lr = _metrics_at_threshold(y_te, p_te_lr, thr_lr)
        m_lr["auc"] = _roc_auc(y_te, p_te_lr)
        m_lr["thr"] = thr_lr
        m_lr["fold"] = fold
        results["logreg"].append(m_lr)

        gb_clf = _train_histgb(x_tr, y_tr, max_depth=6, max_iter=250, lr=0.06)
        p_te_gb = gb_clf.predict_proba(x_te)[:, 1]
        thr_gb = _best_thr_ba(y_te, p_te_gb)
        m_gb = _metrics_at_threshold(y_te, p_te_gb, thr_gb)
        m_gb["auc"] = _roc_auc(y_te, p_te_gb)
        m_gb["thr"] = thr_gb
        m_gb["fold"] = fold
        results["histgb"].append(m_gb)

        print(
            f"fold {fold}: "
            f"LR ba={m_lr['balanced_acc']:.4f} auc={m_lr['auc']:.4f} | "
            f"GB ba={m_gb['balanced_acc']:.4f} auc={m_gb['auc']:.4f}",
            flush=True,
        )

    def _agg(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        keys = ["balanced_acc", "acc", "f1", "auc"]
        out: Dict[str, Dict[str, float]] = {}
        for k in keys:
            vals = np.array([r[k] for r in rows], dtype=np.float64)
            out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                      "min": float(vals.min()), "max": float(vals.max())}
        return out

    summary = {
        "features": feature_names,
        "standardize": standardize,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "n_samples": int(len(y_all)),
        "n_pos": int((y_all == 1).sum()),
        "n_neg": int((y_all == 0).sum()),
        "logreg": {"folds": results["logreg"], "agg": _agg(results["logreg"])},
        "histgb": {"folds": results["histgb"], "agg": _agg(results["histgb"])},
    }

    print("\n=== Aggregate (mean ± std) ===")
    for model_name in ("logreg", "histgb"):
        a = summary[model_name]["agg"]
        print(
            f"{model_name}: ba={a['balanced_acc']['mean']:.4f}±{a['balanced_acc']['std']:.4f}  "
            f"auc={a['auc']['mean']:.4f}±{a['auc']['std']:.4f}  "
            f"f1={a['f1']['mean']:.4f}±{a['f1']['std']:.4f}"
        )

    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
