import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.fusion_expand import expand_features
from src.fusion_features import cache_column, cache_load_keys


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    pred = (p >= threshold).astype(np.float32)
    return float(np.mean((pred == y).astype(np.float32)))


def _balanced_acc(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    pred = (p >= threshold).astype(np.int32)
    y_int = y.astype(np.int32)
    tp = int(((pred == 1) & (y_int == 1)).sum())
    fn = int(((pred == 0) & (y_int == 1)).sum())
    tn = int(((pred == 0) & (y_int == 0)).sum())
    fp = int(((pred == 1) & (y_int == 0)).sum())
    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    return float(0.5 * (tpr + tnr))


def _f1(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    pred = (p >= threshold).astype(np.int32)
    y_int = y.astype(np.int32)
    tp = int(((pred == 1) & (y_int == 1)).sum())
    fp = int(((pred == 1) & (y_int == 0)).sum())
    fn = int(((pred == 0) & (y_int == 1)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return float(2 * precision * recall / (precision + recall + 1e-9))


def _confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, int]:
    pred = (p >= threshold).astype(np.int32)
    y_int = y.astype(np.int32)
    tp = int(((pred == 1) & (y_int == 1)).sum())
    fn = int(((pred == 0) & (y_int == 1)).sum())
    tn = int(((pred == 0) & (y_int == 0)).sum())
    fp = int(((pred == 1) & (y_int == 0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    y_sorted = y[order]
    pos = float((y_sorted == 1).sum())
    neg = float((y_sorted == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5
    tpr = [0.0]
    fpr = [0.0]
    tp = 0.0
    fp = 0.0
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


def _xy(
    rows: List[Dict[str, str]],
    cache: Dict[str, Dict[str, float]],
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in rows:
        vp = row["video_path"]
        if vp not in cache:
            continue
        xs.append([cache[vp][cache_column(name)] for name in feature_names])
        ys.append(float(row["label"]))
    if not xs:
        raise RuntimeError("No matched samples between metadata and cache.")
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _predict_probs(
    rows: List[Dict[str, str]],
    cache: Dict[str, Dict[str, float]],
    feature_names: List[str],
    expansion: str,
    w: np.ndarray,
    b: float,
    use_std: bool,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x, y = _xy(rows, cache, feature_names)
    x = expand_features(x, expansion)
    if use_std and len(mu) == x.shape[1] and len(sigma) == x.shape[1]:
        sigma_safe = np.where(sigma < 1e-9, 1.0, sigma)
        x = (x - mu) / sigma_safe
    p = _sigmoid(x @ w + b)
    return y, p


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained fusion model on val/test split.")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--cache-csv", default="data/feature_cache.csv")
    parser.add_argument("--model-json", default="models/fusion_model.json")
    parser.add_argument("--confusion", action="store_true", help="TP/FP/TN/FN yazdir (model esigi ile).")
    parser.add_argument(
        "--threshold-sweep",
        nargs=3,
        type=float,
        metavar=("START", "END", "STEP"),
        default=None,
        help="Ornek: 0.40 0.65 0.02 — val ve test icin esik tablosu.",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Opsiyonel: metrikler, confusion ve sweep bu dosyaya yazilir.",
    )
    args = parser.parse_args()

    metadata = _read_metadata(Path(args.metadata_csv))
    cache = _read_cache(Path(args.cache_csv))
    model = json.loads(Path(args.model_json).read_text(encoding="utf-8"))

    feature_names = model["feature_names"]
    expansion = str(model.get("feature_expansion") or "none").strip().lower()
    w = np.array(model["weights"], dtype=np.float64)
    b = float(model["bias"])
    t = float(model.get("threshold", 0.5))
    use_std = bool(model.get("standardize", False))
    mu = np.array(model.get("scaler_mean") or [], dtype=np.float64)
    sigma = np.array(model.get("scaler_std") or [], dtype=np.float64)

    report: Dict[str, object] = {
        "model_json": str(Path(args.model_json).resolve()),
        "threshold_model": t,
        "splits": {},
    }
    sweep_spec = args.threshold_sweep
    thresholds: List[float] = []
    if sweep_spec is not None:
        lo, hi, step = float(sweep_spec[0]), float(sweep_spec[1]), float(sweep_spec[2])
        if step <= 0:
            raise SystemExit("threshold-sweep STEP must be > 0")
        x = lo
        while x <= hi + 1e-9:
            thresholds.append(round(x, 6))
            x += step

    for split in ("val", "test"):
        rows = [r for r in metadata if r["split"] == split]
        y, p = _predict_probs(rows, cache, feature_names, expansion, w, b, use_std, mu, sigma)
        metrics = {
            "count": int(len(y)),
            "acc": _accuracy(y, p, threshold=t),
            "balanced_acc": _balanced_acc(y, p, threshold=t),
            "f1": _f1(y, p, threshold=t),
            "auc": _roc_auc(y, p),
        }
        print(f"[{split}] {json.dumps(metrics, ensure_ascii=False)}")
        split_payload: Dict[str, object] = {"metrics": metrics}
        if args.confusion:
            cm = _confusion(y, p, threshold=t)
            split_payload["confusion"] = cm
            print(f"[{split}] confusion @threshold={t} {json.dumps(cm, ensure_ascii=False)}")
        if thresholds:
            rows_tbl = []
            for th in thresholds:
                rows_tbl.append(
                    {
                        "threshold": th,
                        "acc": _accuracy(y, p, threshold=th),
                        "balanced_acc": _balanced_acc(y, p, threshold=th),
                        "f1": _f1(y, p, threshold=th),
                    }
                )
            split_payload["threshold_sweep"] = rows_tbl
            print(f"[{split}] threshold_sweep ({thresholds[0]}..{thresholds[-1]}, step {sweep_spec[2]}):")
            for row in rows_tbl:
                print(
                    "    "
                    + json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                )
        report["splits"][split] = split_payload

    if args.report_json:
        out_path = Path(args.report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved report: {out_path.resolve()}")


if __name__ == "__main__":
    main()

