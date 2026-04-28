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

from src.analyze_video import analyze


FEATURES = ["Sv", "Sl", "Sb", "Sh", "Sa"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _bce_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    pred = (p >= threshold).astype(np.float32)
    return float(np.mean((pred == y).astype(np.float32)))


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


def _best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.1, 0.9, 81):
        f1_val = _f1(y, p, threshold=float(t))
        if f1_val > best_f1:
            best_f1 = f1_val
            best_t = float(t)
    return best_t


def _fit_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 0.05,
    epochs: int = 500,
    l2: float = 1e-3,
) -> Tuple[np.ndarray, float]:
    n, d = x_train.shape
    w = np.zeros((d,), dtype=np.float64)
    b = 0.0
    best_w = w.copy()
    best_b = b
    best_val = float("inf")

    for _ in range(epochs):
        z = x_train @ w + b
        p = _sigmoid(z)
        grad_w = (x_train.T @ (p - y_train)) / n + l2 * w
        grad_b = float(np.mean(p - y_train))
        w -= lr * grad_w
        b -= lr * grad_b

        p_val = _sigmoid(x_val @ w + b)
        val_loss = _bce_loss(y_val, p_val)
        if val_loss < best_val:
            best_val = val_loss
            best_w = w.copy()
            best_b = b

    return best_w, best_b


def _read_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _extract_features(video_path: str) -> Dict[str, float]:
    result = analyze(video_path)
    return result["scores"]


def _build_feature_cache(rows: List[Dict[str, str]], cache_csv: Path) -> None:
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if cache_csv.exists() and cache_csv.stat().st_size > 0:
        with cache_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("video_path"):
                    existing.add(row["video_path"])

    n = len(rows)
    with cache_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not existing:
            writer.writerow(["video_path", "Sv", "Sl", "Sb", "Sh", "Sa", "Sf_pipeline"])
        for idx, row in enumerate(rows, start=1):
            video_path = row["video_path"]
            if video_path in existing:
                continue
            print(f"[{idx}/{n}] ozellik: {video_path}", flush=True)
            try:
                scores = _extract_features(video_path)
                writer.writerow(
                    [
                        video_path,
                        scores.get("Sv", 0.0),
                        scores.get("Sl", 0.0),
                        scores.get("Sb", 0.0),
                        scores.get("Sh", 0.0),
                        scores.get("Sa", 0.0),
                        scores.get("Sf", 0.0),
                    ]
                )
                f.flush()
            except Exception as exc:
                print(f"[WARN] feature extraction failed: {video_path} | {exc}")


def _read_cache(cache_csv: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with cache_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["video_path"]] = {k: float(row[k]) for k in ["Sv", "Sl", "Sb", "Sh", "Sa", "Sf_pipeline"]}
    return out


def _xy(rows: List[Dict[str, str]], cache: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[List[float]] = []
    ys: List[float] = []
    for row in rows:
        video_path = row["video_path"]
        if video_path not in cache:
            continue
        xs.append([cache[video_path][f] for f in FEATURES])
        ys.append(float(row["label"]))
    if not xs:
        raise RuntimeError("No matched rows between metadata and feature cache.")
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train logistic fusion model from metadata.")
    parser.add_argument("--metadata-csv", required=True, help="CSV with columns: video_path,label,split")
    parser.add_argument("--cache-csv", default="data/feature_cache.csv", help="Feature cache path")
    parser.add_argument("--out-model", default="models/fusion_model.json", help="Output model json")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=0,
        help="If >0, limit number of samples per split for quick experiments.",
    )
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="Delete cache CSV before run (clean rebuild for selected rows).",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata_csv)
    cache_path = Path(args.cache_csv)
    out_model = Path(args.out_model)

    if args.reset_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Bos dosya: _build_feature_cache basligi yazar (cift baslik olmaz).
        cache_path.write_text("", encoding="utf-8")
        print(f"Cache sifirlandi: {cache_path}", flush=True)

    rows = _read_metadata(metadata_path)

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    if args.max_per_split > 0:
        train_rows = train_rows[: args.max_per_split]
        val_rows = val_rows[: args.max_per_split]
        test_rows = test_rows[: args.max_per_split]

    # Only extract features for rows used in training (not the full metadata CSV).
    cache_rows = train_rows + val_rows + test_rows
    _build_feature_cache(cache_rows, cache_path)
    cache = _read_cache(cache_path)

    x_train, y_train = _xy(train_rows, cache)
    x_val, y_val = _xy(val_rows, cache)
    x_test, y_test = _xy(test_rows, cache)

    w, b = _fit_logreg(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        lr=args.lr,
        epochs=args.epochs,
    )

    p_val = _sigmoid(x_val @ w + b)
    # Az ornekte esik arama asiri uc degerlere kayar; guvenli varsayilan kullan.
    if len(y_val) < 20:
        threshold = 0.5
    else:
        threshold = float(np.clip(_best_threshold(y_val, p_val), 0.2, 0.8))
    p_test = _sigmoid(x_test @ w + b)

    metrics = {
        "val": {
            "loss": _bce_loss(y_val, p_val),
            "acc": _accuracy(y_val, p_val, threshold),
            "f1": _f1(y_val, p_val, threshold),
            "auc": _roc_auc(y_val, p_val),
        },
        "test": {
            "loss": _bce_loss(y_test, p_test),
            "acc": _accuracy(y_test, p_test, threshold),
            "f1": _f1(y_test, p_test, threshold),
            "auc": _roc_auc(y_test, p_test),
        },
    }

    payload = {
        "feature_names": FEATURES,
        "weights": w.tolist(),
        "bias": float(b),
        "threshold": float(threshold),
        "metrics": metrics,
    }
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved fusion model: {out_model}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

