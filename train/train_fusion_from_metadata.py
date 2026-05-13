import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.analyze_video import analyze
from src.fusion_expand import expand_features
from src.fusion_features import ALL_FUSION_FEATURES, cache_column, cache_load_keys


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


def _precision_recall(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> Tuple[float, float]:
    pred = (p >= threshold).astype(np.int32)
    y_int = y.astype(np.int32)
    tp = int(((pred == 1) & (y_int == 1)).sum())
    fp = int(((pred == 1) & (y_int == 0)).sum())
    fn = int(((pred == 0) & (y_int == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return float(precision), float(recall)


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


def _best_threshold(y: np.ndarray, p: np.ndarray, objective: str = "f1") -> float:
    best_t = 0.5
    best_score = -1.0
    for t in np.linspace(0.1, 0.9, 81):
        t_val = float(t)
        if objective == "balanced_acc":
            score = _balanced_acc(y, p, threshold=t_val)
        else:
            score = _f1(y, p, threshold=t_val)
        if score > best_score + 1e-12:
            best_score = score
            best_t = t_val
        elif abs(score - best_score) <= 1e-12:
            # Tie-break: keep threshold close to 0.5 to avoid degenerate all-positive/all-negative outputs.
            if abs(t_val - 0.5) < abs(best_t - 0.5):
                best_t = t_val
    return best_t


def _fit_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 0.05,
    epochs: int = 500,
    l2: float = 1e-3,
    pos_weight: float = 1.0,
    show_progress: bool = True,
) -> Tuple[np.ndarray, float]:
    n, d = x_train.shape
    w = np.zeros((d,), dtype=np.float64)
    b = 0.0
    best_w = w.copy()
    best_b = b
    best_val = float("inf")

    rng = range(epochs)
    bar = None
    if show_progress and tqdm is not None:
        bar = tqdm(rng, desc="Fusion egitimi (BCE)", unit="epoch", dynamic_ncols=True, mininterval=0.5)

    iterator = bar if bar is not None else rng
    sample_w = np.where(y_train > 0.5, pos_weight, 1.0).astype(np.float64)
    sw_sum = float(np.sum(sample_w))
    for _ in iterator:
        z = x_train @ w + b
        p = _sigmoid(z)
        err = (p - y_train) * sample_w
        grad_w = (x_train.T @ err) / max(sw_sum, 1e-9) + l2 * w
        grad_b = float(np.sum(err) / max(sw_sum, 1e-9))
        w -= lr * grad_w
        b -= lr * grad_b

        p_val = _sigmoid(x_val @ w + b)
        val_loss = _bce_loss(y_val, p_val)
        if val_loss < best_val:
            best_val = val_loss
            best_w = w.copy()
            best_b = b
        if bar is not None:
            bar.set_postfix(val_bce=f"{val_loss:.4f}", best=f"{best_val:.4f}")

    if bar is not None:
        bar.close()

    return best_w, best_b


def _read_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _extract_features(video_path: str) -> Dict[str, float]:
    result = analyze(video_path)
    return result["scores"]


def _load_cached_video_paths(cache_csv: Path) -> Set[str]:
    existing: Set[str] = set()
    if cache_csv.exists() and cache_csv.stat().st_size > 0:
        with cache_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                vp = row.get("video_path", "").strip()
                if vp:
                    existing.add(vp)
    return existing


def _build_feature_cache(
    rows: List[Dict[str, str]],
    cache_csv: Path,
    *,
    show_progress: bool = True,
) -> None:
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_cached_video_paths(cache_csv)
    n_meta = len(rows)
    pending = [r for r in rows if r["video_path"] not in existing]
    n_pending = len(pending)
    n_cached = n_meta - n_pending

    print(
        "\n=== Faz 1: Ozellik cache (CSV) ===\n"
        f"  metadata satiri (bu kosu): {n_meta}\n"
        f"  cache'te hazir:          {n_cached}\n"
        f"  islenecek kalan:        {n_pending}\n"
        f"  cache dosyasi:          {cache_csv.resolve()}\n",
        flush=True,
    )

    if n_pending == 0:
        print("  -> Tum videolar cache'te; ozellik cikarma atlaniyor.\n", flush=True)
        return

    t0 = time.perf_counter()
    with cache_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not existing:
            writer.writerow(["video_path", "Sv", "Sl", "Sb", "Sh", "Sa", "Sf_pipeline"])

        row_iter = pending
        if show_progress and tqdm is not None:
            row_iter = tqdm(
                pending,
                desc="Ozellik cikarma",
                unit="video",
                total=n_pending,
                dynamic_ncols=True,
                mininterval=0.3,
            )

        processed = 0
        errors = 0
        for idx, row in enumerate(row_iter, start=1):
            video_path = row["video_path"]
            if tqdm is None or not show_progress:
                print(f"  [{idx}/{n_pending}] {video_path}", flush=True)

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
                processed += 1
            except Exception as exc:
                errors += 1
                err_nm = type(exc).__name__
                err_line = f"[WARN] ozellik atlandi ({err_nm}): {video_path} | {exc}"
                print(err_line, flush=True)

    elapsed = time.perf_counter() - t0
    avg = elapsed / max(processed, 1)
    print(
        f"\n=== Faz 1 bitti ===\n"
        f"  yazilan satir (basarili): {processed}\n"
        f"  hatali atlama:            {errors}\n"
        f"  sure:                     {elapsed:.1f} s  (ort. {avg:.2f} s/video)\n",
        flush=True,
    )


def _read_cache(cache_csv: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    keys = cache_load_keys()
    with cache_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["video_path"]] = {k: float(row[k]) for k in keys}
    return out


def _count_matched(rows: List[Dict[str, str]], cache: Dict[str, Dict[str, float]]) -> int:
    return sum(1 for r in rows if r["video_path"] in cache)


def _xy(rows: List[Dict[str, str]], cache: Dict[str, Dict[str, float]], feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[List[float]] = []
    ys: List[float] = []
    for row in rows:
        video_path = row["video_path"]
        if video_path not in cache:
            continue
        xs.append([cache[video_path][cache_column(f)] for f in feature_names])
        ys.append(float(row["label"]))
    if not xs:
        raise RuntimeError("No matched rows between metadata and feature cache.")
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _standardize_with_train(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    return (x_train - mu) / sigma, (x_val - mu) / sigma, (x_test - mu) / sigma, mu, sigma


def main() -> None:
    parser = argparse.ArgumentParser(description="Train logistic fusion model from metadata.")
    parser.add_argument("--metadata-csv", required=True, help="CSV with columns: video_path,label,split")
    parser.add_argument("--cache-csv", default="data/feature_cache.csv", help="Feature cache path")
    parser.add_argument("--out-model", default="models/fusion_model.json", help="Output model json")
    parser.add_argument(
        "--features",
        default="Sv",
        help="Comma-separated fusion features. Options: Sv,Sl,Sb,Sh,Sa,Sf (Sf uses cached Sf_pipeline; default: Sv)",
    )
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 regularization strength")
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=1.0,
        help="Positive class weight (>1 boosts fake recall, <1 boosts precision). Ignored if --pos-weight-auto.",
    )
    parser.add_argument(
        "--pos-weight-auto",
        action="store_true",
        help="Set pos_weight to (n_negative / n_positive) on the train split (common default for imbalanced BCE).",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize features using train split mean/std before logistic fit.",
    )
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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm bars (plain log lines only).",
    )
    parser.add_argument(
        "--threshold-objective",
        choices=["f1", "balanced_acc"],
        default="balanced_acc",
        help="Validation metric to choose decision threshold.",
    )
    parser.add_argument(
        "--expansion",
        choices=["none", "poly2"],
        default="none",
        help="Augment base scores before standardize/fit: poly2 adds squares and pairwise products.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata_csv)
    cache_path = Path(args.cache_csv)
    out_model = Path(args.out_model)
    feature_names = [x.strip() for x in args.features.split(",") if x.strip()]
    invalid_features = [x for x in feature_names if x not in ALL_FUSION_FEATURES]
    if not feature_names:
        raise ValueError("No feature selected. Use --features Sv or e.g. --features Sv,Sh,Sa")
    if invalid_features:
        raise ValueError(f"Unknown features: {invalid_features}. Allowed: {ALL_FUSION_FEATURES}")

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
    _build_feature_cache(cache_rows, cache_path, show_progress=not args.no_progress)
    cache = _read_cache(cache_path)

    expansion = str(args.expansion or "none").strip().lower()

    print(
        "\n=== Faz 2: Logistic fusion ===\n"
        f"  features: {feature_names}\n"
        f"  expansion: {expansion}\n"
        f"  train: {len(train_rows)} satir (cache eslesen: {_count_matched(train_rows, cache)})\n"
        f"  val:   {len(val_rows)} satir (cache eslesen: {_count_matched(val_rows, cache)})\n"
        f"  test:  {len(test_rows)} satir (cache eslesen: {_count_matched(test_rows, cache)})\n"
        f"  lr={args.lr} epochs={args.epochs} l2={args.l2} pos_weight={'auto' if args.pos_weight_auto else args.pos_weight} standardize={args.standardize}\n",
        flush=True,
    )

    x_train, y_train = _xy(train_rows, cache, feature_names)
    x_val, y_val = _xy(val_rows, cache, feature_names)
    x_test, y_test = _xy(test_rows, cache, feature_names)

    pos_weight = float(args.pos_weight)
    if args.pos_weight_auto:
        n_pos = float((y_train > 0.5).sum())
        n_neg = float((y_train <= 0.5).sum())
        pos_weight = float(n_neg / max(n_pos, 1.0))
        print(f"  pos_weight_auto: n_pos={int(n_pos)} n_neg={int(n_neg)} -> pos_weight={pos_weight:.4f}\n", flush=True)

    if expansion != "none":
        x_train = expand_features(x_train, expansion)
        x_val = expand_features(x_val, expansion)
        x_test = expand_features(x_test, expansion)

    scaler_mean = None
    scaler_std = None
    if args.standardize:
        x_train, x_val, x_test, scaler_mean, scaler_std = _standardize_with_train(x_train, x_val, x_test)

    w, b = _fit_logreg(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        lr=args.lr,
        epochs=args.epochs,
        l2=args.l2,
        pos_weight=pos_weight,
        show_progress=not args.no_progress,
    )

    p_val = _sigmoid(x_val @ w + b)
    # Az ornekte esik arama asiri uc degerlere kayar; guvenli varsayilan kullan.
    if len(y_val) < 20:
        threshold = 0.5
    else:
        best_thr = _best_threshold(y_val, p_val, objective=args.threshold_objective)
        if args.threshold_objective == "balanced_acc":
            threshold = float(best_thr)
        else:
            threshold = float(np.clip(best_thr, 0.2, 0.8))
    p_test = _sigmoid(x_test @ w + b)

    def _split_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
        pr, rc = _precision_recall(y, p, thr)
        return {
            "loss": _bce_loss(y, p),
            "acc": _accuracy(y, p, thr),
            "balanced_acc": _balanced_acc(y, p, thr),
            "precision": pr,
            "recall": rc,
            "f1": _f1(y, p, thr),
            "auc": _roc_auc(y, p),
        }

    metrics = {
        "val": _split_metrics(y_val, p_val, threshold),
        "test": _split_metrics(y_test, p_test, threshold),
    }

    payload = {
        "feature_names": feature_names,
        "feature_expansion": expansion,
        "weights": w.tolist(),
        "bias": float(b),
        "threshold": float(threshold),
        "standardize": bool(args.standardize),
        "scaler_mean": scaler_mean.tolist() if scaler_mean is not None else None,
        "scaler_std": scaler_std.tolist() if scaler_std is not None else None,
        "train_config": {
            "lr": args.lr,
            "epochs": args.epochs,
            "l2": args.l2,
            "pos_weight": pos_weight,
            "pos_weight_auto": bool(args.pos_weight_auto),
            "threshold_objective": args.threshold_objective,
            "feature_expansion": expansion,
        },
        "metrics": metrics,
    }
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved fusion model: {out_model}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

