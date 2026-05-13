import argparse
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion_features import ALL_FUSION_FEATURES

ALL_FEATURES = ALL_FUSION_FEATURES


def _score_key(model_json: Path, metric: str) -> float:
    payload = json.loads(model_json.read_text(encoding="utf-8"))
    if metric == "val_auc":
        return float(payload["metrics"]["val"]["auc"])
    if metric == "val_f1":
        return float(payload["metrics"]["val"]["f1"])
    if metric == "test_auc":
        return float(payload["metrics"]["test"]["auc"])
    if metric == "val_balanced_acc":
        return float(payload["metrics"]["val"].get("balanced_acc", payload["metrics"]["val"]["acc"]))
    if metric == "test_balanced_acc":
        return float(payload["metrics"]["test"].get("balanced_acc", payload["metrics"]["test"]["acc"]))
    raise ValueError(f"Unsupported metric: {metric}")


def _run_train(
    project_root: Path,
    metadata_csv: Path,
    cache_csv: Path,
    out_model: Path,
    features: List[str],
    epochs: int,
    lr: float,
    l2: float,
    standardize: bool,
    pos_weight: float,
    pos_weight_auto: bool,
    threshold_objective: str,
    expansion: str,
) -> None:
    cmd = [
        sys.executable,
        str(project_root / "train" / "train_fusion_from_metadata.py"),
        "--metadata-csv",
        str(metadata_csv),
        "--cache-csv",
        str(cache_csv),
        "--out-model",
        str(out_model),
        "--features",
        ",".join(features),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--l2",
        str(l2),
        "--threshold-objective",
        threshold_objective,
        "--expansion",
        expansion,
        "--no-progress",
    ]
    if pos_weight_auto:
        cmd.append("--pos-weight-auto")
    else:
        cmd.extend(["--pos-weight", str(pos_weight)])
    if standardize:
        cmd.append("--standardize")
    subprocess.run(cmd, check=True, cwd=str(project_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-search best fusion feature set on cached scores.")
    parser.add_argument("--metadata-csv", default="data/avlips_metadata.csv")
    parser.add_argument("--cache-csv", default="data/feature_cache.csv")
    parser.add_argument("--out-model", default="models/fusion_model.json")
    parser.add_argument("--report-json", default="models/fusion_model_search_report.json")
    parser.add_argument(
        "--metric",
        choices=["val_balanced_acc", "val_auc", "val_f1", "test_auc", "test_balanced_acc"],
        default="val_balanced_acc",
        help="Selection metric. Prefer val_* (not test_*) to avoid cherry-picking the holdout.",
    )
    parser.add_argument("--max-combo-size", type=int, default=6, help=f"1..{len(ALL_FEATURES)}")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument(
        "--pos-weight-auto",
        action="store_true",
        help="Train with pos_weight = n_neg / n_pos on train (recommended for imbalanced labels).",
    )
    parser.add_argument("--threshold-objective", choices=["f1", "balanced_acc"], default="balanced_acc")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument(
        "--expansion",
        choices=["none", "poly2"],
        default="none",
        help="Öğrenilmiş füzyondan önce özellik genişletmesi (poly2: kareler + çarpımlar).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    metadata_csv = (project_root / args.metadata_csv).resolve() if not Path(args.metadata_csv).is_absolute() else Path(args.metadata_csv)
    cache_csv = (project_root / args.cache_csv).resolve() if not Path(args.cache_csv).is_absolute() else Path(args.cache_csv)
    out_model = (project_root / args.out_model).resolve() if not Path(args.out_model).is_absolute() else Path(args.out_model)
    report_json = (project_root / args.report_json).resolve() if not Path(args.report_json).is_absolute() else Path(args.report_json)

    max_combo_size = max(1, min(len(ALL_FEATURES), int(args.max_combo_size)))
    standardize = not args.no_standardize

    search_dir = out_model.parent / "_auto_search"
    search_dir.mkdir(parents=True, exist_ok=True)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    combos: List[Tuple[str, ...]] = []
    for k in range(1, max_combo_size + 1):
        combos.extend(list(itertools.combinations(ALL_FEATURES, k)))

    rows = []
    best_score = -1e18
    best_val_auc = -1e18
    best_path: Path | None = None
    best_features: List[str] = []

    total = len(combos)
    for idx, combo in enumerate(combos, start=1):
        features = list(combo)
        exp_tag = "" if args.expansion == "none" else f"_{args.expansion}"
        model_path = search_dir / f"fusion_{'_'.join(features)}{exp_tag}.json"
        print(f"[{idx}/{total}] training features={features} expansion={args.expansion}", flush=True)
        _run_train(
            project_root=project_root,
            metadata_csv=metadata_csv,
            cache_csv=cache_csv,
            out_model=model_path,
            features=features,
            epochs=args.epochs,
            lr=args.lr,
            l2=args.l2,
            standardize=standardize,
            pos_weight=args.pos_weight,
            pos_weight_auto=bool(args.pos_weight_auto),
            threshold_objective=args.threshold_objective,
            expansion=args.expansion,
        )
        s = _score_key(model_path, args.metric)
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        val_auc = float(payload["metrics"]["val"]["auc"])
        val_ba = float(payload["metrics"]["val"].get("balanced_acc", payload["metrics"]["val"]["acc"]))
        test_ba = float(payload["metrics"]["test"].get("balanced_acc", payload["metrics"]["test"]["acc"]))
        row = {
            "features": features,
            "metric": args.metric,
            "score": s,
            "val_balanced_acc": val_ba,
            "test_balanced_acc": test_ba,
            "val_auc": val_auc,
            "test_auc": float(payload["metrics"]["test"]["auc"]),
            "val_f1": float(payload["metrics"]["val"]["f1"]),
            "test_f1": float(payload["metrics"]["test"]["f1"]),
            "model_path": str(model_path),
        }
        rows.append(row)
        print(
            f"    -> {args.metric}={s:.6f} val_ba={val_ba:.4f} val_auc={row['val_auc']:.6f} test_auc={row['test_auc']:.6f}",
            flush=True,
        )
        replace = False
        if s > best_score + 1e-12:
            replace = True
        elif abs(s - best_score) <= 1e-12 and val_auc > best_val_auc + 1e-12:
            replace = True
        if replace:
            best_score = s
            best_val_auc = val_auc
            best_path = model_path
            best_features = features

    assert best_path is not None
    shutil.copyfile(best_path, out_model)

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    report = {
        "selection_metric": args.metric,
        "pos_weight_auto": bool(args.pos_weight_auto),
        "max_combo_size": max_combo_size,
        "standardize": standardize,
        "expansion": args.expansion,
        "search_count": total,
        "best_features": best_features,
        "best_score": best_score,
        "best_model_source": str(best_path),
        "final_model_path": str(out_model),
        "top10": rows_sorted[:10],
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Auto search done ===", flush=True)
    print(f"best_features={best_features}", flush=True)
    print(f"{args.metric}={best_score:.6f}", flush=True)
    print(f"written: {out_model}", flush=True)
    print(f"report: {report_json}", flush=True)


if __name__ == "__main__":
    main()
