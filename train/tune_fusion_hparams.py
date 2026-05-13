"""
Grid-search lr/l2 (fixed feature set) to maximize val balanced accuracy.
Use after auto_select: refines the logistic head without re-scanning feature combos.
"""

import argparse
import json
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import List


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
    pos_weight_auto: bool,
    pos_weight: float,
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


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-tune fusion lr/l2 for a fixed feature set.")
    parser.add_argument("--metadata-csv", default="data/avlips_metadata.csv")
    parser.add_argument("--cache-csv", default="data/feature_cache.csv")
    parser.add_argument("--out-model", default="models/fusion_model.json")
    parser.add_argument("--report-json", default="models/fusion_hparam_tune_report.json")
    parser.add_argument(
        "--from-model",
        default="models/fusion_model.json",
        help="Read feature_names from this JSON (if --features not set).",
    )
    parser.add_argument("--features", default=None, help="Comma list; overrides --from-model feature_names.")
    parser.add_argument("--epochs", type=int, default=2000, help="More epochs often stabilizes the logistic fit.")
    parser.add_argument("--lrs", default="0.02,0.03,0.05,0.07", help="Comma-separated learning rates.")
    parser.add_argument("--l2s", default="0.005,0.01,0.02,0.03", help="Comma-separated L2 values.")
    parser.add_argument("--pos-weight-auto", action="store_true", help="Recommended for imbalanced labels.")
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--threshold-objective", choices=["f1", "balanced_acc"], default="balanced_acc")
    parser.add_argument("--expansion", choices=["none", "poly2"], default="none")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    metadata_csv = (
        (project_root / args.metadata_csv).resolve()
        if not Path(args.metadata_csv).is_absolute()
        else Path(args.metadata_csv)
    )
    cache_csv = (
        (project_root / args.cache_csv).resolve() if not Path(args.cache_csv).is_absolute() else Path(args.cache_csv)
    )
    out_model = (
        (project_root / args.out_model).resolve() if not Path(args.out_model).is_absolute() else Path(args.out_model)
    )
    report_json = (
        (project_root / args.report_json).resolve()
        if not Path(args.report_json).is_absolute()
        else Path(args.report_json)
    )

    if args.features:
        features = [x.strip() for x in args.features.split(",") if x.strip()]
    else:
        from_path = (project_root / args.from_model).resolve() if not Path(args.from_model).is_absolute() else Path(
            args.from_model
        )
        payload = json.loads(from_path.read_text(encoding="utf-8"))
        features = list(payload["feature_names"])

    lrs = _parse_float_list(args.lrs)
    l2s = _parse_float_list(args.l2s)
    standardize = not args.no_standardize

    tune_dir = out_model.parent / "_hparam_tune"
    tune_dir.mkdir(parents=True, exist_ok=True)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    best_ba = -1.0
    best_auc = -1.0
    best_path: Path | None = None
    best_key: tuple[float, float] | None = None

    total = len(lrs) * len(l2s)
    n = 0
    for lr, l2 in product(lrs, l2s):
        n += 1
        tag = f"lr{lr:g}_l2{l2:g}".replace(".", "p")
        trial_path = tune_dir / f"fusion_tune_{tag}.json"
        print(f"[{n}/{total}] lr={lr} l2={l2} -> {trial_path.name}", flush=True)
        _run_train(
            project_root=project_root,
            metadata_csv=metadata_csv,
            cache_csv=cache_csv,
            out_model=trial_path,
            features=features,
            epochs=args.epochs,
            lr=lr,
            l2=l2,
            standardize=standardize,
            pos_weight_auto=bool(args.pos_weight_auto),
            pos_weight=args.pos_weight,
            threshold_objective=args.threshold_objective,
            expansion=args.expansion,
        )
        payload = json.loads(trial_path.read_text(encoding="utf-8"))
        val_ba = float(payload["metrics"]["val"].get("balanced_acc", payload["metrics"]["val"]["acc"]))
        val_auc = float(payload["metrics"]["val"]["auc"])
        test_ba = float(payload["metrics"]["test"].get("balanced_acc", payload["metrics"]["test"]["acc"]))
        row = {
            "lr": lr,
            "l2": l2,
            "val_balanced_acc": val_ba,
            "val_auc": val_auc,
            "test_balanced_acc": test_ba,
            "val_acc": float(payload["metrics"]["val"]["acc"]),
            "test_acc": float(payload["metrics"]["test"]["acc"]),
            "trial_path": str(trial_path),
        }
        rows.append(row)
        replace = False
        if val_ba > best_ba + 1e-12:
            replace = True
        elif abs(val_ba - best_ba) <= 1e-12 and val_auc > best_auc + 1e-12:
            replace = True
        if replace:
            best_ba = val_ba
            best_auc = val_auc
            best_path = trial_path
            best_key = (lr, l2)

    assert best_path is not None and best_key is not None
    shutil.copyfile(best_path, out_model)
    rows_sorted = sorted(rows, key=lambda r: (r["val_balanced_acc"], r["val_auc"]), reverse=True)
    report = {
        "selection": "max val_balanced_acc, tie-break val_auc",
        "features": features,
        "epochs": args.epochs,
        "pos_weight_auto": bool(args.pos_weight_auto),
        "standardize": standardize,
        "threshold_objective": args.threshold_objective,
        "expansion": args.expansion,
        "grid_count": total,
        "best_lr": best_key[0],
        "best_l2": best_key[1],
        "best_val_balanced_acc": best_ba,
        "best_val_auc": best_auc,
        "final_model_path": str(out_model),
        "top_trials": rows_sorted[:15],
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Hparam tune done ===", flush=True)
    print(f"best lr={best_key[0]} l2={best_key[1]} val_balanced_acc={best_ba:.6f} val_auc={best_auc:.6f}", flush=True)
    print(f"written: {out_model}", flush=True)
    print(f"report: {report_json}", flush=True)


if __name__ == "__main__":
    main()
