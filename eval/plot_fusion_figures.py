"""
Fusion degerlendirme sekilleri (PNG): ROC, kalibrasyon guvenilirlik egrisi, 5-fold bar.

Varsayilan cikti: results/v2/figures/

Kaynakca PDF'leri (makale klasoru):
  C:\\Users\\busra\\Desktop\\projeler\\makale\\
  Ornek dosyalar: Deepfake_Media_Generation_and_Detection_in_the_Gen.pdf,
  FaceForensics / multimodal / calibration ile ilgili Springerer ve MDPI PDF'leri.
  Tez kaynakcasi icin bu klasordeki DOI'li makaleleri acip bibliyografik giris yapin.

Kullanim:
  cd Multimodal-Deepfake-Tespit-Sistemi
  .\\.venv\\Scripts\\python.exe eval\\plot_fusion_figures.py

Opsiyon:
  --no-train   Sadece fusion_cv_allfeats.json ile fold bar grafigi (ROC/kalibrasyon atlanir)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
_eval_dir = ROOT / "eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import fusion_io  # noqa: E402


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )
    return plt


def plot_cv_bars(cv_json: Path, out_png: Path, plt) -> None:
    data = json.loads(cv_json.read_text(encoding="utf-8"))
    folds_lr = [f["balanced_acc"] for f in data["logreg"]["folds"]]
    folds_gb = [f["balanced_acc"] for f in data["histgb"]["folds"]]
    folds_lr_auc = [f["auc"] for f in data["logreg"]["folds"]]
    folds_gb_auc = [f["auc"] for f in data["histgb"]["folds"]]
    x = np.arange(1, len(folds_lr) + 1)
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(x - w / 2, folds_lr, width=w, label="Lojistik", color="#4C72B0")
    axes[0].bar(x + w / 2, folds_gb, width=w, label="HistGB", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Dengeli dogruluk")
    axes[0].set_title("5-fold CV — dengeli dogruluk")
    axes[0].legend()
    axes[0].set_ylim(0.45, 0.72)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - w / 2, folds_lr_auc, width=w, label="Lojistik", color="#4C72B0")
    axes[1].bar(x + w / 2, folds_gb_auc, width=w, label="HistGB", color="#DD8452")
    axes[1].set_xticks(x)
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("5-fold CV — ROC AUC")
    axes[1].legend()
    axes[1].set_ylim(0.45, 0.72)
    axes[1].grid(axis="y", alpha=0.3)

    agg_lr = data["logreg"]["agg"]["balanced_acc"]["mean"]
    agg_gb = data["histgb"]["agg"]["balanced_acc"]["mean"]
    fig.suptitle(
        f"Ozellikler: {', '.join(data['features'])}  |  "
        f"LR ort={agg_lr:.3f}, HistGB ort={agg_gb:.3f}",
        fontsize=10,
        y=1.02,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _train_and_predict(
    metadata_path: Path,
    cache_path: Path,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier

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

    clf = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=250,
        learning_rate=0.06,
        random_state=42,
        class_weight="balanced",
        early_stopping=False,
    )
    clf.fit(x_tr_s, y_tr)
    p_raw = clf.predict_proba(x_te_s)[:, 1]

    cal = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit")
    cal.fit(x_va_s, y_va)
    p_platt = cal.predict_proba(x_te_s)[:, 1]
    return y_te, p_raw, p_platt, y_tr, y_va, x_te_s


def plot_roc(y_te: np.ndarray, p_raw: np.ndarray, p_platt: np.ndarray, out_png: Path, plt) -> None:
    from sklearn.metrics import auc, roc_curve

    fig, ax = plt.subplots(figsize=(5.2, 5))
    for name, p, style in (
        ("HistGB (ham)", p_raw, "-"),
        ("HistGB + Platt", p_platt, "--"),
    ):
        fpr, tpr, _ = roc_curve(y_te, p)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, style, lw=2, label=f"{name}, AUC={a:.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5, label="Rastgele")
    ax.set_xlabel("Yanlis pozitif orani (FPR)")
    ax.set_ylabel("Dogru pozitif orani (TPR)")
    ax.set_title("ROC — test (Sv, Sl, Sb, Sh, Sa, Sf)")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def plot_calibration(
    y_te: np.ndarray,
    p_raw: np.ndarray,
    p_platt: np.ndarray,
    out_png: Path,
    plt,
    n_bins: int = 10,
) -> None:
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(5.2, 5))
    for name, p, color in (
        ("Ham HistGB", p_raw, "#4C72B0"),
        ("Platt (sigmoid)", p_platt, "#DD8452"),
    ):
        prob_true, prob_pred = calibration_curve(y_te, p, n_bins=n_bins, strategy="uniform")
        ax.plot(prob_pred, prob_true, "s-", color=color, lw=2, markersize=6, label=name)
    ax.plot([0, 1], [0, 1], "k:", lw=1.5, alpha=0.6, label="Mukemmel kalibrasyon")
    ax.set_xlabel("Ortalama tahmin olasiligi (bin)")
    ax.set_ylabel("Pozitif orani (gozlenen)")
    ax.set_title(f"Guvenilirlik egrisi — test, {n_bins} esit genislik bin")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def plot_calibration_from_json(cal_json: Path, out_png: Path, plt) -> None:
    """JSON'daki ECE bin ortalamalariyla hafif guvenilirlik diyagrami (yeniden egitim yok)."""
    data = json.loads(cal_json.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5.2, 5))

    def _scatter_bins_from_list(bins: list, label: str, color: str, linestyle: str) -> None:
        px = [b["p_mean"] for b in bins if b["n"] > 0]
        py = [b["y_mean"] for b in bins if b["n"] > 0]
        ns = [b["n"] for b in bins if b["n"] > 0]
        ax.scatter(
            px,
            py,
            s=[max(24, min(400, n * 3)) for n in ns],
            c=color,
            alpha=0.78,
            label=label,
            edgecolors="k",
            linewidths=0.35,
        )
        if len(px) >= 2:
            ax.plot(px, py, linestyle, color=color, alpha=0.65, lw=1.6)

    raw_bins = data.get("raw_histgb", {}).get("test_ece_bins", [])
    if raw_bins:
        _scatter_bins_from_list(raw_bins, "Ham HistGB (JSON bin)", "#4C72B0", "-")
    if "calibrators" in data and "sigmoid" in data["calibrators"]:
        sig_bins = data["calibrators"]["sigmoid"].get("test_ece_bins", [])
        if sig_bins:
            _scatter_bins_from_list(sig_bins, "Platt (JSON bin)", "#DD8452", "--")
    ax.plot([0, 1], [0, 1], "k:", lw=1.5, alpha=0.6, label="Mukemmel kalibrasyon")
    ax.set_xlabel("Ortalama tahmin (bin)")
    ax.set_ylabel("Gozlenen pozitif orani")
    ax.set_title("Kalibrasyon (yalnizca fusion_calibration.json binleri)")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion ROC, kalibrasyon ve CV sekilleri (PNG).")
    parser.add_argument("--metadata-csv", default="data/avlips_metadata.csv")
    parser.add_argument("--cache-csv", default="data/feature_cache.csv")
    parser.add_argument("--features", default="Sv,Sl,Sb,Sh,Sa,Sf")
    parser.add_argument("--cv-json", default="results/v2/fusion_cv_allfeats.json")
    parser.add_argument("--cal-json", default="results/v2/fusion_calibration.json")
    parser.add_argument("--out-dir", default="results/v2/figures")
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="ROC ve sklearn kalibrasyon egrisini atla; CV bar + JSON-bin kalibrasyon",
    )
    parser.add_argument("--refs-dir", default=r"C:\Users\busra\Desktop\projeler\makale", help="Kaynak PDF klasoru (bilgi)")
    args = parser.parse_args()

    plt = _setup_matplotlib()
    feature_names = [s.strip() for s in args.features.split(",") if s.strip()]

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_path = Path(args.cv_json)
    if not cv_path.is_absolute():
        cv_path = ROOT / cv_path
    if cv_path.exists():
        plot_cv_bars(cv_path, out_dir / "cv_fold_balanced_acc_auc.png", plt)
        print(f"OK: {out_dir / 'cv_fold_balanced_acc_auc.png'}")
    else:
        print(f"[WARN] CV JSON yok, atlaniyor: {cv_path}")

    cal_path = Path(args.cal_json)
    if not cal_path.is_absolute():
        cal_path = ROOT / cal_path
    if cal_path.exists():
        plot_calibration_from_json(cal_path, out_dir / "calibration_reliability_from_json_bins.png", plt)
        print(f"OK: {out_dir / 'calibration_reliability_from_json_bins.png'}")
    else:
        print(f"[WARN] Kalibrasyon JSON yok: {cal_path}")

    refs = Path(args.refs_dir)
    if refs.is_dir():
        n_pdf = len(list(refs.glob("*.pdf")))
        print(f"Kaynakca klasoru: {refs} ({n_pdf} PDF)")

    if args.no_train:
        print("--no-train: ROC ve tam kalibrasyon egrisi uretilmedi.")
        return

    metadata_path = Path(args.metadata_csv)
    if not metadata_path.is_absolute():
        metadata_path = ROOT / metadata_path
    cache_path = Path(args.cache_csv)
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path

    y_te, p_raw, p_platt, _, _, _ = _train_and_predict(metadata_path, cache_path, feature_names)
    plot_roc(y_te, p_raw, p_platt, out_dir / "roc_test_histgb_platt.png", plt)
    print(f"OK: {out_dir / 'roc_test_histgb_platt.png'}")
    plot_calibration(y_te, p_raw, p_platt, out_dir / "calibration_reliability_sklearn.png", plt)
    print(f"OK: {out_dir / 'calibration_reliability_sklearn.png'}")


if __name__ == "__main__":
    main()
