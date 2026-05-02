import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple


def _resolve_dataset_root(dataset_root: Path) -> Path:
    """Return a directory that directly contains 0_real and 1_fake.

    Accepts either that layout at ``dataset_root`` or one level down
    (common after unzip: ``.../AVLips/AVLips/0_real``).
    """
    root = dataset_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}\n"
            "Mount or copy AVLips to the pod, then set --dataset-root to the folder "
            "that contains 0_real and 1_fake (not the repo folder)."
        )

    real = root / "0_real"
    fake = root / "1_fake"
    if real.is_dir() and fake.is_dir():
        return root

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "0_real").is_dir() and (child / "1_fake").is_dir():
            print(f"[metadata_builder] Nested layout: using {child}", flush=True)
            return child

    hints: List[str] = []
    if root.is_dir():
        try:
            names = sorted(p.name for p in root.iterdir())
            hints.append(f"Contents of {root}: {names[:40]}{' ...' if len(names) > 40 else ''}")
        except OSError as e:
            hints.append(f"Cannot list {root}: {e}")
        hints.append(
            "On RunPod try: find /workspace -maxdepth 4 -type d -name 0_real 2>/dev/null"
        )

    raise FileNotFoundError(
        f"Expected folders '0_real' and '1_fake' under: {root}\n"
        + ("\n".join(hints) if hints else "")
    )


def _collect_samples(dataset_root: Path) -> List[Tuple[Path, int]]:
    dataset_root = _resolve_dataset_root(dataset_root)
    real_dir = dataset_root / "0_real"
    fake_dir = dataset_root / "1_fake"

    samples: List[Tuple[Path, int]] = []
    samples.extend((p, 0) for p in sorted(real_dir.glob("*.mp4")))
    samples.extend((p, 1) for p in sorted(fake_dir.glob("*.mp4")))
    if not samples:
        raise RuntimeError("No .mp4 files found in dataset.")
    return samples


def _assign_splits(
    samples: List[Tuple[Path, int]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> List[Tuple[Path, int, str]]:
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for sample in samples:
        by_label[sample[1]].append(sample)

    out: List[Tuple[Path, int, str]] = []
    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_test <= 0:
            raise ValueError(
                f"Split ratios leave no test data for label={label}. "
                "Adjust train/val ratios."
            )
        for i, (path, y) in enumerate(items):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            out.append((path, y, split))
    return out


def build_metadata(
    dataset_root: Path,
    out_csv: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    samples = _collect_samples(dataset_root)
    rows = _assign_splits(samples, train_ratio, val_ratio, seed)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label", "split"])
        for video_path, label, split in rows:
            writer.writerow([str(video_path), label, split])

    n_real = sum(1 for _, y, _ in rows if y == 0)
    n_fake = sum(1 for _, y, _ in rows if y == 1)
    print(f"Saved metadata to: {out_csv}")
    print(f"Total: {len(rows)} | real={n_real} fake={n_fake}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/val/test metadata CSV from AVLips-style dataset."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path containing 0_real and 1_fake folders.",
    )
    parser.add_argument(
        "--out-csv",
        default="data/avlips_metadata.csv",
        help="Output metadata CSV path.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or (args.train_ratio + args.val_ratio) >= 1:
        raise ValueError("Ratios must be >0 and train_ratio + val_ratio < 1.")

    build_metadata(
        dataset_root=Path(args.dataset_root),
        out_csv=Path(args.out_csv),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

