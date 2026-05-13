"""Birden fazla video icin predict_video.py cagir; her biri icin JSON kaydet."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Toplu video tahmini (--out-json ile).")
    parser.add_argument("--model-json", default="models/fusion_model.json")
    parser.add_argument("--out-dir", required=True, help="Ornek: results/predict_batch")
    parser.add_argument(
        "--video",
        action="append",
        dest="videos",
        default=[],
        help="Tekrarlanabilir: --video path1 --video path2",
    )
    parser.add_argument(
        "--videos-file",
        default="",
        help="Satir basina bir video yolu (UTF-8).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    pred = root / "infer" / "predict_video.py"
    model_json = Path(args.model_json)
    if not model_json.is_absolute():
        model_json = root / model_json

    paths: list[Path] = []
    for v in args.videos:
        paths.append(Path(v))
    if args.videos_file:
        vf = Path(args.videos_file)
        text = vf.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(Path(line))

    if not paths:
        print("ERR: en az bir --video veya --videos-file gerekli", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for vp in paths:
        vp = vp.expanduser()
        if not vp.is_file():
            print(f"SKIP (yok): {vp}", file=sys.stderr)
            continue
        stem = vp.stem
        out_json = out_dir / f"{stem}.json"
        cmd = [
            py,
            str(pred),
            "--video",
            str(vp.resolve()),
            "--model-json",
            str(model_json.resolve()),
            "--out-json",
            str(out_json.resolve()),
        ]
        print(f"-> {vp.name} ...", flush=True)
        subprocess.run(cmd, check=True, cwd=str(root))
        print(f"   OK {out_json}") 
    print("Done.")


if __name__ == "__main__":
    main()
