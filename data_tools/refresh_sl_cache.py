"""Refresh only the `Sl` column in feature_cache.csv.

Yeni: opsiyonel paralel calistirma (--workers) ve periyodik checkpoint
kaydi (--checkpoint-every). Sureci kesintiye ugratirsan, son checkpoint'ten
--start-offset ile devam edebilirsin.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def _compute_sl(video_path: str) -> float:
    from src.lip_sync import has_speech_like_audio, lip_mismatch_score
    from src.media_io import extract_audio, extract_frames

    with tempfile.TemporaryDirectory(prefix="df_sl_") as tmp:
        audio_path = str(Path(tmp) / "audio.wav")
        frames_dir = str(Path(tmp) / "frames")
        extract_audio(video_path, audio_path)
        extract_frames(video_path, frames_dir)
        if not Path(audio_path).exists() or Path(audio_path).stat().st_size <= 2048:
            return 0.0
        if not has_speech_like_audio(audio_path):
            return 0.0
        return float(lip_mismatch_score(audio_path, frames_dir))


def _worker(item: Tuple[int, str]) -> Tuple[int, str, float, str]:
    """Tek bir video icin Sl hesapla. Hata yakalanir, mesaji geri donulur."""
    idx, vp = item
    if not vp:
        return idx, vp, -1.0, "empty_path"
    try:
        return idx, vp, float(_compute_sl(vp)), ""
    except Exception as exc:  # pragma: no cover - paralel surec
        return idx, vp, -1.0, f"{type(exc).__name__}: {exc}"


def _write_csv(cache_path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(cache_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh only Sl column in feature cache CSV.")
    parser.add_argument("--cache-csv", default="data/feature_cache.csv")
    parser.add_argument("--backup", action="store_true", help="Create .bak copy before overwrite")
    parser.add_argument("--max-rows", type=int, default=0, help="If >0, process only first N rows (debug)")
    parser.add_argument("--start-offset", type=int, default=0, help="Skip first N rows (resume helper)")
    parser.add_argument("--workers", type=int, default=1, help=">1 enables multiprocessing pool")
    parser.add_argument("--checkpoint-every", type=int, default=200, help="CSV'yi her N video'da bir kaydet")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    cache_path = Path(args.cache_csv)
    if not cache_path.is_absolute():
        cache_path = (ROOT / cache_path).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    with cache_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise RuntimeError("Empty cache CSV header.")
        rows = list(reader)

    if "Sl" not in fieldnames:
        raise RuntimeError("Cache CSV has no 'Sl' column.")

    total = len(rows)
    start = max(0, int(args.start_offset))
    end = total if args.max_rows <= 0 else min(total, start + int(args.max_rows))
    target_idx = list(range(start, end))
    n_target = len(target_idx)
    workers = max(1, int(args.workers))
    checkpoint_every = max(50, int(args.checkpoint_every))

    print(
        f"Refresh Sl cache\n"
        f"  cache: {cache_path}\n"
        f"  total rows: {total}\n"
        f"  target rows: {n_target} (offset={start}, end={end})\n"
        f"  workers: {workers}\n"
        f"  checkpoint_every: {checkpoint_every}\n",
        flush=True,
    )
    if n_target == 0:
        print("Nothing to do.", flush=True)
        return

    if args.backup:
        bak = cache_path.with_suffix(cache_path.suffix + ".bak")
        bak.write_bytes(cache_path.read_bytes())
        print(f"Backup: {bak}", flush=True)

    items: List[Tuple[int, str]] = [(i, rows[i].get("video_path", "").strip()) for i in target_idx]

    processed = 0
    failed = 0
    last_ckpt = time.time()
    t0 = time.time()

    def _flush(reason: str) -> None:
        nonlocal last_ckpt
        _write_csv(cache_path, list(fieldnames), rows)
        last_ckpt = time.time()
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, n_target - processed)
        eta = remaining / rate if rate > 0 else float("inf")
        print(
            f"[checkpoint:{reason}] processed={processed}/{n_target} failed={failed} "
            f"rate={rate:.2f} v/s eta={eta/60:.1f} min",
            flush=True,
        )

    try:
        if workers <= 1:
            for it in items:
                idx, vp, sl, err = _worker(it)
                if err:
                    failed += 1
                    print(f"[WARN] Sl refresh failed: {vp} | {err}", flush=True)
                else:
                    rows[idx]["Sl"] = repr(sl) if False else f"{sl}"
                    processed += 1
                if processed and processed % checkpoint_every == 0:
                    _flush("interval")
        else:
            with mp.get_context("spawn").Pool(processes=workers) as pool:
                for idx, vp, sl, err in pool.imap_unordered(_worker, items, chunksize=2):
                    if err:
                        failed += 1
                        print(f"[WARN] Sl refresh failed: {vp} | {err}", flush=True)
                    else:
                        rows[idx]["Sl"] = f"{sl}"
                        processed += 1
                    if processed and processed % checkpoint_every == 0:
                        _flush("interval")
    finally:
        _write_csv(cache_path, list(fieldnames), rows)

    elapsed = time.time() - t0
    print(
        f"\nDone\n"
        f"  processed: {processed}\n"
        f"  failed: {failed}\n"
        f"  elapsed: {elapsed/60:.1f} min\n"
        f"  written: {cache_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
