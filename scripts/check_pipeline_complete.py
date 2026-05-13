"""Sl yenilemesi ve zincirin tamamlanıp tamamlanmadığını hızlı kontrol et."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _csv_data_row_count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _fallback_from_artifacts(
    *,
    metadata: Path,
    cache: Path,
    model_json: Path,
    report_json: Path,
) -> tuple[bool, list[str]]:
    """Log isaretleri yoksa: arama raporu + model + satir sayilari uyumlu mu?"""
    notes: list[str] = []
    if not report_json.exists():
        notes.append("models/fusion_model_search_report.json yok")
        return False, notes
    if not model_json.exists():
        notes.append("models/fusion_model.json yok")
        return False, notes
    if not metadata.exists():
        notes.append("metadata csv yok")
        return False, notes

    try:
        report = json.loads(report_json.read_text(encoding="utf-8"))
        model = json.loads(model_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        notes.append(f"JSON okunamadi: {e}")
        return False, notes

    bf = report.get("best_features")
    mf = model.get("feature_names")
    if not isinstance(bf, list) or not isinstance(mf, list) or bf != mf:
        notes.append(f"best_features rapor/model uyumsuz: rapor={bf!r} model={mf!r}")
        return False, notes

    try:
        best_score = float(report.get("best_score", 0.0))
        val_ba = float(model["metrics"]["val"]["balanced_acc"])
    except (KeyError, TypeError, ValueError) as e:
        notes.append(f"Metrik alanlari eksik veya gecersiz: {e}")
        return False, notes

    if abs(val_ba - best_score) > 1e-4:
        notes.append(f"val_balanced_acc ({val_ba}) ile rapor best_score ({best_score}) uyusmuyor")
        return False, notes

    n_meta = _csv_data_row_count(metadata)
    n_cache = _csv_data_row_count(cache)
    if n_meta <= 0:
        notes.append("metadata satir sayisi 0")
        return False, notes
    if n_cache != n_meta:
        notes.append(f"metadata ({n_meta}) vs cache ({n_cache}) satir sayisi farkli")
        return False, notes

    notes.append("Rapor, fusion_model.json ve metadata/cache satir sayilari tutarli.")
    return True, notes


def main() -> None:
    cache = ROOT / "data" / "feature_cache.csv"
    bak = ROOT / "data" / "feature_cache.csv.bak"
    log = ROOT / "logs" / "sl_fusion_complete.log"
    metadata = ROOT / "data" / "avlips_metadata.csv"
    model_json = ROOT / "models" / "fusion_model.json"
    report_json = ROOT / "models" / "fusion_model_search_report.json"

    if not cache.exists():
        print("ERR: data/feature_cache.csv yok", file=sys.stderr)
        sys.exit(2)

    diffs = None
    if bak.exists():
        rows = list(csv.DictReader(cache.open(encoding="utf-8")))
        backs = list(csv.DictReader(bak.open(encoding="utf-8")))
        diffs = sum(
            1
            for r, s in zip(rows, backs)
            if r.get("video_path") == s.get("video_path") and (r.get("Sl") or "") != (s.get("Sl") or "")
        )
        print(f"Sl farki (cache vs .bak): {diffs} / {len(rows)}")
    else:
        print("Uyari: .bak yok; Sl karsilastirmasi atlandi.")

    ok_chain = False
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        print(f"Log: {log.name} ({log.stat().st_size} byte)")
        marks = ("=== REFRESH OK ===", "=== AUTO_SELECT OK ===", "=== EVAL OK ===", "=== DONE ")
        for mark in marks:
            print(f"  {mark.strip()} : {mark in text}")
        ok_chain = ("=== REFRESH OK ===" in text and "=== AUTO_SELECT OK ===" in text and "=== EVAL OK ===" in text)
    else:
        print("Log yok: logs/sl_fusion_complete.log (zincir henuz calistirilmamis olabilir)")

    if ok_chain:
        print("\nSONUC: Zincir logda tamamlanmis (REFRESH + AUTO_SELECT + EVAL).")
        sys.exit(0)

    ok_fb, fb_notes = _fallback_from_artifacts(
        metadata=metadata,
        cache=cache,
        model_json=model_json,
        report_json=report_json,
    )
    if ok_fb:
        print("\nUYARI: Logda tam zincir isareti yok; dosya bazli kontrol basarili:")
        for line in fb_notes:
            print(f"  - {line}")
        print("\nSONUC: Muhtemelen auto_select + model yazimi tamam (log eksik/legacy olabilir).")
        sys.exit(0)

    for line in fb_notes:
        print(f"Yedek kontrol: {line}", file=sys.stderr)

    if diffs is not None and diffs == 0 and bak.exists():
        print("\nSONUC: Sl hala .bak ile ayni; refresh henuz dosyaya yazilmamis veya calismadi.")
    else:
        print("\nSONUC: Islem devam ediyor veya zincir yarım; logs/sl_fusion_complete.log kontrol et.")
    sys.exit(1)


if __name__ == "__main__":
    main()
