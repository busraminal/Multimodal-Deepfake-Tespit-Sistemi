"""Quick: compare Sl column between two cache CSVs."""
from __future__ import annotations

import csv
import sys


def main() -> None:
    new_path = sys.argv[1] if len(sys.argv) > 1 else "data/feature_cache.csv"
    old_path = sys.argv[2] if len(sys.argv) > 2 else "data/feature_cache.csv.preLS3"
    with open(new_path, encoding="utf-8") as f:
        new = list(csv.DictReader(f))
    with open(old_path, encoding="utf-8") as f:
        old = list(csv.DictReader(f))
    n = min(len(new), len(old))
    print(f"rows: new={len(new)}  old={len(old)}  compared={n}")
    diffs = []
    for i in range(n):
        a = old[i].get("Sl") or ""
        b = new[i].get("Sl") or ""
        try:
            if abs(float(a) - float(b)) > 1e-6:
                diffs.append((i, a, b))
        except ValueError:
            if a != b:
                diffs.append((i, a, b))
    print(f"changed: {len(diffs)}")
    for i, a, b in diffs[:25]:
        print(f"  row {i}: {a} -> {b}")


if __name__ == "__main__":
    main()
