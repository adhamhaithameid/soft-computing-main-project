#!/usr/bin/env python3
"""Environment and dataset readiness check."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "02_data" / "raw" / "epileptic_seizure_recognition" / "epileptic_seizure_data.csv"

required = [
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
]
optional = ["matplotlib", "seaborn"]

print("Dependency check:")
missing = []
for mod in required:
    ok = importlib.util.find_spec(mod) is not None
    print(f"- {mod}: {'OK' if ok else 'MISSING'}")
    if not ok:
        missing.append(mod)

print("\nDataset check:")
if RAW.exists():
    with RAW.open() as f:
        r = csv.reader(f)
        rows = list(r)
    print(f"- CSV found: {RAW}")
    print(f"- columns: {len(rows[0]) if rows else 0}")
    print(f"- rows: {max(0, len(rows)-1)}")
else:
    print(f"- CSV missing: {RAW}")

if missing:
    print("\nMissing dependencies detected. Install with:")
    print("python -m pip install --disable-pip-version-check -r requirements.txt")
    raise SystemExit(1)

print("\nOptional plotting deps:")
for mod in optional:
    ok = importlib.util.find_spec(mod) is not None
    print(f"- {mod}: {'OK' if ok else 'MISSING (plots will be skipped)'}")

print("\nEnvironment ready.")
