#!/usr/bin/env python3
"""Fetch and clean Epileptic Seizure Recognition CSV using standard library only."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "02_data" / "raw" / "epileptic_seizure_recognition"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = RAW_DIR / "epileptic_seizure_data.csv"
METADATA_JSON = RAW_DIR / "metadata.json"

URLS = [
    "https://raw.githubusercontent.com/akshayg056/Epileptic-seizure-detection-/master/data.csv",
    "https://raw.githubusercontent.com/dragonpilee/Epileptic-Seizure-Detection-System/master/data.csv",
    "https://archive.ics.uci.edu/static/public/388/epileptic+seizure+recognition.zip",
]


def _download_binary(url: str, output_path: Path, retries: int = 5) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, output_path.open("wb") as out:
                content_length = resp.headers.get("Content-Length")
                expected = int(content_length) if content_length else None
                read_bytes = 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    read_bytes += len(chunk)

            if expected is not None and read_bytes < expected:
                raise RuntimeError(
                    f"Incomplete download ({read_bytes}/{expected} bytes)"
                )
            return
        except urllib.error.HTTPError as exc:  # pragma: no cover
            # 4xx client errors will not succeed with retries.
            if 400 <= exc.code < 500:
                output_path.unlink(missing_ok=True)
                raise
            last_exc = exc
            output_path.unlink(missing_ok=True)
            print(f"Attempt {attempt}/{retries} failed for {url}: {exc}")
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            output_path.unlink(missing_ok=True)
            print(f"Attempt {attempt}/{retries} failed for {url}: {exc}")

    raise RuntimeError(f"Failed after {retries} attempts: {url}. Last error: {last_exc}")


def _download_csv() -> str:
    for url in URLS:
        try:
            print(f"Trying download: {url}")
            if url.endswith(".csv"):
                _download_binary(url, RAW_CSV)
            elif url.endswith(".zip"):
                zip_path = RAW_DIR / "dataset_tmp.zip"
                _download_binary(url, zip_path)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
                    if not csv_members:
                        raise RuntimeError(f"No CSV found inside zip: {url}")
                    # Prefer the largest CSV member when multiple files exist.
                    csv_member = max(csv_members, key=lambda m: zf.getinfo(m).file_size)
                    with zf.open(csv_member, "r") as src, RAW_CSV.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                zip_path.unlink(missing_ok=True)
            else:
                continue

            if RAW_CSV.exists() and RAW_CSV.stat().st_size > 0:
                print(f"Downloaded to: {RAW_CSV}")
                return url
        except Exception as exc:  # pragma: no cover
            print(f"Download failed from {url}: {exc}")
    raise RuntimeError("Unable to download dataset CSV from configured sources.")


def _clean_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError("CSV is empty.")

    header = rows[0]
    n_cols = len(header)

    clean = [header]
    bad_rows = 0
    for row in rows[1:]:
        if len(row) == n_cols:
            clean.append(row)
        else:
            bad_rows += 1

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(clean)

    target_col = "y" if "y" in header else header[-1]
    return {
        "rows": len(clean) - 1,
        "columns": n_cols,
        "target_column": target_col,
        "dropped_bad_rows": bad_rows,
        "path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if CSV exists.")
    args = parser.parse_args()

    if args.force_download or not RAW_CSV.exists() or RAW_CSV.stat().st_size == 0:
        source_url = _download_csv()
    else:
        source_url = "cached_local_file"

    info = _clean_csv(RAW_CSV)
    info.update(
        {
            "dataset": "Epileptic Seizure Recognition",
            "source": "Kaggle/UCI source",
            "selected_source_url": source_url,
            "source_links": URLS,
        }
    )

    with METADATA_JSON.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("Dataset ready.")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
