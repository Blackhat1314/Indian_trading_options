#!/usr/bin/env python3
"""Package what NeuralEdge trains on into one zip, on the server:

  docker compose run --rm tools python /app/research/neural_export.py
  (or, without Docker: python research/neural_export.py)

→ data/neural/training-data-YYYYMMDD.zip with
    lake/dhan/D/*.parquet            daily bars of every stock and index
    recorder/<day>/news-*.parquet    every recorded headline (the news archive)
    cache/screener/*.json            fundamentals already fetched by the SCREENER desk
Ticks, 1-minute bars and everything else stay on the server (they are large and not needed for the swing model).
Then download the zip (GCP browser SSH: gear menu → Download file) and run research/neural_train.py on the PC.
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from td.config import settings  # noqa: E402


def main() -> None:
    data = Path(settings.data_dir)
    parts = [("lake/dhan/D", "*.parquet"), ("recorder", "*/news-*.parquet"), ("cache/screener", "*.json")]
    files: list[tuple[Path, str]] = []
    for sub, pat in parts:
        base = data / sub
        for f in sorted(base.glob(pat)) if base.exists() else []:
            files.append((f, f.relative_to(data).as_posix()))
    daily = sum(1 for _, a in files if a.startswith("lake/"))
    if not daily:
        sys.exit(f"no daily bars under {data / 'lake' / 'dhan' / 'D'} — run the history backfill first")
    out = data / "neural" / f"training-data-{time.strftime('%Y%m%d')}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f, arc in files:
            z.write(f, arc)
    tmp.replace(out)
    news = sum(1 for _, a in files if a.startswith("recorder/"))
    fund = sum(1 for _, a in files if a.startswith("cache/"))
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB): {daily} daily-bar files, {news} news files, "
          f"{fund} fundamentals files")
    print(f"download it: data/neural/{out.name} inside your tradedesk folder (with Docker, ~/tradedesk/data/…)")


if __name__ == "__main__":
    main()
