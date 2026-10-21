#!/usr/bin/env python3
"""Package what NeuralEdge trains on into one zip, on the server:

  docker compose run --rm tools python /app/research/neural_export.py
  (or, without Docker: python research/neural_export.py)

→ data/neural/training-data-YYYYMMDD.zip with
    lake/dhan/D/*.parquet            daily bars of every stock and index
    intraday/*.parquet               one row per stock and day condensed from the 1-minute bars (td.neural.minute):
                                     the first 15 bars, the 09:30 price, the close, the day's 1-minute summaries —
                                     what the 09:30 model and the daily model's 1-minute inputs need, ~1/50 of the size
    recorder/<day>/news-*.parquet    every recorded headline (the news archive)
    cache/screener/*.json            fundamentals already fetched by the SCREENER desk
    config/*.json                    the live ranker's weights (if fitted) and the TOP5 settings
The condensed tables are cached in data/neural/intraday-cache/ and only rebuilt for stocks whose 1-minute file
changed, so later exports are quick. Then download the zip and run research/neural_train.py on the PC.
"""
from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from td.config import settings  # noqa: E402


def _tables(data: Path) -> list[tuple[Path, str]]:
    """Condensed 1-minute tables, (re)built where the 1-minute file changed since the cached copy."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from td.lake import Lake
    from td.neural import minute as mn
    src = data / "lake" / "dhan" / "1m"
    if not src.exists():
        return []
    cache = data / "neural" / "intraday-cache"
    cache.mkdir(parents=True, exist_ok=True)
    lake = Lake(data / "lake" / "dhan")
    files = sorted(src.glob("*.parquet"))
    out, built = [], 0
    t0 = time.time()
    for i, f in enumerate(files):
        dst = cache / f.name
        stamp = f"{f.stat().st_mtime_ns}:{f.stat().st_size}"
        fresh = False
        if dst.exists():
            try:
                fresh = (pq.read_schema(dst).metadata or {}).get(b"source") == stamp.encode()
            except Exception:
                fresh = False
        if not fresh:
            from td.lake import _unsafe
            sym = _unsafe(f.stem)
            try:
                t = mn.day_table(lake.read("1m", sym))
            except Exception as e:                       # a damaged file must not stop the export
                print(f"  skipped {sym}: {e}")
                continue
            if t.empty:
                continue
            table = pa.Table.from_pandas(t, preserve_index=False)
            table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"source": stamp.encode()})
            tmp = dst.with_suffix(f".{os.getpid()}.tmp")
            pq.write_table(table, tmp, compression="zstd")
            tmp.replace(dst)
            built += 1
        out.append((dst, f"intraday/{dst.name}"))
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            print(f"  1-minute tables {i + 1}/{len(files)} ({built} rebuilt, {time.time() - t0:.0f} s)")
    return out


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
    print("condensing 1-minute bars …")
    tables = _tables(data)
    files += tables
    out = data / "neural" / f"training-data-{time.strftime('%Y%m%d')}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f, arc in files:
            z.write(f, arc)
        weights = data / "config" / "ranker_weights.json"
        if weights.exists():
            z.write(weights, "config/ranker_weights.json")
        z.writestr("config/tdesk.json", json.dumps({
            "min_turnover_cr": settings.min_turnover_cr, "lock_time": settings.lock_time,
            "or_minutes": settings.or_minutes, "top_n": settings.top_n, "exported_at": time.time()}))
    tmp.replace(out)
    try:
        out.chmod(0o644)                                   # created by root inside Docker: let the SSH user read it
    except OSError:
        pass
    news = sum(1 for _, a in files if a.startswith("recorder/"))
    fund = sum(1 for _, a in files if a.startswith("cache/"))
    print(f"\n{out.name}  ({out.stat().st_size / 1e6:.1f} MB): {daily} daily-bar files, {len(tables)} 1-minute "
          f"tables, {news} news files, {fund} fundamentals files")
    if not tables:
        print("  no 1-minute history in the lake — the 09:30 model needs it (ADMIN → history → 1-minute)")
    inside_docker = Path("/.dockerenv").exists() or str(data) == "/app/data"
    if inside_docker:
        print("\nOn the VM the file is in your tradedesk folder:  tradedesk/data/neural/" + out.name)
        print("Its full path (for Download file) is printed by:   realpath ~/tradedesk/data/neural/" + out.name)
    else:
        print(f"\nFull path (for Download file): {out.resolve()}")


if __name__ == "__main__":
    main()
