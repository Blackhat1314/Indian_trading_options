#!/usr/bin/env python3
"""Train NeuralEdge on your PC (GPU if PyTorch sees one) from the server's training-data zip.

  1. on the server:  docker compose run --rm tools python /app/research/neural_export.py
                     → data/neural/training-data-YYYYMMDD.zip   (download it)
  2. on the PC:      python research/neural_train.py --data C:\\path\\training-data-20261021.zip --fetch-fundamentals
                     → <data folder>/neural/model.onnx + model.json
  3. upload both files to the server's data/neural/ folder; the NEURAL desk picks them up within a minute.

Fundamentals: the zip carries the server's Screener.in cache (stocks someone opened on the SCREENER desk).
--fetch-fundamentals fetches the rest from Screener.in politely (one page every 3 s, ~20 minutes for the F&O list
the first time, cached after that). Without fundamentals the network still trains; its fundamental inputs are
masked.

Options: --seeds 3 (networks averaged), --epochs 15, --no-gbm (skip the gradient-boosting baseline, faster),
--device auto|cuda|cpu, --out <folder>.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from td.config import settings  # noqa: E402

log = logging.getLogger("neural_train")


def _data_root(arg: str) -> Path:
    p = Path(arg).expanduser().resolve()
    if p.suffix.lower() == ".zip":
        dest = p.with_suffix("")
        stamp = dest / ".extracted"
        if not stamp.exists() or stamp.stat().st_mtime < p.stat().st_mtime:
            print(f"extracting {p.name} → {dest}")
            with zipfile.ZipFile(p) as z:
                for m in z.infolist():                  # no absolute paths or .. from the archive
                    target = (dest / m.filename).resolve()
                    if dest.resolve() not in target.parents and target != dest.resolve():
                        raise SystemExit(f"refusing unsafe path in the zip: {m.filename}")
                z.extractall(dest)
            stamp.write_text(str(time.time()))
        return dest
    if not p.is_dir():
        raise SystemExit(f"{p} is neither the training-data zip nor a data folder")
    return p


def _frames(root: Path):
    from td.lake import Lake
    from td.research.data import clean_daily
    from td.research.quant import is_index
    lake = Lake(root / "lake" / "dhan")
    syms = lake.symbols("D")
    if not syms:
        raise SystemExit(f"no daily bars in {root / 'lake' / 'dhan' / 'D'} — is this the right folder?")
    frames = {}
    for s in syms:
        if is_index(s):
            continue
        df = clean_daily(lake.read("D", s))
        if len(df):
            frames[s] = df
    index_df = clean_daily(lake.read("D", "NIFTY"))
    return frames, index_df


def _cached_fundamentals(caches: list[Path], syms: list[str]) -> dict[str, dict]:
    """Newest cached copy per stock across the caches (the server's, in the zip, and the PC's own)."""
    from urllib.parse import quote
    out = {}
    for s in syms:
        found = [c / f"{quote(s, safe='')}.{view}.json" for c in caches for view in ("auto", "consolidated", "standalone")]
        found = sorted((f for f in found if f.exists()), key=lambda f: f.stat().st_mtime, reverse=True)
        for f in found:
            if f.exists():
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    out[s] = d.get("fundamentals") or d
                    break
                except (OSError, ValueError):
                    pass
    return out


async def _fetch_fundamentals(cache: Path, syms: list[str], have: dict[str, dict]) -> dict[str, dict]:
    from td.research.screener import ScreenerClient
    client = ScreenerClient(cache_dir=cache, ttl_hours=24 * 30)
    todo = [s for s in syms if s not in have]
    print(f"fetching fundamentals for {len(todo)} stocks from Screener.in (≈{len(todo) * 6 / 60:.0f} min, "
          "cached afterwards) …")
    for i, s in enumerate(todo, 1):
        try:
            have[s] = await client.fundamentals(s)
        except Exception as e:                         # not on Screener, renamed, blocked: train without it
            log.info("fundamentals %s: %s", s, e)
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} done")
    await client.aclose()
    return have


def _table(meta: dict) -> str:
    rows = [f"{'horizon':<10}{'model':<11}{'AUC':>7}{'± SE':>8}{'IC':>8}{'edge':>7}   verdict"]
    for hz in meta["horizons"]:
        first = True
        for name in ("neural", "quantedge", "gbm"):
            m = hz["metrics"].get(name)
            if not m:
                continue
            auc = f"{m['auc']:.3f}" if m.get("auc") is not None else "—"
            se = f"{m['auc_se']:.3f}" if m.get("auc_se") is not None else "—"
            ic = f"{m['ic']:+.3f}" if m.get("ic") is not None else "—"
            rows.append(f"{hz['label'] if first else '':<10}{name:<11}{auc:>7}{se:>8}{ic:>8}"
                        f"{'yes' if m.get('edge') else 'no':>7}   {hz['verdict'] if first else ''}")
            first = False
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="training-data zip from the server, or its extracted folder")
    ap.add_argument("--out", default=None, help="output folder (default: <data>/neural)")
    ap.add_argument("--fetch-fundamentals", action="store_true", help="fetch missing fundamentals from Screener.in")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--no-gbm", action="store_true", help="skip the gradient-boosting baseline")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    root = _data_root(a.data)
    settings.data_dir = root
    settings.data_source = "dhan"
    from td.neural import dataset as ds
    from td.neural import news as nw
    from td.neural import train as tr

    t0 = time.time()
    frames, index_df = _frames(root)
    print(f"{len(frames)} stocks with daily bars, NIFTY {len(index_df)} days")
    # the PC keeps its own Screener cache next to the zips, so the next export does not start from scratch
    pc_cache = root.parent / "screener-cache"
    pc_cache.mkdir(parents=True, exist_ok=True)
    fund = _cached_fundamentals([root / "cache" / "screener", pc_cache], sorted(frames))
    if a.fetch_fundamentals:
        fund = asyncio.run(_fetch_fundamentals(pc_cache, sorted(ds.eligible(frames)), fund))
    print(f"fundamentals for {len(fund)} stocks")
    news_df = nw.load_archive(root / "recorder")
    start = nw.archive_start(news_df)
    print(f"news archive: {len(news_df):,} stock headlines"
          + (f" since {time.strftime('%Y-%m-%d', time.localtime(start))}" if start else " (none yet)"))

    P = ds.build(frames, index_df, fundamentals=fund, news=news_df, news_start=start,
                 progress=lambda f, m: print(f"  dataset {f * 100:3.0f}% {m}"))
    print(f"panel: {P.rows:,} rows × ({ds.SEQ_LEN} days × {P.Z.shape[1]} technical + {P.static.shape[1]} other "
          f"inputs), {len(P.folds)} walk-forward folds")
    for n in P.notes:
        print(f"  note: {n}")
    cfg = tr.TrainConfig(seeds=a.seeds, epochs=a.epochs, batch=a.batch, gbm=not a.no_gbm, device=a.device)
    ens, meta = tr.train(P, cfg, say=lambda f, m: print(f"  {f * 100:3.0f}% {m}"))
    out = Path(a.out) if a.out else root / "neural"
    onnx_path, meta_path = tr.export(ens, P, meta, out)
    print()
    print(_table(meta))
    print(f"\ntrained on {meta['device']} in {(time.time() - t0) / 60:.1f} min")
    print(f"\nmodel files:\n  {onnx_path}\n  {meta_path}")
    print("\nnext: upload both files to the server's data/neural/ folder (the NEURAL desk loads them within a "
          "minute).")


if __name__ == "__main__":
    main()
