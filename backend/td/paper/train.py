"""Build the intraday training set from the data lake and fit the decision models.

    python -m td.paper.train                 # train on all history, save to data/paper/models
    python -m td.paper.train --until 2025-09-30

Each row is one symbol on one day at one decision minute: its features and the gross return a long and a short
trade would have made from that minute to 15:15 under the structural stop (sim.py, pessimistic fills).
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ..lake import Lake
from .daydata import O, load_days
from .features import daily_context, intraday_features
from .intraday import DECISIONS, EXIT_MINUTE, fit_side, prepare, save_models, stop_distance
from .sim import ExitRules, simulate

log = logging.getLogger("td.paper.train")
INDEX = {"NIFTY", "BANKNIFTY", "FINNIFTY", "INDIAVIX", "SENSEX", "MIDCPNIFTY"}
MIN_DAYS = 120


def index_frames(lake: Lake):
    d, b, _ = load_days(lake, "NIFTY")
    D = lake.read("D", "NIFTY").sort_values("ts")
    ctx = daily_context(D)
    close = pd.Series(D["close"].to_numpy(float),
                      index=pd.to_datetime(D["ts"] + 19800, unit="s").dt.strftime("%Y%m%d").astype(int).to_numpy())
    return d, b, ctx, close


def symbol_rows(sym: str, root: str | None = None) -> pd.DataFrame | None:
    """Features + outcomes for every day of one symbol (both decision times)."""
    lake = Lake(root) if root else Lake()
    idates, ibars, ictx, iclose = index_frames(lake)
    dates, bars, printed = load_days(lake, sym)
    D = lake.read("D", sym)
    if D is None or len(D) < 60 or len(dates) < MIN_DAYS:
        return None
    ctx = daily_context(D, iclose)
    rules = ExitRules(exit_minute=EXIT_MINUTE)
    frames = []
    for m in DECISIONS:
        f = intraday_features(bars, m, ctx, dates, printed, ibars, ictx, idates)
        f["symbol"] = sym
        entry = bars[:, m, O].astype(float)
        atr = f["atr"].to_numpy(float)
        ok = np.isfinite(atr) & (atr > 0) & f["full_day"].to_numpy(bool) & np.isfinite(entry) & (entry > 0)
        f["entry"] = entry
        for side, key in ((1, "L"), (-1, "S")):
            dist = stop_distance(f, side, entry, m)
            sel = np.where(ok & np.isfinite(dist) & (dist > 0))[0]
            for col in ("y", "risk", "exit_min", "reason"):
                f[f"{col}_{key}"] = np.nan
            if len(sel):
                res = simulate(bars[sel], np.full(len(sel), m), np.full(len(sel), side), entry[sel] - side * dist[sel],
                               atr[sel], rules)
                f.loc[sel, f"y_{key}"] = res["r"] * res["risk"] / res["entry"]
                f.loc[sel, f"risk_{key}"] = dist[sel] / entry[sel]
                f.loc[sel, f"exit_min_{key}"] = res["exit_min"]
                f.loc[sel, f"reason_{key}"] = res["reason"]
        frames.append(f[ok])
    out = pd.concat(frames, ignore_index=True)
    fl = out.select_dtypes("float64").columns
    out[fl] = out[fl].astype("float32")
    return out


def build_dataset(lake: Lake | None = None, workers: int = 4, symbols: list[str] | None = None) -> pd.DataFrame:
    lake = lake or Lake()
    syms = symbols or [s for s in lake.symbols("1m") if s not in INDEX]
    t = time.time()
    parts = []
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for r in ex.map(symbol_rows, syms, [str(lake.root)] * len(syms), chunksize=2):
                if r is not None:
                    parts.append(r)
    else:
        parts = [r for r in (symbol_rows(s, str(lake.root)) for s in syms) if r is not None]
    if not parts:
        return pd.DataFrame()
    ds = prepare(pd.concat(parts, ignore_index=True))
    log.info("intraday dataset: %d rows, %d symbols, %.0fs", len(ds), ds["symbol"].nunique(), time.time() - t)
    return ds


def train(ds: pd.DataFrame, until: int | None = None) -> dict:
    """Fit long and short models for each decision minute on rows dated up to `until` (yyyymmdd)."""
    from .intraday import liquid
    models = {}
    for m in DECISIONS:
        d = ds[(ds["m"] == m)]
        if until:
            d = d[d["date"] <= until]
        d = d[liquid(d)].reset_index(drop=True)
        for key in ("L", "S"):
            y = d[f"y_{key}"].to_numpy(float)
            meta = {"m": m, "side": key, "train_from": int(d["date"].min()), "train_to": int(d["date"].max()),
                    "symbols": int(d["symbol"].nunique())}
            models[f"m{m}_{key}"] = fit_side(d, y, meta)
            log.info("trained m%d_%s on %d rows (%s..%s)", m, key, models[f"m{m}_{key}"].meta["n"],
                     meta["train_from"], meta["train_to"])
    return models


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--until", help="last training date, YYYY-MM-DD (default: all history)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", help="directory for the model files (default data/paper/models)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ds = build_dataset(workers=a.workers)
    if ds.empty:
        raise SystemExit("no 1-minute history in the lake: run the history pull (admin page) first")
    until = int(a.until.replace("-", "")) if a.until else None
    from pathlib import Path
    path = save_models(train(ds, until), Path(a.out) if a.out else None)
    print(f"models saved to {path}")


if __name__ == "__main__":
    main()
