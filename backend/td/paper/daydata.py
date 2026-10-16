"""Minute bars as day x minute arrays (09:15 .. 15:29 IST, 375 minutes), the shape the intraday maths works on.

Backtests read years of 1-minute bars per symbol; turning them into one float32 block per symbol
(days x 375 x [open, high, low, close, volume]) once, and caching it, makes every later pass a numpy slice.
Missing minutes (no trade printed) carry the last close forward with zero volume.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings

SESSION = 375
OPEN_MIN = 9 * 60 + 15
O, H, L, C, V = range(5)
IST_OFFSET = 19800                       # seconds; IST has no DST


def day_tensor(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1-minute bars -> (dates yyyymmdd int32 [n], bars float32 [n, 375, 5], printed minutes per day int16 [n])."""
    if df is None or df.empty:
        return np.zeros(0, np.int32), np.zeros((0, SESSION, 5), np.float32), np.zeros(0, np.int16)
    ts = df["ts"].to_numpy(np.int64) + IST_OFFSET
    days = ts // 86400
    minute = (ts % 86400) // 60 - OPEN_MIN
    ok = (minute >= 0) & (minute < SESSION)
    days, minute = days[ok], minute[ok]
    vals = df[["open", "high", "low", "close", "volume"]].to_numpy(np.float64)[ok]
    uniq, di = np.unique(days, return_inverse=True)
    n = len(uniq)
    arr = np.full((n, SESSION, 5), np.nan, np.float32)
    arr[di, minute] = vals
    printed = np.zeros(n, np.int16)
    np.add.at(printed, di, 1)
    # carry the last close into empty minutes (open = high = low = close, no volume)
    c = pd.DataFrame(arr[:, :, C].T).ffill().bfill().to_numpy().T          # per day, along minutes
    first_open = pd.DataFrame(arr[:, :, O].T).bfill().to_numpy().T
    empty = np.isnan(arr[:, :, C])
    lead = empty & np.isnan(pd.DataFrame(arr[:, :, C].T).ffill().to_numpy().T)   # before the first print
    fill = np.where(lead, first_open, c)
    for k in (O, H, L, C):
        arr[:, :, k] = np.where(empty, fill, arr[:, :, k])
    arr[:, :, V] = np.where(empty, 0.0, arr[:, :, V])
    dates = pd.to_datetime(uniq, unit="D").strftime("%Y%m%d").astype(np.int32).to_numpy()
    return dates, arr, printed


def cache_dir() -> Path:
    return Path(settings.data_dir) / "cache" / "paper" / "days"


def load_days(lake, symbol: str, refresh: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cached day_tensor of a symbol's 1-minute history (rebuilt when the lake file is newer than the cache)."""
    src = lake.path("1m", symbol)
    p = cache_dir() / f"{symbol.replace('/', '_')}.npz"
    if not refresh and p.exists() and src.exists() and p.stat().st_mtime >= src.stat().st_mtime:
        z = np.load(p)
        return z["dates"], z["bars"], z["printed"]
    dates, bars, printed = day_tensor(lake.read("1m", symbol))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, dates=dates, bars=bars, printed=printed)
    tmp.replace(p)
    return dates, bars, printed
