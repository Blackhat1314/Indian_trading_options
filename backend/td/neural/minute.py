"""One row per stock and trading day, condensed from the lake's 1-minute bars.

Five years of 1-minute bars are ~470k rows per stock — too much to download and mostly not needed. What the two
neural models use from them fits in a compact table, built on the server (`research/neural_export.py`) and shipped
in the training zip as intraday/<SYMBOL>.parquet:

  for the 09:30 → close model (td.neural.intraday)
    o0…o14, h…, l…, c…, v…   the first 15 one-minute bars (09:15 … 09:29), NaN where a minute has no bar
    b0…b14                   the "usual" volume of those 15 bars: mean over the full sessions among the previous
                             14 days with minute data — the same positional curve the live ranker's RVOL uses
                             (td.analytics.context.load_context), so training and live agree
    px_lock                  close of the last bar before 09:30 = the price the picks are entered at
    close, post_high, post_low   last close of the session; high / low after the lock
  for the daily model (DAY_KEYS, known at that day's close)
    m_rvol       realised volatility of 1-minute returns
    m_rskew      realised skewness of 1-minute returns (Amaya et al. 2015: high skew → lower future returns)
    m_last30     return over the last 30 minutes (closing pressure)
    m_close_vwap close vs the day's VWAP
    m_late_vol   share of the day's volume in the last 30 minutes
    m_first30    return over the first 30 minutes

`day` is the IST date as days since 1970-01-01. Every value of a row uses that day's bars only, except b0…b14,
which use earlier days only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SESSION = 375                      # 09:15 … 15:29
FIRST = 15                         # bars before the 09:30 lock
IST_OFFSET = 19800
OPEN_S = 9 * 3600 + 15 * 60
BASE_DAYS = 14                     # sessions in the usual-volume curve (load_context's)
FULL_BARS = SESSION * 0.9          # a "full" session for that curve (build_base_curve's rule)
MIN_DAY_BARS = 300                 # a day's summaries need this many bars
DAY_KEYS: tuple[str, ...] = ("m_rvol", "m_rskew", "m_last30", "m_close_vwap", "m_late_vol", "m_first30")
DAY_LABEL = {"m_rvol": "Intraday volatility (1-min)", "m_rskew": "Intraday return skew (1-min)",
             "m_last30": "Last 30 minutes' move", "m_close_vwap": "Close vs VWAP",
             "m_late_vol": "Volume share of the last 30 minutes", "m_first30": "First 30 minutes' move"}
BAR_COLS = [f"{k}{i}" for k in "ohlcv" for i in range(FIRST)]
BASE_COLS = [f"b{i}" for i in range(FIRST)]
COLUMNS = (["day", "n", "pre", "post"] + BAR_COLS + BASE_COLS
           + ["open", "px_lock", "close", "post_high", "post_low", "vol_day"] + list(DAY_KEYS))


def ist_day(ts) -> np.ndarray:
    return (np.asarray(ts, dtype=np.int64) + IST_OFFSET) // 86400


def day_of_epoch(day: int) -> int:
    """IST midnight epoch of an IST day number."""
    return int(day) * 86400 - IST_OFFSET


def _ffill_idx(present: np.ndarray) -> np.ndarray:
    idx = np.where(present, np.arange(present.shape[1])[None, :], -1)
    return np.maximum.accumulate(idx, axis=1)


def day_table(m: pd.DataFrame) -> pd.DataFrame:
    """1-minute bars of one symbol (lake columns ts, open, high, low, close, volume) → the per-day table."""
    if m is None or m.empty:
        return pd.DataFrame(columns=COLUMNS)
    ts = m["ts"].to_numpy(dtype=np.int64)
    sod = (ts + IST_OFFSET) % 86400
    minute = (sod - OPEN_S) // 60
    keep = (minute >= 0) & (minute < SESSION) & np.isfinite(m["close"].to_numpy(dtype=float)) \
        & (m["close"].to_numpy(dtype=float) > 0)
    if not keep.any():
        return pd.DataFrame(columns=COLUMNS)
    ts, minute = ts[keep], minute[keep]
    vals = m[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)[keep]
    day = ist_day(ts)
    days, di = np.unique(day, return_inverse=True)
    nd = days.size
    A = np.full((nd, SESSION, 5), np.nan)
    A[di, minute] = vals
    o, h, l, c, v = (A[:, :, k] for k in range(5))
    for x in (o, h, l):                                  # a bar with only a close: the close stands in
        bad = ~np.isfinite(x) | (x <= 0)
        x[bad] = c[bad]
    v[np.isfinite(c) & ~np.isfinite(v)] = 0.0
    present = np.isfinite(c)
    n = present.sum(axis=1)
    rows = np.arange(nd)[:, None]
    fi = _ffill_idx(present)
    c_ff = np.where(fi >= 0, c[rows, np.maximum(fi, 0)], np.nan)
    first = np.argmax(present, axis=1)
    day_open = o[np.arange(nd), first]
    pre = present[:, :FIRST].sum(axis=1)
    post = present[:, FIRST:].sum(axis=1)
    px_lock = c_ff[:, FIRST - 1]
    close = c_ff[:, -1]
    with np.errstate(all="ignore"):
        post_high = np.nanmax(np.where(present[:, FIRST:], h[:, FIRST:], -np.inf), axis=1)
        post_low = np.nanmin(np.where(present[:, FIRST:], l[:, FIRST:], np.inf), axis=1)
        post_high[~np.isfinite(post_high)] = np.nan
        post_low[~np.isfinite(post_low)] = np.nan
        vol_day = np.nansum(v, axis=1)
        typical = (h + l + c) / 3.0
        vwap = np.nansum(typical * v, axis=1) / vol_day
        lc = np.log(c_ff)
        r = np.diff(lc, axis=1)
        r[~np.isfinite(r)] = 0.0
        nr = np.maximum((r != 0).sum(axis=1), 1)
        s2 = (r * r).sum(axis=1)
        rvol = np.sqrt(s2)
        rskew = np.sqrt(nr) * (r ** 3).sum(axis=1) / np.where(s2 > 0, s2 ** 1.5, np.nan)
        last30 = np.log(close / c_ff[:, SESSION - 31])
        late = np.nansum(v[:, SESSION - 30:], axis=1) / vol_day
        first30 = np.log(c_ff[:, 29] / day_open)
        close_vwap = np.log(close / vwap)
    summ = np.column_stack([rvol, rskew, last30, close_vwap, late, first30])
    summ[n < MIN_DAY_BARS] = np.nan
    summ[~np.isfinite(summ)] = np.nan

    # usual volume of the first 15 bars: positional, like load_context (k-th bar the day actually had)
    order = np.argsort(~present, axis=1, kind="stable")[:, :FIRST]
    vpos = np.take_along_axis(np.where(present, v, np.nan), order, axis=1)
    full = n >= FULL_BARS
    w = np.where(full[:, None], np.nan_to_num(vpos), 0.0)
    csum = np.vstack([np.zeros((1, FIRST)), np.cumsum(w, axis=0)])
    ccnt = np.r_[0, np.cumsum(full)]
    lo = np.maximum(np.arange(nd) - BASE_DAYS, 0)
    hi = np.arange(nd)
    cnt = ccnt[hi] - ccnt[lo]
    with np.errstate(all="ignore"):
        base = (csum[hi] - csum[lo]) / cnt[:, None]
    base[cnt == 0] = np.nan

    first15 = A[:, :FIRST, :]
    out = pd.DataFrame({"day": days.astype(np.int32), "n": n.astype(np.int16), "pre": pre.astype(np.int16),
                        "post": post.astype(np.int16)})
    bars = {f"{k}{i}": first15[:, i, j].astype(np.float32) for j, k in enumerate("ohlcv") for i in range(FIRST)}
    bcols = {f"b{i}": base[:, i].astype(np.float32) for i in range(FIRST)}
    tail = {"open": day_open, "px_lock": px_lock, "close": close, "post_high": post_high, "post_low": post_low,
            "vol_day": vol_day}
    out = pd.concat([out, pd.DataFrame(bars), pd.DataFrame(bcols), pd.DataFrame(tail),
                     pd.DataFrame(summ.astype(np.float32), columns=list(DAY_KEYS))], axis=1)
    return out[COLUMNS]


def daily_values(table: pd.DataFrame | None, bar_ts: np.ndarray) -> np.ndarray:
    """(n, len(DAY_KEYS)) day summaries for a stock's daily bars (NaN where the day has no minute data)."""
    ts = np.asarray(bar_ts, dtype=np.int64)
    out = np.full((ts.size, len(DAY_KEYS)), np.nan)
    if table is None or table.empty or ts.size == 0:
        return out
    td = table["day"].to_numpy(dtype=np.int64)
    j = np.searchsorted(td, ist_day(ts))
    j = np.minimum(j, td.size - 1)
    hit = td[j] == ist_day(ts)
    out[hit] = table[list(DAY_KEYS)].to_numpy(dtype=float)[j[hit]]
    return out


def first_bars(table: pd.DataFrame) -> np.ndarray:
    """(rows, 15, 5) o/h/l/c/v of the first 15 bars, NaN where missing."""
    a = table[BAR_COLS].to_numpy(dtype=float).reshape(len(table), 5, FIRST)
    return np.transpose(a, (0, 2, 1))
