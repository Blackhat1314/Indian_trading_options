"""Load pre-open context for every instrument from the Parquet lake.

Daily bars  -> ATR, ADV, beta, previous-day levels, 20/252-day extremes
1-min bars  -> 14-session average cumulative-volume curve (for time-of-day RVOL)
If minute history is missing (e.g. backfill not run yet) we fall back to an ADV-scaled
U-shaped intraday curve so RVOL still works on day one.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from ..lake import Lake
from ..timeutil import IST
from .state import SESSION, HistoryContext, build_base_curve

log = logging.getLogger("td.context")


def default_curve(adv: float) -> list[float]:
    m = np.arange(SESSION, dtype=float)
    u = 0.62 + 2.6 * np.exp(-m / 18.0) + 1.3 * np.exp(-(SESSION - m) / 25.0)
    cum = np.cumsum(u / u.sum()) * adv
    return [float(x) for x in cum]


def _daily_before(df: pd.DataFrame, day: date) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = pd.Timestamp(day, tz=IST).timestamp()
    return df[df["ts"] < cutoff]


def load_context(lake: Lake, symbols: list[str], day: date, beta_ref: str = "NIFTY") -> tuple[dict, dict]:
    """Returns ({symbol: HistoryContext}, {symbol: daily DataFrame (before `day`)})."""
    ctx: dict[str, HistoryContext] = {}
    daily_all: dict[str, pd.DataFrame] = {}
    ref = _daily_before(lake.read("D", beta_ref, last_n=400), day)
    ref_ret = ref.set_index("ts")["close"].pct_change() if not ref.empty else None
    for sym in symbols:
        d = _daily_before(lake.read("D", sym, last_n=420), day)
        daily_all[sym] = d.reset_index(drop=True)
        c = HistoryContext()
        if len(d) >= 2:
            h, l, cl = d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
            pc = np.r_[cl[0], cl[:-1]]
            tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
            c.atr = float(pd.Series(tr).tail(14).mean())
            c.atr_pct = c.atr / cl[-1] * 100 if cl[-1] else 0
            c.adv = float(d["volume"].tail(20).mean())
            c.pdh, c.pdl, c.pdc = float(h[-1]), float(l[-1]), float(cl[-1])
            c.hh20 = float(h[-20:].max())
            c.ll20 = float(l[-20:].min())
            c.hh252 = float(h[-252:].max())
            c.prev_day_ret = float(cl[-1] / cl[-2] - 1) * 100
            rng = h[-1] - l[-1]
            c.prev_close_loc = float((cl[-1] - l[-1]) / rng) if rng > 0 else 0.5
            if ref_ret is not None:
                r = d.set_index("ts")["close"].pct_change()
                j = pd.concat([r, ref_ret], axis=1, join="inner").dropna().tail(60)
                if len(j) > 20 and j.iloc[:, 1].var() > 0:
                    c.beta = float(np.clip(j.cov().iloc[0, 1] / j.iloc[:, 1].var(), 0.2, 3.0))
        m = lake.read("1m", sym, last_n=375 * 16)
        if not m.empty:
            cutoff = pd.Timestamp(day, tz=IST).timestamp()
            m = m[m["ts"] < cutoff]
            m["day"] = pd.to_datetime(m["ts"], unit="s", utc=True).dt.tz_convert(IST).dt.date
            days = [g["volume"].tolist() for _, g in m.groupby("day")][-14:]
            c.base_cum = build_base_curve(days)
        if not c.base_cum and c.adv:
            c.base_cum = default_curve(c.adv)
        ctx[sym] = c
    return ctx, daily_all
