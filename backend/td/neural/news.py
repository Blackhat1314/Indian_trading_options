"""Daily news features from the recorder's archive (data/recorder/<day>/news-*.parquet).

The ingestor has recorded every Dhan headline since the app first ran live, so the archive grows on its own; this
module only reads it. A headline counts toward a day only when it arrived by that day's close (15:30 IST): a
prediction made at the close cannot have traded on news that came out in the evening (the label starts at that
close). Arrival time (`rt`, when the ingestor received it) is used, never Dhan's publish time — it is what the
app actually knew, and when.

Features per stock and day (`NEWS_KEYS`): how much news (decayed count over the last 3 trading days) and which
way it leans (decayed mean sign from the news engine's classifier), plus `news_on` — 1 on days the archive
covers, 0 before it started, so the network can tell "no news" from "no archive yet".
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import pyarrow.parquet as pq

from ..analytics.news import classify

log = logging.getLogger("td.neural.news")

NEWS_KEYS: tuple[str, ...] = ("news_n", "news_sent", "news_on")
NEWS_LABEL = {"news_n": "Headlines (last 3 days)", "news_sent": "News tone (last 3 days)",
              "news_on": "News archive covers the day"}
CLOSE_S = 15 * 3600 + 30 * 60          # 15:30 IST after the bar's IST midnight
DECAY = (1.0, 0.5, 0.25)               # weight of the day itself, the day before, two days before


def load_archive(recorder_root: Path | str, since_day: str | None = None) -> pd.DataFrame:
    """Recorded headlines as rows (rt, symbol, sign), one per (headline, symbol), duplicates dropped.
    `since_day` (YYYY-MM-DD) skips older day folders — the server only needs the last few days."""
    root = Path(recorder_root)
    rows: list[tuple[float, str, int, str]] = []
    days = sorted(d for d in root.iterdir() if d.is_dir() and (since_day is None or d.name >= since_day)) \
        if root.exists() else []
    files = [f for d in days for f in sorted(d.glob("news-*.parquet"))]
    for f in files:
        try:
            t = pq.read_table(f, columns=["rt", "json"]).to_pydict()
        except Exception as e:                      # a half-written file must not stop training
            log.debug("news archive %s unreadable: %s", f, e)
            continue
        for rt, js in zip(t["rt"], t["json"]):
            try:
                it = orjson.loads(js)
            except orjson.JSONDecodeError:
                continue
            if it.get("source") == "SIM":           # the simulator invents headlines about real names
                continue
            syms = [str(s).upper() for s in it.get("symbols") or [] if s]
            if not syms:
                continue
            sign = int(classify(it.get("title") or "", it.get("sentiment")).get("sign") or 0)
            key = str(it.get("id") or it.get("title") or "")
            for s in syms:
                rows.append((float(rt), s, sign, key))
    df = pd.DataFrame(rows, columns=["rt", "symbol", "sign", "key"])
    if df.empty:
        return df.drop(columns="key")
    df = df.sort_values("rt").drop_duplicates(["key", "symbol"], keep="first").drop(columns="key")
    return df.reset_index(drop=True)


def archive_start(df: pd.DataFrame) -> float | None:
    return float(df["rt"].min()) if df is not None and not df.empty else None


def daily_features(df: pd.DataFrame | None, symbol: str, bar_ts: np.ndarray, start: float | None) -> np.ndarray:
    """(n, len(NEWS_KEYS)) for one stock's daily bars (IST-midnight epochs)."""
    ts = np.asarray(bar_ts, dtype=np.int64)
    n = ts.size
    out = np.zeros((n, len(NEWS_KEYS)))
    if n == 0 or start is None:
        return out
    cut = ts + CLOSE_S                                # news up to this day's close
    on = cut >= start
    out[:, NEWS_KEYS.index("news_on")] = on.astype(float)
    if df is None or df.empty:
        return out
    mine = df[df["symbol"] == symbol.upper()]
    if mine.empty:
        return out
    # bucket each headline into the first bar whose close is at or after its arrival
    b = np.searchsorted(cut, mine["rt"].to_numpy(dtype=float), side="left")
    keep = b < n
    b, sign = b[keep], mine["sign"].to_numpy(dtype=float)[keep]
    cnt = np.bincount(b, minlength=n).astype(float)
    sgn = np.bincount(b, weights=sign, minlength=n)
    dc = np.zeros(n)
    ds = np.zeros(n)
    for lag, w in enumerate(DECAY):
        if lag:
            dc[lag:] += w * cnt[:-lag]
            ds[lag:] += w * sgn[:-lag]
        else:
            dc += w * cnt
            ds += w * sgn
    out[:, NEWS_KEYS.index("news_n")] = np.where(on, np.log1p(dc), 0.0)
    out[:, NEWS_KEYS.index("news_sent")] = np.where(on, ds / (dc + 1.0), 0.0)
    return out
