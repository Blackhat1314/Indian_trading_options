"""Parquet data lake shared by the backfill tool, analytics and research scripts.

Layout (one file per symbol per interval, append-merge with de-duplication):
    <lake>/<interval>/<SYMBOL>.parquet       interval ∈ D, 1m, 5m, 15m, 60m, opt
Columns: ts (int64 epoch seconds, IST-aware when converted), open, high, low, close,
volume, oi.  DuckDB is used for fast cross-symbol queries.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import settings

log = logging.getLogger("td.lake")
COLUMNS = ["ts", "open", "high", "low", "close", "volume", "oi"]
_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def _lock_for(p: Path) -> threading.Lock:
    with _guard:
        return _locks.setdefault(str(p), threading.Lock())


def _safe(symbol: str) -> str:
    """File-name escape that round-trips (M&M -> M%26M); plain names stay readable."""
    return symbol.replace("%", "%25").replace("&", "%26").replace("/", "%2F")


def _unsafe(stem: str) -> str:
    return stem.replace("%2F", "/").replace("%26", "&").replace("%25", "%")


class Lake:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or settings.lake_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, interval: str, symbol: str) -> Path:
        p = self.root / interval / f"{_safe(symbol)}.parquet"
        if ("&" in symbol or "/" in symbol) and not p.exists():
            # files written by early versions used "AND" for "&" (ambiguous with BANDHANBNK) — migrate
            legacy = self.root / interval / f"{symbol.replace('/', '_').replace('&', 'AND')}.parquet"
            if legacy.exists():
                legacy.rename(p)
        return p

    # ------------------------------------------------------------------ write
    def write(self, interval: str, symbol: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        df = df.copy()
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = 0 if c in ("volume", "oi") else np.nan
        df = df[COLUMNS]
        df["ts"] = df["ts"].astype("int64")
        for c in ("open", "high", "low", "close"):
            df[c] = df[c].astype("float64")
        df["volume"] = df["volume"].fillna(0).astype("int64")
        df["oi"] = df["oi"].fillna(0).astype("int64")
        p = self.path(interval, symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        with _lock_for(p):                      # one writer per file (backfill runs concurrent workers)
            if p.exists():
                old = pq.read_table(p).to_pandas()
                df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
            tmp = p.with_name(f"{p.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
            tmp.replace(p)
        return len(df)

    # ------------------------------------------------------------------- read
    def read(self, interval: str, symbol: str, start_ts: int | None = None, end_ts: int | None = None,
             last_n: int | None = None) -> pd.DataFrame:
        p = self.path(interval, symbol)
        if not p.exists():
            return pd.DataFrame(columns=COLUMNS)
        filters = []
        if start_ts is not None:
            filters.append(("ts", ">=", int(start_ts)))
        if end_ts is not None:
            filters.append(("ts", "<", int(end_ts)))
        df = pq.read_table(p, filters=filters or None).to_pandas()
        if last_n is not None:
            df = df.tail(last_n)
        return df.reset_index(drop=True)

    def symbols(self, interval: str) -> list[str]:
        d = self.root / interval
        if not d.exists():
            return []
        return sorted(_unsafe(p.stem) for p in d.glob("*.parquet"))

    def last_ts(self, interval: str, symbol: str) -> int | None:
        p = self.path(interval, symbol)
        if not p.exists():
            return None
        t = pq.read_table(p, columns=["ts"])
        if t.num_rows == 0:
            return None
        return int(t.column("ts").to_numpy().max())

    def coverage(self) -> list[dict]:
        """Rows / first / last per interval and symbol (DuckDB, fast)."""
        import duckdb

        out = []
        if not self.root.exists():
            return out
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            files = list(d.glob("*.parquet"))
            if not files:
                continue
            con = duckdb.connect()
            q = (f"select filename, count(*) as n, min(ts) as t_first, max(ts) as t_last "
                 f"from read_parquet('{d.as_posix()}/*.parquet', filename=true) group by filename")
            for fn, n, first, last in con.execute(q).fetchall():
                out.append({"interval": d.name, "symbol": Path(fn).stem.replace("AND", "&"), "rows": int(n),
                            "first": int(first), "last": int(last)})
            con.close()
        return out

    def summary(self) -> dict:
        cov = self.coverage()
        by: dict[str, dict] = {}
        for c in cov:
            s = by.setdefault(c["interval"], {"symbols": 0, "rows": 0, "first": None, "last": None})
            s["symbols"] += 1
            s["rows"] += c["rows"]
            s["first"] = c["first"] if s["first"] is None else min(s["first"], c["first"])
            s["last"] = c["last"] if s["last"] is None else max(s["last"], c["last"])
        return by


def bars_from_dhan(payload: dict) -> pd.DataFrame:
    """Dhan historical response (column arrays) -> DataFrame in lake format."""
    if not payload or "timestamp" not in payload:
        return pd.DataFrame(columns=COLUMNS)
    n = len(payload["timestamp"])
    df = pd.DataFrame({
        "ts": payload["timestamp"],
        "open": payload.get("open", [np.nan] * n),
        "high": payload.get("high", [np.nan] * n),
        "low": payload.get("low", [np.nan] * n),
        "close": payload.get("close", [np.nan] * n),
        "volume": payload.get("volume", [0] * n),
        "oi": payload.get("open_interest") or [0] * n,
    })
    return df
