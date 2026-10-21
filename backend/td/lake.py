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
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import settings

try:
    import fcntl
except ImportError:                    # Windows: in-process locking only
    fcntl = None

log = logging.getLogger("td.lake")
COLUMNS = ["ts", "open", "high", "low", "close", "volume", "oi"]
ROW_GROUP = 50_000      # rows per Parquet row group: reads of the last N bars touch only the last group or two
_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def _lock_for(p: Path) -> threading.Lock:
    with _guard:
        return _locks.setdefault(str(p), threading.Lock())


@contextmanager
def _file_lock(p: Path):
    """Exclusive lock across processes (the ingestor's auto-history and the backfill worker can both append to
    the same file); a no-op where flock is unavailable."""
    f = None
    if fcntl is not None:
        try:
            d = p.parent / ".locks"
            d.mkdir(exist_ok=True)
            f = open(d / f"{p.name}.lock", "a+")
            fcntl.flock(f, fcntl.LOCK_EX)
        except OSError:
            if f:
                f.close()
            f = None
    try:
        yield
    finally:
        if f:
            try:
                fcntl.flock(f, fcntl.LOCK_UN)
            finally:
                f.close()


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
        with _lock_for(p), _file_lock(p):       # one writer per file, across threads and processes
            if p.exists():
                old = pq.read_table(p).to_pandas()
                df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
            tmp = p.with_name(f"{p.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd",
                           row_group_size=ROW_GROUP)
            tmp.replace(p)
        return len(df)

    # ------------------------------------------------------------------- read
    def read(self, interval: str, symbol: str, start_ts: int | None = None, end_ts: int | None = None,
             last_n: int | None = None) -> pd.DataFrame:
        p = self.path(interval, symbol)
        if not p.exists():
            return pd.DataFrame(columns=COLUMNS)
        if last_n is not None and start_ts is None and end_ts is None:
            return _tail(p, last_n)
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
        _, _, last = file_stats(p)
        return last

    def first_ts(self, interval: str, symbol: str) -> int | None:
        p = self.path(interval, symbol)
        if not p.exists():
            return None
        try:
            _, first, _ = file_stats(p)
        except (OSError, ValueError, pa.ArrowException):       # being replaced right now
            return None
        return first

    def coverage(self) -> list[dict]:
        """Rows / first / last per interval and symbol, from Parquet footers (no data pages are read)."""
        out = []
        if not self.root.exists():
            return out
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for f in sorted(d.glob("*.parquet")):
                try:
                    rows, first, last = file_stats(f)
                except (OSError, ValueError, pa.ArrowException) as e:    # replaced or half-written: skip
                    log.debug("coverage: %s: %s", f, e)
                    continue
                if rows and first is not None:
                    out.append({"interval": d.name, "symbol": _unsafe(f.stem), "rows": rows, "first": first,
                                "last": last, "bytes": f.stat().st_size})
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


def file_stats(p: Path) -> tuple[int, int | None, int | None]:
    """(rows, first ts, last ts) of a lake file from its footer; falls back to reading the ts column."""
    with open(p, "rb") as fh:
        pf = pq.ParquetFile(fh)
        md = pf.metadata
        if md.num_rows == 0:
            return 0, None, None
        i = md.schema.names.index("ts")
        lo = hi = None
        for g in range(md.num_row_groups):
            st = md.row_group(g).column(i).statistics
            if st is None or not st.has_min_max:
                ts = pf.read(columns=["ts"]).column("ts").to_numpy()
                return md.num_rows, int(ts.min()), int(ts.max())
            lo = st.min if lo is None else min(lo, st.min)
            hi = st.max if hi is None else max(hi, st.max)
        return md.num_rows, int(lo), int(hi)


def _tail(p: Path, n: int) -> pd.DataFrame:
    """Last n rows (files are sorted by ts), reading only the trailing row groups."""
    with open(p, "rb") as fh:
        pf = pq.ParquetFile(fh)
        groups, rows = [], 0
        for g in range(pf.metadata.num_row_groups - 1, -1, -1):
            groups.insert(0, g)
            rows += pf.metadata.row_group(g).num_rows
            if rows >= n:
                break
        if not groups:
            return pd.DataFrame(columns=COLUMNS)
        df = pf.read_row_groups(groups).to_pandas()
    return df.tail(n).reset_index(drop=True) if n > 0 else df.iloc[0:0].reset_index(drop=True)
