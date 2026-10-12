"""Tick / news / CAS recorder.

Dhan keeps no tick or depth history, so everything the feed delivers is written to
Parquet (one folder per day). After a few months this becomes a proprietary dataset for
order-flow, news and closing-auction research.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from ..timeutil import from_epoch

log = logging.getLogger("td.recorder")

TICK_COLS = ["rt", "t", "seg", "sid", "ltp", "ltq", "ltt", "atp", "vol", "tbq", "tsq", "oi", "o", "h", "l"]


class Recorder:
    def __init__(self, root: Path, enabled: bool = True, flush_secs: float = 30.0):
        self.root = Path(root)
        self.enabled = enabled
        self.flush_secs = flush_secs
        self._ticks: list[dict] = []
        self._other: dict[str, list[dict]] = {"news": [], "cas": [], "depth": []}
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self.rows_written = 0

    def ticks(self, clock: float, packets: list[dict]) -> None:
        if not self.enabled:
            return
        with self._lock:
            for p in packets:
                if p.get("t") in ("quote", "full", "tick", "oi", "prev"):
                    self._ticks.append({c: p.get(c) for c in TICK_COLS} | {"rt": clock})

    def add(self, kind: str, clock: float, item: dict) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._other.setdefault(kind, []).append({"rt": clock, "json": orjson.dumps(item).decode()})

    def maybe_flush(self, clock: float, force: bool = False) -> None:
        if not self.enabled:
            return
        if not force and time.monotonic() - self._last < self.flush_secs:
            return
        self._last = time.monotonic()
        with self._lock:
            ticks, self._ticks = self._ticks, []
            other = {k: v for k, v in self._other.items() if v}
            self._other = {k: [] for k in self._other}
        day = from_epoch(clock).date().isoformat()
        stamp = from_epoch(clock).strftime("%H%M%S")
        d = self.root / day
        try:
            if ticks:
                d.mkdir(parents=True, exist_ok=True)
                cols = {c: [r.get(c) for r in ticks] for c in TICK_COLS}
                pq.write_table(pa.table(cols), d / f"ticks-{stamp}-{len(ticks)}.parquet", compression="zstd")
                self.rows_written += len(ticks)
            for kind, rows in other.items():
                d.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.table({"rt": [r["rt"] for r in rows], "json": [r["json"] for r in rows]}),
                               d / f"{kind}-{stamp}.parquet", compression="zstd")
        except Exception:
            log.exception("recorder flush failed")
