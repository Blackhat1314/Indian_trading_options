"""Per-instrument live state built from normalized feed packets.

Market time comes from the ingestor batch clock (receipt time in live mode, simulated
time in sim mode), so the same code runs live, accelerated or in replay.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..timeutil import from_epoch, minute_of_session

SESSION = 375


@dataclass
class HistoryContext:
    """Facts known before the open, loaded from the data lake."""
    atr: float = 0.0              # 14-day ATR (₹)
    atr_pct: float = 0.0
    adv: float = 0.0              # 20-day average volume (shares)
    beta: float = 1.0             # 60-day beta to NIFTY
    pdh: float = 0.0              # previous day high / low / close
    pdl: float = 0.0
    pdc: float = 0.0
    hh20: float = 0.0             # 20-day high / low (excl. today)
    ll20: float = 0.0
    hh252: float = 0.0
    base_cum: list[float] = field(default_factory=list)   # avg cumulative volume by session minute (len 375)
    prev_day_ret: float = 0.0
    prev_close_loc: float = 0.5   # where yesterday closed inside its range (0..1)


class InstState:
    __slots__ = ("sid", "seg", "symbol", "kind", "underlying", "sector", "lot", "pc", "poi", "ltp", "open", "high",
                 "low", "vol", "atp", "tbq", "tsq", "oi", "oih", "oil", "ts", "depth", "bars", "_last_vol",
                 "orh", "orl", "ctx", "first_ts", "lock_px", "max_since_lock", "min_since_lock", "dirty",
                 "cas", "updates")

    def __init__(self, sid: int, seg: str, symbol: str, kind: str, underlying: str, sector: str, lot: int = 1):
        self.sid, self.seg, self.symbol, self.kind = sid, seg, symbol, kind
        self.underlying, self.sector, self.lot = underlying, sector, lot
        self.pc = 0.0
        self.poi = 0
        self.ltp = 0.0
        self.open = self.high = self.low = 0.0
        self.vol = 0
        self.atp = 0.0
        self.tbq = self.tsq = 0
        self.oi = self.oih = self.oil = 0
        self.ts = 0.0
        self.first_ts = 0.0
        self.depth: list | None = None
        self.bars: dict[int, list] = {}          # minute -> [o,h,l,c,v]
        self._last_vol = 0
        self.orh = 0.0
        self.orl = 0.0
        self.ctx = HistoryContext()
        self.lock_px = 0.0
        self.max_since_lock = 0.0
        self.min_since_lock = 0.0
        self.dirty = False
        self.cas: dict | None = None
        self.updates = 0

    # --------------------------------------------------------------- update
    def apply(self, p: dict, clock: float, or_minutes: int = 15) -> None:
        t = p.get("t")
        if t == "prev":
            self.pc = p.get("pc") or self.pc
            self.poi = p.get("poi") or self.poi
            return
        if t == "oi":
            self.oi = p["oi"]
            self.oih = max(self.oih, self.oi)
            self.oil = self.oi if not self.oil else min(self.oil, self.oi)
            return
        if t == "cas":
            self.cas = p
            return
        ltp = p.get("ltp")
        if not ltp:
            return
        self.updates += 1
        self.ltp = ltp
        self.ts = clock
        if not self.first_ts:
            self.first_ts = clock
        if "o" in p and p["o"]:
            self.open = p["o"]
            self.high = p.get("h") or max(self.high, ltp)
            self.low = p.get("l") or (ltp if not self.low else min(self.low, ltp))
        else:
            if not self.open:
                self.open = ltp
            self.high = max(self.high, ltp)
            self.low = ltp if not self.low else min(self.low, ltp)
        if "vol" in p:
            self.vol = p["vol"]
            self.atp = p.get("atp") or self.atp
            self.tbq = p.get("tbq", self.tbq)
            self.tsq = p.get("tsq", self.tsq)
        if t == "full":
            self.oi = p.get("oi", self.oi)
            self.oih = p.get("oih", self.oih)
            self.oil = p.get("oil", self.oil)
            self.depth = p.get("depth")
        if self.lock_px:
            self.max_since_lock = max(self.max_since_lock, ltp)
            self.min_since_lock = min(self.min_since_lock, ltp) if self.min_since_lock else ltp
        self.dirty = True
        # minute bars
        m = minute_of_session(from_epoch(clock))
        if 0 <= m < SESSION:
            dv = max(0, self.vol - self._last_vol) if self._last_vol else 0
            self._last_vol = self.vol
            b = self.bars.get(m)
            if b is None:
                self.bars[m] = [ltp, ltp, ltp, ltp, dv]
            else:
                if ltp > b[1]:
                    b[1] = ltp
                if ltp < b[2]:
                    b[2] = ltp
                b[3] = ltp
                b[4] += dv
            if m < or_minutes:
                self.orh = max(self.orh, ltp)
                self.orl = ltp if not self.orl else min(self.orl, ltp)
        elif self.vol:
            self._last_vol = self.vol

    # -------------------------------------------------------------- derived
    @property
    def chg(self) -> float:
        return (self.ltp / self.pc - 1.0) * 100 if self.pc and self.ltp else 0.0

    @property
    def gap(self) -> float:
        return (self.open / self.pc - 1.0) * 100 if self.pc and self.open else 0.0

    @property
    def vwap(self) -> float:
        return self.atp or self.ltp

    @property
    def turnover_cr(self) -> float:
        return self.vwap * self.vol / 1e7

    @property
    def oi_chg_pct(self) -> float:
        return (self.oi / self.poi - 1.0) * 100 if self.poi and self.oi else 0.0

    def rvol(self, minute: float) -> float:
        base = self.ctx.base_cum
        if not base or minute < 0:
            if self.ctx.adv:
                frac = max(0.02, min(1.0, (minute + 1) / SESSION))
                return self.vol / (self.ctx.adv * frac) if self.ctx.adv * frac else 0.0
            return 0.0
        m = min(SESSION - 1, max(0, int(minute)))
        frac = minute - int(minute)
        b = base[m] if m == 0 else base[m - 1] + (base[m] - base[m - 1]) * max(0.0, min(1.0, frac))
        return self.vol / b if b > 0 else 0.0

    def bars_list(self) -> list[list]:
        return [[m, *b] for m, b in sorted(self.bars.items())]


def build_base_curve(minute_df_by_day: list[list[int]]) -> list[float]:
    """Average cumulative volume curve from a list of per-day minute volume arrays (len 375)."""
    if not minute_df_by_day:
        return []
    import numpy as np

    days = [list(v[:SESSION]) + [0] * (SESSION - len(v[:SESSION])) for v in minute_df_by_day
            if len(v) >= SESSION * 0.9]
    arr = np.array([np.cumsum(np.asarray(v, dtype=float)) for v in days])
    if arr.size == 0:
        return []
    return [float(x) for x in arr.mean(axis=0)]


def safe(x: float, nd: int = 2) -> float:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return 0.0
    return round(float(x), nd)
