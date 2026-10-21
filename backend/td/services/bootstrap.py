"""Automatic history for live mode (runs inside the ingestor, DATA_SOURCE=dhan).

The live desks need a little history before they can rank anything:
  previous close      — % change, gap, breadth, movers (Dhan's feed doesn't send it for every instrument)
  previous futures OI — the F&O build-up board (LB / SB / SC / LU)
  ~1 year daily bars  — ATR, beta, average volume, previous-day levels, every swing setup
  ~30 days of 1-min   — the time-of-day volume curve behind RVOL
  today's 1-min so far— opening range and charts when the app is started after 09:15

This pulls exactly that from Dhan's historical API, incrementally (only what the lake is
missing), through the same rate limiter and daily budget as everything else. Prev close / prev OI
go out as normal "prev" packets; analytics reloads its context when a stage finishes.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta

from .. import bus
from ..config import settings
from ..dhan.instruments import Instrument, Universe
from ..holidays import is_trading_day
from ..lake import Lake, bars_from_dhan
from ..ratelimit import BudgetExhausted
from ..timeutil import IST, from_epoch, minute_of_session, now_ist

log = logging.getLogger("td.bootstrap")

HIST = {  # kind/segment -> (exchangeSegment, instrument) for the historical API
    ("EQ", "NSE_EQ"): ("NSE_EQ", "EQUITY"),
    ("IDX", "IDX_I"): ("IDX_I", "INDEX"),
    ("FUT", "NSE_FNO"): ("NSE_FNO", "FUTSTK"),
    ("MACRO", "MCX_COMM"): ("MCX_COMM", "FUTCOM"),
    ("MACRO", "NSE_CURRENCY"): ("NSE_CURRENCY", "FUTCUR"),
}


def k_seed(key: int) -> str:
    return f"td:seed1m:{key}"


class HistoryBootstrap:
    def __init__(self, r, rest, universe: Universe, pub, lake: Lake | None = None):
        self.r, self.rest, self.universe, self.pub = r, rest, universe, pub
        self.lake = lake or Lake()
        self.status = {"stage": "waiting", "done": 0, "total": 0, "errors": 0, "updated": None, "last_error": ""}
        self.sem = asyncio.Semaphore(5)
        self.pub.stats["history"] = dict(self.status)       # visible at once (the backfill worker waits on it)

    # ------------------------------------------------------------ helpers
    def _st(self, stage: str, done: int = 0, total: int = 0):
        self.status.update(stage=stage, done=done, total=total, updated=time.time())
        self.pub.stats["history"] = dict(self.status)

    async def _call(self, coro):
        async with self.sem:
            try:
                return await coro
            except BudgetExhausted:
                raise
            except Exception as e:                       # one bad symbol must not stop the rest
                self.status["errors"] += 1
                self.status["last_error"] = str(e)[:160]
                log.debug("history call failed: %s", e)
                return None

    async def _run(self, stage: str, items: list, fn):
        self._st(stage, 0, len(items))
        done = 0

        async def one(x):
            nonlocal done
            await fn(x)
            done += 1
            if done % 10 == 0 or done == len(items):
                self._st(stage, done, len(items))
        await asyncio.gather(*(one(x) for x in items))

    async def _event(self, what: str):
        await self.r.xadd(bus.STREAM_EVENTS, {"j": bus.dumps({"type": "history", "what": what, "ts": time.time()})},
                          maxlen=2000, approximate=True)

    def _prev(self, inst: Instrument, pc: float | None, poi: int | None = None):
        if pc:
            self.pub.add([{"t": "prev", "seg": _code(inst.seg), "sid": inst.sid, "pc": round(float(pc), 2),
                           "poi": int(poi or 0)}])

    # ------------------------------------------------------------ stages
    async def daily(self, today: date):
        """Daily bars for equities, indices and macro futures (incremental) + their previous close."""
        insts = [i for i in self.universe.instruments if i.kind in ("EQ", "IDX")]

        async def one(i: Instrument):
            seg, kind = HIST[(i.kind, i.seg)]
            last = self.lake.last_ts("D", i.symbol)
            start = today - timedelta(days=settings.auto_history_days)
            if last is not None:
                start = max(start, from_epoch(last).date() + timedelta(days=1))
            while start < today and not is_trading_day(start):  # nothing new over a weekend or holiday
                start += timedelta(days=1)
            if start < today:
                payload = await self._call(self.rest.daily(i.sid, seg, kind, start, today))
                df = bars_from_dhan(payload or {})
                if not df.empty:
                    df = df[df["ts"].map(lambda t: from_epoch(t).date() < today)]
                    await asyncio.to_thread(self.lake.write, "D", i.symbol, df)
            d = self.lake.read("D", i.symbol, last_n=3)
            if not d.empty:
                d = d[d["ts"].map(lambda t: from_epoch(t).date() < today)]
                if not d.empty:
                    self._prev(i, d["close"].iat[-1])
        await self._run("daily bars", insts, one)

    async def futures_prev(self, today: date):
        """Yesterday's close and open interest for each near-month future (F&O build-up board)."""
        insts = [i for i in self.universe.instruments if i.kind in ("FUT", "MACRO") and (i.kind, i.seg) in HIST]

        async def one(i: Instrument):
            seg, kind = HIST[(i.kind, i.seg)]
            payload = await self._call(self.rest.daily(i.sid, seg, kind, today - timedelta(days=12), today, oi=True,
                                                       expiry_code=0))
            df = bars_from_dhan(payload or {})
            if df.empty:
                return
            df = df[df["ts"].map(lambda t: from_epoch(t).date() < today)]
            if not df.empty:
                self._prev(i, df["close"].iat[-1], int(df["oi"].iat[-1] or 0))
        await self._run("futures prev close / OI", insts, one)

    async def minutes(self, today: date):
        """1-minute bars for the last ~30 days (RVOL volume curve), only where the lake has fewer than 10 sessions."""
        insts = self.universe.equities()

        async def one(i: Instrument):
            # off the event loop: with years of 1-minute history a file read takes long enough to stall the feed
            m = await asyncio.to_thread(self.lake.read, "1m", i.symbol, None, None, 375 * 12)
            start = datetime.combine(today - timedelta(days=30), datetime.min.time())
            if not m.empty:
                days = {from_epoch(t).date() for t in m["ts"].iloc[::60]}
                last = from_epoch(int(m["ts"].iat[-1])).date()
                if len(days) >= 10 and last >= today - timedelta(days=4):
                    return
                if len(days) >= 10:
                    start = datetime.combine(last + timedelta(days=1), datetime.min.time())
            end = datetime.combine(today - timedelta(days=1), datetime.max.time()).replace(microsecond=0)
            if start >= end:
                return
            payload = await self._call(self.rest.intraday(i.sid, "NSE_EQ", "EQUITY", 1,
                                                          start.replace(hour=9, minute=0), end))
            df = bars_from_dhan(payload or {})
            if not df.empty:
                df = df[df["ts"].map(lambda t: from_epoch(t).date() < today)]
                await asyncio.to_thread(self.lake.write, "1m", i.symbol, df)
        await self._run("1-minute history", insts, one)

    async def today_so_far(self, today: date):
        """Today's 1-minute bars up to now, so a late start still has the opening range and full charts."""
        now = now_ist()
        if now.date() != today or now.time() < datetime.strptime("09:16", "%H:%M").time():
            return
        insts = self.universe.equities()

        async def one(i: Instrument):
            payload = await self._call(self.rest.intraday(
                i.sid, "NSE_EQ", "EQUITY", 1, datetime.combine(today, datetime.min.time()).replace(hour=9, minute=15),
                now.replace(tzinfo=None, microsecond=0)))
            df = bars_from_dhan(payload or {})
            if df.empty:
                return
            rows = []
            for t, o, h, l, c, v in zip(df["ts"], df["open"], df["high"], df["low"], df["close"], df["volume"]):
                dt = from_epoch(int(t))
                if dt.date() != today:
                    continue
                m = minute_of_session(dt)
                if 0 <= m < 375:
                    rows.append([m, float(o), float(h), float(l), float(c), int(v)])
            if rows:
                await self.r.set(k_seed(i.key), bus.dumps({"day": today.isoformat(), "bars": rows}), ex=86400)
        await self._run("today's 1-minute bars", insts, one)

    # ------------------------------------------------------------ orchestration
    async def run_once(self):
        today = now_ist().date()
        log.info("history bootstrap for %s: %d instruments", today, len(self.universe.instruments))
        try:
            await self.daily(today)
            await self._event("daily")
            await self.futures_prev(today)
            await self.today_so_far(today)
            await self._event("today")
            await self.minutes(today)
            await self._event("minutes")
            self._st("ready", 1, 1)
            log.info("history bootstrap done (%d failed calls)", self.status["errors"])
        except BudgetExhausted as e:
            self.status["last_error"] = f"daily data budget exhausted: {e}"
            self._st("budget exhausted")
            log.warning("history bootstrap stopped: %s", e)

    async def run(self, stop: asyncio.Event):
        if not settings.auto_history:
            self._st("disabled (AUTO_HISTORY=false)")
            return
        await asyncio.sleep(3)                              # let the feed connect first
        while not stop.is_set():
            await self.run_once()
            # next run tomorrow 08:40 IST (fresh daily bars, prev close and prev OI before the open)
            nxt = datetime.combine(now_ist().date() + timedelta(days=1), datetime.min.time()).replace(
                hour=8, minute=40, tzinfo=IST)
            wait = max(60.0, (nxt - now_ist()).total_seconds())
            try:
                await asyncio.wait_for(stop.wait(), timeout=wait)
            except asyncio.TimeoutError:
                pass


def _code(seg: str) -> int:
    from ..dhan.instruments import SEG_CODE
    return SEG_CODE.get(seg, 1)
