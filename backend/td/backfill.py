"""History pulls from Dhan into the Parquet lake — the engine behind backfill/backfill.py (CLI) and the
admin page's backfill worker (td.services.backfiller, /admin → HISTORY).

Resumable across days. Every finished request is remembered in data/backfill/state.json as the date range
it covered, and a request is skipped (or shrunk to what is still missing) when earlier runs already covered
its range. Chunk boundaries are counted back from today, so they move every day: matching exact request
keys would download everything again tomorrow. Only settled days count as covered — yesterday and today
are always fetched again, because Dhan publishes a day's bars some time after the close.

Requests run newest-first and interleave symbols: a pull cut short by the daily quota leaves every symbol
with the most recent years, and concurrent writers never wait on the same Parquet file.
"""
from __future__ import annotations

import asyncio
import errno
import json
import logging
import math
import os
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from .config import settings
from .dhan.rest import DhanError
from .lake import Lake, bars_from_dhan
from .ratelimit import BudgetExhausted
from .timeutil import from_epoch

log = logging.getLogger("td.backfill")

INTERVALS = {"D": None, "1": 1, "5": 5, "15": 15, "25": 25, "60": 60}
LAKE_NAME = {"D": "D", "1": "1m", "5": "5m", "15": "15m", "25": "25m", "60": "60m"}
INTRADAY_CHUNK_DAYS = 90       # Dhan: max 90 days per intraday request
DAILY_CHUNK_DAYS = 3660       # one daily request returns 10+ years (no documented cap): 1 call per symbol
INTRADAY_MAX_YEARS = 5         # Dhan keeps 5 years of intraday bars
MAX_ATTEMPTS = 4               # per request, for rate-limit / network errors (then it is reported as failed)
EMPTY_OK_DAYS = 14             # a bar-less 200 for a longer range is suspicious: not recorded as covered…
EMPTY_STRIKES = 2              # …unless it ends before the stock's first trading day, or comes back empty again in a
                               # later run (Dhan really has nothing there): then it is recorded, not retried forever
TEST_SYMBOL = re.compile(r"NSETEST$")          # exchange test scrips Dhan lists among the F&O underlyings
AUTH_CODES = {"806", "807", "808", "809", "810", "DH-901", "DH-902", "DH-903"}
RATE_CODES = {"805", "DH-904", "RL001"}
NO_DATA_CODES = {"DH-907"}


def min_reserve() -> int:
    """Calls a pull may never touch: what the live app itself needs in a trading day (news headlines every
    20 s in market hours on the Pro plan, the 08:40 history refresh, a restart)."""
    return 2000 if settings.is_pro else 800


def state_dir() -> Path:
    return Path(settings.data_dir) / "backfill"


# ------------------------------------------------------------------ targets & jobs
@dataclass(frozen=True)
class Target:
    symbol: str
    sid: int
    seg: str
    instrument: str


@dataclass
class Job:
    symbol: str
    sid: int
    seg: str
    instrument: str
    interval: str
    start: str          # first day (inclusive)
    end: str            # last day (exclusive)

    @property
    def key(self) -> str:
        return f"{self.interval}|{self.symbol}|{self.start}|{self.end}"

    @property
    def label(self) -> str:
        return f"{self.symbol} {LAKE_NAME[self.interval]} {self.start} → {self.end}"


def _get(obj: Any, name: str) -> Any:
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def targets(instruments: Iterable[Any], symbols: set[str] | None = None, include_index: bool = True) -> list[Target]:
    """Stocks (NSE_EQ) and indices to pull, in universe order. Accepts Instrument objects or the JSON rows the
    ingestor publishes; exchange test scrips (011NSETEST …) are skipped."""
    out = []
    for i in instruments:
        kind, sym = _get(i, "kind"), _get(i, "symbol")
        if TEST_SYMBOL.search(sym) or (symbols and sym not in symbols):
            continue
        if kind == "EQ":
            out.append(Target(sym, int(_get(i, "sid")), "NSE_EQ", "EQUITY"))
        elif kind == "IDX" and include_index:
            out.append(Target(sym, int(_get(i, "sid")), "IDX_I", "INDEX"))
    return out


def settled_end(today: date) -> date:
    """Exclusive end of the days whose bars are final: everything before yesterday."""
    return today - timedelta(days=1)


class Coverage:
    """Date ranges [start, end) already fetched, per (interval, symbol), merged."""

    def __init__(self):
        self.ranges: dict[tuple[str, str], list[list[date]]] = {}

    def add(self, interval: str, symbol: str, start: date, end: date) -> None:
        if end <= start:
            return
        rs = sorted(self.ranges.get((interval, symbol), []) + [[start, end]])
        merged = [list(rs[0])]
        for s, e in rs[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        self.ranges[(interval, symbol)] = merged

    def missing(self, interval: str, symbol: str, start: date, end: date) -> tuple[date, date] | None:
        """The smallest single range that still needs fetching inside [start, end), or None if all covered."""
        gaps: list[tuple[date, date]] = []
        cur = start
        for s, e in self.ranges.get((interval, symbol), []):
            if e <= cur:
                continue
            if s >= end:
                break
            if s > cur:
                gaps.append((cur, s))
            cur = max(cur, e)
            if cur >= end:
                break
        if cur < end:
            gaps.append((cur, end))
        return (gaps[0][0], gaps[-1][1]) if gaps else None


def parse_key(key: str) -> tuple[str, str, date, date] | None:
    try:
        iv, sym, s, e = key.split("|")
        return iv, sym, date.fromisoformat(s), date.fromisoformat(e)
    except ValueError:
        return None


def coverage_from_state(st: dict, today: date) -> Coverage:
    """Done and empty requests as coverage. Ranges are clipped to settled days, which also drops the unsettled
    tail that requests recorded by older versions (exact keys up to tomorrow) claimed."""
    cov = Coverage()
    settled = settled_end(today)
    for bucket in ("done", "empty"):
        for key in st.get(bucket, {}):
            p = parse_key(key)
            if p:
                cov.add(p[0], p[1], p[2], min(p[3], settled))
    return cov


def make_jobs(tgts: list[Target], intervals: list[str], years: float, today: date, cov: Coverage | None = None,
              since: dict[tuple[str, str], date] | None = None, recent: bool = False) -> list[Job]:
    """Requests for [today - years, end), newest chunk first, symbols interleaved; with `cov`, ranges already
    covered are skipped and partly covered chunks shrink to the part still missing.

    `end` is the settled boundary (the day before yesterday, inclusive): the live app's auto-history keeps the
    last few days current, so a finished pull plans nothing. `recent=True` (CLI `update`) runs through today."""
    jobs: list[Job] = []
    end_all = today + timedelta(days=1) if recent else settled_end(today)
    for iv in intervals:
        if iv not in INTERVALS:
            raise ValueError(f"unknown interval {iv!r} (use {', '.join(INTERVALS)})")
        yrs = years if iv == "D" else min(years, INTRADAY_MAX_YEARS)
        start_all = today - timedelta(days=int(365.25 * yrs))
        step = timedelta(days=DAILY_CHUNK_DAYS if iv == "D" else INTRADAY_CHUNK_DAYS)
        chunks = []
        e = end_all
        while e > start_all:
            s = max(start_all, e - step)
            chunks.append((s, e))
            e = s
        for s, e in chunks:
            for t in tgts:
                lo = s
                if since and (iv, t.symbol) in since:
                    lo = max(lo, since[(iv, t.symbol)])
                if lo >= e:
                    continue
                rng = (lo, e) if cov is None else cov.missing(iv, t.symbol, lo, e)
                if rng:
                    jobs.append(Job(t.symbol, t.sid, t.seg, t.instrument, iv, rng[0].isoformat(), rng[1].isoformat()))
    return jobs


def summarize(jobs: list[Job], budget_left: int | None = None, reserve: int = 0, rate: float = 3.0) -> dict:
    """Plan numbers for the CLI and the admin desk."""
    by: dict[str, int] = {}
    for j in jobs:
        by[j.interval] = by.get(j.interval, 0) + 1
    n = len(jobs)
    per_day = max(1, settings.limit_data_per_day - reserve)
    today_calls = None if budget_left is None else max(0, budget_left - reserve)
    days = 0                                   # quota days the pull touches, today included
    if n:
        first = min(n, today_calls) if today_calls is not None else min(n, per_day)
        days = (1 if first else 0) + math.ceil((n - first) / per_day)
    # each intraday request takes ~2.5 s at Dhan, so throughput tops out near concurrency / 2.5 s
    eff = min(rate, concurrency_for(rate) / 2.5)
    return {"requests": n, "by_interval": by, "calls_left_today": today_calls, "per_day": per_day,
            "days_of_quota": days, "run_minutes": round(n / max(0.05, eff) / 60, 1)}


def concurrency_for(rate: float) -> int:
    return max(2, min(8, math.ceil(rate * 2.5)))


# ------------------------------------------------------------------ state file
def load_state(path: Path | None = None) -> dict:
    p = path or state_dir() / "state.json"
    st = {"done": {}, "empty": {}, "failed": {}}
    if p.exists():
        try:
            st.update(json.loads(p.read_text()))
        except (OSError, ValueError):
            bad = p.with_suffix(f".bad.{int(time.time())}")
            p.replace(bad)                      # kept for inspection, never overwritten by the next save
            log.warning("unreadable %s — moved to %s, starting a fresh backfill state", p, bad.name)
    for k in ("done", "empty", "failed"):
        st.setdefault(k, {})
    return st


def compact(st: dict, today: date) -> dict:
    """Merge adjacent/overlapping recorded ranges (state grows by one small range per symbol per day otherwise)."""
    settled = settled_end(today)
    for bucket in ("done", "empty"):
        groups: dict[tuple[str, str], list[tuple[date, date, int]]] = {}
        keep: dict[str, Any] = {}
        for key, v in st.get(bucket, {}).items():
            p = parse_key(key)
            if not p:
                keep[key] = v
                continue
            iv, sym, s, e = p
            e = min(e, settled)
            if e > s:
                groups.setdefault((iv, sym), []).append((s, e, int(v) if isinstance(v, (int, float)) else 1))
        for (iv, sym), rs in groups.items():
            rs.sort()
            cur = list(rs[0])
            for s, e, v in rs[1:]:
                if s <= cur[1]:
                    cur[1] = max(cur[1], e)
                    cur[2] += v
                else:
                    keep[f"{iv}|{sym}|{cur[0]}|{cur[1]}"] = cur[2]
                    cur = [s, e, v]
            keep[f"{iv}|{sym}|{cur[0]}|{cur[1]}"] = cur[2]
        st[bucket] = keep
    return st


def save_state(st: dict, path: Path | None = None) -> None:
    p = path or state_dir() / "state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")       # per process: two writers never share a temp file
    tmp.write_text(json.dumps(st))
    tmp.replace(p)


def record(st: dict, bucket: str, job: Job, value: int, today: date) -> None:
    """Remember the settled part of a finished request."""
    s, e = date.fromisoformat(job.start), min(date.fromisoformat(job.end), settled_end(today))
    if e > s:
        st[bucket][f"{job.interval}|{job.symbol}|{s}|{e}"] = value


# ------------------------------------------------------------------ run
class Stopped(Exception):
    """Raised by a gate to end the run (stop requested)."""


@dataclass
class Progress:
    total: int = 0
    done: int = 0
    ok: int = 0
    empty: int = 0
    failed: int = 0
    retried: int = 0
    rows: int = 0
    started: float = field(default_factory=time.time)
    current: str = ""
    by_interval: dict = field(default_factory=dict)      # interval -> [done, total]
    last_error: str = ""
    errors: list = field(default_factory=list)          # recent failures [ts, label, message]
    recent: list = field(default_factory=list)          # completion times of the last requests (speed)

    def tick(self) -> None:
        self.recent = (self.recent + [time.time()])[-60:]

    def as_dict(self) -> dict:
        d = asdict(self)
        del d["recent"]
        # speed over the last minutes only, so pauses (market hours, live app busy) don't skew it
        now = time.time()
        r = [t for t in self.recent if now - t < 180]
        d["rate"] = round((len(r) - 1) / max(1e-6, r[-1] - r[0]), 2) if len(r) >= 2 else 0.0
        left = self.total - self.done
        d["eta_s"] = round(left / d["rate"]) if d["rate"] > 0 and left > 0 else None
        return d


class Pacer:
    """Evenly spaced requests (no bursts) at a rate that may change between calls; cooldown after a 429."""

    def __init__(self):
        self.next = 0.0
        self.lock = asyncio.Lock()
        self.backoff = 5.0

    async def wait(self, rate: float) -> None:
        async with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + 1.0 / max(0.05, rate)
        await asyncio.sleep(max(0.0, t - now))

    def cooldown(self) -> float:
        wait = self.backoff
        self.next = max(self.next, time.monotonic() + wait)
        self.backoff = min(60.0, self.backoff * 2)
        return wait

    def relax(self) -> None:
        self.backoff = 5.0


def classify(e: DhanError) -> str:
    code = str(getattr(e, "code", "") or "")
    status = getattr(e, "status", None)
    msg = str(e).lower()
    if code in AUTH_CODES:
        return "auth"
    if code in NO_DATA_CODES or "no data present" in msg:
        return "nodata"
    if status == 429 or code in RATE_CODES or "retries exhausted" in msg:
        return "rate"
    if "network error" in msg or (isinstance(status, int) and status >= 500):
        return "network"
    return "error"


async def _first(coro, ended: asyncio.Event):
    """Result of `coro`, or None as soon as `ended` is set (the coroutine is cancelled)."""
    task = asyncio.ensure_future(coro)
    waiter = asyncio.ensure_future(ended.wait())
    try:
        done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        return None
    finally:
        for t in (task, waiter):
            if not t.done():
                t.cancel()


async def fetch(rest, job: Job, retries: int = 1) -> dict:
    s, e = date.fromisoformat(job.start), date.fromisoformat(job.end)
    if job.interval == "D":
        return await rest.daily(job.sid, job.seg, job.instrument, s, e, retries=retries)
    last = e - timedelta(days=1)
    return await rest.intraday(job.sid, job.seg, job.instrument, INTERVALS[job.interval],
                               datetime(s.year, s.month, s.day, 9, 0), datetime(last.year, last.month, last.day,
                                                                                 23, 59, 59), retries=retries)


async def run_jobs(jobs: list[Job], rest, lake: Lake, st: dict, *, today: date, rate: float = 3.0,
                   reserve: int = 800, concurrency: int | None = None, stop: asyncio.Event | None = None,
                   gate: Callable[[], Awaitable[float]] | None = None,
                   on_progress: Callable[[Progress], Any] | None = None,
                   save: Callable[[dict], None] = save_state) -> tuple[str, Progress]:
    """Run requests until done, stopped or out of quota. Returns (reason, progress) where reason is one of
    complete | reserve | budget | stopped | auth: … | disk: …

    `gate` is awaited before every request and returns the rate to use now (it may sleep to pause, or raise
    Stopped); without it the fixed `rate` applies. Requests stop when the shared daily budget would drop
    below `reserve` calls."""
    prog = Progress(total=len(jobs))
    for j in jobs:
        prog.by_interval.setdefault(j.interval, [0, 0])[1] += 1
    q: deque[Job] = deque(jobs)
    attempts: dict[str, int] = {}
    pacer = Pacer()
    reason: list[str] = []
    ended = asyncio.Event()                 # set with the first reason: wakes workers parked in a paused gate
    st["failed"] = {}
    if on_progress:
        on_progress(prog)                    # totals are known before the first request (which may wait on a gate)

    def finish(job: Job) -> None:
        prog.done += 1
        prog.tick()
        prog.by_interval[job.interval][0] += 1
        if prog.done % 10 == 0:
            save(st)
        if on_progress:
            on_progress(prog)

    first_daily: dict[str, date | None] = {}
    strikes: dict[str, int] = st.setdefault("empty_strikes", {})

    def listed_on(symbol: str) -> date | None:
        """The stock's first daily bar in the lake: its first trading day (listing) once the daily pull, which runs
        newest chunk first and before the 1-minute requests, has reached it."""
        if symbol not in first_daily:
            try:
                d = lake.read(LAKE_NAME["D"], symbol)
            except Exception:
                d = None
            first_daily[symbol] = (from_epoch(int(d["ts"].min())).date()
                                   if d is not None and len(d) else None)
        return first_daily[symbol]

    def empty_long(job: Job) -> None:
        """A bar-less answer for a long range: before the stock existed → nothing to fetch, recorded; Dhan having
        nothing there twice in separate runs → recorded; otherwise retried next run (it can be a passing hiccup)."""
        first = listed_on(job.symbol)
        if first and date.fromisoformat(job.end) <= first:
            record(st, "empty", job, 1, today)
            prog.empty += 1
            strikes.pop(job.key, None)
            log.info("%s: before %s's first trading day %s — nothing to fetch", job.label, job.symbol, first)
        elif strikes.get(job.key, 0) + 1 >= EMPTY_STRIKES:
            record(st, "empty", job, 1, today)
            prog.empty += 1
            strikes.pop(job.key, None)
            log.info("%s: empty again — Dhan has no data for this range, recorded as empty", job.label)
        else:
            strikes[job.key] = strikes.get(job.key, 0) + 1
            fail(job, "empty response for a long range (retried next run)")

    def fail(job: Job, msg: str) -> None:
        st["failed"][job.key] = msg[:200]
        prog.failed += 1
        prog.last_error = msg[:200]
        prog.errors = (prog.errors + [[time.time(), job.label, msg[:200]]])[-20:]
        log.warning("%s failed: %s", job.label, msg[:200])

    async def worker() -> None:
        while q and not reason:
            if stop is not None and stop.is_set():
                reason.append("stopped")
                ended.set()
                return
            job = q.popleft()
            try:
                now_rate = await _first(gate(), ended) if gate else rate
            except Stopped:
                q.appendleft(job)
                reason.append("stopped")
                ended.set()
                return
            if reason or now_rate is None:
                q.appendleft(job)
                return
            await pacer.wait(now_rate)
            # checked right before the call; concurrent workers can still overshoot the reserve by a few calls
            if reason or await rest.data_budget.remaining() <= reserve:
                q.appendleft(job)
                if not reason:
                    reason.append("reserve")
                ended.set()
                return
            prog.current = job.label
            try:
                df = bars_from_dhan(await fetch(rest, job))
                if df.empty and (date.fromisoformat(job.end) - date.fromisoformat(job.start)).days > EMPTY_OK_DAYS:
                    await asyncio.to_thread(listed_on, job.symbol)        # file read off the event loop
                    empty_long(job)
                elif df.empty:
                    record(st, "empty", job, 1, today)
                    prog.empty += 1
                else:
                    await asyncio.to_thread(lake.write, LAKE_NAME[job.interval], job.symbol, df)
                    if job.interval == "D":                              # keep the first-trading-day view current
                        first = from_epoch(int(df["ts"].min())).date()
                        known = first_daily.get(job.symbol)
                        if job.symbol in first_daily and (known is None or first < known):
                            first_daily[job.symbol] = first
                    strikes.pop(job.key, None)
                    record(st, "done", job, len(df), today)
                    prog.ok += 1
                    prog.rows += len(df)
                pacer.relax()
            except BudgetExhausted:
                q.appendleft(job)
                reason.append("budget")
                ended.set()
                return
            except DhanError as e:
                kind = classify(e)
                if kind == "nodata":
                    record(st, "empty", job, 1, today)
                    prog.empty += 1
                elif kind == "auth":
                    q.appendleft(job)
                    reason.append(f"auth: {e}")
                    ended.set()
                    return
                elif kind in ("rate", "network") and attempts.get(job.key, 0) < MAX_ATTEMPTS:
                    attempts[job.key] = attempts.get(job.key, 0) + 1
                    prog.retried += 1
                    if kind == "rate":
                        log.info("rate limited — cooling down %.0f s", pacer.cooldown())
                    q.append(job)                    # try again later in this run
                    continue
                else:
                    fail(job, str(e))
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    q.appendleft(job)
                    reason.append(f"disk: {e}")
                    ended.set()
                    return
                fail(job, f"write failed: {e}")
            except Exception as e:                    # one bad response must not stop the pull
                log.exception("%s", job.label)
                fail(job, f"{type(e).__name__}: {e}")
            finish(job)

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency or concurrency_for(rate))]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:                       # never leave siblings running on a client the caller is closing
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        save(st)
    prog.current = ""
    if on_progress:
        on_progress(prog)
    return (reason[0] if reason else "complete"), prog


# ------------------------------------------------------------------ lake coverage (admin desk)
def lake_coverage(lake: Lake, intervals: Iterable[str] = ("D", "1m")) -> dict:
    """Per-interval summary and per-symbol first/last/rows (Parquet footers: milliseconds, no data read)."""
    want = set(intervals)
    out: dict = {"intervals": {}, "symbols": {}, "updated": time.time()}
    for c in lake.coverage():
        iv = c["interval"]
        if iv not in want:
            continue
        out["symbols"].setdefault(c["symbol"], {})[iv] = [c["first"], c["last"], c["rows"]]
        tot = out["intervals"].setdefault(iv, {"symbols": 0, "rows": 0, "first": None, "last": None, "bytes": 0})
        tot["symbols"] += 1
        tot["rows"] += c["rows"]
        tot["bytes"] += c["bytes"]
        tot["first"] = c["first"] if tot["first"] is None else min(tot["first"], c["first"])
        tot["last"] = c["last"] if tot["last"] is None else max(tot["last"], c["last"])
    return out


# ------------------------------------------------------------------ admin job spec
@dataclass
class JobSpec:
    """What the admin desk asked for. Lives in data/backfill/job.json so a redeploy resumes it."""
    intervals: list[str] = field(default_factory=lambda: ["D", "1"])
    years: float = 5.0
    rate: float = 3.0
    reserve: int = 2500                 # the live app itself uses ~2,000 calls a trading day (news, 08:40 refresh)
    market_hours: str = "slow"          # slow (1 req/s) | pause | full
    symbols: list[str] = field(default_factory=list)       # empty = whole F&O universe
    active: bool = True
    created: float = field(default_factory=time.time)
    by: str = "local"

    @classmethod
    def clean(cls, d: dict) -> "JobSpec":
        """Validate and clamp user input; raises ValueError with a readable message."""
        ivs = [str(x) for x in (d.get("intervals") or [])]
        if not ivs or any(x not in INTERVALS for x in ivs):
            raise ValueError(f"intervals must be some of {', '.join(INTERVALS)}")
        ivs = sorted(set(ivs), key=list(INTERVALS).index)
        try:
            years = float(d.get("years", 5))
            rate = float(d.get("rate", 3))
            reserve = int(d.get("reserve", 2500))
        except (TypeError, ValueError) as e:
            raise ValueError("years, rate and reserve must be numbers") from e
        if not 0.05 <= years <= 20:
            raise ValueError("years must be between 0.05 and 20")
        if not 0.2 <= rate <= 5:
            raise ValueError("rate must be between 0.2 and 5 requests per second (Dhan allows 5)")
        if not min_reserve() <= reserve < settings.limit_data_per_day:
            raise ValueError(f"reserve must be between {min_reserve()} (what the live app itself uses in a day) "
                             f"and {settings.limit_data_per_day - 1}")
        mh = str(d.get("market_hours", "slow"))
        if mh not in ("slow", "pause", "full"):
            raise ValueError("market_hours must be slow, pause or full")
        syms = []
        for s in d.get("symbols") or []:
            s = str(s).strip().upper()
            if s and not re.fullmatch(r"[A-Z0-9&\-]{1,30}", s):
                raise ValueError(f"bad symbol {s!r}")
            if s:
                syms.append(s)
        if len(syms) > 1000:
            raise ValueError("at most 1000 symbols")
        return cls(intervals=ivs, years=years, rate=rate, reserve=reserve, market_hours=mh,
                   symbols=sorted(set(syms)), active=bool(d.get("active", True)),
                   created=float(d.get("created") or time.time()), by=str(d.get("by", "local"))[:40])

    def as_dict(self) -> dict:
        return asdict(self)


def load_job(path: Path | None = None) -> JobSpec | None:
    p = path or state_dir() / "job.json"
    if not p.exists():
        return None
    try:
        return JobSpec.clean(json.loads(p.read_text()))
    except (OSError, ValueError) as e:
        log.warning("ignoring %s: %s", p, e)
        return None


def save_job(job: JobSpec | None, path: Path | None = None) -> None:
    p = path or state_dir() / "job.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if job is None:
        p.unlink(missing_ok=True)
        return
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(job.as_dict(), indent=1))
    tmp.replace(p)


# ------------------------------------------------------------------ one pull at a time
LOCK_KEY = "td:admin:backfill:lock"


class RunLock:
    """Redis lease so the admin worker and the CLI never pull (and write the same files) at the same time."""

    def __init__(self, r, owner: str, ttl: int = 90):
        self.r, self.owner, self.ttl = r, owner, ttl

    async def acquire(self) -> bool:
        return bool(await self.r.set(LOCK_KEY, self.owner, nx=True, ex=self.ttl))

    async def holder(self) -> str | None:
        v = await self.r.get(LOCK_KEY)
        return v.decode() if isinstance(v, bytes) else v

    async def refresh(self) -> bool:
        """Extend the lease; if Redis lost the key (restart) take it back unless someone else holds it now."""
        holder = await self.holder()
        if holder == self.owner:
            await self.r.expire(LOCK_KEY, self.ttl)
            return True
        if holder is None:
            return await self.acquire()
        return False

    async def release(self) -> None:
        if await self.holder() == self.owner:
            await self.r.delete(LOCK_KEY)
