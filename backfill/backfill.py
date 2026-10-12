#!/usr/bin/env python3
"""TradeDesk backfill — pulls up to 5 years of Dhan history into the shared Parquet lake.

Separate from the live system on purpose: run it once (it resumes automatically across
days because of Dhan's 7,000 data-calls/day limit), then `update` daily after the close.
The lake is read by the live analytics, the swing scanner, the research scripts — and by
anything else you build (it's plain Parquet).

  python backfill/backfill.py plan    --intervals D,1 --years 5           # calls needed + ETA
  python backfill/backfill.py run     --intervals D,1,5 --years 5         # resumable
  python backfill/backfill.py update                                      # incremental top-up
  python backfill/backfill.py options --underlyings NIFTY,BANKNIFTY --strikes 2 --years 1
  python backfill/backfill.py status                                      # lake coverage
  python backfill/backfill.py sim                                         # synthetic lake (no Dhan)

Rate limits honoured: 5 req/s, daily budget (default 7,000, shared through Redis with the
live ingestor) minus a reserve you choose with --reserve.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from td.config import settings  # noqa: E402
from td.dhan.instruments import Universe, dhan_universe, sim_universe  # noqa: E402
from td.lake import Lake, bars_from_dhan  # noqa: E402
from td.ratelimit import BudgetExhausted  # noqa: E402
from td.timeutil import IST, now_ist  # noqa: E402

log = logging.getLogger("backfill")
STATE_DIR = Path(settings.data_dir) / "backfill"
INTERVALS = {"D": None, "1": 1, "5": 5, "15": 15, "25": 25, "60": 60}
LAKE_NAME = {"D": "D", "1": "1m", "5": "5m", "15": "15m", "25": "25m", "60": "60m"}
INTRADAY_CHUNK_DAYS = 90       # Dhan: max 90 days per intraday request
DAILY_CHUNK_DAYS = 365


@dataclass
class Job:
    symbol: str
    sid: int
    seg: str
    instrument: str
    interval: str
    start: str
    end: str

    @property
    def key(self) -> str:
        return f"{self.interval}|{self.symbol}|{self.start}|{self.end}"


def load_state() -> dict:
    p = STATE_DIR / "state.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"done": {}, "empty": {}, "failed": {}}


def save_state(st: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_DIR / "state.tmp"
    tmp.write_text(json.dumps(st))
    tmp.replace(STATE_DIR / "state.json")


def targets(universe: Universe, symbols: set[str] | None, include_index: bool = True) -> list[tuple[str, int, str, str]]:
    out = []
    for i in universe.instruments:
        if symbols and i.symbol not in symbols:
            continue
        if i.kind == "EQ":
            out.append((i.symbol, i.sid, "NSE_EQ", "EQUITY"))
        elif i.kind == "IDX" and include_index:
            out.append((i.symbol, i.sid, "IDX_I", "INDEX"))
    return out


def make_jobs(universe: Universe, intervals: list[str], years: float, symbols: set[str] | None,
              since: dict[tuple[str, str], int] | None = None) -> list[Job]:
    today = now_ist().date()
    start_all = today - timedelta(days=int(365.25 * years))
    jobs: list[Job] = []
    for sym, sid, seg, inst in targets(universe, symbols):
        for iv in intervals:
            start = start_all
            if since and (iv, sym) in since:
                start = max(start_all, datetime.fromtimestamp(since[(iv, sym)], tz=IST).date())
            step = DAILY_CHUNK_DAYS if iv == "D" else INTRADAY_CHUNK_DAYS
            cur = start
            while cur <= today:
                end = min(today + timedelta(days=1), cur + timedelta(days=step))
                jobs.append(Job(sym, sid, seg, inst, iv, cur.isoformat(), end.isoformat()))
                cur = end
    return jobs


async def run_jobs(jobs: list[Job], reserve: int, concurrency: int = 5) -> dict:
    from td.bus import connect
    from td.dhan.rest import DhanError, DhanRest

    redis = None
    try:
        redis = connect()
        await redis.ping()
    except Exception:
        redis = None
        log.info("Redis not reachable — using a local budget file (start Redis to share the budget with the ingestor)")
    rest = DhanRest(redis=redis, budget_file=STATE_DIR / "budget.json")
    lake = Lake(Path(settings.data_dir) / "lake" / "dhan")
    st = load_state()
    pending = [j for j in jobs if j.key not in st["done"] and j.key not in st["empty"]]
    log.info("%d jobs pending (%d already done)", len(pending), len(jobs) - len(pending))
    q: asyncio.Queue = asyncio.Queue()
    for j in pending:
        q.put_nowait(j)
    stats = {"ok": 0, "empty": 0, "failed": 0, "rows": 0, "stopped": None}
    t0 = time.time()
    stop = asyncio.Event()

    async def worker():
        while not q.empty() and not stop.is_set():
            j: Job = await q.get()
            remaining = await rest.data_budget.remaining()
            if remaining <= reserve:
                stats["stopped"] = f"daily budget reserve reached ({remaining} left) — run again tomorrow to resume"
                stop.set()
                return
            try:
                if j.interval == "D":
                    payload = await rest.daily(j.sid, j.seg, j.instrument, date.fromisoformat(j.start),
                                               date.fromisoformat(j.end))
                else:
                    s = datetime.fromisoformat(j.start).replace(hour=9, minute=15)
                    e = datetime.fromisoformat(j.end).replace(hour=15, minute=30)
                    payload = await rest.intraday(j.sid, j.seg, j.instrument, INTERVALS[j.interval], s, e)
                df = bars_from_dhan(payload)
                if df.empty:
                    st["empty"][j.key] = 1
                    stats["empty"] += 1
                else:
                    await asyncio.to_thread(lake.write, LAKE_NAME[j.interval], j.symbol, df)
                    st["done"][j.key] = len(df)
                    stats["ok"] += 1
                    stats["rows"] += len(df)
            except BudgetExhausted as e:
                stats["stopped"] = str(e)
                stop.set()
                return
            except DhanError as e:
                if getattr(e, "code", None) in (807, 808, 809, 810, "DH-901", 806, "806"):
                    stats["stopped"] = f"auth/subscription error: {e}"
                    stop.set()
                    return
                st["failed"][j.key] = str(e)[:200]
                stats["failed"] += 1
            done = stats["ok"] + stats["empty"] + stats["failed"]
            if done % 25 == 0:
                save_state(st)
                rate = done / max(1e-6, time.time() - t0)
                log.info("progress %d/%d  rows=%d  %.1f req/s  budget left=%d", done, len(pending), stats["rows"],
                         rate, await rest.data_budget.remaining())
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    save_state(st)
    await rest.close()
    if redis:
        await redis.aclose()
    return stats


async def run_options(underlyings: list[str], strikes: int, years: float, interval: int, reserve: int) -> dict:
    """Expired options (rolling ATM±N, CE & PE) — 30 days per call. Stored as lake interval 'opt'."""
    from td.dhan.rest import DhanError, DhanRest
    from td.universe_data import INDICES
    import pandas as pd

    rest = DhanRest(budget_file=STATE_DIR / "budget.json")
    lake = Lake(Path(settings.data_dir) / "lake" / "dhan")
    st = load_state()
    today = now_ist().date()
    start = today - timedelta(days=int(365.25 * years))
    out = {"ok": 0, "failed": 0, "rows": 0}
    for und in underlyings:
        sid = INDICES.get(und, (None,))[0]
        if sid is None:
            log.warning("options backfill supports index underlyings only (%s skipped)", und)
            continue
        for k in range(-strikes, strikes + 1):
            strike = "ATM" if k == 0 else f"ATM{k:+d}"
            for cp in ("CALL", "PUT"):
                cur = start
                while cur < today:
                    end = min(today, cur + timedelta(days=30))
                    key = f"opt|{und}|{strike}|{cp}|{cur}|{end}"
                    if key in st["done"]:
                        cur = end
                        continue
                    if await rest.data_budget.remaining() <= reserve:
                        return out | {"stopped": "budget reserve reached"}
                    try:
                        p = await rest.rolling_option(sid, "NSE_FNO", "OPTIDX", "WEEK" if und == "NIFTY" else "MONTH",
                                                      1, strike, cp, cur, end, interval)
                        side = (p.get("data") or {}).get("ce" if cp == "CALL" else "pe") or {}
                        if side.get("timestamp"):
                            df = pd.DataFrame({c: side.get(c) for c in ("timestamp", "open", "high", "low", "close",
                                                                         "volume", "oi", "iv", "spot", "strike")
                                               if side.get(c) is not None}).rename(columns={"timestamp": "ts"})
                            path = lake.root / "opt" / f"{und}_{strike}_{cp}.parquet"
                            path.parent.mkdir(parents=True, exist_ok=True)
                            if path.exists():
                                df = pd.concat([pd.read_parquet(path), df]).drop_duplicates("ts").sort_values("ts")
                            df.to_parquet(path, index=False)
                            out["rows"] += len(df)
                        st["done"][key] = 1
                        out["ok"] += 1
                    except DhanError as e:
                        st["failed"][key] = str(e)[:200]
                        out["failed"] += 1
                    cur = end
                save_state(st)
    await rest.close()
    return out


def cmd_plan(args, universe):
    intervals = args.intervals.split(",")
    jobs = make_jobs(universe, intervals, args.years, _symbols(args))
    st = load_state()
    pending = [j for j in jobs if j.key not in st["done"] and j.key not in st["empty"]]
    per_day = max(1, settings.limit_data_per_day - args.reserve)
    print(f"instruments: {len(targets(universe, _symbols(args)))}   intervals: {intervals}   years: {args.years}")
    print(f"requests total: {len(jobs)}   pending: {len(pending)}")
    print(f"daily budget: {settings.limit_data_per_day} - reserve {args.reserve} = {per_day} calls/day")
    print(f"=> ~{len(pending) / per_day:.1f} day(s) of quota;  at 5 req/s ~{len(pending) / 5 / 60:.0f} min of run time")
    by = {}
    for j in pending:
        by[j.interval] = by.get(j.interval, 0) + 1
    for k, v in by.items():
        print(f"   {k:>3}: {v} requests")


def _symbols(args) -> set[str] | None:
    return {s.strip().upper() for s in args.symbols.split(",")} if getattr(args, "symbols", "") else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--intervals", default="D,1")
        p.add_argument("--years", type=float, default=5)
        p.add_argument("--symbols", default="")
        p.add_argument("--reserve", type=int, default=800, help="calls to leave for the live system today")
    p = sub.add_parser("update")
    p.add_argument("--intervals", default="D,1")
    p.add_argument("--reserve", type=int, default=800)
    p = sub.add_parser("options")
    p.add_argument("--underlyings", default="NIFTY,BANKNIFTY")
    p.add_argument("--strikes", type=int, default=2)
    p.add_argument("--years", type=float, default=1)
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--reserve", type=int, default=800)
    sub.add_parser("status")
    p = sub.add_parser("sim")
    p.add_argument("--days", type=int, default=320)
    p.add_argument("--minute-days", type=int, default=25)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "status":
        for sub_ in ("dhan", "sim"):
            lake = Lake(Path(settings.data_dir) / "lake" / sub_)
            summ = lake.summary()
            if summ:
                print(f"[{sub_}] {lake.root}")
                for iv, s in summ.items():
                    f = datetime.fromtimestamp(s["first"], tz=IST).date()
                    l = datetime.fromtimestamp(s["last"], tz=IST).date()
                    print(f"   {iv:>4}: {s['symbols']:4d} symbols  {s['rows']:>12,d} rows  {f} → {l}")
        st = load_state()
        print(f"backfill state: done={len(st['done'])} empty={len(st['empty'])} failed={len(st['failed'])}")
        return

    if args.cmd == "sim":
        from td.sim.simulator import generate_history, history_end_for, sim_trading_date

        uni = sim_universe()
        end = history_end_for(sim_trading_date())
        hist = generate_history(uni, end, args.days, args.minute_days)
        lake = Lake(Path(settings.data_dir) / "lake" / "sim")
        for iv, by in hist.items():
            for sym, df in by.items():
                lake.write(iv, sym, df)
        print(f"synthetic lake written to {lake.root}")
        return

    if not settings.dhan_client_id or not settings.dhan_access_token:
        if args.cmd == "plan":
            from td.universe_data import FNO_STOCKS
            print("(no Dhan credentials yet — estimating from the built-in F&O list)")
            cmd_plan(args, sim_universe(len(FNO_STOCKS)))
            return
        sys.exit("Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env first.")
    universe = dhan_universe()

    if args.cmd == "plan":
        cmd_plan(args, universe)
    elif args.cmd == "run":
        jobs = make_jobs(universe, args.intervals.split(","), args.years, _symbols(args))
        stats = asyncio.run(run_jobs(jobs, args.reserve))
        print(json.dumps(stats, indent=2))
    elif args.cmd == "update":
        lake = Lake(Path(settings.data_dir) / "lake" / "dhan")
        since = {}
        for iv in args.intervals.split(","):
            for sym in lake.symbols(LAKE_NAME[iv]):
                ts = lake.last_ts(LAKE_NAME[iv], sym)
                if ts:
                    since[(iv, sym)] = ts
        jobs = make_jobs(universe, args.intervals.split(","), 5, None, since)
        stats = asyncio.run(run_jobs(jobs, args.reserve))
        print(json.dumps(stats, indent=2))
    elif args.cmd == "options":
        stats = asyncio.run(run_options(args.underlyings.split(","), args.strikes, args.years, args.interval,
                                        args.reserve))
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
