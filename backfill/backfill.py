"""TradeDesk backfill — pulls up to 5 years of Dhan history into the shared Parquet lake.

The same engine runs behind the admin page (/admin → HISTORY, admin login), which is the easy way to do this: it pulls
in a background worker, paced so the live desks are not affected, and carries on by itself after midnight
IST when a day's quota runs out. This CLI is for scripts and cron. Only one pull runs at a time (a Redis
lock shared with the admin worker), and both resume from the same state (data/backfill/state.json).

  python backfill/backfill.py plan    --intervals D,1 --years 5           # calls needed + ETA
  python backfill/backfill.py run     --intervals D,1 --years 5           # resumable, across days too
  python backfill/backfill.py update                                      # incremental top-up
  python backfill/backfill.py options --underlyings NIFTY,BANKNIFTY --strikes 2 --years 1
  python backfill/backfill.py status                                      # lake coverage
  python backfill/backfill.py sim                                         # synthetic lake (no Dhan)

Rate limits honoured: --rate requests/s (default 3, leaving room for the live ingestor under Dhan's 5/s),
and the daily budget (default 7,000, shared through Redis with the live ingestor) minus --reserve
(default 2,500: the live app itself uses ~2,000 calls on a trading day).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from td import backfill as bf  # noqa: E402
from td.config import settings  # noqa: E402
from td.dhan.instruments import Universe, dhan_universe, sim_universe  # noqa: E402
from td.lake import Lake  # noqa: E402
from td.timeutil import IST, now_ist  # noqa: E402

log = logging.getLogger("backfill")
STATE_DIR = bf.state_dir()
LAKE_NAME = bf.LAKE_NAME
load_state, save_state = bf.load_state, bf.save_state


def _lake() -> Lake:
    return Lake(Path(settings.data_dir) / "lake" / "dhan")


def _since(intervals: list[str]) -> dict:
    """Last bar per symbol in the lake (for `update`)."""
    lake = _lake()
    since = {}
    for iv in intervals:
        for sym in lake.symbols(LAKE_NAME[iv]):
            ts = lake.last_ts(LAKE_NAME[iv], sym)
            if ts:
                since[(iv, sym)] = datetime.fromtimestamp(ts, tz=IST).date()
    return since


def _jobs(universe: Universe, intervals: list[str], years: float, symbols: set[str] | None,
          since: dict | None = None) -> list[bf.Job]:
    today = now_ist().date()
    st = bf.compact(bf.load_state(), today)
    return bf.make_jobs(bf.targets(universe.instruments, symbols), intervals, years, today,
                        bf.coverage_from_state(st, today), since, recent=since is not None)


class _Session:
    """One CLI pull: Redis (the daily budget shared with the live ingestor), the pull lock shared with the admin
    worker, the token the ingestor keeps renewed, and a Dhan client. `stop` is set if the lock is lost, so two
    pulls never write together."""

    async def __aenter__(self) -> "_Session":
        from td.bus import connect
        from td.dhan.rest import DhanRest
        from td.dhan.token import TokenKeeper

        self.redis = None
        try:
            r = connect()
            await r.ping()
            self.redis = r
        except Exception:
            log.info("Redis not reachable — using a local budget file (start Redis to share the budget with the "
                     "ingestor)")
        self.stop = asyncio.Event()
        self.lock = bf.RunLock(self.redis, f"cli@{socket.gethostname()}:{os.getpid()}") if self.redis else None
        if self.lock and not await self.lock.acquire():
            holder = await self.lock.holder()
            await self.redis.aclose()
            sys.exit(f"Another pull is running ({holder}). Stop it on the admin page or wait for it to finish.")
        self.keeper = asyncio.create_task(self._keep()) if self.lock else None
        # DHAN_ACCESS_TOKEN in .env lapses after a day: follow the stored one (Redis, else data/auth), also when
        # it is renewed during a long run; DhanRest reads it on every call
        self.tokens = TokenKeeper(self.redis, role="follower")
        await self.tokens.start()
        self.rest = DhanRest(redis=self.redis, budget_file=STATE_DIR / "budget.json")
        return self

    async def _keep(self) -> None:
        while True:
            await asyncio.sleep(30)
            try:
                ok = await self.lock.refresh()
            except Exception:                      # Redis hiccup: try again next time
                continue
            if not ok:
                log.warning("lost the pull lock — stopping so two pulls never write the same files")
                self.stop.set()
                return

    async def __aexit__(self, *exc) -> None:
        if self.keeper:
            self.keeper.cancel()
        await self.tokens.stop()
        await self.rest.close()
        if self.lock:
            await self.lock.release()
        if self.redis:
            await self.redis.aclose()


async def run_jobs(jobs: list[bf.Job], reserve: int, rate: float) -> dict:
    today = now_ist().date()
    st = bf.compact(bf.load_state(), today)
    log.info("%d requests pending", len(jobs))
    last = [0]

    def progress(p: bf.Progress) -> None:
        if p.done - last[0] >= 25 or p.done == p.total:
            last[0] = p.done
            log.info("progress %d/%d  rows=%d  %.1f req/s", p.done, p.total, p.rows, p.as_dict()["rate"])

    async with _Session() as ses:
        reason, prog = await bf.run_jobs(jobs, ses.rest, _lake(), st, today=today, rate=rate, reserve=reserve,
                                         stop=ses.stop, on_progress=progress)
    stopped = {"complete": None, "reserve": "daily budget reserve reached — run again tomorrow to resume",
               "budget": "daily budget exhausted — run again tomorrow to resume"}.get(reason, reason)
    return {"ok": prog.ok, "empty": prog.empty, "failed": prog.failed, "rows": prog.rows, "stopped": stopped}


async def run_options(underlyings: list[str], strikes: int, years: float, interval: int, reserve: int) -> dict:
    """Expired options (rolling ATM±N, CE & PE) — 30 days per call. Stored as lake interval 'opt'."""
    async with _Session() as ses:
        return await _options(ses, underlyings, strikes, years, interval, reserve)


async def _options(ses: "_Session", underlyings: list[str], strikes: int, years: float, interval: int,
                   reserve: int) -> dict:
    from td.dhan.rest import DhanError
    from td.universe_data import INDICES
    import pandas as pd

    rest = ses.rest
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
                    if ses.stop.is_set():
                        save_state(st)
                        return out | {"stopped": "lost the pull lock"}
                    if await rest.data_budget.remaining() <= reserve:
                        save_state(st)
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


async def _stored_token() -> None:
    """Apply the token stored in data/auth (the one the ingestor renews), if any."""
    from td.dhan.token import TokenKeeper

    await TokenKeeper(None, role="follower").sync()


def cmd_plan(args, universe):
    intervals = args.intervals.split(",")
    jobs = _jobs(universe, intervals, args.years, _symbols(args))
    s = bf.summarize(jobs, None, args.reserve, args.rate)
    print(f"instruments: {len(bf.targets(universe.instruments, _symbols(args)))}   intervals: {intervals}   "
          f"years: {args.years}")
    print(f"requests pending: {s['requests']}  (ranges already in the lake are skipped)")
    print(f"daily budget: {settings.limit_data_per_day} - reserve {args.reserve} = {s['per_day']} calls/day")
    print(f"=> ~{s['days_of_quota']} day(s) of quota;  ~{s['run_minutes']:.0f} min of run time at {args.rate} req/s")
    for k, v in s["by_interval"].items():
        print(f"   {k:>3}: {v} requests")


def _symbols(args) -> set[str] | None:
    return {s.strip().upper() for s in args.symbols.split(",") if s.strip()} if getattr(args, "symbols", "") \
        else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--intervals", default="D,1")
        p.add_argument("--years", type=float, default=5)
        p.add_argument("--symbols", default="")
        p.add_argument("--reserve", type=int, default=2500,
                       help="calls to leave for the live system today (it uses ~2,000 on a trading day)")
        p.add_argument("--rate", type=float, default=3.0, help="requests per second (Dhan allows 5 in total)")
    p = sub.add_parser("update")
    p.add_argument("--intervals", default="D,1")
    p.add_argument("--reserve", type=int, default=2500)
    p.add_argument("--rate", type=float, default=3.0)
    p = sub.add_parser("options")
    p.add_argument("--underlyings", default="NIFTY,BANKNIFTY")
    p.add_argument("--strikes", type=int, default=2)
    p.add_argument("--years", type=float, default=1)
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--reserve", type=int, default=2500)
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

    if settings.dhan_client_id and not settings.dhan_access_token:
        asyncio.run(_stored_token())         # .env may hold none once the token was pasted on /admin
    if not settings.dhan_client_id or not settings.dhan_access_token:
        if args.cmd == "plan":
            from td.universe_data import FNO_STOCKS
            print("(no Dhan credentials yet — estimating from the built-in F&O list)")
            cmd_plan(args, sim_universe(len(FNO_STOCKS)))
            return
        sys.exit("Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env first (or paste a token on /admin → SYSTEM).")
    universe = dhan_universe()

    if args.cmd == "plan":
        cmd_plan(args, universe)
    elif args.cmd == "run":
        jobs = _jobs(universe, args.intervals.split(","), args.years, _symbols(args))
        stats = asyncio.run(run_jobs(jobs, args.reserve, args.rate))
        print(json.dumps(stats, indent=2))
    elif args.cmd == "update":
        ivs = args.intervals.split(",")
        jobs = _jobs(universe, ivs, 5, None, _since(ivs))
        stats = asyncio.run(run_jobs(jobs, args.reserve, args.rate))
        print(json.dumps(stats, indent=2))
    elif args.cmd == "options":
        stats = asyncio.run(run_options(args.underlyings.split(","), args.strikes, args.years, args.interval,
                                        args.reserve))
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
