"""BACKFILLER — downloads Dhan history into the lake for the ADMIN desk (F9), in its own process.

Kept apart from the live services on purpose: it never touches the market feed, runs at low CPU priority,
takes a paced share of Dhan's 5 req/s data limit, leaves a reserve of the daily 7,000 calls for the
ingestor, waits while the ingestor loads its own history (start-up and 08:40 IST) and slows down — or
pauses — in market hours. If it crashes, the live desks carry on.

The gateway sends commands on a Redis list; status goes to td:admin:backfill:status. The job is kept in
data/backfill/job.json, so a restart or redeploy resumes it, and a pull that needs more than one day of
quota carries on by itself after midnight IST.
Run:  python -m td.services.backfiller
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import time
from collections import deque
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path

from .. import __version__, bus
from .. import backfill as bf
from ..config import settings
from ..dhan.rest import DhanRest
from ..lake import Lake
from ..ratelimit import DailyBudget
from ..timeutil import IST, now_ist

log = logging.getLogger("td.backfiller")
IDLE_STAGES = {"ready", "budget exhausted", "disabled (AUTO_HISTORY=false)"}   # ingestor auto-history at rest
PREMARKET = (dtime(8, 35), dtime(8, 50))          # the ingestor refreshes its history at 08:40 IST
MARKET = (dtime(9, 0), dtime(15, 40))             # pre-open to just after the close
COVERAGE_EVERY = 300                              # s, lake coverage rescan while pulling


class RingLog(logging.Handler):
    """Recent log lines for the admin desk."""

    def __init__(self, n: int = 80):
        super().__init__(logging.INFO)
        self.lines: deque = deque(maxlen=n)

    def emit(self, rec: logging.LogRecord) -> None:
        try:
            self.lines.append([round(rec.created, 1), rec.levelname, rec.getMessage()[:300]])
        except Exception:
            pass


def _next_quota_day(now: datetime) -> float:
    """Ten minutes after the next IST midnight, when the daily budget resets."""
    nxt = datetime.combine(now.date() + timedelta(days=1), dtime(0, 10), tzinfo=IST)
    return nxt.timestamp()


def _hm(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=IST).strftime("%d %b %H:%M")


class Backfiller:
    def __init__(self):
        self.r = bus.connect()
        self.lake = Lake(Path(settings.data_dir) / "lake" / "dhan")
        self.budget = DailyBudget("dhan_data", settings.limit_data_per_day, redis=self.r)
        self.job: bf.JobSpec | None = bf.load_job()
        self.state, self.reason = "starting", ""
        self.progress: dict | None = None
        self.result: dict | None = None
        self.coverage: dict | None = None
        self.resume_at: float | None = None
        self.started = time.time()
        self.owner = f"admin-worker@{socket.gethostname()}:{os.getpid()}"
        self.stop = asyncio.Event()          # process shutdown
        self.halt = asyncio.Event()          # end the current pull (stop / new settings)
        self.wake = asyncio.Event()          # a command arrived: re-evaluate now
        self.ring = RingLog()
        logging.getLogger("td").addHandler(self.ring)
        self._ing: tuple[float, dict | None] = (0.0, None)
        self._cov_task: asyncio.Task | None = None
        self._cov_at = 0.0
        self._uni: list | None = None
        self._rest: DhanRest | None = None
        self._usage: dict = {}
        self._calls_seen = 0
        self._bg: set[asyncio.Task] = set()
        self._usage_saved = 0.0

    # ------------------------------------------------------------ helpers
    def _set(self, state: str, reason: str = "") -> None:
        if (state, reason) != (self.state, self.reason):
            self.state, self.reason = state, reason

    def unavailable(self) -> str | None:
        if settings.data_source != "dhan":
            return "Simulator mode — history pulls need DATA_SOURCE=dhan (the simulator writes its own synthetic lake)."
        if not (settings.dhan_client_id and settings.dhan_access_token):
            return "DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN are not set in .env."
        return None

    async def _idle(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self.wake.wait(), timeout=max(0.1, seconds))
        except asyncio.TimeoutError:
            pass
        self.wake.clear()

    async def _universe(self) -> list | None:
        """The ingestor's instrument list — only while a live (dhan-mode) ingestor is running: a list left in
        Redis by the simulator has made-up security ids. Otherwise resolve it from Dhan's scrip master."""
        ing = await self._ingestor()
        if ing and ing.get("mode") == "dhan":
            raw = await self.r.get(bus.KEY_UNIVERSE)
            if raw:
                self._uni = bus.loads(raw)
                return self._uni
        if self._uni is None and settings.data_source == "dhan":
            try:
                from ..dhan.instruments import dhan_universe
                self._uni = (await asyncio.to_thread(dhan_universe)).to_json()
            except Exception as e:
                log.warning("instrument list unavailable: %s", e)
        return self._uni

    async def _ingestor(self) -> dict | None:
        ts, val = self._ing
        if time.time() - ts > 2:
            raw = await self.r.get(bus.k_status("ingestor"))
            val = bus.loads(raw) if raw else None
            self._ing = (time.time(), val)
        return val

    # ------------------------------------------------------------ usage floor
    def _usage_path(self) -> Path:
        return bf.state_dir() / "usage.json"

    def _read_usage(self) -> dict:
        try:
            p = self._usage_path()
            return json.loads(p.read_text()) if p.exists() else {}
        except (OSError, ValueError):
            return {}

    def _write_usage(self) -> None:
        try:
            p = self._usage_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self._usage))
        except OSError as e:
            log.debug("usage file: %s", e)

    def _today_usage(self) -> dict:
        day = now_ist().date().isoformat()
        if self._usage.get("date") != day:
            u = self._read_usage()
            self._usage = u if u.get("date") == day else {"date": day, "calls": 0, "seen": 0}
        return self._usage

    def _save_usage(self, force: bool = False) -> None:
        """This worker's own calls per IST day, on disk (see _floor)."""
        if not self._rest or (not force and time.time() - self._usage_saved < 10):
            return
        seen = self._rest.calls.get("data", 0)
        delta, self._calls_seen = seen - self._calls_seen, seen
        u = self._today_usage()
        u["calls"] = int(u.get("calls", 0)) + delta
        self._write_usage()
        self._usage_saved = time.time()

    async def _floor(self) -> int:
        """Today's shared call counter, never below what was already spent. Redis may restart (docker runs it
        without persistence) and forget the ingestor's and our calls; the highest value seen today and our own
        calls are kept on disk and written back."""
        u = self._today_usage()
        used = await self.budget.used()
        floor = max(int(u.get("seen", 0)), int(u.get("calls", 0)))
        if used < floor:
            await self.r.set(self.budget._key(), floor, ex=3 * 86400)
            log.info("daily call counter restored to %d (Redis had %d)", floor, used)
            used = floor
        if used > int(u.get("seen", 0)):
            u["seen"] = used
            self._write_usage()
        return used

    async def _budget_floor(self) -> None:
        self._calls_seen = 0
        await self._floor()

    # ------------------------------------------------------------ pacing
    async def _pause_reason(self) -> str | None:
        now = now_ist()
        if now.weekday() < 5:
            t = now.timetz().replace(tzinfo=None)
            if PREMARKET[0] <= t < PREMARKET[1]:
                return "pre-market: the live app refreshes its own history at 08:40 IST"
            if self.job and self.job.market_hours == "pause" and MARKET[0] <= t < MARKET[1]:
                return "market hours — paused until 15:40 IST (choose Slow to keep pulling at 1 request/s)"
        ing = await self._ingestor()
        if ing:
            stage = ((ing.get("stats") or {}).get("history") or {}).get("stage")
            if stage and stage not in IDLE_STAGES and time.time() - float(ing.get("updated") or 0) < 30:
                return f"the live app is loading its own history ({stage}) — waiting so both stay under Dhan's limit"
        return None

    def _in_market(self) -> bool:
        now = now_ist()
        t = now.timetz().replace(tzinfo=None)
        return now.weekday() < 5 and MARKET[0] <= t < MARKET[1]

    async def _gate(self) -> float:
        """Awaited before every request: pauses while the live app needs Dhan, returns the rate to use now."""
        while True:
            if self.halt.is_set() or self.stop.is_set():
                raise bf.Stopped()
            why = await self._pause_reason()
            if why is None:
                break
            self._set("paused", why)
            try:
                await asyncio.wait_for(self.halt.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass
        job = self.job
        if job and job.market_hours == "slow" and self._in_market():
            self._set("running", "market hours — slowed to 1 request/s")
            return min(job.rate, 1.0)
        self._set("running", "")
        return job.rate if job else 1.0

    # ------------------------------------------------------------ coverage
    def _refresh_coverage(self, force: bool = False) -> None:
        if self._cov_task and not self._cov_task.done():
            return
        if not force and time.time() - self._cov_at < COVERAGE_EVERY:
            return
        self._cov_at = time.time()

        async def scan():
            try:
                self.coverage = await asyncio.to_thread(bf.lake_coverage, self.lake)
            except Exception as e:
                log.warning("lake coverage scan failed: %s", e)
        self._cov_task = asyncio.create_task(scan())

    # ------------------------------------------------------------ commands
    async def _commands(self) -> None:
        while not self.stop.is_set():
            try:
                item = await self.r.brpop([bus.KEY_ADMIN_CMD], timeout=2)
            except Exception as e:
                log.debug("command read failed: %s", e)
                await asyncio.sleep(2)
                continue
            if not item:
                continue
            try:
                cmd = json.loads(item[1])
                await self._handle(cmd)
            except Exception:
                log.exception("bad admin command")

    async def _handle(self, cmd: dict) -> None:
        op, by = cmd.get("op"), str(cmd.get("by", "local"))[:40]
        if op == "start":
            try:
                spec = bf.JobSpec.clean({**(cmd.get("job") or {}), "active": True, "created": time.time(), "by": by})
            except ValueError as e:
                log.warning("start rejected: %s", e)
                return
            busy = self.state in ("running", "paused", "blocked", "waiting")
            self.job, self.resume_at = spec, None
            bf.save_job(spec)
            log.info("%s by %s: %s · %g years · %.1f req/s · reserve %d · market hours %s · %s",
                     "settings changed" if busy else "pull started", by, "+".join(bf.LAKE_NAME[i] for i in spec.intervals),
                     spec.years, spec.rate, spec.reserve, spec.market_hours,
                     f"{len(spec.symbols)} symbols" if spec.symbols else "all F&O symbols")
            self.halt.set()                           # a pull in progress ends; the main loop restarts it
            self.wake.set()
        elif op == "stop":
            if self.job and self.job.active:
                self.job.active = False
                bf.save_job(self.job)
                log.info("pull stopped by %s — progress is kept; Start resumes it", by)
            self.resume_at = None
            self.halt.set()
            self.wake.set()
        elif op == "plan":
            rid = str(cmd.get("id", ""))[:64]

            async def answer():               # off the command loop: resolving instruments can take seconds
                try:
                    res = await self._plan(bf.JobSpec.clean(cmd.get("job") or {}))
                except Exception as e:
                    res = {"error": str(e)}
                await self.r.set(bus.k_admin_plan(rid), json.dumps(res), ex=60)
            t = asyncio.create_task(answer())
            self._bg.add(t)
            t.add_done_callback(self._bg.discard)
        elif op == "coverage":
            self._refresh_coverage(force=True)

    async def _plan(self, spec: bf.JobSpec) -> dict:
        why = self.unavailable()
        if why:
            return {"error": why}
        uni = await self._universe()
        if not uni:
            return {"error": "the instrument list is not available yet (is the ingestor running?)"}
        tg = bf.targets(uni, set(spec.symbols) or None)
        today = now_ist().date()
        jobs = bf.make_jobs(tg, spec.intervals, spec.years, today, bf.coverage_from_state(bf.load_state(), today))
        out = bf.summarize(jobs, await self.budget.remaining(), spec.reserve, spec.rate)
        out["symbols"] = len(tg)
        out["unknown"] = sorted(set(spec.symbols) - {t.symbol for t in tg})
        return out

    # ------------------------------------------------------------ pulls
    async def _keep_lock(self, lock: bf.RunLock) -> None:
        while True:
            await asyncio.sleep(30)
            if not await lock.refresh():
                log.warning("lost the backfill lock — stopping this pull")
                self.halt.set()
                return

    def _on_progress(self, prog: bf.Progress) -> None:
        self.progress = prog.as_dict()
        self._save_usage()
        self._refresh_coverage()

    async def _run(self, job: bf.JobSpec) -> None:
        uni = await self._universe()
        if not uni:
            self._set("waiting", "waiting for the instrument list (start the ingestor)")
            await self._idle(15)
            return
        lock = bf.RunLock(self.r, self.owner)
        if not await lock.acquire():
            self._set("blocked", f"another pull holds the lock ({await lock.holder()}) — this one starts when it ends")
            await self._idle(30)
            return
        keeper = asyncio.create_task(self._keep_lock(lock))
        self._rest = rest = DhanRest(redis=self.r)
        try:
            await self._budget_floor()
            today = now_ist().date()
            st = bf.compact(bf.load_state(), today)
            bf.save_state(st)
            tg = bf.targets(uni, set(job.symbols) or None)
            jobs = bf.make_jobs(tg, job.intervals, job.years, today, bf.coverage_from_state(st, today))
            if not jobs:
                self._done(job, "complete", bf.Progress())
                return
            by = {}
            for j in jobs:
                by[bf.LAKE_NAME[j.interval]] = by.get(bf.LAKE_NAME[j.interval], 0) + 1
            log.info("pulling %d requests (%s) for %d symbols", len(jobs),
                     ", ".join(f"{k} {v}" for k, v in by.items()), len(tg))
            self._set("running", "")
            reason, prog = await bf.run_jobs(jobs, rest, self.lake, st, today=today, rate=job.rate,
                                             reserve=job.reserve, stop=self.halt, gate=self._gate,
                                             on_progress=self._on_progress)
            self._done(job, reason, prog)
        except Exception as e:
            log.exception("pull crashed")
            self._set("error", f"{type(e).__name__}: {e}")
            self.resume_at = time.time() + 600
        finally:
            keeper.cancel()
            self._save_usage(force=True)
            await rest.close()
            self._rest = None
            await lock.release()
            self._refresh_coverage(force=True)

    def _done(self, job: bf.JobSpec, reason: str, prog: bf.Progress) -> None:
        self.progress = prog.as_dict()
        if job is not self.job:                  # replaced by new settings while finishing: nothing to record
            return
        self.result = {"reason": reason, "at": time.time(), "ok": prog.ok, "empty": prog.empty,
                       "failed": prog.failed, "rows": prog.rows, "requests": prog.done}
        if reason == "complete":
            job.active = False
            bf.save_job(job)
            msg = f"pull complete — {prog.ok} requests with data, {prog.empty} empty, {prog.rows:,} rows"
            if prog.failed:
                msg += f"; {prog.failed} failed (Start again to retry them)"
            log.info(msg)
            self._set("done", msg)
        elif reason in ("reserve", "budget"):
            self.resume_at = _next_quota_day(now_ist())
            log.info("today's Dhan quota is used down to the reserve — resuming %s IST", _hm(self.resume_at))
        elif reason.startswith("auth"):
            self.resume_at = time.time() + 1800
            log.error("Dhan rejected the credentials (%s) — paste a fresh DHAN_ACCESS_TOKEN into .env and restart; "
                      "retrying in 30 min", reason[5:].strip())
            self._set("error", f"Dhan credentials rejected: {reason[5:].strip()}")
        elif reason.startswith("disk"):
            job.active = False
            bf.save_job(job)
            log.error("disk full — pull stopped (%s)", reason)
            self._set("error", "disk full — free some space, then Start again")
        elif reason == "stopped" and not job.active:
            self._set("stopped", "stopped — progress is kept; Start resumes where it left off")

    async def _main(self) -> None:
        while not self.stop.is_set():
            try:
                await self._step()
            except asyncio.CancelledError:
                raise
            except Exception as e:                  # e.g. Redis briefly unreachable: keep the loop alive
                log.exception("backfill loop error")
                self._set("error", f"{type(e).__name__}: {e} — retrying in 15 s")
                await asyncio.sleep(15)

    async def _step(self) -> None:
        why = self.unavailable()
        job = self.job
        if why:
            self._set("unavailable", why)
            await self._idle(30)
        elif not job or not job.active:
            if self.state not in ("done", "stopped", "error"):
                self._set("idle", "")
            await self._idle(30)
        elif self.resume_at and time.time() < self.resume_at:
            if self.state != "error":
                self._set("waiting_quota", f"today's Dhan quota is used down to the reserve — resumes "
                                           f"{_hm(self.resume_at)} IST")
            await self._idle(min(60.0, self.resume_at - time.time()))
        else:
            self.resume_at = None
            self.halt.clear()                    # a Stop / new settings from here on ends this pull
            await self._run(job)

    async def _status(self) -> None:
        while not self.stop.is_set():
            try:
                job = self.job
                payload = {
                    "worker": {"pid": os.getpid(), "host": socket.gethostname(), "version": __version__,
                               "started": self.started, "updated": time.time()},
                    "source": settings.data_source, "state": self.state, "reason": self.reason,
                    "job": job.as_dict() if job else None, "progress": self.progress, "result": self.result,
                    "resume_at": self.resume_at,
                    "budget": {"used": await self._floor(), "limit": settings.limit_data_per_day,
                               "reserve": job.reserve if job else None},
                    "coverage": self.coverage, "log": list(self.ring.lines)[-60:],
                    "limits": {"data_per_sec": settings.limit_data_per_sec, "min_reserve": bf.min_reserve()},
                }
                await self.r.set(bus.KEY_ADMIN_STATUS, json.dumps(payload), ex=20)
            except Exception:
                log.debug("status publish failed", exc_info=True)
            await asyncio.sleep(2)

    async def run(self) -> None:
        log.info("backfill worker up (%s) — job: %s", settings.data_source,
                 "resuming" if self.job and self.job.active else "none")
        self._refresh_coverage(force=True)
        tasks = [asyncio.create_task(t) for t in (self._commands(), self._status(), self._main())]
        await self.stop.wait()
        self.halt.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if hasattr(os, "nice"):
        try:
            os.nice(10)                   # Parquet rewrites must never compete with the live services for CPU
        except OSError:
            pass
    svc = Backfiller()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, svc.stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    try:
        await svc.run()
    finally:
        try:
            await svc.r.delete(bus.KEY_ADMIN_STATUS)
        except Exception:
            pass
        await svc.r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
