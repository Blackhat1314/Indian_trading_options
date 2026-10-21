"""History pulls (td.backfill), the admin backfill worker and the admin endpoints."""
import asyncio
import json
from datetime import date, datetime, timedelta

import pytest

from td import backfill as bf
from td.dhan.rest import DhanError
from td.lake import Lake
from td.timeutil import IST

TODAY = date(2026, 10, 13)
UNI = [{"sid": 13, "seg": "IDX_I", "symbol": "NIFTY", "kind": "IDX"},
       {"sid": 2885, "seg": "NSE_EQ", "symbol": "RELIANCE", "kind": "EQ"},
       {"sid": 14747, "seg": "NSE_EQ", "symbol": "011NSETEST", "kind": "EQ"},
       {"sid": 52001, "seg": "NSE_FNO", "symbol": "RELIANCE-FUT", "kind": "FUT"}]


class Budget:
    def __init__(self, left=10_000):
        self.left = left

    async def remaining(self):
        return self.left


class FakeRest:
    """Dhan historical API stand-in: one bar per weekday (daily) or 3 bars per weekday (intraday)."""

    def __init__(self, left=10_000, fail=None):
        self.data_budget = Budget(left)
        self.calls = []
        self.fail = fail or {}          # (symbol sid, start date) -> list of exceptions to raise first

    def _bars(self, start: date, end: date, per_day: int):
        ts, d = [], start
        while d < end:
            if d.weekday() < 5:
                t0 = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST).timestamp()
                ts += [int(t0 + 60 * m) for m in range(per_day)]
            d += timedelta(days=1)
        n = len(ts)
        return {"timestamp": ts, "open": [1.0] * n, "high": [1.0] * n, "low": [1.0] * n, "close": [1.0] * n,
                "volume": [1] * n}

    def _maybe_fail(self, sid, start):
        errs = self.fail.get((sid, start))
        if errs:
            raise errs.pop(0)

    async def daily(self, sid, seg, kind, start, end, oi=False, expiry_code=None, retries=3):
        self.calls.append(("D", sid, start, end))
        self.data_budget.left -= 1
        self._maybe_fail(sid, start)
        return self._bars(start, min(end, TODAY), 1)

    async def intraday(self, sid, seg, kind, interval, start, end, oi=False, retries=3):
        assert (end.date() - start.date()).days < bf.INTRADAY_CHUNK_DAYS     # Dhan's 90-day cap
        self.calls.append(("1", sid, start.date(), end.date()))
        self.data_budget.left -= 1
        self._maybe_fail(sid, start.date())
        return self._bars(start.date(), min(end.date() + timedelta(days=1), TODAY), 3)


def run(coro):
    return asyncio.run(coro)


def test_targets_skip_test_scrips_and_futures():
    t = bf.targets(UNI)
    assert [x.symbol for x in t] == ["NIFTY", "RELIANCE"]
    assert t[0].seg == "IDX_I" and t[0].instrument == "INDEX" and t[1].instrument == "EQUITY"
    assert [x.symbol for x in bf.targets(UNI, {"RELIANCE"})] == ["RELIANCE"]


def test_coverage_merge_and_missing():
    c = bf.Coverage()
    c.add("1", "X", date(2026, 1, 1), date(2026, 2, 1))
    c.add("1", "X", date(2026, 2, 1), date(2026, 3, 1))           # adjacent -> merged
    c.add("1", "X", date(2026, 4, 1), date(2026, 5, 1))
    assert c.ranges[("1", "X")] == [[date(2026, 1, 1), date(2026, 3, 1)], [date(2026, 4, 1), date(2026, 5, 1)]]
    assert c.missing("1", "X", date(2026, 1, 5), date(2026, 2, 20)) is None
    assert c.missing("1", "X", date(2026, 2, 10), date(2026, 4, 10)) == (date(2026, 3, 1), date(2026, 4, 1))
    assert c.missing("1", "X", date(2026, 2, 10), date(2026, 6, 1)) == (date(2026, 3, 1), date(2026, 6, 1))
    assert c.missing("1", "Y", date(2026, 1, 1), date(2026, 1, 2)) == (date(2026, 1, 1), date(2026, 1, 2))


def test_make_jobs_newest_first_interleaved_and_capped():
    jobs = bf.make_jobs(bf.targets(UNI), ["D", "1"], 7, TODAY)
    d = [j for j in jobs if j.interval == "D"]
    m = [j for j in jobs if j.interval == "1"]
    assert jobs.index(m[0]) > jobs.index(d[-1])                      # all daily bars first
    assert [j.symbol for j in m[:2]] == ["NIFTY", "RELIANCE"]        # symbols interleaved
    assert m[0].end == bf.settled_end(TODAY).isoformat()             # newest chunk first, settled days only
    assert bf.make_jobs(bf.targets(UNI), ["1"], 1, TODAY, recent=True)[0].end == (TODAY + timedelta(days=1)).isoformat()
    assert min(date.fromisoformat(j.start) for j in m) >= TODAY - timedelta(days=int(365.25 * 5))   # 5y cap
    assert min(date.fromisoformat(j.start) for j in d) == TODAY - timedelta(days=int(365.25 * 7))
    assert all((date.fromisoformat(j.end) - date.fromisoformat(j.start)).days <= 90 for j in m)


def test_resume_next_day_downloads_only_what_is_missing(tmp_path):
    """Chunk boundaries move every day; a pull cut short by the quota must not start over tomorrow."""
    lake = Lake(tmp_path / "lake")
    st = {"done": {}, "empty": {}, "failed": {}}
    tg = bf.targets(UNI)
    jobs = bf.make_jobs(tg, ["1"], 2, TODAY)
    rest = FakeRest(left=len(jobs) // 2 + 100)                       # quota for about half of it
    reason, prog = run(bf.run_jobs(jobs, rest, lake, st, today=TODAY, reserve=100, rate=50, concurrency=1,
                                   save=lambda s: None))
    assert reason == "reserve" and 0 < prog.done < len(jobs)
    tomorrow = TODAY + timedelta(days=1)
    left = bf.make_jobs(tg, ["1"], 2, tomorrow, bf.coverage_from_state(st, tomorrow))
    # only the chunks never fetched, plus the one day that settled overnight
    assert len(left) <= len(jobs) - prog.done + len(tg)
    assert any(j.end == bf.settled_end(tomorrow).isoformat() and j.start == bf.settled_end(TODAY).isoformat()
               for j in left)
    rest2 = FakeRest()
    reason, prog2 = run(bf.run_jobs(left, rest2, lake, st, today=tomorrow, rate=50, save=lambda s: None))
    assert reason == "complete" and prog2.failed == 0
    # finished: nothing left to plan
    assert bf.make_jobs(tg, ["1"], 2, tomorrow, bf.coverage_from_state(st, tomorrow)) == []
    df = lake.read("1m", "RELIANCE")
    assert df["ts"].is_monotonic_increasing and not df["ts"].duplicated().any()


def test_old_state_keys_count_as_coverage():
    """state.json written by the old CLI (exact chunk keys up to tomorrow) is honoured, minus the unsettled tail."""
    st = {"done": {"1|RELIANCE|2026-07-15|2026-10-14": 9000, "D|RELIANCE|2025-10-13|2026-10-14": 250},
          "empty": {"1|NIFTY|2021-10-13|2022-01-11": 1}, "failed": {"D|NIFTY|2026-10-12|2026-10-14": "no data"}}
    cov = bf.coverage_from_state(st, TODAY)
    assert cov.missing("1", "RELIANCE", date(2026, 8, 1), date(2026, 10, 14)) == (date(2026, 10, 12),
                                                                                 date(2026, 10, 14))
    assert cov.missing("1", "NIFTY", date(2021, 11, 1), date(2022, 1, 1)) is None
    assert cov.missing("D", "NIFTY", date(2026, 10, 12), date(2026, 10, 14)) is not None    # failures don't count
    c = bf.compact(json.loads(json.dumps(st)), TODAY)
    assert c["done"] == {"1|RELIANCE|2026-07-15|2026-10-12": 9000, "D|RELIANCE|2025-10-13|2026-10-12": 250}


def test_compact_merges_daily_slivers():
    st = {"done": {"1|X|2026-01-01|2026-04-01": 10, "1|X|2026-04-01|2026-06-30": 20, "1|X|2026-06-29|2026-07-01": 1,
                   "opt|NIFTY|ATM|CALL|2026-01-01|2026-01-31": 1}, "empty": {}, "failed": {}}
    bf.compact(st, TODAY)
    assert st["done"] == {"1|X|2026-01-01|2026-07-01": 31, "opt|NIFTY|ATM|CALL|2026-01-01|2026-01-31": 1}


def test_run_jobs_error_handling(tmp_path):
    lake = Lake(tmp_path / "lake")
    jobs = bf.make_jobs(bf.targets(UNI), ["1"], 1, TODAY)
    nifty_first, rel_first, rel_second = jobs[0], jobs[1], jobs[3]
    s = date.fromisoformat
    fail = {
        (13, s(nifty_first.start)): [DhanError("x: retries exhausted", status=429)],         # retried, then ok
        (2885, s(rel_first.start)): [DhanError("x: no data present", code="DH-907")],        # recorded as empty
        (2885, s(rel_second.start)): [DhanError("bad", code="DH-905")],                       # reported as failed
    }
    st = {"done": {}, "empty": {}, "failed": {}}
    reason, prog = run(bf.run_jobs(jobs, FakeRest(fail=fail), lake, st, today=TODAY, rate=50,
                                   save=lambda s: None))
    assert reason == "complete"
    assert prog.retried == 1 and prog.empty == 1 and prog.failed == 1 and prog.done == len(jobs)
    assert list(st["failed"]) == [rel_second.key] and prog.ok == len(jobs) - 2
    # credentials rejected -> the pull stops at once
    auth = {(13, s(nifty_first.start)): [DhanError("expired", code="DH-901")]}
    reason, _ = run(bf.run_jobs(jobs, FakeRest(fail=auth), lake, {"done": {}, "empty": {}, "failed": {}},
                                today=TODAY, rate=50, concurrency=1, save=lambda s: None))
    assert reason.startswith("auth")


def test_gate_stop_and_rate():
    lake_calls = []

    async def go():
        stop = asyncio.Event()
        rates = []

        async def gate():
            rates.append(1)
            if len(rates) > 3:
                stop.set()
                raise bf.Stopped()
            return 100.0

        class L:
            def write(self, *a):
                lake_calls.append(a)

        jobs = bf.make_jobs(bf.targets(UNI), ["1"], 5, TODAY)
        return await bf.run_jobs(jobs, FakeRest(), L(), {"done": {}, "empty": {}, "failed": {}}, today=TODAY,
                                 concurrency=1, stop=stop, gate=gate, save=lambda s: None)
    reason, prog = run(go())
    assert reason == "stopped" and prog.done == 3


def test_jobspec_validation():
    s = bf.JobSpec.clean({"intervals": ["1", "D", "1"], "years": "5", "rate": 2, "symbols": [" reliance ", "M&M"]})
    assert s.intervals == ["D", "1"] and s.symbols == ["M&M", "RELIANCE"] and s.market_hours == "slow"
    for bad in ({"intervals": []}, {"intervals": ["2"]}, {"intervals": ["D"], "rate": 9},
                {"intervals": ["D"], "reserve": 7000}, {"intervals": ["D"], "reserve": 500},
                {"intervals": ["D"], "market_hours": "fast"},
                {"intervals": ["D"], "symbols": ["DROP TABLE"]}, {"intervals": ["D"], "years": 0}):
        with pytest.raises(ValueError):
            bf.JobSpec.clean(bad)


def test_job_file_round_trip(tmp_path):
    p = tmp_path / "job.json"
    bf.save_job(bf.JobSpec.clean({"intervals": ["D"], "years": 3}), p)
    j = bf.load_job(p)
    assert j.intervals == ["D"] and j.years == 3 and j.active
    bf.save_job(None, p)
    assert bf.load_job(p) is None


def test_summarize():
    jobs = bf.make_jobs(bf.targets(UNI), ["D", "1"], 5, TODAY)
    assert sum(j.interval == "D" for j in jobs) == 2                  # one daily request per symbol
    s = bf.summarize(jobs, budget_left=1000, reserve=800, rate=3)
    assert s["requests"] == len(jobs) == 2 + 2 * 21 and s["calls_left_today"] == 200 and s["days_of_quota"] == 1
    assert bf.summarize(jobs, budget_left=800, reserve=800)["days_of_quota"] == 1          # all tomorrow
    assert bf.summarize(jobs * 300, budget_left=7000, reserve=800)["days_of_quota"] == 3   # 13,200 requests
    assert bf.summarize([], budget_left=0)["days_of_quota"] == 0


def test_lake_coverage_and_locks(tmp_path):
    lake = Lake(tmp_path / "lake")
    rest = FakeRest()
    st = {"done": {}, "empty": {}, "failed": {}}
    run(bf.run_jobs(bf.make_jobs(bf.targets(UNI), ["D"], 1, TODAY), rest, lake, st, today=TODAY, rate=50,
                    save=lambda s: None))
    lake.write("D", "M&M", lake.read("D", "RELIANCE"))
    cov = bf.lake_coverage(lake)
    assert set(cov["symbols"]) == {"NIFTY", "RELIANCE", "M&M"}
    assert cov["intervals"]["D"]["symbols"] == 3 and cov["intervals"]["D"]["bytes"] > 0
    assert (tmp_path / "lake" / "D" / ".locks").is_dir()               # cross-process write locks
    assert sorted(lake.symbols("D")) == ["M&M", "NIFTY", "RELIANCE"]   # lock files are not symbols


# ------------------------------------------------------------------ redis-backed: lock, worker, endpoints
fakeredis = pytest.importorskip("fakeredis")


def test_run_lock():
    async def go():
        r = fakeredis.aioredis.FakeRedis()
        a, b = bf.RunLock(r, "a"), bf.RunLock(r, "b")
        assert await a.acquire() and not await b.acquire()
        assert await b.holder() == "a" and not await b.refresh()
        await b.release()                     # not the owner: no effect
        assert await a.holder() == "a"
        await r.flushall()                    # Redis restarted: the owner takes the lease back
        assert await a.refresh() and await a.holder() == "a"
        await a.release()
        assert await b.acquire()
    run(go())


def test_unreadable_state_is_kept(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{not json")
    st = bf.load_state(p)
    assert st == {"done": {}, "empty": {}, "failed": {}} and not p.exists()
    assert len(list(tmp_path.glob("state.bad.*"))) == 1


def _worker(monkeypatch, tmp_path):
    from td.config import settings
    from td.services import backfiller as wk
    r = fakeredis.aioredis.FakeRedis()
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "data_source", "dhan")
    monkeypatch.setattr(settings, "dhan_client_id", "1")
    monkeypatch.setattr(settings, "dhan_access_token", "t")
    monkeypatch.setattr(wk.bus, "connect", lambda *a, **k: r)
    return wk, r


def test_worker_commands(monkeypatch, tmp_path):
    wk, r = _worker(monkeypatch, tmp_path)

    async def go():
        import time as _t
        from td import bus
        await r.set(bus.KEY_UNIVERSE, json.dumps(UNI))
        await r.set(bus.k_status("ingestor"), json.dumps({"mode": "dhan", "updated": _t.time()}))
        w = wk.Backfiller()
        await w._handle({"op": "plan", "id": "p1", "job": {"intervals": ["D", "1"], "years": 1}})
        await asyncio.gather(*w._bg)                       # plans are answered off the command loop
        plan = json.loads(await r.get(bus.k_admin_plan("p1")))
        assert plan["symbols"] == 2 and plan["requests"] == 2 * (1 + 5)
        await w._handle({"op": "start", "by": "me", "job": {"intervals": ["D"], "years": 2, "rate": 2}})
        assert w.job.active and w.job.by == "me" and bf.load_job().years == 2
        await w._handle({"op": "start", "job": {"intervals": ["D"], "rate": 99}})     # rejected
        assert w.job.rate == 2
        await w._handle({"op": "stop", "by": "me"})
        assert not w.job.active and not bf.load_job().active and w.halt.is_set()
    run(go())


def test_worker_pauses_for_live_history_and_market_hours(monkeypatch, tmp_path):
    wk, r = _worker(monkeypatch, tmp_path)

    async def go():
        import time as _t
        from td import bus
        w = wk.Backfiller()
        w.job = bf.JobSpec.clean({"intervals": ["D"], "market_hours": "pause"})
        night = datetime(2026, 10, 13, 1, 0, tzinfo=IST)
        monkeypatch.setattr(wk, "now_ist", lambda: night)
        assert await w._pause_reason() is None
        await r.set(bus.k_status("ingestor"), json.dumps({"updated": _t.time(),
                                                          "stats": {"history": {"stage": "daily bars"}}}))
        w._ing = (0.0, None)
        assert "loading its own history" in await w._pause_reason()
        await r.set(bus.k_status("ingestor"), json.dumps({"updated": _t.time(),
                                                          "stats": {"history": {"stage": "ready"}}}))
        w._ing = (0.0, None)
        noon = datetime(2026, 10, 13, 12, 0, tzinfo=IST)
        monkeypatch.setattr(wk, "now_ist", lambda: noon)
        assert "market hours" in await w._pause_reason()
        w.job.market_hours = "slow"
        assert await w._pause_reason() is None and await w._gate() == 1.0
        monkeypatch.setattr(wk, "now_ist", lambda: datetime(2026, 10, 13, 8, 40, tzinfo=IST))
        assert "08:40" in await w._pause_reason()
        monkeypatch.setattr(wk, "now_ist", lambda: datetime(2026, 10, 17, 12, 0, tzinfo=IST))   # Saturday
        assert await w._pause_reason() is None and await w._gate() == w.job.rate
    run(go())


def test_worker_restores_budget_floor(monkeypatch, tmp_path):
    wk, r = _worker(monkeypatch, tmp_path)

    async def go():
        w = wk.Backfiller()
        day = wk.now_ist().date().isoformat()
        (tmp_path / "backfill").mkdir(exist_ok=True)
        (tmp_path / "backfill" / "usage.json").write_text(json.dumps({"date": day, "calls": 1234}))
        await w._budget_floor()                              # Redis restarted: counter is 0 < 1234
        assert await w.budget.used() == 1234
        await r.set(w.budget._key(), 5000)                   # the ingestor spends more…
        assert await w._floor() == 5000
        await r.flushall()                                   # …then Redis restarts again
        w._usage = {}
        assert await w._floor() == 5000 and await w.budget.used() == 5000
    run(go())


def test_empty_long_range_is_not_recorded(tmp_path):
    class Empty(FakeRest):
        async def intraday(self, *a, **k):
            await super().intraday(*a, **k)
            return {}
    jobs = bf.make_jobs(bf.targets(UNI, {"RELIANCE"}), ["1"], 1, TODAY)
    st = {"done": {}, "empty": {}, "failed": {}}
    _, prog = run(bf.run_jobs(jobs, Empty(), Lake(tmp_path), st, today=TODAY, rate=50, save=lambda s: None))
    longs = [j for j in jobs if (date.fromisoformat(j.end) - date.fromisoformat(j.start)).days > bf.EMPTY_OK_DAYS]
    assert prog.failed == len(longs) and len(st["failed"]) == len(longs)
    assert not any(k in st["empty"] for k in st["failed"])


def test_gate_pause_does_not_outlive_the_pull():
    """Quota runs out while other workers are parked in a paused gate: the pull must end at once."""
    async def go():
        calls = []

        async def gate():
            calls.append(1)
            if len(calls) > 2:
                await asyncio.sleep(3600)          # the others: parked (e.g. market hours, paused)
            return 100.0

        class L:
            def write(self, *a):
                pass

        jobs = bf.make_jobs(bf.targets(UNI), ["1"], 5, TODAY)
        rest = FakeRest(left=2)                    # worker 1 makes one call, worker 2 then finds the reserve
        return await asyncio.wait_for(bf.run_jobs(jobs, rest, L(), {"done": {}, "empty": {}, "failed": {}},
                                                  today=TODAY, reserve=1, concurrency=4, gate=gate,
                                                  save=lambda s: None), timeout=10)
    reason, prog = run(go())
    assert reason == "reserve" and prog.done == 1


def test_lake_tail_reads_and_footer_coverage(tmp_path):
    import pandas as pd
    from td import lake as lk
    lake = Lake(tmp_path)
    n = lk.ROW_GROUP * 2 + 123
    df = pd.DataFrame({"ts": range(1_000_000, 1_000_000 + n), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 1, "oi": 0})
    lake.write("1m", "BIG", df)
    t = lake.read("1m", "BIG", last_n=500)
    assert len(t) == 500 and t["ts"].iat[-1] == 1_000_000 + n - 1 and t["ts"].iat[0] == 1_000_000 + n - 500
    assert len(lake.read("1m", "BIG", last_n=lk.ROW_GROUP + 10)) == lk.ROW_GROUP + 10      # spans row groups
    assert lake.read("1m", "BIG", last_n=0).empty and len(lake.read("1m", "BIG")) == n
    assert lake.last_ts("1m", "BIG") == 1_000_000 + n - 1
    assert lk.file_stats(lake.path("1m", "BIG")) == (n, 1_000_000, 1_000_000 + n - 1)
    cov = {c["symbol"]: c for c in lake.coverage()}
    assert cov["BIG"]["rows"] == n and cov["BIG"]["bytes"] > 0
    lake.write("D", "M&M", df.head(3))
    assert "M&M" in {c["symbol"] for c in lake.coverage()}                  # escaped names come back as is


def test_worker_ignores_a_simulator_universe(monkeypatch, tmp_path):
    wk, r = _worker(monkeypatch, tmp_path)

    async def go():
        import time as _t
        from td import bus
        from td.dhan import instruments
        sim = [{"sid": 1, "seg": "NSE_EQ", "symbol": "HDFCBANK", "kind": "EQ"}]
        await r.set(bus.KEY_UNIVERSE, json.dumps(sim))
        await r.set(bus.k_status("ingestor"), json.dumps({"mode": "sim", "updated": _t.time()}))

        class U:
            def to_json(self):
                return UNI
        monkeypatch.setattr(instruments, "dhan_universe", lambda *a: U())
        w = wk.Backfiller()
        assert await w._universe() == UNI                                       # not the sim list
        await r.set(bus.k_status("ingestor"), json.dumps({"mode": "dhan", "updated": _t.time()}))
        w._ing = (0.0, None)
        assert await w._universe() == sim                                       # a live ingestor's list
    run(go())


def test_worker_keeps_the_newer_job(monkeypatch, tmp_path):
    wk, r = _worker(monkeypatch, tmp_path)

    async def go():
        w = wk.Backfiller()
        old = bf.JobSpec.clean({"intervals": ["D"], "years": 1})
        w.job = old
        await w._handle({"op": "start", "job": {"intervals": ["1"], "years": 2}})   # arrives mid-run
        assert w.halt.is_set()
        w._done(old, "complete", bf.Progress())                                 # the old run finishing
        assert w.job.intervals == ["1"] and w.job.active and bf.load_job().intervals == ["1"]
    run(go())


# --------------------------------------------------------------- stocks with a shorter history than the window
NEWCO = [{"sid": 7001, "seg": "NSE_EQ", "symbol": "NEWCO", "kind": "EQ"}]


class Since(FakeRest):
    """Dhan with bars only from `first` (daily and intraday) for every symbol — like a stock listed that day, or
    one whose intraday archive starts there (`daily_from` lets daily bars start earlier)."""

    def __init__(self, first: date, daily_from: date | None = None, hole: tuple[date, date] | None = None, **kw):
        super().__init__(**kw)
        self.first, self.daily_from, self.hole = first, daily_from or first, hole

    async def daily(self, sid, seg, kind, start, end, oi=False, expiry_code=None, retries=3):
        self.calls.append(("D", sid, start, end))
        self.data_budget.left -= 1
        return self._bars(max(start, self.daily_from), min(end, TODAY), 1)

    async def intraday(self, sid, seg, kind, interval, start, end, oi=False, retries=3):
        self.calls.append(("1", sid, start.date(), end.date()))
        self.data_budget.left -= 1
        s, e = start.date(), min(end.date() + timedelta(days=1), TODAY)
        if self.hole and s < self.hole[1] and e > self.hole[0]:
            return {}                                        # a bar-less answer in the middle of the history
        return self._bars(max(s, self.first), e, 3) if e > self.first else {}


def test_stock_listed_later_is_not_failed_or_asked_again(tmp_path):
    lake, listed = Lake(tmp_path / "lake"), date(2025, 3, 3)
    st = {"done": {}, "empty": {}, "failed": {}}
    jobs = bf.make_jobs(bf.targets(NEWCO), ["D", "1"], 5, TODAY)
    rest = Since(listed)
    # first pull: the daily request (one call, first in the queue) shows the listing day; every intraday range
    # before it comes back empty and is recorded as covered instead of failing on every run
    _, prog = run(bf.run_jobs(jobs, rest, lake, st, today=TODAY, rate=50, concurrency=1, save=lambda s: None))
    before = [j for j in jobs if j.interval == "1" and date.fromisoformat(j.end) <= listed]
    assert before and prog.failed == 0 and prog.before_history == len(before)
    assert prog.ok == len(jobs) - len(before)
    # the plan knows the listing day: nothing is asked for again — and a fresh state still skips it
    assert bf.listing_floor(st, lake, TODAY) == {"NEWCO": listed}
    assert bf.make_jobs(bf.targets(NEWCO), ["D", "1"], 5, TODAY, bf.coverage_from_state(st, TODAY)) == []
    fresh = bf.make_jobs(bf.targets(NEWCO), ["1"], 5, TODAY, floor=bf.listing_floor(st, lake, TODAY))
    assert fresh and min(date.fromisoformat(j.start) for j in fresh) == listed


def test_old_stock_past_the_intraday_archive_is_recorded(tmp_path):
    lake = Lake(tmp_path / "lake")
    jobs = bf.make_jobs(bf.targets(NEWCO), ["1"], 5, TODAY)
    oldest = jobs[-1]                                        # newest first: the last job is the window's oldest chunk
    st = {"done": {}, "empty": {}, "failed": {}}
    rest = Since(date.fromisoformat(oldest.end), daily_from=TODAY - timedelta(days=4000))
    _, prog = run(bf.run_jobs(jobs, rest, lake, st, today=TODAY, rate=50, concurrency=1, save=lambda s: None))
    assert prog.failed == 0 and prog.before_history == 1 and oldest.key in st["empty"]


def test_empty_range_inside_the_history_still_fails(tmp_path):
    """A stock that was trading (daily bars) answering an empty middle range is suspicious: retried next run."""
    lake = Lake(tmp_path / "lake")
    jobs = bf.make_jobs(bf.targets(NEWCO), ["D", "1"], 5, TODAY)
    mid = [j for j in jobs if j.interval == "1"][6]                       # ~1.5 years back
    hole = (date.fromisoformat(mid.start), date.fromisoformat(mid.end))
    st = {"done": {}, "empty": {}, "failed": {}}
    rest = Since(TODAY - timedelta(days=4000), hole=hole)
    _, prog = run(bf.run_jobs(jobs, rest, lake, st, today=TODAY, rate=50, concurrency=1, save=lambda s: None))
    assert prog.failed == 1 and list(st["failed"]) == [mid.key] and prog.before_history == 0
