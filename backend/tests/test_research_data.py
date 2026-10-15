"""Research data layer (td.research.data): symbols, search, daily history with the one-call Dhan top-up, live spot,
news and market context — fakeredis, a temp lake and a fake Dhan client, no network."""
import asyncio
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from td import bus
from td.config import settings
from td.dhan.instruments import ikey
from td.dhan.rest import DhanError
from td.lake import Lake
from td.ratelimit import DailyBudget
from td.research import data as rdata
from td.research.data import ResearchData, clean_daily, day_start
from td.timeutil import IST

fakeredis = pytest.importorskip("fakeredis")

TODAY = date(2026, 10, 15)                     # a Thursday
UNI = [{"symbol": "RELIANCE", "kind": "EQ", "sid": 2885, "seg": "NSE_EQ", "sector": "ENERGY",
        "name": "Reliance Industries", "underlying": "RELIANCE"},
       {"symbol": "TCS", "kind": "EQ", "sid": 11536, "seg": "NSE_EQ", "sector": "IT", "name": "Tata Consultancy",
        "underlying": "TCS"},
       {"symbol": "NIFTY", "kind": "IDX", "sid": 13, "seg": "IDX_I", "sector": "INDEX", "name": "",
        "underlying": "NIFTY"},
       {"symbol": "RELIANCE-FUT", "kind": "FUT", "sid": 52001, "seg": "NSE_FNO", "sector": "ENERGY",
        "underlying": "RELIANCE"}]
MASTER = "\n".join([
    "EXCH_ID,SEGMENT,SECURITY_ID,ISIN,INSTRUMENT,UNDERLYING_SECURITY_ID,UNDERLYING_SYMBOL,SYMBOL_NAME,DISPLAY_NAME,"
    "INSTRUMENT_TYPE,SERIES",
    "NSE,E,7508,INE251B01027,EQUITY,,ZENTEC,ZEN TECHNOLOGIES LTD,Zen Technologies,ES,EQ",
    'NSE,E,100,INE885A01032,EQUITY,,ARE&M,"AMARA RAJA ENERGY MOB LTD","Amara Raja Energy & Mobility",ES,EQ',
    "NSE,E,9001,INE000X01011,EQUITY,,DUALCO,DUAL CO LTD,Dual Co (BE),ES,BE",
    "NSE,E,9002,INE000X01011,EQUITY,,DUALCO,DUAL CO LTD,Dual Co,ES,EQ",
    "NSE,E,1000,IN2220200306,EQUITY,,656MH32,SDL MH 6.56% 2032,SDL MH 6.56% 2032,DBT,SG",
    "BSE,E,500325,INE002A01018,EQUITY,,RELIANCE,RELIANCE INDUSTRIES,Reliance Industries,ES,A",
    "NSE,E,10176,INF200KA1FS1,EQUITY,,SETFNIF50,SBI-ETF NIFTY 50,SBI Nifty 50 ETF,ETF,EQ",
]) + "\n"


def midnight(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=IST).timestamp())


def weekdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_bars(start: date, end: date, px: float = 100.0, seed: int = 0) -> pd.DataFrame:
    days = weekdays(start, end)
    rng = np.random.default_rng(seed)
    close = px * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    return pd.DataFrame({"ts": [midnight(d) for d in days], "open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": 1000, "oi": 0})


class FakeRest:
    """Dhan historical API stand-in: one bar per weekday in [start, end] (end included, like a live partial bar)."""

    def __init__(self, log: list, fail: Exception | None = None, offset: int = 0, delay: float = 0.0):
        self.log, self.fail, self.offset, self.delay = log, fail, offset, delay
        self.closed = False

    async def daily(self, sid, seg, instrument, start, end, **kw):
        self.log.append({"sid": sid, "seg": seg, "instrument": instrument, "start": start, "end": end})
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail
        days = weekdays(start, end)
        px = np.linspace(500, 600, len(days)) if days else []
        return {"timestamp": [midnight(d) + self.offset for d in days], "open": list(px), "high": list(px),
                "low": list(px), "close": list(px), "volume": [10] * len(days)}

    async def close(self):
        self.closed = True
        self.log.append("closed")


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "data_source", "dhan")
    monkeypatch.setattr(settings, "limit_data_per_day", 7000)
    monkeypatch.setattr(settings, "research_fetch_reserve", 500)
    monkeypatch.setattr(rdata, "now_ist", lambda: datetime(2026, 10, 15, 11, 0, tzinfo=IST))
    monkeypatch.setattr(rdata, "download_scrip_master", lambda: MASTER)
    return tmp_path


def setup(tmp_path, fail=None, offset=0, delay=0.0, universe=UNI):
    """(ResearchData, lake, calls) on a fresh FakeRedis — call inside the event loop."""
    r = fakeredis.aioredis.FakeRedis()
    lake = Lake(tmp_path / "lake")
    calls: list = []
    rd = ResearchData(r, lake, rest_factory=lambda: FakeRest(calls, fail, offset, delay))
    return rd, lake, calls, r


async def put_universe(r, rows=UNI):
    await r.set(bus.KEY_UNIVERSE, bus.dumps(rows))


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ cleaning
def test_clean_daily():
    raw = pd.DataFrame({"ts": [3, 1, 2, 2, 4, 5], "open": [1, np.nan, 2, 2.5, 4, 5], "high": [1, 1, 2, 2.5, 4, 5],
                        "low": [1, 1, 2, 2.5, 4, 5], "close": [3.0, 1.0, 2.0, 2.2, np.nan, -1.0],
                        "volume": [1, np.nan, 1, 1, 1, 1], "oi": 0})
    d = clean_daily(raw)
    assert list(d.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert d["ts"].tolist() == [1, 2, 3]                                  # sorted, NaN / negative closes gone
    assert d.loc[d["ts"] == 2, "close"].item() == 2.2                     # duplicate ts: the last one wins
    assert d.loc[d["ts"] == 1, "open"].item() == 1.0                      # missing open -> close
    assert d["volume"].tolist() == [0, 1, 1]
    assert clean_daily(None).empty and clean_daily(pd.DataFrame()).empty


def test_day_start_is_ist_midnight():
    t = midnight(date(2026, 10, 12))
    assert t == 1791743400                                                # the contract's example
    assert day_start([t, t + 19800, t + 86399]).tolist() == [t, t, t]


# ------------------------------------------------------------------ symbols
def test_resolve_and_search_sim(env, monkeypatch):
    monkeypatch.setattr(settings, "data_source", "sim")

    async def main():
        rd, lake, calls, r = setup(env)
        lake.write("D", "RELIANCE", make_bars(date(2025, 1, 1), date(2026, 10, 14)))
        lake.write("D", "LAKEONLY", make_bars(date(2025, 1, 1), date(2026, 10, 14)))
        await put_universe(r)
        info = await rd.resolve("reliance")
        assert (info.symbol, info.sid, info.seg, info.kind, info.in_universe) == \
            ("RELIANCE", 2885, "NSE_EQ", "EQ", True)
        assert info.name == "Reliance Industries" and info.sector == "ENERGY"
        nifty = await rd.resolve("NIFTY")
        assert nifty.kind == "IDX" and nifty.name == "Nifty 50"
        lo = await rd.resolve("LAKEONLY")
        assert not lo.in_universe and lo.sid is None
        with pytest.raises(KeyError, match="simulator"):
            await rd.resolve("ZENTEC")                                    # sim mode never reads the scrip master
        with pytest.raises(KeyError):
            await rd.resolve("RELIANCE-FUT")                              # futures are not research symbols
        hits = await rd.search("rel")
        assert hits[0] == {"symbol": "RELIANCE", "name": "Reliance Industries", "sector": "ENERGY",
                           "in_universe": True, "has_history": True}
        assert [h["symbol"] for h in await rd.search("tata con")] == ["TCS"]          # by company name
        assert await rd.search("  ") == [] and len(await rd.search("e", limit=2)) == 2
        assert {h["symbol"]: h["has_history"] for h in await rd.search("lake")} == {"LAKEONLY": True}
        # no universe in Redis (gateway without the ingestor): the built-in sim universe still resolves
        await r.delete(bus.KEY_UNIVERSE)
        assert (await rd.resolve("HDFCBANK")).in_universe
    run(main())


def test_scrip_master_symbols(env):
    async def main():
        rd, lake, calls, r = setup(env)
        await put_universe(r)
        z = await rd.resolve("zentec")
        assert (z.symbol, z.sid, z.seg, z.name, z.in_universe) == ("ZENTEC", 7508, "NSE_EQ", "Zen Technologies", False)
        assert (await rd.resolve("ARE&M")).name == "Amara Raja Energy & Mobility"
        assert (await rd.resolve("DUALCO")).sid == 9002                  # main-board series preferred over BE
        for bad in ("656MH32", "SETFNIF50", "NOSUCH"):                    # bonds, ETFs, unknown
            with pytest.raises(KeyError):
                await rd.resolve(bad)
        hits = await rd.search("amara")
        assert hits and hits[0]["symbol"] == "ARE&M" and not hits[0]["in_universe"] and not hits[0]["has_history"]
        assert (await rd.search("arem"))[0]["symbol"] == "ARE&M"           # punctuation-insensitive
        assert (await rd.search("RELIANCE"))[0]["in_universe"]            # the universe row wins over the master
    run(main())


def test_scrip_master_failure_is_reported(env, monkeypatch):
    def boom():
        raise OSError("no network")
    monkeypatch.setattr(rdata, "download_scrip_master", boom)

    async def main():
        rd, lake, calls, r = setup(env)
        await put_universe(r)
        assert (await rd.resolve("TCS")).in_universe                      # the universe still works
        with pytest.raises(KeyError, match="scrip master"):
            await rd.resolve("ZENTEC")
        assert await rd.search("zen") == []
    run(main())


# ------------------------------------------------------------------ daily history + Dhan top-up
def test_empty_lake_fetches_ten_years_once(env):
    async def main():
        rd, lake, calls, r = setup(env)
        await put_universe(r)
        df, meta = await rd.daily("ZENTEC")
        fetches = [c for c in calls if c != "closed"]
        assert len(fetches) == 1 and calls[-1] == "closed"               # one call, client closed
        c = fetches[0]
        assert (c["sid"], c["seg"], c["instrument"], c["end"]) == (7508, "NSE_EQ", "EQUITY", TODAY)
        assert TODAY - c["start"] >= timedelta(days=3650) and c["start"].weekday() < 5
        assert meta["fetched"] and meta["source"] == "lake+dhan" and meta["warning"] is None
        assert df["ts"].iat[-1] == midnight(date(2026, 10, 14))           # today's forming bar is dropped
        assert lake.last_ts("D", "ZENTEC") == midnight(date(2026, 10, 14))
        assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
        df2, meta2 = await rd.daily("ZENTEC")                             # fresh now: no second call
        assert len([c for c in calls if c != "closed"]) == 1 and not meta2["fetched"] and len(df2) == len(df)
    run(main())


def test_stale_lake_fetches_only_the_gap(env):
    async def main():
        rd, lake, calls, r = setup(env)
        await put_universe(r)
        lake.write("D", "RELIANCE", make_bars(date(2022, 1, 3), date(2026, 10, 2)))   # Friday two weeks back
        df, meta = await rd.daily("RELIANCE")
        c = [x for x in calls if x != "closed"]
        assert len(c) == 1 and c[0]["start"] == date(2026, 10, 5)         # last bar + 1 day, skipping the weekend
        assert meta["fetched"] and df["ts"].iat[-1] == midnight(date(2026, 10, 14))
        assert df["ts"].is_unique and df["ts"].is_monotonic_increasing
    run(main())


def test_fresh_lake_makes_no_call(env):
    async def main():
        rd, lake, calls, r = setup(env)
        await put_universe(r)
        lake.write("D", "TCS", make_bars(date(2023, 1, 2), date(2026, 10, 14)))
        df, meta = await rd.daily("TCS")
        assert calls == [] and not meta["fetched"] and meta["source"] == "lake" and meta["warning"] is None
        assert len(df) == len(weekdays(date(2023, 1, 2), date(2026, 10, 14)))
        df, meta = await rd.daily("TCS", fetch=False)
        assert calls == []
    run(main())


def test_short_history_refetches_full_range_once_per_day(env):
    async def main():
        rd, lake, calls, r = setup(env)
        await put_universe(r)
        lake.write("D", "TCS", make_bars(date(2025, 6, 2), date(2026, 10, 14)))      # 1.4 years, up to date
        df, meta = await rd.daily("TCS")
        c = [x for x in calls if x != "closed"]
        assert len(c) == 1 and TODAY - c[0]["start"] >= timedelta(days=3650)
        assert meta["fetched"] and df["ts"].iat[0] < midnight(date(2017, 1, 1))
        # a stock that really is young: Dhan has nothing older, so no refetch for the rest of the day
        rd2, lake2, calls2, r2 = setup(env)
        await put_universe(r2)
        rd2.rest_factory = lambda: FakeRest(calls2)
        lake2.write("D", "ZENTEC", make_bars(date(2025, 6, 2), date(2026, 10, 14)))

        async def young(sid, seg, instrument, start, end, **kw):
            calls2.append(start)
            return {"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
        rd2.rest_factory = lambda: type("R", (), {"daily": staticmethod(young)})()
        for _ in range(3):
            df, meta = await rd2.daily("ZENTEC")
            assert "only 1.4 years" in meta["warning"] and not meta["fetched"]
        assert len(calls2) == 1
    run(main())


def test_budget_reserve_blocks_fetch(env):
    async def main():
        rd, lake, calls, r = setup(env)
        await put_universe(r)
        lake.write("D", "RELIANCE", make_bars(date(2022, 1, 3), date(2026, 10, 2)))
        b = DailyBudget("dhan_data", 7000, r)
        await r.set(b._key(), 6600)                                       # 400 left < 500 reserve
        df, meta = await rd.daily("RELIANCE")
        assert calls == [] and not meta["fetched"]
        w = meta["warning"]
        assert "budget" in w and "400 calls left" in w and "using the lake" in w
        assert not df.empty                                               # the lake is still served
    run(main())


def test_failure_becomes_warning_with_negative_cache(env):
    async def main():
        rd, lake, calls, r = setup(env, fail=DhanError("/charts/historical: Invalid token", code="DH-901", status=401))
        await put_universe(r)
        lake.write("D", "RELIANCE", make_bars(date(2022, 1, 3), date(2026, 10, 2)))
        df, meta = await rd.daily("RELIANCE")
        assert len([c for c in calls if c != "closed"]) == 1 and "closed" in calls
        assert "access token" in meta["warning"] and "using the lake" in meta["warning"] and not df.empty
        df, meta = await rd.daily("RELIANCE")                             # within 10 minutes: no retry
        assert len([c for c in calls if c != "closed"]) == 1 and "access token" in meta["warning"]
        rd._neg["RELIANCE"] = (time.monotonic() - 1, rd._neg["RELIANCE"][1])        # 10 minutes later
        await rd.daily("RELIANCE")
        assert len([c for c in calls if c != "closed"]) == 2
        # any other exception is contained too
        rd3, lake3, calls3, r3 = setup(env, fail=RuntimeError("boom"))
        await put_universe(r3)
        df, meta = await rd3.daily("ZENTEC")
        assert df.empty and "no history available" in meta["warning"] and "no daily history" in meta["warning"]
    run(main())


def test_dhan_timestamps_are_normalised(env):
    async def main():
        rd, lake, calls, r = setup(env, offset=19800)                     # UTC-midnight stamps (05:30 IST)
        await put_universe(r)
        lake.write("D", "RELIANCE", make_bars(date(2022, 1, 3), date(2026, 10, 7)))
        df, meta = await rd.daily("RELIANCE")
        assert meta["fetched"]
        assert (df["ts"] == day_start(df["ts"])).all() and df["ts"].is_unique
        assert df["ts"].iat[-1] == midnight(date(2026, 10, 14))
    run(main())


def test_concurrent_requests_share_one_fetch(env):
    async def main():
        rd, lake, calls, r = setup(env, delay=0.05)
        await put_universe(r)
        res = await asyncio.gather(*(rd.daily("ZENTEC") for _ in range(4)))
        assert len([c for c in calls if c != "closed"]) == 1
        assert all(len(df) == len(res[0][0]) for df, _ in res)
    run(main())


def test_sim_mode_never_fetches(env, monkeypatch):
    monkeypatch.setattr(settings, "data_source", "sim")

    async def main():
        rd, lake, calls, r = setup(env)
        await put_universe(r)
        lake.write("D", "RELIANCE", make_bars(date(2025, 7, 1), date(2026, 10, 14)))
        df, meta = await rd.daily("RELIANCE")
        assert calls == [] and not meta["fetched"]
        assert "simulated" in meta["warning"] and "only 1.3 years" in meta["warning"]
        assert meta["warnings"] == meta["warning"].split("; ")
        df, meta = await rd.daily("TCS")                                  # nothing in the lake
        assert df.empty and calls == [] and "no daily history" in meta["warning"]
    run(main())


def test_index_daily(env):
    async def main():
        rd, lake, calls, r = setup(env)
        await put_universe(r)
        lake.write("D", "NIFTY", make_bars(date(2022, 1, 3), date(2026, 10, 14), px=20000))
        lake.write("D", "INDIAVIX", make_bars(date(2022, 1, 3), date(2026, 10, 14), px=12))
        nifty = await rd.index_daily()
        assert len(nifty) > 1000 and calls == []
        vix = await rd.index_daily("INDIAVIX")                            # not in this universe: lake read
        assert len(vix) > 1000
        assert (await rd.index_daily("NOPE")).empty
    run(main())


# ------------------------------------------------------------------ live spot, news, context
def test_spot_live_only_when_fresh(env):
    async def main():
        rd, lake, calls, r = setup(env)
        await put_universe(r)
        lake.write("D", "RELIANCE", make_bars(date(2023, 1, 2), date(2026, 10, 14)))
        lake.write("D", "NIFTY", make_bars(date(2023, 1, 2), date(2026, 10, 14), px=20000))
        last = round(float(lake.read("D", "RELIANCE")["close"].iat[-1]), 2)
        assert await rd.spot("RELIANCE") == (last, "close")              # no quotes at all
        clock = midnight(TODAY) + 11 * 3600
        quotes = {"ts": clock, "q": {2885: [1234.56, 0.5, 100, 1230.0, 0],
                                     ikey("IDX_I", 13): [25000.5, 0.1, 0, 0, 0]}}
        await bus.publish_snapshot(r, "quotes", quotes)
        assert (await rd.spot("RELIANCE"))[1] == "close"                  # analytics worker not running
        await r.set(bus.k_status("analytics"), b"{}", ex=30)
        assert await rd.spot("RELIANCE") == (1234.56, "live")
        assert await rd.spot("NIFTY") == (25000.5, "live")                # index keyed by its unique key
        quotes["ts"] = midnight(date(2026, 10, 13)) + 11 * 3600           # session older than the last bar
        await bus.publish_snapshot(r, "quotes", quotes)
        assert (await rd.spot("RELIANCE"))[1] == "close"
        assert await rd.spot("ZENTEC") == (None, None)                    # no history, not in the live feed
    run(main())


def test_news_for_a_symbol(env):
    async def main():
        rd, lake, calls, r = setup(env)
        await put_universe(r)
        now = time.time()
        items = [
            {"id": "a", "ts": now - 3600, "title": "Reliance bags large order", "symbols": ["RELIANCE"],
             "source": "DHAN", "sentiment": "positive"},
            {"id": "b", "ts": now - 40 * 86400, "title": "Reliance old story", "symbols": ["RELIANCE"]},
            {"id": "c", "ts": now - 7200, "title": "TCS results beat", "symbols": ["TCS"]},
            {"id": "d", "ts": now - 60, "title": "Reliance net profit falls", "symbols": ["RELIANCE"]},
            {"id": "e", "ts": now - 30, "title": "reliance bags large order", "symbols": ["RELIANCE"]},   # dupe
        ]
        for it in items:
            await r.xadd(bus.STREAM_NEWS, {"j": bus.dumps(it)})
        await bus.publish_snapshot(r, "news", {"items": [
            {"id": "d", "ts": now - 60, "title": "Reliance net profit falls", "impacts": {"RELIANCE": -1}},
            {"id": "g", "ts": now - 900, "title": "Crude oil prices surge", "impacts": {"RELIANCE": 1, "ONGC": 1},
             "type": "MACRO", "sentiment": "positive"}]})
        news = await rd.news("reliance")
        assert [n["id"] for n in news] == ["e", "d", "g"]     # newest first; too old and repeated headlines dropped
        d = news[1]
        assert d["type"] == "RESULTS" and d["sentiment"] == "negative" and d["impact"] == -1
        assert news[2]["impact"] == 1 and news[2]["type"] == "MACRO"
        assert set(d) >= {"ts", "date", "title", "text", "source", "sentiment", "type", "impact"}
        assert len(await rd.news("RELIANCE", limit=1)) == 1
        assert [n["id"] for n in await rd.news("RELIANCE", days=60)] == ["e", "d", "g", "b"]
        assert await rd.news("INFY") == []
    run(main())


def test_market_context(env):
    async def main():
        rd, lake, calls, r = setup(env)
        await put_universe(r)
        empty = await rd.market_context("RELIANCE")
        assert empty == {"live": False, "clock": {}, "market": {}, "sector": {}, "swing": {}}
        await bus.publish_snapshot(r, "clock", {"ts": 1, "phase": "OPEN", "date": "2026-10-15", "time": "11:00:00",
                                                "source": "dhan", "minute": 105})
        await bus.publish_snapshot(r, "market", {
            "ts": 1, "phase": "OPEN", "indices": [{"name": "NIFTY", "chg": 0.4}],
            "breadth": {"adv": 120, "dec": 80}, "movers": {},
            "sectors": [{"sector": "IT", "name": "IT", "chg": -0.3, "members": [{"symbol": "TCS"}]},
                        {"sector": "ENERGY", "name": "Energy", "chg": 1.1, "members": [{"symbol": "RELIANCE"}]}]})
        await bus.publish_snapshot(r, "swing", {"ts": 1, "rules": {"target": 6}, "stats": {},
                                                "rows": [{"symbol": "RELIANCE", "setup": "BREAKOUT", "trend": 3,
                                                          "stats": {"n": 5}}]})
        await r.set(bus.k_status("analytics"), b"{}", ex=30)
        ctx = await rd.market_context("RELIANCE")
        assert ctx["live"] and ctx["clock"]["phase"] == "OPEN" and ctx["clock"]["date"] == "2026-10-15"
        assert ctx["market"]["indices"][0]["name"] == "NIFTY" and ctx["market"]["breadth"]["adv"] == 120
        assert all("members" not in s for s in ctx["market"]["sectors"])
        assert ctx["sector"]["name"] == "Energy"
        assert ctx["swing"]["setup"] == "BREAKOUT" and ctx["swing"]["rules"] == {"target": 6}
        assert "stats" not in ctx["swing"]
        allrows = await rd.market_context()
        assert allrows["sector"] == {} and allrows["swing"]["rows"][0]["symbol"] == "RELIANCE"
        assert (await rd.market_context("TCS"))["swing"] == {}
    run(main())


def test_history_info_shape(env):
    async def main():
        rd, lake, calls, r = setup(env)
        await put_universe(r)
        lake.write("D", "TCS", make_bars(date(2016, 1, 4), date(2026, 10, 14)))
        h = await rd.history_info("tcs")
        assert set(h) == {"symbol", "name", "sector", "bars", "first", "last", "years", "spot", "spot_source",
                          "fetched", "warning", "closes"}
        assert (h["symbol"], h["name"], h["sector"]) == ("TCS", "Tata Consultancy", "IT")
        assert h["first"] == "2016-01-04" and h["last"] == "2026-10-14" and h["years"] == pytest.approx(10.78, abs=0.01)
        assert h["spot_source"] == "close" and h["spot"] == h["closes"][-1][1] and not h["fetched"]
        span = (h["closes"][-1][0] - h["closes"][0][0]) / (365.25 * 86400)
        assert 4.9 < span <= 5.0 and all(isinstance(t, int) and isinstance(c, float) for t, c in h["closes"])
        with pytest.raises(KeyError):
            await rd.history_info("NOSUCH")
    run(main())
