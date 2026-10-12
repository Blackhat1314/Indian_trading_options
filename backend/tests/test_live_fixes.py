"""Regression tests for live-mode issues seen with real Dhan data."""
import asyncio
from datetime import date, datetime, timedelta

from td.dhan.instruments import Instrument, Universe, ikey
from td.lake import Lake
from td.services.bootstrap import HistoryBootstrap, k_seed
from td.timeutil import IST


def test_keys_unique_across_segments():
    nifty = Instrument(sid=13, seg="IDX_I", symbol="NIFTY", kind="IDX", underlying="NIFTY")
    abb = Instrument(sid=13, seg="NSE_EQ", symbol="ABB", kind="EQ", underlying="ABB")
    fut = Instrument(sid=13, seg="NSE_FNO", symbol="X-FUT", kind="FUT", underlying="X")
    assert len({nifty.key, abb.key, fut.key}) == 3 and abb.key == 13
    assert ikey(0, 13) == nifty.key and ikey("NSE_FNO", 13) == fut.key
    # survives the JSON round trip the ingestor publishes
    u = Universe.from_json(Universe([nifty, abb]).to_json())
    assert [i.key for i in u.instruments] == [nifty.key, abb.key]


def test_publisher_rewrites_packet_ids():
    from td.services.ingestor import Publisher

    class Rec:
        def ticks(self, *a):
            pass

    pub = Publisher(None, Rec())
    pub.add([{"t": "tick", "seg": 0, "sid": 13, "ltp": 22400.0}, {"t": "quote", "seg": 1, "sid": 13, "ltp": 5100.0},
             {"t": "prev", "seg": 0, "sid": 13, "pc": 22500.0, "poi": 0}])
    assert pub.buf[0]["sid"] == ikey(0, 13) and pub.buf[0]["dsid"] == 13
    assert pub.buf[1]["sid"] == 13
    assert ikey(0, 13) in pub.prev_updates and 13 not in pub.prev_updates


class FakeRest:
    """Dhan historical API stand-in: daily bars for the last 20 weekdays, 1-minute bars per day."""

    def __init__(self, today: date):
        self.today = today
        self.calls = []

    def _days(self, start: date, end: date):
        d = start
        while d < end:
            if d.weekday() < 5:
                yield d
            d += timedelta(days=1)

    async def daily(self, sid, seg, kind, start, end, oi=False, expiry_code=None):
        self.calls.append(("daily", sid, seg, kind, expiry_code))
        days = list(self._days(max(start, self.today - timedelta(days=30)), end))
        ts = [int(datetime(d.year, d.month, d.day, tzinfo=IST).timestamp()) for d in days]
        n = len(ts)
        out = {"timestamp": ts, "open": [100.0 + i for i in range(n)], "high": [101.0 + i for i in range(n)],
               "low": [99.0 + i for i in range(n)], "close": [100.5 + i for i in range(n)], "volume": [1000] * n}
        if oi:
            out["open_interest"] = [5000 + i for i in range(n)]
        return out

    async def intraday(self, sid, seg, kind, interval, start, end, oi=False):
        self.calls.append(("intraday", sid, start.date()))
        ts, d = [], start.date()
        while d <= end.date():
            if d.weekday() < 5:
                t0 = datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST).timestamp()
                last = 375 if d < self.today else 30
                ts += [int(t0 + 60 * m) for m in range(last)]
            d += timedelta(days=1)
        n = len(ts)
        return {"timestamp": ts, "open": [10.0] * n, "high": [11.0] * n, "low": [9.0] * n, "close": [10.5] * n,
                "volume": [100] * n}


class FakeRedis:
    def __init__(self):
        self.kv, self.events = {}, []

    async def set(self, k, v, ex=None):
        self.kv[k] = v

    async def xadd(self, stream, fields, maxlen=None, approximate=None):
        self.events.append(fields)


def test_history_bootstrap(tmp_path, monkeypatch):
    from td.services import bootstrap as bs
    from td.services.ingestor import Publisher

    class Rec:
        def ticks(self, *a):
            pass

    today = date(2026, 10, 12)
    monkeypatch.setattr(bs, "now_ist", lambda: datetime(2026, 10, 12, 10, 0, tzinfo=IST))
    uni = Universe([Instrument(sid=13, seg="IDX_I", symbol="NIFTY", kind="IDX", underlying="NIFTY"),
                    Instrument(sid=13, seg="NSE_EQ", symbol="ABB", kind="EQ", underlying="ABB"),
                    Instrument(sid=50013, seg="NSE_FNO", symbol="ABB-FUT", kind="FUT", underlying="ABB")])
    pub = Publisher(None, Rec())
    r = FakeRedis()
    rest = FakeRest(today)
    hb = HistoryBootstrap(r, rest, uni, pub, Lake(tmp_path))
    asyncio.run(hb.run_once())
    prev = pub.prev_updates
    assert prev[ikey("IDX_I", 13)][0] > 0 and prev[13][0] > 0            # NIFTY and ABB kept apart
    assert prev[ikey("NSE_FNO", 50013)][1] > 0                           # yesterday's futures OI
    assert ("daily", 50013, "NSE_FNO", "FUTSTK", 0) in rest.calls
    lake = Lake(tmp_path)
    assert lake.last_ts("D", "ABB") is not None and lake.last_ts("1m", "ABB") is not None
    seed = r.kv.get(k_seed(13))
    assert seed and b'"bars"' in seed
    assert [e["j"] for e in r.events]
    assert hb.status["stage"] == "ready"


def test_redis_timeout_survives_slow_loop():
    """redis-py >= 6 defaults to a 5 s socket timeout; a busy event loop must not kill blocking XREADs."""
    from td import bus
    r = bus.connect("redis://localhost:6379/0")
    assert r.connection_pool.connection_kwargs["socket_timeout"] >= 30


def test_supervise_restarts_crashed_loop():
    from td.services.analytics import supervise
    calls = []

    async def loop():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("boom")

    async def run():
        import td.services.analytics as a
        orig = a.asyncio.sleep

        async def fast(_):
            await orig(0)
        a.asyncio.sleep = fast
        try:
            await supervise(loop)
        finally:
            a.asyncio.sleep = orig
    asyncio.run(run())
    assert len(calls) == 3


def test_missing_iv_filled():
    from td.analytics import optmath
    row = {"strike": 21500.0, "ce_ltp": 932.05, "ce_iv": 0.0, "ce_delta": 0.0, "ce_gamma": 0, "ce_theta": 0,
           "ce_vega": 0, "pe_ltp": 0.0, "pe_iv": 0.0, "pe_delta": 0.0, "pe_gamma": 0, "pe_theta": 0, "pe_vega": 0}
    optmath.fill_missing_greeks([row], 22423.1, 2 / 365)
    assert row["ce_iv"] and 0.9 < row["ce_delta"] <= 1.0
    assert row["pe_iv"] is None and row["pe_delta"] == 0.0
