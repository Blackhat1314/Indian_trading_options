"""NSE holidays, no-trading days, pick outcomes that stay decided, capped stops and short-side chips."""
import asyncio
from datetime import date, datetime

import fakeredis.aioredis
import pytest

from td import bus
from td import holidays
from td.analytics import intraday
from td.analytics.state import InstState
from td.config import settings
from td.services import analytics as an
from td.services.gateway import _real_session
from td.timeutil import IST, phase


def ts(d: date, h: int, m: int, s: int = 0) -> float:
    return datetime(d.year, d.month, d.day, h, m, s, tzinfo=IST).timestamp()


@pytest.fixture
def live(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_source", "dhan")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "nse_holidays", "")


def engine(n: int = 30, moved: bool = True) -> an.Engine:
    e = an.Engine(fakeredis.aioredis.FakeRedis())
    for i in range(n):
        s = InstState(i + 1, "NSE_EQ", f"S{i}", "EQ", f"S{i}", "BANK")
        s.pc = 100.0
        s.ltp = 100.0 + (0.5 + 0.1 * i if moved else 0.0)
        e.eqs.append(s)
        e.eq_by_symbol[s.symbol] = s
        e.states[s.sid] = s
    e.rows = [{"symbol": "S0"}]          # maybe_lock only needs a ranking to exist before its guards
    return e


# ------------------------------------------------------------------ holidays
def test_holiday_calendar_live_only(monkeypatch, live):
    dussehra = date(2026, 10, 20)
    assert holidays.holiday_name(dussehra) == "Dussehra" and not holidays.is_trading_day(dussehra)
    assert phase(datetime(2026, 10, 20, 10, 0, tzinfo=IST)) == "CLOSED"
    assert phase(datetime(2026, 10, 21, 10, 0, tzinfo=IST)) == "OPEN"
    monkeypatch.setattr(settings, "nse_holidays", "2027-01-26, bad-date")             # next year's list via .env
    assert holidays.is_holiday(date(2027, 1, 26)) and not holidays.is_holiday(date(2027, 1, 27))
    monkeypatch.setattr(settings, "data_source", "sim")                                # the simulator trades daily
    assert not holidays.is_holiday(dussehra) and phase(datetime(2026, 10, 20, 10, 0, tzinfo=IST)) == "OPEN"


def test_no_lock_and_no_picks_on_a_holiday(live):
    e = engine()
    e.clock = ts(date(2026, 10, 20), 9, 30, 5)
    asyncio.run(e.maybe_lock())
    assert not e.top.get("locked")
    v = e.top_view()
    assert v["closed"] == "Dussehra" and v["long"] == [] and v["short"] == []


def test_no_lock_when_no_price_moved(live):
    """A closure missing from the list: every stock still at yesterday's close at 09:30 → no lock, no card."""
    e = engine(moved=False)
    e.clock = ts(date(2026, 10, 21), 9, 30, 5)
    asyncio.run(e.maybe_lock())
    assert not e.top.get("locked") and "no trading" in e.top["closed"]
    assert "no trading" in e.top_view()["closed"]


def test_scorecard_skips_a_day_where_nothing_moved_after_the_lock(live):
    e = engine()
    for s in e.eqs:
        s.lock_px = s.ltp
    e.top = {"locked": True, "long": [], "short": [], "lock_scores": {}}
    e.clock = ts(date(2026, 10, 21), 15, 31)
    asyncio.run(e.maybe_scorecard())
    assert e.scorecard_done and asyncio.run(e.r.llen(bus.KEY_SCORECARD)) == 0


def test_api_ignores_fake_holiday_scorecards(live):
    real = {"date": "2026-10-21", "ic": 0.18, "long_avg": -1.36, "short_avg": 1.89, "universe_avg": -1.59,
            "picks": ["A"], "shorts": ["B"]}
    flat = {"date": "2026-10-22", "ic": -0.18, "long_avg": 0.0, "short_avg": 0.0, "universe_avg": 0.0,
            "picks": [], "shorts": []}
    holiday = dict(real, date="2026-10-20")
    assert _real_session(real) and not _real_session(flat) and not _real_session(holiday)


# ------------------------------------------------------------------ picks
def pick_state(bars: dict[int, list], ltp: float) -> InstState:
    s = InstState(1, "NSE_EQ", "X", "EQ", "X", "BANK")
    s.pc, s.ltp, s.lock_px = 100.0, ltp, 100.0
    s.bars = bars
    s.max_since_lock = max(b[1] for m, b in bars.items() if m >= 15)
    s.min_since_lock = min(b[2] for m, b in bars.items() if m >= 15)
    return s


def outcome(s: InstState, side: str = "long"):
    e = an.Engine(fakeredis.aioredis.FakeRedis())
    e.top = {"locked": True, "lock_ts": ts(date(2026, 10, 21), 9, 30, 2)}
    p = {"entry": 100.0, "target": 102.0, "invalidation": 99.0} if side == "long" else \
        {"entry": 100.0, "target": 98.0, "invalidation": 101.0}
    return e._outcome(s, p, side)


def test_a_hit_stop_stays_invalid_when_the_price_comes_back(live):
    bars = {10: [100, 100.5, 99.8, 100, 1], 15: [100, 100.2, 99.9, 100, 1],
            20: [99.6, 99.7, 98.9, 99.2, 1],                          # 09:35: stop at 99 touched
            40: [100.5, 102.4, 100.4, 102.1, 1]}                      # 09:55: target touched later
    assert outcome(pick_state(bars, ltp=100.8)) == ("INVALID", "09:35")


def test_target_first_stays_target(live):
    bars = {15: [100, 100.2, 99.9, 100, 1], 30: [101.5, 102.2, 101.4, 102, 1], 50: [99.5, 99.6, 98.5, 98.7, 1]}
    assert outcome(pick_state(bars, ltp=98.7)) == ("TARGET", "09:45")
    short_bars = {15: [100, 100.2, 99.9, 100, 1], 25: [99, 99.1, 97.9, 98, 1]}
    assert outcome(pick_state(short_bars, ltp=99.5), "short") == ("TARGET", "09:40")
    assert outcome(pick_state({15: [100, 100.3, 99.5, 100, 1]}, ltp=100.2)) == ("LIVE", None)


def test_stop_is_capped_by_atr(live):
    e = an.Engine(fakeredis.aioredis.FakeRedis())
    s = InstState(1, "NSE_EQ", "X", "EQ", "X", "BANK")
    s.ltp = 570.4
    e.eq_by_symbol["X"] = s
    row = {"sid": 1, "symbol": "X", "sector": "BANK", "ltp": 570.4, "atr": 10.0, "orl": 552.0, "orh": 575.0,
           "score": 1.5, "score_s": -1.5, "f": {}, "rvol": 2.2, "gap": 0.4, "chg": 1.2, "headline": None,
           "oi_tag": "SC", "orb": 0.5, "vwap_atr": 0.3, "news": 0.0, "resid": 1.4, "news_type": None}
    long = e._pick_view(row, "long")
    assert long["invalidation"] == 564.4 and long["target"] == 580.4          # 0.6 ATR, not the 552 OR low
    tight = e._pick_view(dict(row, orl=568.0), "long")
    assert tight["invalidation"] == 568.0                                    # a tighter OR low is kept
    short = e._pick_view(dict(row, orh=590.0), "short")
    assert short["invalidation"] == 576.4 and short["target"] == 560.4


def test_short_chips_say_relative_weakness():
    r = {"rvol": 2.6, "gap": 0.2, "orb": -1.2, "vwap_atr": -0.4, "oi_tag": "SB", "news": 0.0, "news_type": None,
         "resid": -1.8}
    chips = intraday.factor_chips(r, "short")
    assert "RW" in chips and "RS" not in chips
    assert "RS" in intraday.factor_chips(dict(r, resid=1.8, orb=1.2, vwap_atr=0.4), "long")


def test_restored_lock_without_prices_sends_null_summary(live):
    """Right after a restart (evening rebuild) the lock comes back from Redis before any price: the desk must get
    null for the since-lock summary, not {} (it read {}.leaders and crashed)."""
    e = engine()
    for s in e.eqs:
        s.ltp = 0.0
    e.top = {"locked": True, "lock_time": "09:30:00", "long": [{"symbol": "S1", "entry": 100.0, "target": 101.0,
                                                              "invalidation": 99.5, "chips": ["RVOL 2.1x"]}],
             "short": []}
    v = e.top_view()
    assert v["universe_since_lock"] is None and v["long"][0]["symbol"] == "S1"
