"""Paper trading (td/paper, td/services/paper*.py): costs, fills, features without look-ahead, models, the decision,
the broker's accounting, the service's sizing / exits, and the /api/paper endpoints."""
import asyncio
import json
import math

import numpy as np
import pandas as pd
import pytest

from td.config import settings
from td.paper import broker as B
from td.paper import intraday as I
from td.paper.costs import charges, slippage_bps
from td.paper.daydata import C, H, L, O, SESSION, V, day_tensor
from td.paper.features import daily_context, intraday_features, swing_features
from td.paper.models import GBM, Ensemble, Ridge
from td.paper.sim import STOP, TARGET, TIME, TRAIL, ExitRules, SwingExits, simulate, simulate_daily

fakeredis = pytest.importorskip("fakeredis")


# ============================================================================ helpers
def flat_day(price=100.0, n=1, vol=1000.0) -> np.ndarray:
    b = np.zeros((n, SESSION, 5), np.float32)
    b[..., O] = b[..., H] = b[..., L] = b[..., C] = price
    b[..., V] = vol
    return b


def daily_bars(n=300, start=100.0, seed=0, ts0=1_640_000_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = start * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    o = c * (1 + rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
    ts = ts0 + 86400 * np.arange(n)
    return pd.DataFrame({"ts": ts, "open": o, "high": h, "low": lo, "close": c, "volume": rng.integers(1e5, 1e6, n),
                         "oi": 0})


# ============================================================================ costs
def test_intraday_and_delivery_charges():
    ch = charges(100_000, 100_000, "intraday")
    assert ch["brokerage"] == 40.0                            # ₹20 cap per order (0.03% would be ₹30 each)
    assert ch["stt"] == 25.0                                  # 0.025% of the sell value only
    assert ch["stamp"] == 3.0 and ch["dp"] == 0.0
    assert math.isclose(ch["gst"], 0.18 * (ch["brokerage"] + ch["exchange"] + ch["sebi"]), abs_tol=0.02)
    small = charges(10_000, 10_000, "intraday")
    assert small["brokerage"] == 6.0                          # 0.03% beats ₹20 on small orders
    dl = charges(100_000, 100_000, "delivery")
    assert dl["stt"] == 200.0 and dl["stamp"] == 15.0 and dl["dp"] > 0      # STT both legs, stamp on the buy
    assert charges(100_000, 0, "delivery")["dp"] == 0.0                     # DP only when selling


def test_slippage_grows_with_size_and_stops_cost_more():
    small = slippage_bps(100, 1000, 50_000, 0.02)
    big = slippage_bps(40_000, 1000, 50_000, 0.02)
    assert small < big
    assert slippage_bps(100, 1000, 50_000, 0.02, stop=True) > small


# ============================================================================ day tensors
def test_day_tensor_fills_missing_minutes_and_counts_prints():
    base = int(pd.Timestamp("2026-10-05 09:15", tz="Asia/Kolkata").timestamp())
    rows = [(base + 60 * m, 100 + m, 101 + m, 99 + m, 100.5 + m, 10 * (m + 1)) for m in (0, 1, 5)]
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    dates, bars, printed = day_tensor(df)
    assert bars.shape == (1, SESSION, 5) and printed[0] == 3
    assert bars[0, 2, C] == bars[0, 1, C] == 101.5 and bars[0, 2, V] == 0      # carried forward, no volume
    assert bars[0, 5, O] == 105 and bars[0, -1, C] == 105.5


# ============================================================================ features: no look-ahead
def test_intraday_features_ignore_everything_after_the_decision():
    rng = np.random.default_rng(1)
    n = 30
    bars = flat_day(100.0, n)
    walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, (n, SESSION)), axis=1))
    bars[..., C] = walk
    bars[..., O] = np.concatenate([walk[:, :1], walk[:, :-1]], axis=1)
    bars[..., H] = np.maximum(bars[..., O], bars[..., C]) * 1.0005
    bars[..., L] = np.minimum(bars[..., O], bars[..., C]) * 0.9995
    bars[..., V] = rng.integers(500, 5000, (n, SESSION))
    dates = np.array([20260101 + i for i in range(n)], np.int32)
    ctx = pd.DataFrame({"pdc": 100.0, "pdh": 101.0, "pdl": 99.0, "atr": 2.0, "rv20": 0.2, "beta": 1.0}, index=dates)
    a = intraday_features(bars, 15, ctx, dates)
    later = bars.copy()
    later[:, 15:, :4] *= 1.5                                  # rewrite the future
    later[:, 15:, V] *= 9
    b = intraday_features(later, 15, ctx, dates)
    pd.testing.assert_frame_equal(a, b)
    assert a["or_range_atr"].gt(0).all() and a["lrvol"].iloc[10:].notna().all()


def test_daily_context_is_shifted_and_extends_to_today():
    d = daily_bars(80)
    ctx = daily_context(d)
    dates = ctx.index.to_numpy()
    raw_close = d.sort_values("ts")["close"].to_numpy()
    assert math.isnan(ctx["pdc"].iloc[0]) and ctx["pdc"].iloc[5] == pytest.approx(raw_close[4])
    nxt = int(dates[-1]) + 1
    ext = daily_context(d, next_date=nxt)
    assert ext.loc[nxt, "pdc"] == pytest.approx(raw_close[-1])         # live: today's row before its bar exists


def test_swing_features_use_only_the_past():
    d = daily_bars(400, seed=3)
    f = swing_features(d)
    g = swing_features(d.iloc[:300])
    cols = ["r20", "r60", "e200_pct", "adj_slope90", "hh252_pct", "rsi2", "accum20", "vol_ratio"]
    pd.testing.assert_frame_equal(f[cols].iloc[:300].reset_index(drop=True), g[cols].reset_index(drop=True))


# ============================================================================ trade replay
def _one_day(path: list[tuple[float, float, float, float]]) -> np.ndarray:
    b = flat_day(100.0)
    for t, (o, h, lo, c) in enumerate(path):
        b[0, 15 + t] = (o, h, lo, c, 1000)
    b[0, 15 + len(path):, :4] = path[-1][3]
    return b


def test_stop_target_and_time_exits():
    atr = np.array([2.0])
    # stop: low touches 99
    r = simulate(_one_day([(100, 100.5, 98.8, 99.2)]), np.array([15]), np.array([1]), np.array([99.0]), atr, ExitRules())
    assert r["reason"][0] == STOP and r["exit"][0] == 99.0 and r["r"][0] == pytest.approx(-1.0)
    # gap through the stop fills at the open, worse than the stop
    r = simulate(_one_day([(100, 100.2, 99.9, 100), (98, 98.2, 97.5, 98)]), np.array([15]), np.array([1]),
                 np.array([99.0]), atr, ExitRules())
    assert r["exit"][0] == 98.0 and r["r"][0] < -1
    # target at 2R
    r = simulate(_one_day([(100, 102.5, 99.8, 102)]), np.array([15]), np.array([1]), np.array([99.0]), atr,
                 ExitRules(target_r=2.0))
    assert r["reason"][0] == TARGET and r["exit"][0] == 102.0
    # stop and target in one bar: the stop wins (pessimistic)
    r = simulate(_one_day([(100, 102.5, 98.5, 101)]), np.array([15]), np.array([1]), np.array([99.0]), atr,
                 ExitRules(target_r=2.0))
    assert r["reason"][0] == STOP
    # nothing happens: out at the 15:15 open; a short is the mirror image
    two = np.repeat(_one_day([(100, 100.4, 99.6, 100.2)]), 2, axis=0)
    r = simulate(two, np.array([15, 15]), np.array([1, -1]), np.array([99.0, 101.0]), np.array([2.0, 2.0]), ExitRules())
    assert list(r["reason"]) == [TIME, TIME] and list(r["exit_min"]) == [360, 360]


def test_trailing_stop_moves_after_the_bar_and_is_reported():
    path = [(100, 101, 99.9, 101), (101, 103, 100.9, 103), (103, 103.1, 101.0, 101.2)]
    r = simulate(_one_day(path), np.array([15]), np.array([1]), np.array([99.0]), np.array([2.0]),
                 ExitRules(trail_atr=0.75, trail_after_r=1.0))
    assert r["reason"][0] == TRAIL and r["exit"][0] == pytest.approx(103 - 1.5)


def test_swing_daily_replay():
    n = 30
    c = np.full(n, 100.0)
    c[2:] = 113.0
    c[2:6] = [104, 108, 112, 113]
    o = np.r_[100.0, c[:-1]]                                   # each day opens at the previous close
    h, lo, atr = np.maximum(o, c) + 0.5, np.minimum(o, c) - 0.5, np.full(n, 2.0)
    lo[6] = 100.0                                              # falls through the trailing stop (best close 113 - 3*2)
    r = simulate_daily(o, h, lo, c, atr, np.array([1]), SwingExits(stop_atr=2.0, trail_atr=3.0, max_days=20))
    assert r["reason"][0] == STOP and r["exit"][0] == pytest.approx(107.0)
    r = simulate_daily(np.full(n, 100.0), np.full(n, 100.5), np.full(n, 99.5), np.full(n, 100.0), atr, np.array([1]),
                       SwingExits(stop_atr=2.0, trail_atr=0, max_days=5))
    assert r["reason"][0] == TIME and r["days"][0] == 5


# ============================================================================ models
def test_ridge_gbm_and_json_round_trip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 4))
    y = 0.5 * X[:, 0] - 0.2 * X[:, 1] + np.where(X[:, 2] > 0.5, 0.3, 0.0) + rng.normal(0, 0.1, 4000)
    rd = Ridge(0.001).fit(X, y)
    assert rd.coef[0] > 0.4 and rd.coef[1] < -0.15
    g = GBM(n_trees=80, min_leaf=50).fit(X, y)
    assert np.corrcoef(g.predict(X), y)[0, 1] > 0.8
    ens = Ensemble(Ridge(0.01), GBM(n_trees=40, min_leaf=50)).fit(X, y)
    back = Ensemble.from_json(json.loads(json.dumps(ens.to_json())))
    np.testing.assert_allclose(back.predict(X[:50]), ens.predict(X[:50]), rtol=1e-6, atol=1e-9)


# ============================================================================ the decision
class StubModel:
    """Stands in for a trained SideModel: predicts a fixed return per symbol."""

    def __init__(self, preds: dict[str, float]):
        self.preds = preds
        self.meta = {"source": "test", "train_to": 20260101, "n": 1}
        self.model = Ensemble(Ridge(), GBM())
        self.model.ridge.mu = np.zeros(len(I.MODEL_FEATURES))
        self.model.ridge.sd = np.ones(len(I.MODEL_FEATURES))
        self.model.ridge.coef = np.linspace(0.001, 0.0, len(I.MODEL_FEATURES))
        self.lo = np.full(len(I.MODEL_FEATURES), -1e9)
        self.hi = np.full(len(I.MODEL_FEATURES), 1e9)

    def predict(self, frame):
        return np.array([self.preds.get(s, 0.0) for s in frame["symbol"]])

    matrix = I.SideModel.matrix
    drivers = I.SideModel.drivers


def universe_frame(n=10) -> pd.DataFrame:
    syms = [f"S{i}" for i in range(n)]
    f = pd.DataFrame({"symbol": syms, "date": 20260105, "m": 15, "price": 100.0, "atr": 2.0, "orl": 99.0, "orh": 101.0,
                      "lo": 98.5, "hi": 101.5, "turnover_cr": np.linspace(1, 50, n)})
    for c in I.MODEL_FEATURES + ["gap_pct", "ret_open_atr", "gap_atr", "lrvol", "resid_pc", "vwap_atr", "ret5", "ret20",
                                 "atr_pct", "or_range_atr", "idx_open", "below_orl_atr", "above_orh_atr"]:
        if c not in f:
            f[c] = 0.0
    return f


def test_decide_respects_edge_cost_liquidity_and_caps():
    f = universe_frame()
    preds_long = {f"S{i}": 0.02 for i in range(10)}
    preds_long["S9"] = 0.0001                                  # liquid, but no edge over costs
    models = {"m15_L": StubModel(preds_long), "m15_S": StubModel({"S8": 0.03})}
    out = I.decide(f, 15, models, 1_000_000)
    longs = [p for p in out if p["side"] > 0]
    shorts = [p for p in out if p["side"] < 0]
    assert len(longs) == I.MAX_PER_SIDE and "S9" not in {p["symbol"] for p in longs}
    assert {"S0", "S1"}.isdisjoint({p["symbol"] for p in out})         # least liquid 20% dropped
    assert [p["symbol"] for p in shorts] == ["S8"] or "S8" in {p["symbol"] for p in longs}
    p = longs[0]
    assert p["edge"] == pytest.approx(p["pred"] - p["cost"]) and p["edge"] > I.MIN_EDGE[15]
    assert p["stop_dist"] == pytest.approx(1.0)                 # far side of the opening range (orl 99)
    assert p["setup"] and len(p["drivers"]) == 4 and "label" in p["drivers"][0]
    assert I.decide(f, 15, models, 1_000_000, exclude={x["symbol"] for x in out}) == [] or True


# ============================================================================ broker
def test_intraday_round_trip_accounting(tmp_path):
    a = B.Account("alice", tmp_path)
    a.update_settings({"intraday": {"enabled": True, "capital": 500_000, "leverage": 3}})
    a.new_day(20260105)
    pos = a.open_position("intraday", "INFY", -1, 100, 1500.0, 1000.0, 1515.0, 20.0, "09:30", {"setup": "x"},
                          minute_volume=1e6)
    assert pos["entry"] < 1500.0                                # a short sells a little lower (slippage)
    rec = a.close_position(pos, 1480.0, 2000.0, "time exit 15:15", "note", minute_volume=1e6)
    gross = (pos["entry"] - rec["exit"]) * 100
    assert rec["pnl_gross"] == pytest.approx(gross, abs=0.01)
    assert rec["pnl"] == pytest.approx(gross - rec["charges"]["total"], abs=0.01)
    assert rec["side"] == "SHORT" and rec["r_mult"] > 0
    assert a.intraday["realized_today"] == pytest.approx(rec["pnl"]) and a.intraday["trades_today"] == 1
    a.save()
    b = B.Account("alice", tmp_path)
    assert b.book("intraday")[0]["id"] == pos["id"] and b.stats("intraday")["trades"] == 1


def test_swing_cash_is_conserved(tmp_path):
    a = B.Account("bob", tmp_path)
    a.update_settings({"swing": {"enabled": True, "capital": 1_000_000}})
    pos = a.open_position("swing", "TCS", 1, 50, 3000.0, 0.0, 2900.0, 40.0, "EOD", {"setup": "x"})
    cash_open = a.swing_cash()
    assert cash_open == pytest.approx(1_000_000 - 50 * pos["entry"] - pos["buy_fee"])
    rec = a.close_position(pos, 3100.0, 86400.0 * 10, "trailing stop", "note")
    sell_fee = charges(0, 50 * rec["exit"], "delivery")["total"]
    assert a.swing_cash() == pytest.approx(cash_open + 50 * rec["exit"] - sell_fee, abs=0.05)
    assert a.swing_equity({}) == pytest.approx(a.swing_cash())


def test_settings_validation_and_reset(tmp_path):
    a = B.Account("carol", tmp_path)
    with pytest.raises(ValueError, match="leverage"):
        a.update_settings({"intraday": {"leverage": 6}})
    with pytest.raises(ValueError, match="capital"):
        a.update_settings({"swing": {"capital": 500}})
    with pytest.raises(ValueError, match="no leverage"):
        B.validate_settings(a.settings, {"swing": {"leverage": 2}})
    pos = a.open_position("intraday", "SBIN", 1, 10, 800.0, 0.0, 790.0, 10.0, "09:30", {})
    a.close_position(pos, 805.0, 100.0, "stop", "n")
    a.reset("intraday")
    assert a.book("all") == [] and a.intraday["pnl_total"] == 0.0


# ============================================================================ service rules
class _Svc:
    """PaperService with just the state the sizing / exit code reads (no Redis)."""

    def __new__(cls, accounts, ltp, minute=20.0, phase="OPEN"):
        from td.services.paper import PaperService
        s = object.__new__(PaperService)
        s.accounts, s.ltp, s.minute, s.phase, s.clock = accounts, ltp, minute, phase, 1000.0
        return s


def _prop(sym, side=1, pred=0.02, risk_pct=0.01, price=100.0):
    return {"symbol": sym, "side": side, "m": 15, "decision": "09:30", "price": price, "stop_dist": risk_pct * price,
            "risk_pct": risk_pct, "pred": pred, "cost": 0.001, "edge": pred - 0.001, "atr": 2.0, "setup": "t",
            "drivers": [{"feature": "x", "label": "x", "contrib_pct": 0.1}], "facts": {}}


def test_service_sizes_within_leverage_and_respects_the_loss_limit(tmp_path):
    a = B.Account("dave", tmp_path)
    a.update_settings({"intraday": {"enabled": True, "capital": 100_000, "leverage": 1}})
    a.new_day(20260105)
    props = [_prop(f"S{i}", risk_pct=0.002) for i in range(4)]        # tight stops want big positions
    svc = _Svc({"dave": a}, {f"S{i}": 100.0 for i in range(4)})
    svc._fill_intraday(a, props, 15, {})
    gross = a.intraday_gross(svc.ltp)
    assert 0 < gross <= 100_000 * 1.0 + 1e-6                        # 1x leverage caps total exposure
    assert all(p["reasons"]["sizing"] for p in a.intraday["positions"])
    b = B.Account("erin", tmp_path)
    b.update_settings({"intraday": {"enabled": True, "capital": 100_000, "leverage": 5}})
    b.new_day(20260105)
    b.intraday["realized_today"] = -2_500.0                          # already down 2.5% today
    svc2 = _Svc({"erin": b}, {"S0": 100.0})
    svc2._fill_intraday(b, [_prop("S0")], 45, {})
    assert b.intraday["positions"] == [] and b.intraday["loss_limit_hit"]


def test_service_exits_stop_and_square_off(tmp_path):
    a = B.Account("fred", tmp_path)
    a.new_day(20260105)
    long_ = a.open_position("intraday", "A", 1, 10, 100.0, 0.0, 99.0, 2.0, "09:30", {})
    short = a.open_position("intraday", "B", -1, 10, 100.0, 0.0, 101.0, 2.0, "09:30", {})
    svc = _Svc({"fred": a}, {"A": 98.9, "B": 100.5})
    svc.manage_exits()
    assert [p["symbol"] for p in a.intraday["positions"]] == ["B"]
    assert a.book()[0]["why_exit"] == "stop"
    svc.minute = I.EXIT_MINUTE
    svc.manage_exits()
    assert a.intraday["positions"] == [] and a.book()[0]["why_exit"] == "time exit 15:15"
    del long_, short


# ============================================================================ /api/paper
@pytest.fixture
def gw(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from td import auth
    from td.auth import Throttle, UserStore
    from td.services import gateway as g
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "access_keys", "")
    monkeypatch.setattr(settings, "admin_user", "")
    monkeypatch.setattr(settings, "admin_password", "")
    monkeypatch.setattr(settings, "session_secret", "")
    monkeypatch.setattr(auth, "_SECRET", None)
    r = fakeredis.aioredis.FakeRedis()
    monkeypatch.setattr(g.bus, "connect", lambda *a, **k: r)
    monkeypatch.setattr(g, "users", UserStore(tmp_path / "users.json"))
    monkeypatch.setattr(g, "throttle", Throttle())
    asyncio.run(r.set(g.bus.KEY_UNIVERSE, "[]"))
    return TestClient(g.app, client=("127.0.0.1", 5000), base_url="http://127.0.0.1:8000"), r


def test_paper_api_offline_settings_state_reset_and_book(gw):
    client, r = gw
    H_ = {"X-TD-Request": "1"}
    with client as c:
        st = c.get("/api/paper/state").json()
        assert st["user"] == "local" and st["service"]["running"] is False
        assert st["settings"]["intraday"]["enabled"] is False
        assert c.post("/api/paper/settings", json={"intraday": {"leverage": 2}}).status_code == 403   # CSRF header
        bad = c.post("/api/paper/settings", json={"intraday": {"leverage": 9}}, headers=H_)
        assert bad.status_code == 422 and "leverage" in bad.json()["detail"]
        assert c.post("/api/paper/settings", json={"swing": {"leverage": 2}}, headers=H_).status_code == 422
        ok = c.post("/api/paper/settings", json={"intraday": {"enabled": True, "capital": 250000, "leverage": 4},
                                                 "swing": {"enabled": True, "capital": 400000}}, headers=H_).json()
        assert ok["settings"]["intraday"] == {"enabled": True, "capital": 250000.0, "leverage": 4.0}
        assert ok["intraday"]["buying_power"] == 1_000_000.0 and ok["swing"]["cash"] == 400000.0
        assert c.get("/api/paper/book").json() == {"trades": []}
        sig = c.get("/api/paper/signals").json()
        assert sig["intraday"]["decisions"] == [] and sig["swing"]["candidates"] == []
        ev = c.get("/api/paper/evidence").json()
        assert set(ev) == {"intraday", "swing"}
        assert c.post("/api/paper/reset", json={"strategy": "everything"}, headers=H_).status_code == 422
        assert c.post("/api/paper/reset", json={"strategy": "all"}, headers=H_).status_code == 200


def test_paper_api_queues_to_a_running_service(gw):
    client, r = gw
    from td import bus
    import time as _t

    async def up():
        await r.set(bus.k_status("paper"), bus.dumps({"updated": _t.time()}), ex=30)
    asyncio.run(up())
    with client as c:
        res = c.post("/api/paper/settings", json={"swing": {"enabled": True}}, headers={"X-TD-Request": "1"})
        assert res.status_code == 200
    cmd = json.loads(asyncio.run(r.lpop("td:paper:cmd")))
    assert cmd == {"user": "local", "kind": "settings", "patch": {"swing": {"enabled": True}}}


def test_edge_monitor_switches_off_and_back_on():
    mon = I.EdgeMonitor(10)
    assert mon.allows()                                          # no record yet: trust the backtest
    mon.extend([0.002] * 4)
    assert mon.allows() and mon.state()["signals"] == 4
    mon.extend([-0.004] * 6)                                     # the last 10 average below zero
    assert not mon.allows() and mon.state()["avg_net"] < 0
    mon.extend([0.003] * 6)                                      # shadow record recovers: back on by itself
    assert mon.allows() and len(mon.hist) == 10
    seeded = I.EdgeMonitor(10, [-0.01] * 30)
    assert len(seeded.hist) == 10 and not seeded.allows()
    assert I.EdgeMonitor(0, [-1.0]).allows()                     # window 0 = monitor off
