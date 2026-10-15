"""/api/research/... endpoints (td.services.research_api) on the real gateway app: fakeredis, a temp lake with
synthetic bars, and fake Monte Carlo / quant / Screener / AI services, so these tests do not depend on the model
modules — only on the contract shapes the router passes through."""
import asyncio
import json
import math
import sys
import zlib
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from td.auth import Throttle, UserStore
from td.config import settings
from td.lake import Lake
from td.services import research_api as api
from td.timeutil import IST

fakeredis = pytest.importorskip("fakeredis")
H = {"X-TD-Request": "1"}
UNI = [{"symbol": "RELIANCE", "kind": "EQ", "sid": 2885, "seg": "NSE_EQ", "sector": "ENERGY",
        "name": "Reliance Industries", "underlying": "RELIANCE"},
       {"symbol": "M&M", "kind": "EQ", "sid": 2031, "seg": "NSE_EQ", "sector": "AUTO", "name": "Mahindra & Mahindra",
        "underlying": "M&M"},
       {"symbol": "YOUNG", "kind": "EQ", "sid": 9, "seg": "NSE_EQ", "sector": "OTHER", "name": "Young Ltd",
        "underlying": "YOUNG"},
       {"symbol": "NIFTY", "kind": "IDX", "sid": 13, "seg": "IDX_I", "sector": "INDEX", "name": "",
        "underlying": "NIFTY"}]


def bars(start: date, end: date, px: float, seed: int = 1) -> pd.DataFrame:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    close = px * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, len(days))))
    ts = [int(datetime(d.year, d.month, d.day, tzinfo=IST).timestamp()) for d in days]
    return pd.DataFrame({"ts": ts, "open": close, "high": close, "low": close, "close": close, "volume": 5, "oi": 0})


class FakeMC:
    """Records the call; returns numpy scalars like the real module (the router must serialise them)."""

    def __init__(self):
        self.calls = []

    def simulate(self, df, **kw):
        self.calls.append(("simulate", len(df), kw))
        if len(df) < 250:
            raise ValueError(f"need at least 250 daily bars in the lookback window, have {len(df)}")
        return {"asof": "2026-10-14", "spot": np.float64(kw["spot"]), "spot_source": kw["spot_source"],
                "horizon_days": np.int64(kw["horizon_days"]), "paths": kw["paths"], "model": kw["model"],
                "drift": kw["drift"], "seed": kw["seed"], "elapsed_ms": np.int64(12),
                "fan": [{"t": 0, "p50": np.float64(kw["spot"]), "mean": float("nan")}], "warnings": ["model note"]}

    def backtest(self, df, **kw):
        self.calls.append(("backtest", len(df), kw))
        if kw["lookback_years"] > 4:
            raise ValueError("not enough history for 40 origins with a 5-year lookback")
        return {"model": kw["model"], "drift": kw["drift"], "horizon_days": kw["horizon_days"],
                "origins": kw["origins"], "paths": kw["paths"], "coverage": [], "pit_hist": np.zeros(10),
                "verdict": "ok", "warnings": []}


class FakeQuant:
    def __init__(self, auc=0.58, status="ready"):
        self.auc, self.status = auc, status
        self.trained = []
        self.board = []

    def _info(self):
        return {"status": self.status, "progress": 0.0, "message": "", "trained_at": None, "symbols": 0,
                "rows": 0, "features": [], "metrics": []}

    async def result(self, symbol):
        return {"symbol": symbol, "status": self.status, "model": self._info(), "warnings": [],
                "horizons": [{"h": 1, "exp_ret": 0.001, "ret_sd": 0.02, "oos_auc": 0.5},
                             {"h": 21, "exp_ret": 0.01, "ret_sd": 0.08, "oos_auc": self.auc}]}

    async def info(self):
        return self._info()

    async def train(self, force=True):
        self.trained.append(force)
        self.status = "training"
        await asyncio.sleep(0.01)
        self.status = "ready"

    async def leaderboard(self, h, n):
        self.board.append((h, n))
        return {"h": h, "asof": "2026-10-14", "up": [], "down": [], "model": self._info()}


class ScreenerError(RuntimeError):
    """Same name as td.research.screener.ScreenerError (the router maps it by name)."""


class FakeScreener:
    def __init__(self):
        self.calls = []

    async def fundamentals(self, symbol, view="auto", refresh=False):
        self.calls.append((symbol, view, refresh))
        if symbol == "NOPE":
            raise KeyError("NOPE is not on Screener.in")
        if symbol == "DOWN":
            raise ScreenerError("HTTP 503")
        return {"symbol": symbol, "name": symbol.title(), "view": "consolidated" if view == "auto" else view,
                "warnings": []}


class FakeAi:
    def __init__(self):
        self.started = []
        self.jobs = {"job1": {"id": "job1", "symbol": "RELIANCE", "status": "running", "steps": [], "result": None,
                              "error": None, "cached": False, "started": 1.0, "finished": None}}
        self.reports = {"RELIANCE": {"symbol": "RELIANCE", "ai": False}}

    async def config(self, user, is_admin):
        return {"enabled": False, "provider": "anthropic", "model": "", "reason": "no key", "daily_limit": 40,
                "used_today": 0, "user_limit": 40 if is_admin else 10, "user_used_today": 0, "who": user}

    async def start(self, symbol, user, is_admin, refresh=False):
        self.started.append((symbol, user, is_admin, refresh))
        return {**self.jobs["job1"], "symbol": symbol}

    def job(self, id):
        return self.jobs.get(id)

    async def latest(self, symbol):
        return self.reports.get(symbol)


@pytest.fixture
def app(monkeypatch, tmp_path):
    """(gateway module, TestClient factory, fakes) — sim mode, temp lake, no accounts (local mode)."""
    from fastapi.testclient import TestClient
    from td import auth
    from td.services import gateway as gw
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "data_source", "sim")
    monkeypatch.setattr(settings, "access_keys", "")
    monkeypatch.setattr(settings, "admin_user", "")
    monkeypatch.setattr(settings, "admin_password", "")
    monkeypatch.setattr(settings, "session_secret", "")
    monkeypatch.setattr(auth, "_SECRET", None)
    r = fakeredis.aioredis.FakeRedis()
    lake = Lake(tmp_path / "lake")
    lake.write("D", "RELIANCE", bars(date(2022, 10, 3), date(2026, 10, 14), 1400))
    lake.write("D", "M&M", bars(date(2022, 10, 3), date(2026, 10, 14), 3000, seed=2))
    lake.write("D", "NIFTY", bars(date(2022, 10, 3), date(2026, 10, 14), 25000, seed=3))
    lake.write("D", "YOUNG", bars(date(2026, 6, 1), date(2026, 10, 14), 50, seed=4))
    monkeypatch.setattr(gw.bus, "connect", lambda *a, **k: r)
    monkeypatch.setattr(gw, "Lake", lambda *a, **k: lake)
    monkeypatch.setattr(gw, "users", UserStore(tmp_path / "users.json"))
    monkeypatch.setattr(gw, "throttle", Throttle())
    asyncio.run(r.set(gw.bus.KEY_UNIVERSE, json.dumps(UNI)))
    if hasattr(gw.app.state, "research"):
        monkeypatch.delattr(gw.app.state, "research")
    fakes = {"mc": FakeMC(), "quant": FakeQuant(), "screener": FakeScreener(), "ai": FakeAi()}

    def client():
        return TestClient(gw.app, client=("127.0.0.1", 5000), base_url="http://127.0.0.1:8000")

    def install(**over):
        f = {**fakes, **over}
        gw.app.state.research = api.Research(gw.app.state.redis, gw.app.state.lake, **f)
        return gw.app.state.research
    yield gw, client, fakes, install
    if hasattr(gw.app.state, "research"):
        del gw.app.state.research


def test_search_and_history(app):
    gw, client, fakes, install = app
    with client() as c:
        install()
        hits = c.get("/api/research/search", params={"q": "rel"}).json()
        assert hits[0] == {"symbol": "RELIANCE", "name": "Reliance Industries", "sector": "ENERGY",
                           "in_universe": True, "has_history": True}
        assert c.get("/api/research/search", params={"q": "mahindra"}).json()[0]["symbol"] == "M&M"
        assert c.get("/api/research/search").json() == []
        r = c.get("/api/research/search", params={"q": "x", "limit": 0})
        assert r.status_code == 422 and isinstance(r.json()["detail"], str) and "limit" in r.json()["detail"]
        h = c.get("/api/research/history/reliance?fetch=1").json()
        assert h["symbol"] == "RELIANCE" and h["bars"] > 1000 and h["last"] == "2026-10-14"
        assert h["spot_source"] == "close" and "simulated" in h["warning"] and h["closes"][-1][1] == h["spot"]
        assert c.get("/api/research/history/M%26M").json()["name"] == "Mahindra & Mahindra"
        r = c.get("/api/research/history/NOSUCH")
        assert r.status_code == 404 and "NOSUCH" in r.json()["detail"]
        r = c.get("/api/research/history/BAD%20SYM")
        assert r.status_code == 422 and isinstance(r.json()["detail"], str)


def test_lazy_container_is_built_per_app(app):
    gw, client, fakes, install = app
    with client() as c:
        assert c.get("/api/research/search", params={"q": "nif"}).json()[0]["symbol"] == "NIFTY"
        rs = gw.app.state.research
        assert isinstance(rs, api.Research) and rs.redis is gw.app.state.redis
        c.get("/api/research/search", params={"q": "nif"})
        assert gw.app.state.research is rs                                  # one container per app


def test_montecarlo(app):
    gw, client, fakes, install = app
    mc = fakes["mc"]
    with client() as c:
        install()
        body = {"horizon_days": 252, "paths": 5000, "model": "garch", "drift": "capm", "lookback_years": 3,
                "targets": [1500, 1200]}
        assert c.post("/api/research/montecarlo/RELIANCE", json=body).status_code == 403        # CSRF header
        r = c.post("/api/research/montecarlo/reliance", json=body, headers=H)
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["symbol"] == "RELIANCE" and out["horizon_days"] == 252 and out["elapsed_ms"] == 12
        assert out["fan"][0]["mean"] is None                                # NaN -> null
        assert "model note" in out["warnings"] and any("simulated" in w for w in out["warnings"])
        name, n, kw = mc.calls[-1]
        last_ts = int(gw.app.state.lake.read("D", "RELIANCE")["ts"].iat[-1])
        assert kw["seed"] == zlib.crc32(b"RELIANCE:2026-10-14") == api.default_seed("RELIANCE", last_ts)
        assert kw["spot_source"] == "close" and kw["model"] == "garch" and kw["targets"] == [1500.0, 1200.0]
        assert kw["index_df"] is not None and len(kw["index_df"]) > 1000 and kw["quant_drift"] is None
        assert kw["rf"] == settings.risk_free_rate and kw["erp"] == settings.equity_risk_premium
        assert n > 1000 and list(kw) and "lookback_years" in kw
        # explicit seed and the default body
        assert c.post("/api/research/montecarlo/M%26M", json={"seed": 7}, headers=H).json()["seed"] == 7
        assert mc.calls[-1][2]["model"] == "ensemble" and mc.calls[-1][2]["horizon_days"] == 252
        # validation: bounds, unknown fields, too many targets, bad model
        for bad, field in (({"horizon_days": 1000}, "horizon_days"), ({"paths": 100}, "paths"),
                           ({"lookback_years": 11}, "lookback_years"), ({"targets": [1] * 7}, "targets"),
                           ({"targets": [-5]}, "targets"), ({"model": "magic"}, "model"),
                           ({"drift": "up"}, "drift"), ({"bogus": 1}, "bogus")):
            r = c.post("/api/research/montecarlo/RELIANCE", json=bad, headers=H)
            assert r.status_code == 422 and field in r.json()["detail"], (bad, r.text)
        # unknown symbol / too little history
        r = c.post("/api/research/montecarlo/NOSUCH", json={}, headers=H)
        assert r.status_code == 404
        r = c.post("/api/research/montecarlo/YOUNG", json={}, headers=H)
        assert r.status_code == 422 and "250 daily bars" in r.json()["detail"]


def test_montecarlo_uses_live_spot(app):
    gw, client, fakes, install = app
    with client() as c:
        install()
        r = gw.app.state.redis
        asyncio.run(r.set(gw.bus.k_status("analytics"), b"{}"))
        asyncio.run(gw.bus.publish_snapshot(r, "quotes", {"ts": 1792046400, "q": {2885: [1450.25, 1, 1, 1, 0]}}))
        out = c.post("/api/research/montecarlo/RELIANCE", json={}, headers=H).json()
        assert out["spot"] == 1450.25 and out["spot_source"] == "live"


def test_montecarlo_quant_drift(app):
    gw, client, fakes, install = app
    mc = fakes["mc"]
    with client() as c:
        install()
        out = c.post("/api/research/montecarlo/RELIANCE", json={"drift": "quant"}, headers=H).json()
        qd = mc.calls[-1][2]["quant_drift"]
        assert qd == pytest.approx(math.expm1(12 * (0.01 + 0.5 * 0.08 ** 2)))
        assert not any("no reliable" in w for w in out["warnings"])
        install(quant=FakeQuant(auc=0.505))
        out = c.post("/api/research/montecarlo/RELIANCE", json={"drift": "quant"}, headers=H).json()
        assert any("no reliable 1-month edge" in w for w in out["warnings"])
        install(quant=FakeQuant(status="training"))
        out = c.post("/api/research/montecarlo/RELIANCE", json={"drift": "quant"}, headers=H).json()
        assert mc.calls[-1][2]["quant_drift"] is None and any("training" in w for w in out["warnings"])


def test_quant_drift_math():
    assert api.quant_drift(None)[0] is None
    assert api.quant_drift({"status": "ready", "horizons": []})[0] is None
    d, note = api.quant_drift({"status": "ready", "horizons": [{"h": 21, "exp_ret": 0.0, "ret_sd": 0.0,
                                                                "oos_auc": 0.6}]})
    assert d == 0.0 and note is None
    d, note = api.quant_drift({"status": "ready", "horizons": [{"h": 21, "exp_ret": 0.0, "ret_sd": 0.1,
                                                                "oos_auc": None}]})
    assert d == pytest.approx(math.expm1(0.06)) and "not been measured" in note   # flat median -> σ²/2 mean
    d, _ = api.quant_drift({"status": "ready", "horizons": [{"h": 21, "exp_ret": 5.0, "ret_sd": 1.0,
                                                            "oos_auc": 0.6}]})
    assert math.isfinite(d)                                               # extreme input stays finite


def test_backtest(app):
    gw, client, fakes, install = app
    mc = fakes["mc"]
    with client() as c:
        install()
        body = {"horizon_days": 63, "model": "bootstrap", "drift": "quant", "lookback_years": 2, "origins": 20,
                "paths": 1000}
        assert c.post("/api/research/montecarlo/RELIANCE/backtest", json=body).status_code == 403
        r = c.post("/api/research/montecarlo/RELIANCE/backtest", json=body, headers=H)
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["symbol"] == "RELIANCE" and out["pit_hist"] == [0.0] * 10
        kw = mc.calls[-1][2]
        assert mc.calls[-1][0] == "backtest" and kw["drift"] == "capm" and kw["origins"] == 20
        assert any("look-ahead" in w for w in out["warnings"])
        for bad, field in (({"origins": 5}, "origins"), ({"origins": 81}, "origins"), ({"paths": 6000}, "paths"),
                           ({"horizon_days": 4}, "horizon_days")):
            r = c.post("/api/research/montecarlo/RELIANCE/backtest", json=bad, headers=H)
            assert r.status_code == 422 and field in r.json()["detail"]
        r = c.post("/api/research/montecarlo/RELIANCE/backtest", json={"lookback_years": 5}, headers=H)
        assert r.status_code == 422 and "not enough history" in r.json()["detail"]


def test_quant_endpoints(app):
    gw, client, fakes, install = app
    q = fakes["quant"]
    with client() as c:
        install()
        out = c.get("/api/research/quant/reliance").json()
        assert out["symbol"] == "RELIANCE" and out["status"] == "ready"
        assert c.get("/api/research/quant/NOSUCH").status_code == 404
        assert c.get("/api/research/quant-model").json()["status"] == "ready"
        assert c.get("/api/research/quant-leaderboard").json()["h"] == 21 and q.board[-1] == (21, 15)
        c.get("/api/research/quant-leaderboard?h=5&n=3")
        assert q.board[-1] == (5, 3)
        assert c.get("/api/research/quant-leaderboard?h=2").status_code == 422
        assert c.get("/api/research/quant-leaderboard?n=0").status_code == 422
        # training is admin-only: local mode is a viewer
        assert c.post("/api/research/quant-model/train", json={}, headers=H).status_code == 403
        assert c.post("/api/auth/setup", json={"username": "boss", "password": "bosspassword"},
                      headers=H).status_code == 200                       # now an admin session (cookie)
        assert c.post("/api/research/quant-model/train", json={}).status_code == 403          # CSRF still needed
        r = c.post("/api/research/quant-model/train", json={}, headers=H)
        assert r.status_code == 200 and r.json()["status"] in ("training", "ready") and q.trained == [True]


def test_login_required_once_accounts_exist(app):
    gw, client, fakes, install = app
    with client() as c:
        install()
        c.post("/api/auth/setup", json={"username": "boss", "password": "bosspassword"}, headers=H)
        c.cookies.clear()
        assert c.get("/api/research/search?q=rel").status_code == 401
        assert c.get("/api/research/history/RELIANCE").status_code == 401
        assert c.post("/api/research/montecarlo/RELIANCE", json={}, headers=H).status_code == 401


def test_screener_endpoint(app):
    gw, client, fakes, install = app
    s = fakes["screener"]
    with client() as c:
        install()
        out = c.get("/api/research/screener/reliance").json()
        assert out["symbol"] == "RELIANCE" and s.calls[-1] == ("RELIANCE", "auto", False)
        c.get("/api/research/screener/M%26M?refresh=1&view=standalone")
        assert s.calls[-1] == ("M&M", "standalone", True)
        r = c.get("/api/research/screener/NOPE")                         # unresolved: still asked (BSE-only, aliases)
        assert r.status_code == 404 and "not on Screener.in" in r.json()["detail"] and s.calls[-1][0] == "NOPE"
        r = c.get("/api/research/screener/DOWN")
        assert r.status_code == 502 and "HTTP 503" in r.json()["detail"]
        r = c.get("/api/research/screener/RELIANCE?view=weird")
        assert r.status_code == 422 and "view" in r.json()["detail"]


def test_shutdown_closes_the_shared_dhan_client(app):
    """The history layer keeps one Dhan client for every top-up; the app's shutdown closes it (inside the app's
    own lifespan, so Redis is still open) and the next top-up builds a fresh one."""
    gw, client, fakes, install = app

    class Rest:
        closed = 0

        async def close(self):
            Rest.closed += 1
    with client() as c:
        rs = install()
        rs.data._rest = Rest()
        assert c.get("/api/research/search", params={"q": "rel"}).status_code == 200
        assert Rest.closed == 0
    assert Rest.closed == 1 and rs.data._rest is None
    with client() as c:                                                    # restart: same container, no client yet
        assert c.get("/api/research/history/RELIANCE").status_code == 200
    assert Rest.closed == 1


def test_ai_endpoints(app):
    gw, client, fakes, install = app
    ai = fakes["ai"]
    with client() as c:
        install()
        cfg = c.get("/api/research/ai-config").json()
        assert cfg["who"] == "local" and cfg["user_limit"] == 10
        assert c.post("/api/research/ai/RELIANCE", json={"refresh": True}).status_code == 403
        job = c.post("/api/research/ai/reliance", json={"refresh": True}, headers=H).json()
        assert job["id"] == "job1" and ai.started[-1] == ("RELIANCE", "local", False, True)
        c.post("/api/research/ai/RELIANCE", headers=H)                       # body is optional
        assert ai.started[-1][3] is False
        assert c.post("/api/research/ai/NOSUCH", json={}, headers=H).status_code == 404
        assert c.post("/api/research/ai/RELIANCE", json={"refresh": "maybe"}, headers=H).status_code == 422
        assert c.get("/api/research/ai-job/job1").json()["status"] == "running"
        r = c.get("/api/research/ai-job/nope")
        assert r.status_code == 404 and "expired" in r.json()["detail"]
        assert c.get("/api/research/ai-job/bad%20id").status_code == 422
        assert c.get("/api/research/ai/RELIANCE/latest").json()["symbol"] == "RELIANCE"
        r = c.get("/api/research/ai/M%26M/latest")
        assert r.status_code == 404 and "M&M" in r.json()["detail"]


def test_missing_module_is_a_503_not_a_crash(app, monkeypatch):
    gw, client, fakes, install = app
    import td.research as pkg
    for mod in ("montecarlo", "quant"):
        monkeypatch.setitem(sys.modules, f"td.research.{mod}", None)     # import fails
        monkeypatch.delattr(pkg, mod, raising=False)    # also when an earlier test already imported it
    with client() as c:
        install(mc=None, quant=None)
        r = c.post("/api/research/montecarlo/RELIANCE", json={}, headers=H)
        assert r.status_code == 503 and "Monte Carlo" in r.json()["detail"]
        assert c.get("/api/research/quant-model").status_code == 503
        assert c.get("/api/research/history/RELIANCE").status_code == 200  # the rest keeps working
        assert c.get("/api/research/screener/RELIANCE").status_code == 200


def test_unexpected_errors_say_what_failed(app):
    gw, client, fakes, install = app

    class Broken(FakeMC):
        def simulate(self, df, **kw):
            raise ZeroDivisionError("bad maths")
    class Buggy(FakeMC):
        def simulate(self, df, **kw):
            return {}["fan"]                                                # a bug, not an unknown symbol
    with client() as c:
        install(mc=Broken())
        r = c.post("/api/research/montecarlo/RELIANCE", json={}, headers=H)
        assert r.status_code == 500 and "Monte Carlo simulation failed" in r.json()["detail"]
        install(mc=Buggy())
        r = c.post("/api/research/montecarlo/RELIANCE", json={}, headers=H)
        assert r.status_code == 500 and "KeyError" in r.json()["detail"]
