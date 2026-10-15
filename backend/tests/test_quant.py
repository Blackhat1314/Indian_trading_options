"""quant (QuantEdge): features are causal, the walk-forward finds nothing in random walks and finds planted
predictability, the math helpers match known answers, the model JSON round-trips and QuantService answers in the
QuantResult / QuantModelInfo / QuantLeaderboard shapes of frontend/src/lib/research.ts."""
import asyncio
import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest

from td.research import quant as q


# ---------------------------------------------------------------------------- synthetic data
def bars(r: np.ndarray, p0: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """Daily bars on consecutive weekdays from 2021-01-04 (IST-midnight epochs) for log returns r."""
    n = r.size + 1
    days = np.busday_offset(np.datetime64("2021-01-04"), np.arange(n), roll="forward")
    ts = days.astype(np.int64) * 86400 - 19800
    c = p0 * np.exp(np.r_[0.0, np.cumsum(r)])
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"ts": ts, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": np.round(1e6 * np.exp(0.3 * rng.standard_normal(n)))})


def panel(n_stocks: int, n_bars: int, phi: float, seed: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Stocks = 0.5 × a common market random walk + idiosyncratic AR(1) noise with coefficient phi."""
    rng = np.random.default_rng(seed)
    m = 0.01 * rng.standard_normal(n_bars - 1)
    frames = {}
    for i in range(n_stocks):
        e = 0.018 * rng.standard_normal(n_bars - 1) + 0.5 * m
        r = np.empty_like(e)
        r[0] = e[0]
        for t in range(1, e.size):
            r[t] = phi * r[t - 1] + e[t]
        frames[f"S{i:02d}"] = bars(r, seed=seed * 100 + i)
    return frames, bars(m, p0=10000.0, seed=seed)


@pytest.fixture(scope="module")
def null_model():
    frames, nifty = panel(16, 1000, 0.0, seed=7)
    return q.train_model(frames, nifty, data_source="test")


@pytest.fixture(scope="module")
def ar_model():
    frames, nifty = panel(16, 1000, 0.15, seed=7)
    return q.train_model(frames, nifty, data_source="test")


# ---------------------------------------------------------------------------- math helpers
def _t_cdf(x: float, nu: float) -> float:
    c = math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)) / math.sqrt(nu * math.pi)
    g = np.linspace(0.0, x, 20001)
    pdf = c * (1 + g * g / nu) ** (-(nu + 1) / 2)
    return 0.5 + float(np.sum((pdf[1:] + pdf[:-1]) / 2) * (g[1] - g[0]))


@pytest.mark.parametrize("nu", [3.0, 5.0, 10.0, 30.0])
@pytest.mark.parametrize("p", [0.8413447460685429, 0.975])
def test_t_quantile_matches_numeric_cdf(nu, p):
    assert _t_cdf(q.t_quantile(p, nu), nu) == pytest.approx(p, abs=2e-3)
    assert q.t_quantile(1 - p, nu) == pytest.approx(-q.t_quantile(p, nu))
    # unit-variance scaling: t quantile × √((ν−2)/ν); ν → ∞ is the normal quantile
    assert q.unit_t_quantile(p, nu) == pytest.approx(q.t_quantile(p, nu) * math.sqrt((nu - 2) / nu))
    assert q.unit_t_quantile(p, math.inf) == pytest.approx(float(q.tm.norm_ppf(p)))


def test_h_day_nu_matches_kurtosis():
    # excess kurtosis 6/(ν−4) of one day falls as 1/h for an h-day sum
    nu, h = 6.0, 5
    nu_h = q.h_day_nu(nu, h)
    assert 6 / (nu_h - 4) == pytest.approx(6 / (nu - 4) / h)
    assert q.h_day_nu(nu, 1) == nu


def test_auc_matches_brute_force_with_ties():
    rng = np.random.default_rng(3)
    s = np.round(rng.standard_normal(300), 1)                  # many ties
    y = rng.random(300) < 1 / (1 + np.exp(-s))
    pos, neg = s[y], s[~y]
    brute = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (pos.size * neg.size)
    assert q.auc_score(s, y) == pytest.approx(brute)
    order = np.argsort(s, kind="mergesort")
    grp = np.cumsum(np.r_[True, s[order][1:] != s[order][:-1]]) - 1
    assert q._weighted_auc_sorted(y[order], np.ones(300), grp) == pytest.approx(brute)
    assert q.auc_score(np.arange(4.0), np.array([0, 0, 1, 1])) == 1.0
    assert q.auc_score(np.arange(4.0), np.array([1, 1, 0, 0])) == 0.0
    assert q.auc_score(np.ones(4), np.array([0, 1, 0, 1])) == 0.5
    assert q.auc_score(np.arange(3.0), np.ones(3)) is None


def test_logistic_fit_recovers_coefficients_and_ridge_shrinks():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((20000, 3))
    w_true, b_true = np.array([0.5, -1.0, 0.25]), 0.3
    y = (rng.random(20000) < 1 / (1 + np.exp(-(b_true + X @ w_true)))).astype(float)
    b0, w, conv = q.logistic_fit(X, y)
    assert conv
    assert b0 == pytest.approx(b_true, abs=0.06)
    np.testing.assert_allclose(w, w_true, atol=0.06)
    b1, w1, _ = q.logistic_fit(X, y, alpha=1.0)
    assert np.all(np.abs(w1) < np.abs(w))                      # intercept unpenalised, weights shrink
    assert abs(b1 - b0) < 0.2


def test_calibration_spread():
    rng = np.random.default_rng(9)
    lt = 0.8 * rng.standard_normal(40000)
    p_true = 1 / (1 + np.exp(-lt))
    y = (rng.random(lt.size) < p_true).astype(float)
    base = float(y.mean())
    over = 1 / (1 + np.exp(-(q._logit(base) + 2 * (lt - q._logit(base)))))   # twice too confident
    assert q.calib_fit(p_true, base, y) == pytest.approx(1.0, abs=0.08)
    assert q.calib_fit(over, base, y) == pytest.approx(0.5, abs=0.05)
    assert q.calib_fit(rng.random(lt.size) * 0.4 + 0.3, base, y) < 0.1       # unrelated scores → base rate
    flat = q.calib_apply(np.array([0.2, 0.5, 0.9]), 0.56, 0.0)
    np.testing.assert_allclose(flat, 0.56)


def test_edge_status_rules():
    good = {"auc": 0.56, "auc_se": 0.01, "fold_auc": [0.55, 0.57, 0.56, 0.56], "calib_b": 0.6}
    assert q.edge_status(good) == (True, None)
    ok, note = q.edge_status({**good, "brier": 0.2530, "brier_base": 0.2525})   # ranks, but loses on probability
    assert ok and note.startswith("weak edge")
    for bad in ({**good, "auc": 0.515}, {**good, "auc_se": 0.04}, {**good, "fold_auc": [0.6, 0.6, 0.6, 0.49]},
                {**good, "calib_b": 0.0}, {}):
        ok, note = q.edge_status(bad)
        assert not ok and "no reliable edge at this horizon" in note


def test_kalman_null_scale_matches_white_noise():
    rng = np.random.default_rng(1)
    k = q.tm.kalman_llt(np.cumsum(rng.standard_normal(20000)), q_level=q.KF_Q_LEVEL, q_slope=q.KF_Q_SLOPE,
                        r=q.KF_R)
    assert np.std(k["slope"][2000:]) == pytest.approx(q.kf_slope_null_sd(), rel=0.1)


# ---------------------------------------------------------------------------- features
def test_features_are_causal():
    frames, nifty = panel(1, 1100, 0.05, seed=11)
    df = frames["S00"]
    cut = 800
    hmm_end = int(df["ts"].iat[600])
    full = q.compute_features(df, nifty, hmm_end_ts=hmm_end)
    short = q.compute_features(df.iloc[:cut], nifty, hmm_end_ts=hmm_end)           # index data runs ahead
    short2 = q.compute_features(df.iloc[:cut], nifty.iloc[:cut], hmm_end_ts=hmm_end)
    assert full.hmm_fit_end_ts is not None and full.hmm_fit_end_ts <= hmm_end
    for f in (short, short2):
        for name in ("raw", "z", "level", "band_sd", "slope", "bull_prob", "sig_fast"):
            np.testing.assert_allclose(getattr(f, name), getattr(full, name)[:cut], rtol=1e-10, atol=1e-12,
                                       equal_nan=True, err_msg=name)
    # every feature becomes available after the warm-up, and the stored HMM reproduces the fitted one
    assert np.isfinite(full.z[q.FEATURE_WARMUP:]).all()
    again = q.compute_features(df, nifty, hmm=q.hmm_params(full.hmm, full.hmm_fit_end_ts))
    np.testing.assert_allclose(again.z, full.z, rtol=1e-10, atol=1e-12, equal_nan=True)
    # the trend t-stat is on the random-walk null scale (not the filter's own, ~8x smaller, slope se)
    kt = full.raw[300:, q.FI["kalman_t"]]
    assert 0.5 < np.nanstd(kt) < 2.0


# ---------------------------------------------------------------------------- walk-forward
def test_random_walks_show_no_edge(null_model):
    model, _ = null_model
    assert len(model.metrics) == 3
    for m in model.metrics:
        assert 0.45 <= m["auc"] <= 0.56, m
        assert 0.45 <= m["auc_pooled"] <= 0.56, m
        assert m["folds"] >= 4 and m["n_test"] > 1000
        assert abs(m["ic"]) < 0.05
    ev = model.evaluation
    assert ev["hmm_fit_end"] < ev["test_start"]                 # HMMs fitted before the evaluation window
    assert len(ev["folds"]) >= 4


def test_planted_autocorrelation_is_found(ar_model):
    model, feats = ar_model
    m1 = model.horizons[1].metrics
    assert m1["auc"] > 0.55, m1
    assert m1["edge"] is True and model.horizons[1].calib_b > 0.3
    assert all(a > 0.53 for a in m1["fold_auc"])
    # the fitted model uses the AR(1) features: last-day return / AR signal carry the largest weights
    top = [q.FEATURE_KEYS[j] for j in np.argsort(-np.abs(model.horizons[1].coef))[:3]]
    assert {"ret_1", "ar1_signal"} & set(top)
    # a stock that just rose should get a higher next-day P(up) than one that just fell
    fe = feats["S00"]
    z = fe.z[-1].copy()
    hi, lo = z.copy(), z.copy()
    hi[q.FI["ret_1"]], lo[q.FI["ret_1"]] = 2.0, -2.0
    hm = model.horizons[1]
    assert hm.p_up(hi[None, :])[0] > hm.p_up(lo[None, :])[0]


def test_model_json_round_trip(ar_model, tmp_path):
    model, feats = ar_model
    path = tmp_path / "research" / "quant_model.json"
    q.save_model(model, path)
    json.loads(path.read_text())                                # strict JSON (no NaN)
    back = q.load_model(path)
    assert back.metrics == model.metrics and back.symbols == model.symbols and back.rows == model.rows
    assert back.hmm.keys() == model.hmm.keys() and back.hmm_end_ts == model.hmm_end_ts
    z = np.vstack([feats[s].z[-1] for s in model.symbols])
    for h in q.HORIZONS:
        np.testing.assert_allclose(back.horizons[h].p_up(z), model.horizons[h].p_up(z), rtol=1e-12)
    raw = json.loads(path.read_text())
    raw["features"] = raw["features"][:-1]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        q.load_model(path)


def test_predict_without_enough_history(ar_model):
    model, _ = ar_model
    frames, nifty = panel(1, 200, 0.0, seed=4)
    with pytest.raises(ValueError):
        q.predict(model, "SHORT", frames["S00"].iloc[:40], nifty)
    out = q.predict(model, "SHORT", frames["S00"], nifty)
    assert out["horizons"] == [] and any("needs about" in w for w in out["warnings"])
    assert out["chart"] is not None and out["components"]


# ---------------------------------------------------------------------------- service
class FakeLake:
    def __init__(self, frames):
        self.frames = frames

    def symbols(self, interval):
        return sorted(self.frames)


class FakeData:
    """The slice of ResearchData that QuantService uses."""

    def __init__(self, frames, nifty):
        self.frames = {**frames, "NIFTY": nifty}
        self.lake = FakeLake(self.frames)
        self.calls = []

    async def daily(self, symbol, fetch=True, min_years=0.0):
        self.calls.append((symbol, fetch))
        if symbol not in self.frames:
            raise KeyError(symbol)
        return self.frames[symbol].copy(), {"fetched": False, "warning": None, "warnings": [], "source": "lake"}

    async def index_daily(self, name="NIFTY"):
        return self.frames[name].copy()


HORIZON_KEYS = {"h", "label", "p_up", "exp_ret", "ret_sd", "range68", "range95", "price_median", "signal",
                "confidence", "oos_auc", "oos_acc", "base_rate"}
INFO_KEYS = {"status", "progress", "message", "trained_at", "symbols", "rows", "features", "metrics"}
METRIC_KEYS = {"h", "auc", "acc", "brier", "brier_base", "logloss", "logloss_base", "base_rate", "ic", "n_test",
               "folds"}


def _finite(xs):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def test_service_result_shapes(tmp_path):
    frames, nifty = panel(8, 1000, 0.1, seed=21)
    frames["SHORTY"] = frames["S00"].iloc[-150:].reset_index(drop=True)   # too short for the panel
    data = FakeData(frames, nifty)

    async def run():
        qs = q.QuantService(data, model_path=tmp_path / "quant_model.json")
        first = await qs.result("S01")                         # starts training, never waits for it
        assert first["status"] == "training" and first["horizons"] == []
        assert first["model"]["status"] == "training" and first["chart"] is not None
        info = await qs.train(force=False)                     # joins the running job
        assert info["status"] == "ready", info
        return qs, info, await qs.result("s01"), await qs.result("SHORTY"), await qs.leaderboard(5, 3)

    _, info, res, short, board = asyncio.run(run())
    assert all(fetch is False for s, fetch in data.calls if s != "NIFTY")
    assert (tmp_path / "quant_model.json").exists()
    assert set(info) >= INFO_KEYS and info["symbols"] == 8 and info["features"] == list(q.FEATURE_KEYS)
    assert [m["h"] for m in info["metrics"]] == [1, 5, 21] and all(set(m) >= METRIC_KEYS for m in info["metrics"])

    json.dumps(res, allow_nan=False)
    assert res["symbol"] == "S01" and res["status"] == "ready" and res["spot"] > 0 and len(res["asof"]) == 10
    assert [h["h"] for h in res["horizons"]] == [1, 5, 21]
    for h in res["horizons"]:
        assert set(h) >= HORIZON_KEYS and h["label"] == q.H_LABELS[h["h"]]
        assert 0 < h["p_up"] < 1 and h["signal"] in ("UP", "DOWN", "NEUTRAL")
        assert h["confidence"] in ("low", "medium", "high")
        lo95, lo68, hi68, hi95 = h["range95"][0], h["range68"][0], h["range68"][1], h["range95"][1]
        assert lo95 < lo68 < h["price_median"] < hi68 < hi95
        if not h["edge"]:
            assert h["signal"] == "NEUTRAL" and h["confidence"] == "low"
    for c in res["components"]:
        assert set(c) == {"key", "label", "value", "display", "z", "reading", "tone"}
        assert c["tone"] in ("up", "down", "neutral") and math.isfinite(c["value"])
    assert {c["key"] for c in res["components"]} >= {"trend", "momentum", "regime", "vol_regime", "rsi"}
    assert 0 < len(res["contributions"]) <= 10
    assert set(res["contributions"][0]) == {"feature", "label", "value", "contribution"}
    reg = res["regime"]
    assert reg["state"] in ("bull", "bear", "sideways") and sum(reg["probs"]) == pytest.approx(1, abs=1e-3)
    assert len(reg["mu_annual"]) == len(reg["vol_annual"]) == len(reg["expected_duration"]) == 2
    assert set(res["vol"]) == {"garch_now", "garch_21d", "long_run", "ewma", "realized_21", "percentile"}
    assert 0 <= res["vol"]["percentile"] <= 1
    ch = res["chart"]
    n = len(ch["ts"])
    assert 200 <= n <= 252 and all(len(ch[k]) == n for k in ("close", "trend", "upper", "lower", "bull_prob"))
    assert _finite(ch["close"] + ch["trend"] + ch["upper"] + ch["lower"] + ch["bull_prob"])
    assert all(lo <= t <= up for lo, t, up in zip(ch["lower"], ch["trend"], ch["upper"]))
    fc = ch["forecast"]
    assert [p["t"] for p in fc] == list(range(22)) and fc[0]["p50"] == res["spot"]
    assert all(a["ts"] < b["ts"] for a, b in itertools.pairwise(fc))
    assert all(p["lo95"] <= p["lo68"] <= p["p50"] <= p["hi68"] <= p["hi95"] for p in fc)
    assert {m["h"] for m in res["stock_metrics"]} == {1, 5, 21}
    assert set(res["model"]) >= INFO_KEYS

    assert short["horizons"] == [] and short["status"] == "ready"
    assert any("needs about" in w for w in short["warnings"])

    assert board["h"] == 5 and set(board) >= {"h", "asof", "up", "down", "model"}
    assert len(board["up"]) == len(board["down"]) == 3
    assert set(board["up"][0]) == {"symbol", "sector", "close", "p_up", "exp_ret", "signal"}
    assert board["up"][0]["p_up"] >= board["up"][-1]["p_up"] >= board["down"][0]["p_up"]
    assert "SHORTY" not in [x["symbol"] for x in board["up"] + board["down"]]


def test_service_reports_training_failure(tmp_path):
    frames, nifty = panel(2, 400, 0.0, seed=2)                  # far too little history to train
    data = FakeData(frames, nifty)

    async def run():
        qs = q.QuantService(data, model_path=tmp_path / "m.json")
        info = await qs.train()
        res = await qs.result("S00")
        return info, res, await qs.leaderboard(21, 5)

    info, res, board = asyncio.run(run())
    assert info["status"] == "error" and "3 years" in info["message"]
    assert res["status"] == "error" and res["horizons"] == [] and res["components"]
    assert board["up"] == [] and board["model"]["status"] == "error"
    with pytest.raises(ValueError):
        asyncio.run(q.QuantService(data, model_path=tmp_path / "m.json").leaderboard(2, 5))
