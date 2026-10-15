"""montecarlo: simulated distributions match theory where it is known (GBM), every model runs and returns the
McResult shape with ordered percentiles, results are deterministic per seed, the drift is what was asked for,
and the calibration backtest is honest — well calibrated on data that really is a random walk, and blind to
anything after each origin."""
import itertools
import math

import numpy as np
import pandas as pd
import pytest

from td.research import montecarlo as mc
from td.research import tsmath as tm

RF, ERP = 0.065, 0.06
MC_KEYS = {"asof", "spot", "spot_source", "horizon_days", "paths", "model", "drift", "seed", "elapsed_ms", "history",
           "fan", "sample_paths", "terminal", "risk", "horizons", "targets", "inputs", "warnings"}
BT_KEYS = {"model", "drift", "horizon_days", "origins", "paths", "elapsed_ms", "coverage", "pit_hist", "crps",
           "crps_naive", "skill", "direction_hit", "effective_n", "points", "verdict", "warnings"}


def _bars(r: np.ndarray, start: str = "2016-01-04", price: float = 100.0) -> pd.DataFrame:
    """Daily bars on consecutive weekdays (IST-midnight epochs) whose log returns are exactly r."""
    days = np.busday_offset(np.datetime64(start, "D"), np.arange(r.size + 1), roll="forward")
    ts = days.astype(np.int64) * 86400 - mc.IST_OFFSET_S
    close = price * np.exp(np.r_[0.0, np.cumsum(r)])
    return pd.DataFrame({"ts": ts, "open": close, "high": close, "low": close, "close": close,
                         "volume": 1000, "oi": 0})


def _exact_normal(n: int, sigma: float, mu: float, seed: int) -> np.ndarray:
    z = np.random.default_rng(seed).standard_normal(n)
    return mu + sigma * (z - z.mean()) / z.std(ddof=1)


def _garch_returns(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    omega, alpha, beta, nu = 4e-6, 0.08, 0.90, 5.0
    z = tm.t_unit_sample(rng, nu, n + 300)
    v, out = omega / (1 - alpha - beta), np.empty(n + 300)
    for t in range(n + 300):
        out[t] = 0.0003 + math.sqrt(v) * z[t]
        v = omega + alpha * (out[t] - 0.0003) ** 2 + beta * v
    return out[300:]


def _finite(obj) -> bool:
    if isinstance(obj, dict):
        return all(_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_finite(v) for v in obj)
    if isinstance(obj, float):
        return math.isfinite(obj)
    return True


@pytest.fixture(scope="module")
def garch_df():
    return _bars(_garch_returns(900, seed=11))


# ------------------------------------------------------------------ simulate: theory
@pytest.mark.parametrize("drift", ["zero", "capm"])
def test_gbm_terminal_mean_and_median_match_theory(drift):
    sigma, h, n = 0.015, 252, 20000
    df = _bars(_exact_normal(600, sigma, 0.0004, seed=1))
    res = mc.simulate(df, horizon_days=h, paths=n, model="gbm", drift=drift, lookback_years=5, seed=3,
                      index_df=df, rf=RF, erp=ERP)        # the stock vs itself: beta exactly 1
    spot = res["spot"]
    d = 0.0 if drift == "zero" else RF + ERP
    assert res["inputs"]["drift_annual"] == pytest.approx(d, abs=1e-4)
    assert res["inputs"]["sigma_annual"] == pytest.approx(sigma * math.sqrt(252), abs=1e-4)
    if drift == "capm":
        assert res["inputs"]["beta"] == pytest.approx(1.0, abs=1e-9)
    T = h / 252
    exp_mean = spot * (1 + d) ** T
    exp_median = spot * math.exp(T * math.log1p(d) - 0.5 * sigma ** 2 * h)
    sd_ratio = math.sqrt(math.exp(sigma ** 2 * h) - 1)                 # lognormal sd / mean
    term = res["terminal"]
    assert term["mean"] == pytest.approx(exp_mean, abs=4 * exp_mean * sd_ratio / math.sqrt(n))
    assert term["median"] == pytest.approx(exp_median, rel=0.01)
    # the fan's 5-95 % spread is the lognormal one: log(p95/p5) = 2·1.645·σ·√h
    one_y = next(x for x in res["horizons"] if x["label"] == "1Y")
    assert math.log(one_y["p95"] / one_y["p5"]) == pytest.approx(2 * 1.6449 * sigma * math.sqrt(h), rel=0.03)
    assert one_y["prob_up"] == pytest.approx(1 - tm.norm_cdf(math.log(spot / exp_median) / (sigma * math.sqrt(h))),
                                             abs=0.015)


def test_every_model_keeps_the_mean_on_the_drift(garch_df):
    """The drift is applied identically: E[S_T] = spot·(1 + D)^T for fat-tailed, clustered, jumpy and switching
    models alike (the convexity correction is model-specific)."""
    for model in mc.MODELS:
        res = mc.simulate(garch_df, horizon_days=126, paths=8000, model=model, drift="zero", lookback_years=5,
                          seed=4, rf=RF, erp=ERP)
        sd = res["terminal"]["std"] / math.sqrt(res["paths"])
        assert res["terminal"]["mean"] == pytest.approx(res["spot"], abs=4.5 * sd), model


# ------------------------------------------------------------------ simulate: shape & invariants
@pytest.mark.parametrize("model", mc.MODELS)
def test_each_model_runs_and_returns_the_contract_shape(garch_df, model):
    spot = float(garch_df["close"].iat[-1])
    targets = [spot * 1.15, spot * 0.85, spot * 1.4]
    res = mc.simulate(garch_df, horizon_days=300, paths=1200, model=model, drift="capm", lookback_years=3,
                      targets=targets, seed=9, rf=RF, erp=ERP)
    assert set(res) == MC_KEYS
    assert res["model"] == model and res["paths"] == 1200 and _finite(res)
    fan = res["fan"]
    ts = [p["t"] for p in fan]
    assert len(fan) <= 200 and ts[0] == 0 and ts[-1] == 300 and ts == sorted(ts)
    assert {21, 63, 126, 252} <= set(ts)                                # checkpoints are exact fan points
    assert all(fan[0][k] == res["spot"] for k in mc.FAN_KEYS + ("mean",))
    for p in fan:
        vals = [p[k] for k in mc.FAN_KEYS]
        assert vals == sorted(vals)
    assert len(res["sample_paths"]) == 30 and all(len(sp) == len(fan) for sp in res["sample_paths"])
    term = res["terminal"]
    assert len(term["edges"]) == 61 and len(term["counts"]) == 60
    assert term["edges"] == sorted(term["edges"]) and sum(term["counts"]) <= res["paths"]
    assert term["p1"] <= term["p5"] <= term["p25"] <= term["median"] <= term["p75"] <= term["p95"] <= term["p99"]
    assert [h["label"] for h in res["horizons"]] == ["1M", "3M", "6M", "1Y", "300D"]
    for h in res["horizons"]:
        assert h["p5"] <= h["p25"] <= h["p50"] <= h["p75"] <= h["p95"]
    risk = res["risk"]
    assert risk["var99"] >= risk["var95"] and risk["cvar95"] >= risk["var95"] and risk["cvar99"] >= risk["var99"]
    assert 0 <= risk["max_dd_median"] <= risk["max_dd_p90"] < 1
    assert [t["direction"] for t in res["targets"]] == ["up", "down", "up"]
    for t in res["targets"]:
        assert 0 <= t["prob_end_beyond"] <= t["prob_touch"] <= 1
    assert res["targets"][2]["prob_touch"] <= res["targets"][0]["prob_touch"]   # farther level, rarer touch
    inp = res["inputs"]
    assert inp["bars"] > 250 and inp["first"] < inp["last"] == res["asof"]
    assert (inp["garch"] is not None) == (model in ("garch", "ensemble"))
    assert (inp["regime"] is not None) == (model in ("regime", "ensemble"))
    assert (inp["jump"] is not None) == (model in ("jump", "ensemble"))
    if model == "ensemble":
        assert [c["model"] for c in inp["components"]] == list(mc.ENSEMBLE)
        assert sum(c["weight"] for c in inp["components"]) == pytest.approx(1.0)
    else:
        assert inp["components"] is None


def test_garch_inputs_are_sane(garch_df):
    g = mc.simulate(garch_df, horizon_days=21, paths=500, model="garch", drift="zero", seed=1)["inputs"]["garch"]
    assert 0 < g["alpha"] < 0.3 and 0.5 < g["beta"] < 1 and g["persistence"] < 0.999
    assert g["long_run_vol"] == pytest.approx(
        mc.simulate(garch_df, horizon_days=21, paths=500, model="gbm", drift="zero", seed=1)["inputs"]["sigma_annual"],
        rel=0.01)                                        # variance targeting: long-run vol = the window's vol


def test_deterministic_for_a_seed(garch_df):
    kw = {"horizon_days": 63, "paths": 800, "model": "ensemble", "drift": "capm", "lookback_years": 3,
          "targets": [50.0]}
    a = mc.simulate(garch_df, seed=21, **kw)
    b = mc.simulate(garch_df, seed=21, **kw)
    c = mc.simulate(garch_df, seed=22, **kw)
    a.pop("elapsed_ms"), b.pop("elapsed_ms"), c.pop("elapsed_ms")
    assert a == b
    assert a["fan"] != c["fan"]


def test_fan_widens_like_sqrt_t():
    df = _bars(_exact_normal(800, 0.02, 0.0, seed=2))
    res = mc.simulate(df, horizon_days=504, paths=6000, model="gbm", drift="zero", seed=5)
    by = {h["days"]: math.log(h["p95"] / h["p5"]) for h in res["horizons"]}
    assert by[252] / by[63] == pytest.approx(2.0, rel=0.06)
    assert by[504] / by[126] == pytest.approx(2.0, rel=0.06)


def test_future_dates_are_weekdays_after_the_last_bar():
    df = _bars(_exact_normal(400, 0.01, 0.0, seed=3), start="2020-01-06")
    res = mc.simulate(df, horizon_days=30, paths=200, model="gbm", drift="zero", seed=1)
    ts = [p["ts"] for p in res["fan"]]
    assert ts[0] == int(df["ts"].iat[-1]) and all(b > a for a, b in itertools.pairwise(ts))
    wd = [pd.Timestamp(t + mc.IST_OFFSET_S, unit="s").weekday() for t in ts]
    assert max(wd) <= 4
    assert len(res["fan"]) == 31                         # short horizons keep every step


# ------------------------------------------------------------------ drift
def test_drift_schedules():
    r = _exact_normal(500, 0.02, 0.003, seed=4)          # +75 %/yr historical mean, 32 % vol
    capm = math.log1p(RF + ERP)
    d = mc._drift("capm", 100, r, 1.0, 104, RF, ERP, None, True)
    assert d.kind == "capm" and np.allclose(d.g, capm / 252) and d.beta == pytest.approx(1.0)
    q = mc._drift("quant", 100, r, 1.0, 104, RF, ERP, 0.40, True)
    assert np.allclose(q.g[:21], math.log1p(0.40) / 252) and np.allclose(q.g[62:], capm / 252)
    assert np.all(np.diff(q.g[20:63]) < 0)               # day 21 → 63: linear fade from the quant view to CAPM
    fb = mc._drift("quant", 100, r, 1.0, 104, RF, ERP, None, True)
    assert fb.kind == "capm" and any("CAPM" in w for w in fb.warnings)
    hist = mc._drift("historical", 100, r, 1.0, 104, RF, ERP, None, True)
    raw = math.expm1(252 * (r.mean() + 0.5 * r.var(ddof=1)))
    assert RF + ERP < hist.annual < raw and hist.annual < 0.40     # shrunk hard: 2 years say little
    assert "noisy" in hist.note
    z = mc._drift("zero", 100, r, None, 0, RF, ERP, None, False)
    assert np.all(z.g == 0) and not z.warnings
    nb = mc._drift("capm", 100, r, None, 0, RF, ERP, None, False)
    assert nb.beta is None and nb.annual == pytest.approx(RF + ERP) and "β = 1" in nb.warnings[0]


def test_beta_vs_index_and_blume_adjustment():
    rng = np.random.default_rng(6)
    rm = rng.normal(0.0004, 0.01, 800)
    stock = _bars(1.5 * rm + rng.normal(0, 0.004, 800))
    index = _bars(rm, price=10000.0)
    res = mc.simulate(stock, horizon_days=21, paths=300, model="gbm", drift="capm", index_df=index, seed=1,
                      rf=RF, erp=ERP)
    beta_adj = res["inputs"]["beta"]
    assert beta_adj == pytest.approx(0.67 * 1.5 + 0.33, abs=0.08)
    assert res["inputs"]["drift_annual"] == pytest.approx(RF + beta_adj * ERP, abs=1e-3)
    missing = mc.simulate(stock, horizon_days=21, paths=300, model="gbm", drift="capm", seed=1, rf=RF, erp=ERP)
    assert missing["inputs"]["beta"] is None and any("NIFTY" in w for w in missing["warnings"])


# ------------------------------------------------------------------ data hygiene & limits
def test_rejects_short_history_and_warns_on_short_lookback():
    with pytest.raises(ValueError, match="at least 250"):
        mc.simulate(_bars(_exact_normal(200, 0.01, 0, seed=1)), horizon_days=21, paths=200, model="gbm",
                    drift="zero")
    res = mc.simulate(_bars(_exact_normal(300, 0.01, 0, seed=1)), horizon_days=21, paths=200, model="gbm",
                      drift="zero")
    assert any("less than 2 years" in w for w in res["warnings"])
    with pytest.raises(ValueError):
        mc.simulate(_bars(_exact_normal(300, 0.01, 0, seed=1)), horizon_days=800, paths=200, model="gbm",
                    drift="zero")
    with pytest.raises(ValueError):
        mc.simulate(_bars(_exact_normal(300, 0.01, 0, seed=1)), horizon_days=21, paths=200, model="nope",
                    drift="zero")


def test_one_year_lookback_works_despite_short_calendar_years():
    df = _bars(_exact_normal(800, 0.01, 0, seed=8))
    df = df[np.arange(len(df)) % 16 != 5].reset_index(drop=True)      # ~16 holidays a year: ≈ 245 sessions
    res = mc.simulate(df, horizon_days=21, paths=200, model="gbm", drift="zero", lookback_years=1)
    assert res["inputs"]["bars"] == mc.MIN_BARS                         # widened from the calendar year
    assert not any("only" in w for w in res["warnings"])


def test_paths_are_capped():
    n, w = mc._cap_paths(50000, 756)
    assert n * 756 <= mc.MAX_CELLS and n == mc.MAX_CELLS // 756 and "reduced" in w
    assert mc._cap_paths(10000, 756) == (10000, None)


def test_bad_prints_are_removed():
    r = _exact_normal(500, 0.012, 0.0, seed=9)
    r[200], r[201] = r[200] + math.log(0.6), r[201] - math.log(0.6)    # one bar printed at 60 % of its value
    res = mc.simulate(_bars(r), horizon_days=21, paths=300, model="gbm", drift="zero", seed=1)
    assert any("bad print" in w for w in res["warnings"])
    assert res["inputs"]["sigma_annual"] < 0.012 * math.sqrt(252) * 1.15
    crash = _exact_normal(500, 0.012, 0.0, seed=9)
    crash[200] = math.log(0.6)                                          # a real one-way crash is kept
    res2 = mc.simulate(_bars(crash), horizon_days=21, paths=300, model="gbm", drift="zero", seed=1)
    assert not any("bad print" in w for w in res2["warnings"])


# ------------------------------------------------------------------ backtest
def test_crps_closed_form_matches_sample_estimator():
    rng = np.random.default_rng(1)
    mu, s = math.log(100.0), 0.2
    x = np.exp(mu + s * rng.standard_normal(200_000))
    for y in (70.0, 100.0, 135.0):
        assert mc._crps_sample(x, y) == pytest.approx(mc._crps_lognormal(mu, s, y), rel=0.01)


def test_backtest_on_a_true_random_walk_is_calibrated():
    sigma = 0.015
    df = _bars(_exact_normal(2000, sigma, -0.5 * sigma ** 2, seed=12))   # martingale prices
    bt = mc.backtest(df, horizon_days=10, model="gbm", drift="zero", lookback_years=2, origins=60, paths=1000,
                     seed=1)
    assert set(bt) == BT_KEYS and _finite(bt)
    cov = {c["level"]: c["hit_rate"] for c in bt["coverage"]}
    assert 0.75 <= cov[0.9] <= 0.98
    assert cov[0.5] <= cov[0.8] <= cov[0.9]
    assert bt["origins"] == 60 and len(bt["points"]) == 60
    assert sum(bt["pit_hist"]) == pytest.approx(1.0, abs=1e-3) and len(bt["pit_hist"]) == 10
    assert abs(bt["skill"]) < 0.05                       # same model as the naive benchmark, up to MC noise
    assert bt["effective_n"] >= 40 and "90% bands held" in bt["verdict"]
    ens = mc.backtest(df, horizon_days=21, model="ensemble", drift="zero", lookback_years=2, origins=25, paths=800,
                      seed=1)
    assert 0.75 <= {c["level"]: c["hit_rate"] for c in ens["coverage"]}[0.9] <= 1.0


def test_backtest_has_no_look_ahead():
    """Forecasts at an origin depend only on bars up to it: rewriting the future (stock AND index) leaves them
    bit-for-bit identical, and a fit on data truncated at the origin reproduces them."""
    rng = np.random.default_rng(13)
    rm = rng.normal(0.0003, 0.01, 1300)
    stock, index = _bars(1.2 * rm + rng.normal(0, 0.01, 1300)), _bars(rm, price=10000.0)
    kw = {"horizon_days": 21, "model": "ensemble", "drift": "capm", "lookback_years": 2, "origins": 12,
          "paths": 400, "seed": 5, "rf": RF, "erp": ERP}
    base = mc.backtest(stock, index_df=index, **kw)
    k = 1000                                             # rewrite every bar after k
    s2, i2 = stock.copy(), index.copy()
    s2.loc[k + 1:, "close"] *= np.exp(np.cumsum(rng.normal(0.002, 0.01, len(s2) - k - 1)))
    i2.loc[k + 1:, "close"] *= np.exp(np.cumsum(rng.normal(-0.002, 0.01, len(i2) - k - 1)))
    other = mc.backtest(s2, index_df=i2, **kw)
    cut_ts = int(stock["ts"].iat[k])
    same = [(a, b) for a, b in zip(base["points"], other["points"]) if a["ts"] <= cut_ts]
    assert len(same) >= 4
    for a, b in same:
        assert (a["p5"], a["p50"], a["p95"], a["spot"]) == (b["p5"], b["p50"], b["p95"], b["spot"])
    later = [(a, b) for a, b in zip(base["points"], other["points"]) if a["ts"] > cut_ts]
    assert later and any(a["p50"] != b["p50"] for a, b in later)

    p = same[-1][0]
    i = int(np.searchsorted(stock["ts"].to_numpy(), p["ts"]))
    s = mc._series(stock.iloc[: i + 1])
    ix = mc._series(index[index["ts"] <= p["ts"]])
    sim, _ = mc._origin_forecast(s, ix, i, 21, "ensemble", "capm", 2.0, 400, RF, ERP, 5)
    assert np.round(np.quantile(sim, [0.05, 0.5, 0.95]), 2).tolist() == [p["p5"], p["p50"], p["p95"]]


def test_backtest_needs_enough_history():
    df = _bars(_exact_normal(400, 0.01, 0, seed=1))
    with pytest.raises(ValueError, match="years of history before each origin"):
        mc.backtest(df, horizon_days=252, model="gbm", drift="zero", lookback_years=2, origins=20, paths=200)
    bt = mc.backtest(df, horizon_days=63, model="gbm", drift="quant", lookback_years=1, origins=20, paths=200)
    assert bt["drift"] == "capm" and any("look-ahead" in w for w in bt["warnings"])
