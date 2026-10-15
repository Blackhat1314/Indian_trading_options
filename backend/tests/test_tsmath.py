"""tsmath: estimators recover known parameters on synthetic data, filters are causal (appending data never
changes earlier outputs), fits are deterministic, and degenerate inputs degrade gracefully."""
import math
import time

import numpy as np
import pandas as pd
import pytest

from td.research import tsmath as tm


def _sim_garch(n, mu, omega, alpha, beta, nu, seed, burn=500):
    rng = np.random.default_rng(seed)
    z = tm.t_unit_sample(rng, nu, n + burn)
    v = omega / (1 - alpha - beta)
    r = np.empty(n + burn)
    for t in range(n + burn):
        e = math.sqrt(v) * z[t]
        r[t] = mu + e
        v = omega + alpha * e * e + beta * v
    return r[burn:]


def _sim_regimes(n, seed):
    """Bull: +0.2 %/day, 0.8 % sd, stays 98 %; bear: −0.3 %/day, 2.5 % sd, stays 97 %."""
    rng = np.random.default_rng(seed)
    stay = (0.98, 0.97)
    s = np.empty(n, dtype=int)
    s[0] = 0
    u = rng.random(n)
    for t in range(1, n):
        s[t] = s[t - 1] if u[t] < stay[s[t - 1]] else 1 - s[t - 1]
    mu, sd = np.array([0.002, -0.003]), np.array([0.008, 0.025])
    return mu[s] + sd[s] * rng.standard_normal(n), s


@pytest.fixture(scope="module")
def garch_sample():
    return _sim_garch(4000, 3e-4, 4e-6, 0.08, 0.90, 6.0, seed=11)


@pytest.fixture(scope="module")
def garch_fit(garch_sample):
    return tm.garch11_fit(garch_sample)


@pytest.fixture(scope="module")
def regimes():
    return _sim_regimes(1250, seed=1)


@pytest.fixture(scope="module")
def hmm2(regimes):
    return tm.GaussianHMM(2).fit(regimes[0], seed=7)


# ------------------------------------------------------------------ returns & volatility
def test_log_returns():
    assert np.allclose(tm.log_returns([100, 110, 99]), [math.log(1.1), math.log(0.9)])
    for bad in ([100, 0, 5], [100, np.nan, 5], [100, -1]):
        with pytest.raises(ValueError):
            tm.log_returns(bad)


def test_ewma_vol_matches_definition_and_is_causal():
    rng = np.random.default_rng(1)
    r = 0.02 * rng.standard_normal(3000)
    v = tm.ewma_vol(r, lam=0.94)
    assert v.shape == r.shape
    assert v[0] == pytest.approx(abs(r[0]))
    t = 57
    w = 0.94 ** np.arange(t, -1, -1)
    assert v[t] == pytest.approx(math.sqrt((w * r[: t + 1] ** 2).sum() / w.sum()), rel=1e-12)
    assert abs(np.median(v[200:]) - 0.02) < 0.0015
    assert np.array_equal(tm.ewma_vol(r[:1000]), v[:1000])
    r2 = r.copy()
    r2[5] = np.nan                                         # a missing return carries no weight, no NaN spill
    assert np.isfinite(tm.ewma_vol(r2)[5:]).all()


def test_yang_zhang_matches_close_to_close_on_gbm():
    rng = np.random.default_rng(21)
    n, sig, steps = 800, 0.02, 78
    o, h, low, c = (np.empty(n) for _ in range(4))
    prev = 100.0
    for t in range(n):
        op = prev * math.exp(math.sqrt(0.3) * sig * rng.standard_normal())   # 30 % of variance overnight
        path = op * np.exp(np.cumsum(math.sqrt(0.7 / steps) * sig * rng.standard_normal(steps)))
        o[t], h[t], low[t], c[t] = op, max(op, path.max()), min(op, path.min()), path[-1]
        prev = c[t]
    yz = tm.yang_zhang_vol(o, h, low, c, window=21)
    assert np.isnan(yz[:21]).all() and np.isfinite(yz[21:]).all()
    cc = np.std(np.diff(np.log(c)))
    assert abs(np.mean(yz[21:]) / cc - 1) < 0.08
    assert abs(np.mean(yz[21:]) / sig - 1) < 0.12
    assert np.array_equal(tm.yang_zhang_vol(o[:300], h[:300], low[:300], c[:300]), yz[:300], equal_nan=True)


# ------------------------------------------------------------------ optimiser
def test_nelder_mead_rosenbrock():
    def rosen(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    x, fx, it, conv = tm.nelder_mead(rosen, [-1.2, 1.0], step=0.5, max_iter=5000, tol=1e-12)
    assert conv and fx < 1e-9 and np.allclose(x, [1, 1], atol=1e-4) and it > 0


def test_nelder_mead_treats_errors_as_infeasible():
    def f(x):
        if x[0] < 0:
            raise ValueError("infeasible")
        return (x[0] - 2) ** 2 + (x[1] + 1) ** 2

    x, fx, _, conv = tm.nelder_mead(f, [1.0, 1.0], step=1.0)
    assert conv and np.allclose(x, [2, -1], atol=1e-4)
    _, fx, _, conv = tm.nelder_mead(lambda x: math.nan, [0.0])
    assert fx == math.inf and not conv


# ------------------------------------------------------------------ Student-t
def test_t_unit_sample_and_df_fit():
    rng = np.random.default_rng(2)
    z = tm.t_unit_sample(rng, 5.0, 200_000)
    assert abs(z.var() - 1) < 0.05
    assert 4.0 < tm.fit_t_df(z[:20_000]) < 6.5
    assert tm.fit_t_df(rng.standard_normal(20_000)) > 30
    assert abs(tm.t_unit_sample(rng, math.inf, 50_000).var() - 1) < 0.03
    assert math.isnan(tm.fit_t_df([1.0, 2.0])) and math.isnan(tm.fit_t_df(np.ones(50)))
    with pytest.raises(ValueError):
        tm.t_unit_sample(rng, 2.0, 5)


# ------------------------------------------------------------------ GARCH
def test_garch_recovers_simulated_params(garch_fit):
    f = garch_fit
    assert f.converged and f.note == "" and f.n_obs == 4000 and f.dist == "t"
    assert abs(f.alpha - 0.08) < 0.03
    assert abs(f.beta - 0.90) < 0.04
    assert 4.0 < f.nu < 9.0
    assert f.persistence == pytest.approx(f.alpha + f.beta) and f.persistence < 0.999
    assert 0.6 < f.long_run_var / (4e-6 / 0.02) < 1.6
    assert f.long_run_var == pytest.approx(f.omega / (1 - f.persistence))
    assert abs(f.mu - 3e-4) < 6e-4 and np.isfinite(f.loglik)


def test_garch_normal_innovations(garch_sample):
    f = tm.garch11_fit(garch_sample, dist="normal")
    assert f.converged and f.nu == math.inf and abs(f.alpha + f.beta - 0.98) < 0.04
    assert tm.garch11_simulate(f, 5, 10, np.random.default_rng(0)).shape == (10, 5)


def test_garch_variance_targeting(garch_sample, garch_fit):
    f = tm.garch11_fit(garch_sample, variance_targeting=True)
    assert f.converged and f.long_run_var == pytest.approx(garch_sample.var(), rel=1e-9)
    assert abs(f.alpha - 0.08) < 0.03 and abs(f.beta - 0.90) < 0.04 and 4.0 < f.nu < 9.0
    assert f.loglik <= garch_fit.loglik + 1e-6                # a constrained fit cannot beat the free MLE


def test_garch_simulate_variance_ceiling(garch_fit):
    f = garch_fit
    cap = 0.5 * f.long_run_var
    sims = tm.garch11_simulate(f, 200, 3000, np.random.default_rng(9), last_var=20 * f.long_run_var, max_var=cap)
    assert sims.var() < 1.1 * cap
    assert np.array_equal(sims, tm.garch11_simulate(f, 200, 3000, np.random.default_rng(9),
                                                    last_var=20 * f.long_run_var, max_var=cap))


def test_garch_filter_is_causal_and_matches_fit(garch_fit, garch_sample):
    s2 = tm.garch11_filter(garch_fit, garch_sample)
    assert s2.shape == garch_sample.shape and (s2 > 0).all()
    assert s2[-1] == pytest.approx(garch_fit.last_var, rel=1e-9)
    assert garch_sample[-1] - garch_fit.mu == pytest.approx(garch_fit.last_resid, rel=1e-9)
    assert np.array_equal(tm.garch11_filter(garch_fit, garch_sample[:1500]), s2[:1500])


def test_garch_forecast_decays_to_long_run(garch_fit):
    f = garch_fit
    fc = tm.garch11_forecast(f, 3000)
    assert fc[0] == pytest.approx(f.omega + f.alpha * f.last_resid ** 2 + f.beta * f.last_var)
    assert fc[-1] == pytest.approx(f.long_run_var, rel=1e-6)
    gap = np.abs(fc - f.long_run_var)
    assert (np.diff(gap) <= 1e-18).all()                  # monotone approach
    hot = tm.garch11_forecast(f, 10, last_var=9 * f.long_run_var, last_resid=0.0)
    assert hot[0] > hot[-1] > f.long_run_var


def test_garch_simulate_moments_and_determinism(garch_fit):
    f = garch_fit
    lrv = f.long_run_var
    sims = tm.garch11_simulate(f, 250, 4000, np.random.default_rng(3), last_var=lrv, last_resid=math.sqrt(lrv))
    assert sims.shape == (4000, 250) and np.isfinite(sims).all()
    assert 0.8 < sims.var() / lrv < 1.2                    # started at stationarity → unconditional variance
    assert abs(sims.mean() - f.mu) < 5e-4
    again = tm.garch11_simulate(f, 250, 4000, np.random.default_rng(3), last_var=lrv, last_resid=math.sqrt(lrv))
    assert np.array_equal(sims, again)
    hot = tm.garch11_simulate(f, 5, 20_000, np.random.default_rng(4), last_var=10 * lrv, last_resid=0.0)
    assert hot[:, 0].var() > 5 * lrv                       # starts from the current (high) variance


def test_garch_fallbacks_never_raise():
    f = tm.garch11_fit(np.random.default_rng(0).normal(0, 0.01, 50))
    assert not f.converged and f.alpha == 0 and f.beta == 0 and "only 50" in f.note
    assert f.long_run_var == pytest.approx(f.last_var) and f.long_run_var > 0 and f.nu > 2
    z = tm.garch11_fit(np.zeros(500))
    assert not z.converged and "zero variance" in z.note and np.isfinite(z.long_run_var)
    e = tm.garch11_fit([])
    assert not e.converged and e.n_obs == 0
    sims = tm.garch11_simulate(f, 20, 100, np.random.default_rng(1))
    assert np.isfinite(sims).all()


def test_garch_survives_nasty_real_world_returns():
    r = _sim_garch(1250, 5e-4, 1e-5, 0.10, 0.85, 4.0, seed=5)
    r[300] = math.log(0.5)                                 # unadjusted 1:1 bonus
    r[700] = 0.4                                           # +49 % day
    r[900:930] = 0.0                                       # suspended / illiquid run
    r[1000] = np.nan
    f = tm.garch11_fit(r)
    assert f.converged and "1 non-finite" in f.note and f.n_obs == 1249
    assert np.isfinite([f.mu, f.omega, f.alpha, f.beta, f.nu, f.loglik, f.last_var]).all()
    assert f.omega > 0 and f.alpha >= 0 and f.beta >= 0 and f.persistence < 0.999 and 2.1 <= f.nu <= 100


# ------------------------------------------------------------------ Hurst
def test_hurst_of_random_walk_increments_is_half():
    rng = np.random.default_rng(8)
    x = rng.standard_normal(4000)
    assert abs(tm.hurst_rs(x) - 0.5) < 0.08
    assert abs(tm.hurst_dfa(x) - 0.5) < 0.08
    windows = [tm.hurst_dfa(rng.standard_normal(252)) for _ in range(60)]
    assert abs(np.mean(windows) - 0.5) < 0.04             # rolling-window use: no material bias


def _fgn(n, hurst, rng):
    """Fractional Gaussian noise with a known Hurst exponent (Davies–Harte circulant embedding)."""
    k = np.arange(n + 1, dtype=float)
    g = 0.5 * ((k + 1) ** (2 * hurst) - 2 * k ** (2 * hurst) + np.abs(k - 1) ** (2 * hurst))
    c = np.r_[g, g[-2:0:-1]]
    lam = np.maximum(np.fft.fft(c).real, 0.0)
    w = rng.standard_normal(c.size) + 1j * rng.standard_normal(c.size)
    return (np.fft.fft(np.sqrt(lam) * w) / math.sqrt(c.size)).real[:n]


def test_hurst_recovers_long_memory():
    persistent = _fgn(4000, 0.7, np.random.default_rng(1))
    assert abs(np.var(persistent) - 1) < 0.15 and abs(tm.autocorr(persistent) - (2 ** 0.4 - 1)) < 0.05
    assert 0.62 < tm.hurst_dfa(persistent) < 0.78
    assert 0.56 < tm.hurst_rs(persistent) < 0.75           # R/S is pulled toward 0.5 (documented)
    anti = _fgn(4000, 0.3, np.random.default_rng(2))
    assert 0.22 < tm.hurst_dfa(anti) < 0.38 and 0.22 < tm.hurst_rs(anti) < 0.40


def test_hurst_flags_misuse_and_short_input():
    x = np.random.default_rng(9).standard_normal(4000)
    assert tm.hurst_dfa(np.cumsum(x)) > 1.3                 # prices passed instead of returns
    assert tm.hurst_dfa(np.diff(x)) < 0.3                  # over-differenced → strongly anti-persistent
    assert math.isnan(tm.hurst_dfa(x[:30])) and math.isnan(tm.hurst_rs(x[:15]))


# ------------------------------------------------------------------ Kalman local linear trend
def test_kalman_slope_tracks_known_trend():
    rng = np.random.default_rng(10)
    slope = np.r_[np.full(300, 0.002), np.full(300, -0.001)]
    y = 5.0 + np.cumsum(slope) + np.cumsum(0.003 * rng.standard_normal(600))
    k = tm.kalman_llt(y, q_level=0.003 ** 2, trend_days=40)
    assert set(k) >= {"level", "slope", "slope_se", "P", "level_se"}
    assert k["P"].shape == (600, 2, 2) and np.isnan(k["slope"][0])
    assert abs(k["slope"][299] - 0.002) < 0.001
    assert abs(k["slope"][-1] + 0.001) < 0.001
    assert (k["slope_se"][5:] > 0).all() and np.isfinite(k["level"]).all()
    assert k["r"] == 0.0 and k["q_slope"] == pytest.approx(0.003 ** 2 * (2 / 41) ** 2 / (1 - 2 / 41))


def test_kalman_is_forward_only():
    rng = np.random.default_rng(11)
    y = np.cumsum(0.01 * rng.standard_normal(800))
    full = tm.kalman_llt(y, q_level=1e-4, q_slope=1e-7, r=1e-5)
    part = tm.kalman_llt(np.r_[y[:500], 9.0 + y[500:]], q_level=1e-4, q_slope=1e-7, r=1e-5)
    for key in ("level", "slope", "slope_se", "P"):
        assert np.array_equal(full[key][:500], part[key][:500], equal_nan=True)


def test_kalman_mle_recovers_noise_and_calibrated_se():
    rng = np.random.default_rng(0)
    n, ql, qs, r = 1500, 1e-4, 1e-7, 4e-5
    lev, slo = np.empty(n), np.empty(n)
    lev[0], slo[0] = 5.0, 0.001
    for t in range(1, n):
        slo[t] = slo[t - 1] + math.sqrt(qs) * rng.standard_normal()
        lev[t] = lev[t - 1] + slo[t - 1] + math.sqrt(ql) * rng.standard_normal()
    y = lev + math.sqrt(r) * rng.standard_normal(n)
    est = tm.kalman_llt_fit(y)
    assert est["converged"]
    assert 0.6 < est["q_level"] / ql < 1.5 and 0.5 < est["r"] / r < 2.0 and 0.2 < est["q_slope"] / qs < 5
    k = tm.kalman_llt(y)                                   # parameters from the MLE
    zs = (k["slope"][50:] - slo[50:]) / k["slope_se"][50:]
    assert 0.8 < zs.std() < 1.25


def test_kalman_mle_on_deterministic_trend():
    rng = np.random.default_rng(12)
    t = np.arange(800)
    k = tm.kalman_llt(0.002 * t + 0.01 * rng.standard_normal(800))
    assert abs(k["slope"][-1] - 0.002) < 1e-4
    with pytest.raises(ValueError):
        tm.kalman_llt([1.0, np.nan, 2.0, 3.0])


# ------------------------------------------------------------------ OU
def test_ou_half_life_recovery():
    rng = np.random.default_rng(13)
    kappa, mu, sd_eq, n = 0.05, 3.0, 0.1, 8000
    b = math.exp(-kappa)
    x = np.empty(n)
    x[0] = mu
    eps = sd_eq * math.sqrt(1 - b * b) * rng.standard_normal(n)
    for t in range(1, n):
        x[t] = mu + b * (x[t - 1] - mu) + eps[t]
    out = tm.ou_fit(x)
    assert out["mean_reverting"]
    assert abs(out["half_life"] / (math.log(2) / kappa) - 1) < 0.25
    assert abs(out["mu"] - mu) < 0.03 and abs(out["sigma_eq"] / sd_eq - 1) < 0.1
    assert abs(out["sigma"] / (sd_eq * math.sqrt(2 * kappa)) - 1) < 0.1


def test_ou_without_mean_reversion():
    x = 100 * 1.001 ** np.arange(300)                      # explosive: b > 1
    out = tm.ou_fit(x)
    assert not out["mean_reverting"] and out["kappa"] == 0.0 and out["half_life"] == math.inf
    assert math.isnan(tm.ou_fit([1.0, 2.0])["kappa"])


# ------------------------------------------------------------------ HMM
def test_hmm_separates_two_regimes(hmm2, regimes):
    r, s = regimes
    m = hmm2
    assert m.means[0] < m.means[1]                         # sorted: state 0 = bear
    assert np.allclose(np.sqrt(m.vars), [0.025, 0.008], rtol=0.15)
    assert abs(m.means[0] + 0.003) < 0.002 and abs(m.means[1] - 0.002) < 0.001
    assert np.allclose(np.diag(m.trans), [0.97, 0.98], atol=0.02)
    assert np.allclose(m.trans.sum(axis=1), 1) and m.startprob.sum() == pytest.approx(1)
    truth = (s == 0).astype(int)                           # simulated state 0 is the bull (sorted index 1)
    assert (m.viterbi(r) == truth).mean() > 0.9
    assert ((m.filter(r)[:, 1] > 0.5) == truth).mean() > 0.9
    assert ((m.smooth(r)[:, 1] > 0.5) == truth).mean() > 0.9
    assert m.converged and np.isfinite(m.loglik) and m.score(r) == pytest.approx(m.loglik, rel=1e-6)


def test_hmm_filter_is_causal(hmm2, regimes):
    r = regimes[0]
    full = hmm2.filter(r)
    assert full.shape == (r.size, 2) and np.allclose(full.sum(axis=1), 1)
    for cut in (1, 2, 100, 701, 1024, 1025):
        assert np.array_equal(hmm2.filter(r[:cut]), full[:cut])
    other_future = np.r_[r[:700], np.random.default_rng(3).normal(0.05, 0.1, 400)]
    assert np.array_equal(hmm2.filter(other_future)[:700], full[:700])


def test_hmm_fit_is_deterministic(hmm2, regimes):
    again = tm.GaussianHMM(2).fit(regimes[0], seed=7)
    for a in ("means", "vars", "trans", "startprob"):
        assert np.array_equal(getattr(hmm2, a), getattr(again, a))


def test_hmm_clip_z_keeps_regimes_despite_outliers(regimes):
    r, s = regimes
    dirty = r.copy()
    dirty[[100, 400, 401, 900]] = [0.45, -0.69, 0.3, -0.5]   # bonus seams / crash days
    plain = tm.GaussianHMM(2).fit(dirty, seed=7)
    robust = tm.GaussianHMM(2).fit(dirty, seed=7, clip_z=6)
    truth = s == 0
    assert np.diag(plain.trans).min() < 0.5                # an "outlier state", not a regime
    assert np.allclose(np.diag(robust.trans), [0.97, 0.98], atol=0.03)
    assert ((robust.filter(dirty)[:, 1] > 0.5) == truth).mean() > 0.9


def test_hmm_three_states_and_errors(regimes):
    m3 = tm.GaussianHMM(3).fit(regimes[0], seed=1)
    assert (np.diff(m3.means) >= 0).all() and np.isfinite(m3.vars).all() and np.allclose(m3.trans.sum(axis=1), 1)
    with pytest.raises(ValueError):
        tm.GaussianHMM(2).fit(np.zeros(500))
    with pytest.raises(ValueError):
        tm.GaussianHMM(2).fit(np.ones(20))
    with pytest.raises(ValueError):
        tm.GaussianHMM(4)
    with pytest.raises(RuntimeError):
        tm.GaussianHMM(2).filter(regimes[0])


def test_hmm_simulate(hmm2):
    rets, states = hmm2.simulate(400, 3000, np.random.default_rng(5), start_probs=[1.0, 0.0])
    assert rets.shape == states.shape == (3000, 400)
    assert (states[:, 0] == 0).mean() == pytest.approx(hmm2.trans[0, 0], abs=0.02)   # first step transitions
    occ = (states[:, -100:] == 1).mean()
    assert occ == pytest.approx(hmm2.startprob[1], abs=0.04)                        # → stationary mix
    bull = states == 1
    assert rets[bull].mean() == pytest.approx(hmm2.means[1], abs=3e-4)
    again = hmm2.simulate(400, 3000, np.random.default_rng(5), start_probs=[1.0, 0.0])
    assert np.array_equal(rets, again[0])


# ------------------------------------------------------------------ bootstrap
def test_stationary_bootstrap_block_lengths():
    n, length = 5000, 2000
    idx = tm.stationary_bootstrap_indices(n, length, 1200, 10, np.random.default_rng(6))   # > 1 chunk
    assert idx.shape == (1200, length) and idx.min() >= 0 and idx.max() < n
    breaks = (idx[:, 1:] != (idx[:, :-1] + 1) % n).sum(axis=1)
    assert np.mean(length / (breaks + 1)) == pytest.approx(10, rel=0.08)
    assert np.array_equal(idx, tm.stationary_bootstrap_indices(n, length, 1200, 10, np.random.default_rng(6)))
    iid = tm.stationary_bootstrap_indices(n, length, 50, 1, np.random.default_rng(7))
    assert (iid[:, 1:] == (iid[:, :-1] + 1) % n).mean() < 0.01
    assert tm.stationary_bootstrap_indices(10, 0, 5, 3, np.random.default_rng(0)).shape == (5, 0)


# ------------------------------------------------------------------ normal distribution
def test_norm_cdf_and_ppf():
    xs = np.linspace(-30, 8, 2001)
    ref = np.array([0.5 * math.erfc(-x / math.sqrt(2)) for x in xs])
    assert np.allclose(tm.norm_cdf(xs), ref, rtol=1e-15, atol=0)
    back = tm.norm_ppf(tm.norm_cdf(xs[xs <= 5]))
    assert np.max(np.abs(back - xs[xs <= 5])) < 1e-9
    assert tm.norm_ppf(0.975) == pytest.approx(1.959963984540054, rel=1e-14)
    assert tm.norm_ppf(0.5) == 0.0 and isinstance(tm.norm_ppf(0.3), float)
    assert tm.norm_ppf(0.0) == -math.inf and tm.norm_ppf(1.0) == math.inf
    assert math.isnan(tm.norm_ppf(1.5)) and math.isnan(tm.norm_ppf(-0.1))
    ps = np.array([1e-300, 1e-10, 0.01, 0.3, 0.7, 0.99, 1 - 1e-10])
    assert np.allclose(tm.norm_cdf(tm.norm_ppf(ps)), ps, rtol=1e-9)


# ------------------------------------------------------------------ small statistics
def test_max_drawdown():
    p = np.array([[100, 120, 90, 130, 65], [100, 101, 102, 103, 104]], dtype=float)
    assert np.allclose(tm.max_drawdown(p), [0.5, 0.0])
    assert np.allclose(tm.max_drawdown(p.T, axis=0), [0.5, 0.0])
    assert tm.max_drawdown([100, 80, 120]) == pytest.approx(0.2)


def test_rolling_beta():
    rng = np.random.default_rng(14)
    rm = 0.01 * rng.standard_normal(2000)
    r = 1.3 * rm + 0.005 * rng.standard_normal(2000)
    b = tm.rolling_beta(r, rm, 250)
    assert np.isnan(b[:249]).all() and np.isfinite(b[249:]).all()
    assert abs(np.mean(b[249:]) - 1.3) < 0.05
    assert b[600] == pytest.approx(np.polyfit(rm[351:601], r[351:601], 1)[0], rel=1e-9)
    assert np.array_equal(tm.rolling_beta(r[:700], rm[:700], 250), b[:700], equal_nan=True)


def _rsi_reference(c, n=14):
    d = np.diff(c)
    g, lo = np.maximum(d, 0), np.maximum(-d, 0)
    out = np.full(len(c), np.nan)
    ag, al = g[:n].mean(), lo[:n].mean()
    for t in range(n, len(c)):
        if t > n:
            ag = (ag * (n - 1) + g[t - 1]) / n
            al = (al * (n - 1) + lo[t - 1]) / n
        out[t] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def test_rsi():
    c = 100 * np.exp(np.cumsum(0.01 * np.random.default_rng(15).standard_normal(500)))
    out = tm.rsi(c)
    assert np.isnan(out[:14]).all()
    assert np.allclose(out[14:], _rsi_reference(c)[14:], rtol=1e-10)
    assert np.array_equal(tm.rsi(c[:200]), out[:200], equal_nan=True)
    assert (tm.rsi(np.arange(1.0, 40.0))[14:] == 100).all()
    assert (tm.rsi(np.full(40, 5.0))[14:] == 50).all()


def test_autocorr():
    rng = np.random.default_rng(16)
    e = rng.standard_normal(20_000)
    x = np.empty_like(e)
    x[0] = 0.0
    for t in range(1, x.size):
        x[t] = 0.5 * x[t - 1] + e[t]
    assert abs(tm.autocorr(x, 1) - 0.5) < 0.03
    assert tm.autocorr(x[:300], 2) == pytest.approx(pd.Series(x[:300]).autocorr(2), rel=1e-12)
    assert math.isnan(tm.autocorr(np.ones(50))) and math.isnan(tm.autocorr([1.0, 2.0]))


# ------------------------------------------------------------------ speed (contract: GARCH < 0.5 s, HMM < 1 s)
def test_fits_are_fast_on_five_years():
    r = _sim_garch(1250, 4e-4, 8e-6, 0.09, 0.88, 5.0, seed=17)
    t0 = time.perf_counter()
    tm.garch11_fit(r)
    t_garch = time.perf_counter() - t0
    t0 = time.perf_counter()
    tm.GaussianHMM(2).fit(r, seed=0)
    t_hmm = time.perf_counter() - t0
    assert t_garch < 1.0 and t_hmm < 2.0, (t_garch, t_hmm)     # 2× headroom over the targets for slow CI
