"""Shared numerical library for the research desks (Monte Carlo, quant model, AI dossier).

The project deliberately carries no scipy / statsmodels / sklearn, so the handful of estimators the desks need —
GARCH(1,1)-t, a Gaussian HMM, a local-linear-trend Kalman filter, OU, Hurst, the stationary bootstrap, normal
cdf/ppf and Nelder–Mead — live here in plain numpy, written for robustness on real NSE daily data (fat tails,
gap days, runs of zero returns) and for speed (GARCH fit on 5 years ≈ 0.1 s, HMM fit ≈ 0.1 s).

Conventions
- Inputs are 1-D float sequences ordered oldest → newest. "Per step" means per bar (per trading day for daily
  bars): volatilities are per-step σ (multiply by √252 to annualise), variances per-step σ².
- Every filter is causal: the value at index t uses inputs[: t + 1] only, so appending data never changes
  earlier outputs (the scans below are grouped by index, which keeps that true bit for bit).
- Fitted parameters use all the data they are given; fit on a training window when they feed a prediction.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

log = logging.getLogger("td.research.tsmath")

_LOG_2PI = math.log(2.0 * math.pi)


def _vec(x, name: str = "x") -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    return a


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-min(max(x, -50.0), 50.0)))


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def _linear_recurrence(b: np.ndarray, a: float) -> np.ndarray:
    """x[t] = a·x[t-1] + b[t] with x[-1] = 0 and constant a, by recursive doubling (⌈log2 n⌉ numpy passes).

    The grouping of terms at index t depends only on t, never on len(b), so outputs are bitwise identical
    when data is appended."""
    x = np.array(b, dtype=float)
    s = 1
    while s < x.size:
        x[s:] = x[s:] + (a ** s) * x[:-s]
        s *= 2
    return x


# --------------------------------------------------------------------------- returns & volatility
def log_returns(close) -> np.ndarray:
    """Per-step log returns ln(c[t] / c[t-1]); length len(close) − 1. Prices must be finite and > 0."""
    c = _vec(close, "close")
    if not (np.isfinite(c).all() and (c > 0).all()):
        raise ValueError("close prices must be finite and > 0")
    return np.diff(np.log(c))


def ewma_vol(r, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA volatility (zero-mean), per step.

    out[t] = sqrt(Σ_{k≤t} λ^{t-k} r[k]² / Σ_{k≤t} λ^{t-k}) — the σ estimate through r[t], i.e. the forecast for
    step t + 1. The normalising sum makes the warm-up unbiased (no seed value, so strictly causal). NaN returns
    carry zero weight; outputs before the first finite return are NaN."""
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1)")
    r = _vec(r, "r")
    ok = np.isfinite(r)
    num = _linear_recurrence(np.where(ok, (1.0 - lam) * r * r, 0.0), lam)
    den = _linear_recurrence((1.0 - lam) * ok, lam)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(num / den)


def yang_zhang_vol(o, h, l, c, window: int = 21) -> np.ndarray:
    """Yang–Zhang (2000) range-based volatility over a trailing window, per step (daily σ for daily bars).

    Combines overnight (open vs previous close), open-to-close and Rogers–Satchell variances; unbiased under
    drift and opening jumps and ~7× more efficient than close-to-close. out[t] uses bars t-window+1..t (plus the
    close of bar t-window for the first overnight gap); out[:window] is NaN. Highs/lows that fail to bracket
    open and close (bad prints) are widened to do so. Prices must be > 0."""
    o, h, l, c = (_vec(v, n) for v, n in ((o, "o"), (h, "h"), (l, "l"), (c, "c")))
    n = c.size
    if not (o.size == h.size == l.size == n):
        raise ValueError("o, h, l, c must have the same length")
    out = np.full(n, np.nan)
    if window < 2 or n <= window:
        return out
    h = np.maximum(h, np.maximum(o, c))
    l = np.minimum(l, np.minimum(o, c))
    with np.errstate(invalid="ignore", divide="ignore"):
        lo, lh, ll, lc = np.log(o), np.log(h), np.log(l), np.log(c)
    overnight = lo[1:] - lc[:-1]                          # element i belongs to bar i + 1
    oc = lc[1:] - lo[1:]
    u, d = lh[1:] - lo[1:], ll[1:] - lo[1:]
    rs = u * (u - oc) + d * (d - oc)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var = (sliding_window_view(overnight, window).var(axis=1, ddof=1)
           + k * sliding_window_view(oc, window).var(axis=1, ddof=1)
           + (1.0 - k) * sliding_window_view(rs, window).mean(axis=1))
    out[window:] = np.sqrt(np.maximum(var, 0.0))
    return out


# --------------------------------------------------------------------------- optimiser
def _safe_eval(f: Callable[[np.ndarray], float], x: np.ndarray) -> float:
    try:
        v = float(f(x))
    except (ArithmeticError, ValueError):
        return math.inf
    return v if math.isfinite(v) else math.inf


def nelder_mead(f: Callable[[np.ndarray], float], x0, step=None, max_iter: int = 2000,
                tol: float = 1e-9) -> tuple[np.ndarray, float, int, bool]:
    """Minimise f with the Nelder–Mead simplex (reflection 1, expansion 2, contraction ½, shrink ½).

    step: initial simplex edge per coordinate (scalar or array); default 5 % of |x0| (0.00025 where x0 = 0).
    Converged when the simplex's f spread ≤ tol·(1 + |f_best|) and its coordinate spread ≤ √tol·(1 + |x_best|∞).
    Non-finite values or arithmetic errors from f count as +inf, so f may reject infeasible points that way.
    Returns (x_best, f_best, iterations, converged)."""
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    if step is None:
        step = np.where(x0 != 0, 0.05 * np.abs(x0), 0.00025)
    step = np.broadcast_to(np.asarray(step, dtype=float), (n,))
    sim = np.tile(x0, (n + 1, 1))
    sim[1:] += np.diag(step)
    fs = np.array([_safe_eval(f, s) for s in sim])
    xtol = math.sqrt(tol)
    it, converged = 0, False
    while True:
        order = np.argsort(fs, kind="stable")
        sim, fs = sim[order], fs[order]
        if not math.isfinite(fs[0]):
            break
        if (fs[-1] - fs[0] <= tol * (1.0 + abs(fs[0]))
                and np.max(np.abs(sim[1:] - sim[0])) <= xtol * (1.0 + np.max(np.abs(sim[0])))):
            converged = True
            break
        if it >= max_iter:
            break
        it += 1
        centroid = sim[:-1].mean(axis=0)
        xr = 2.0 * centroid - sim[-1]
        fr = _safe_eval(f, xr)
        if fr < fs[0]:
            xe = 3.0 * centroid - 2.0 * sim[-1]
            fe = _safe_eval(f, xe)
            sim[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
            continue
        if fr < fs[-2]:
            sim[-1], fs[-1] = xr, fr
            continue
        if fr < fs[-1]:                                   # outside contraction
            xc = centroid + 0.5 * (xr - centroid)
            fc = _safe_eval(f, xc)
            if fc <= fr:
                sim[-1], fs[-1] = xc, fc
                continue
        else:                                             # inside contraction
            xc = centroid + 0.5 * (sim[-1] - centroid)
            fc = _safe_eval(f, xc)
            if fc < fs[-1]:
                sim[-1], fs[-1] = xc, fc
                continue
        sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0])       # shrink toward the best vertex
        fs[1:] = [_safe_eval(f, s) for s in sim[1:]]
    return sim[0].copy(), float(fs[0]), it, converged


# --------------------------------------------------------------------------- Student-t helpers
def t_unit_sample(rng: np.random.Generator, nu: float, size) -> np.ndarray:
    """Student-t draws scaled to unit variance (requires nu > 2); nu = inf (or > 1e6) gives standard normals."""
    if not nu > 2.0:
        raise ValueError("nu must be > 2 for a unit-variance Student-t")
    if nu > 1e6:
        return rng.standard_normal(size)
    return rng.standard_t(nu, size) * math.sqrt((nu - 2.0) / nu)


def _t_unit_const(nu: float) -> float:
    return math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * math.log(math.pi * (nu - 2.0))


def fit_t_df(z, lo: float = 2.1, hi: float = 100.0) -> float:
    """MLE degrees of freedom ν ∈ [lo, hi] of a unit-variance Student-t fitted to z.

    z is standardised internally (sample mean 0, variance 1), so pass residuals or returns alike. Golden-section
    search over log(ν − 2). A result at `hi` means tails no fatter than normal. NaN for < 10 finite values or
    zero variance."""
    z = _vec(z, "z")
    z = z[np.isfinite(z)]
    if z.size < 10 or not z.std() > 0:
        return math.nan
    z = (z - z.mean()) / z.std()
    z2 = z * z
    n = z.size

    def nll(u: float) -> float:
        nu = 2.0 + math.exp(u)
        return -(n * _t_unit_const(nu) - 0.5 * (nu + 1.0) * float(np.log1p(z2 / (nu - 2.0)).sum()))

    a, b = math.log(lo - 2.0), math.log(hi - 2.0)
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(c), nll(d)
    while b - a > 1e-5:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(d)
    return float(min(max(2.0 + math.exp((a + b) / 2.0), lo), hi))


# --------------------------------------------------------------------------- GARCH(1,1)
_P_MAX = 0.999                    # alpha + beta < 0.999: keeps the long-run variance finite
_NU_LO, _NU_HI = 2.1, 100.0
_GARCH_MIN_OBS = 100


@dataclass
class GarchFit:
    """GARCH(1,1) on per-step log returns:  r_t = mu + e_t,  e_t = sigma_t·z_t,
    sigma2_t = omega + alpha·e_{t-1}² + beta·sigma2_{t-1},  z_t ~ Student-t(nu) scaled to unit variance
    (nu = inf for dist="normal").

    Units: mu per step (log return); omega, long_run_var, last_var in (log return)² per step; last_resid in log
    return units. last_var / last_resid belong to the LAST OBSERVED return, so the next-step variance is
    omega + alpha·last_resid² + beta·last_var. converged=False marks a fallback or an optimiser that hit its
    iteration cap; `note` says which. loglik is in the original return units."""
    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float
    loglik: float
    converged: bool
    persistence: float
    long_run_var: float
    last_var: float
    last_resid: float
    n_obs: int = 0
    dist: str = "t"
    note: str = ""


def _garch_var(e2: np.ndarray, omega: float, alpha: float, beta: float, var0: float) -> np.ndarray:
    b = np.empty(e2.size)
    b[0] = var0
    b[1:] = omega + alpha * e2[:-1]
    return _linear_recurrence(b, beta)


def _garch_loglik(x: np.ndarray, mu: float, omega: float, alpha: float, beta: float,
                  nu: float) -> tuple[float, np.ndarray, np.ndarray]:
    """(loglik, e, sigma2); sigma2[0] starts at the model's long-run variance (data-free, hence causal)."""
    e = x - mu
    e2 = e * e
    p = alpha + beta
    s2 = _garch_var(e2, omega, alpha, beta, omega / (1.0 - p) if p < 1.0 else float(e2.mean()))
    if not s2.min() > 0:
        return -math.inf, e, s2
    n = x.size
    if math.isinf(nu):
        ll = -0.5 * (n * _LOG_2PI + float(np.log(s2).sum()) + float((e2 / s2).sum()))
    else:
        ll = (n * _t_unit_const(nu) - 0.5 * float(np.log(s2).sum())
              - 0.5 * (nu + 1.0) * float(np.log1p(e2 / (s2 * (nu - 2.0))).sum()))
    return ll, e, s2


def _garch_unpack(th: np.ndarray, normal: bool) -> tuple[float, float, float, float, float]:
    # Clipping makes the objective exactly flat past the bounds, so the simplex contracts instead of drifting.
    p = _P_MAX * _sigmoid(min(max(th[1], -12.0), 12.0))
    a = _sigmoid(min(max(th[2], -12.0), 12.0))
    lrv = math.exp(min(max(th[3], -12.0), 12.0))
    nu = math.inf if normal else _NU_LO + (_NU_HI - _NU_LO) * _sigmoid(min(max(th[4], -9.0), 9.0))
    alpha = p * a
    return float(th[0]), lrv * (1.0 - p), alpha, p - alpha, nu


def _garch_fallback(r: np.ndarray, dist: str, note: str) -> GarchFit:
    """Constant-variance model (alpha = beta = 0) — what the data supports when a GARCH fit is not possible."""
    n = r.size
    mu = float(r.mean()) if n else 0.0
    var = max(float(r.var()) if n > 1 else 0.0, 1e-12)
    if dist == "normal":
        nu = math.inf
    else:
        nu = fit_t_df((r - mu) / math.sqrt(var)) if n else math.nan
        nu = 30.0 if math.isnan(nu) else max(nu, 2.5)
    ll = _garch_loglik(r, mu, var, 0.0, 0.0, nu)[0] if n else math.nan
    log.debug("garch fallback (%s)", note)
    return GarchFit(mu=mu, omega=var, alpha=0.0, beta=0.0, nu=nu, loglik=ll, converged=False, persistence=0.0,
                    long_run_var=var, last_var=var, last_resid=float(r[-1] - mu) if n else 0.0, n_obs=n,
                    dist=dist, note=note)


def garch11_fit(r, dist: str = "t", max_iter: int = 3000, *, variance_targeting: bool = False) -> GarchFit:
    """Maximum-likelihood GARCH(1,1) with Student-t (default) or normal innovations.

    r: per-step log returns (non-finite values are dropped). The fit runs on returns standardised by their sample
    mean/sd (well-conditioned for the simplex) and maps back. Parameters are transformed so omega > 0,
    alpha, beta ≥ 0, alpha + beta < 0.999 and nu ∈ (2.1, 100); start values use variance targeting
    (long-run variance = sample variance) with a few (alpha, beta) candidates, then Nelder–Mead with restarts.
    variance_targeting=True keeps that constraint in the fit (omega = sample var·(1 − alpha − beta), Engle &
    Mezrich): one parameter fewer and a long-run variance that cannot drift away from what the window actually
    showed — useful when fat-tailed episodes (circuit-limit runs, crashes) pull the free MLE's long-run vol far
    above the sample's. Fewer than 100 returns, zero variance or a failed optimisation give the constant-variance
    fallback with converged=False and a note — never an exception."""
    if dist not in ("t", "normal"):
        raise ValueError("dist must be 't' or 'normal'")
    r = _vec(r, "r")
    finite = np.isfinite(r)
    dropped = int((~finite).sum())
    r = r[finite]
    n = r.size
    if n < _GARCH_MIN_OBS:
        return _garch_fallback(r, dist, f"only {n} returns (need {_GARCH_MIN_OBS}) — constant-variance fallback")
    m, s = float(r.mean()), float(r.std())
    if not s > 1e-10:
        return _garch_fallback(r, dist, "returns have ~zero variance — constant-variance fallback")
    x = (r - m) / s                                        # sample variance of x is exactly 1
    normal = dist == "normal"
    free = np.array([True, True, True, not variance_targeting, not normal])

    def full(th: np.ndarray) -> np.ndarray:                # fixed slots stay 0: long-run var = 1 (targeting)
        out = np.zeros(5)
        out[free] = th
        return out

    def nll(th: np.ndarray) -> float:
        return -_garch_loglik(x, *_garch_unpack(full(th), normal))[0]

    nu0 = 30.0 if normal else min(max(fit_t_df(x), 3.0), 50.0)
    t_nu = _logit((nu0 - _NU_LO) / (_NU_HI - _NU_LO))
    starts = []
    for alpha0, beta0 in ((0.05, 0.90), (0.08, 0.90), (0.10, 0.80), (0.03, 0.95), (0.15, 0.65)):
        p0 = alpha0 + beta0
        th = np.array([0.0, _logit(p0 / _P_MAX), _logit(alpha0 / p0), 0.0, t_nu])[free]
        starts.append((nll(th), th))
    best_f, best = min(starts, key=lambda s_: s_[0])
    step = np.array([0.05, 0.6, 0.6, 0.3, 0.6])[free]
    converged = False
    for _ in range(4):                                     # restart from the optimum until it stops moving
        th, f, _, converged = nelder_mead(nll, best, step=step, max_iter=max_iter, tol=1e-10)
        improved = best_f - f
        if f <= best_f:
            best, best_f = th, f
        if improved <= 1e-7 * (1.0 + abs(best_f)):
            break
        step = step * 0.5
    if not math.isfinite(best_f):
        return _garch_fallback(r, dist, "GARCH likelihood not finite — constant-variance fallback")

    mu_x, omega_x, alpha, beta, nu = _garch_unpack(full(best), normal)
    ll_x, e_x, s2_x = _garch_loglik(x, mu_x, omega_x, alpha, beta, nu)
    p = alpha + beta
    omega = omega_x * s * s
    note = "" if converged else "optimiser hit its iteration cap; best point returned"
    if dropped:
        note = (note + "; " if note else "") + f"{dropped} non-finite returns dropped"
    return GarchFit(mu=m + s * mu_x, omega=omega, alpha=alpha, beta=beta, nu=nu, loglik=ll_x - n * math.log(s),
                    converged=bool(converged), persistence=p, long_run_var=omega / (1.0 - p),
                    last_var=float(s2_x[-1]) * s * s, last_resid=float(e_x[-1]) * s, n_obs=n, dist=dist, note=note)


def garch11_filter(fit: GarchFit, r, var0: float | None = None) -> np.ndarray:
    """Conditional variances sigma2[t] of r[t] given r[:t] (per step, same length as r).

    Causal: sigma2[t] uses r[:t] only; the recursion starts at var0 (default: the fit's long-run variance, the
    same start the fit used, so filtering the training returns reproduces fit.last_var)."""
    r = _vec(r, "r")
    if r.size == 0:
        return np.empty(0)
    e = r - fit.mu
    v0 = fit.long_run_var if var0 is None else float(var0)
    return _garch_var(e * e, fit.omega, fit.alpha, fit.beta, v0)


def garch11_forecast(fit: GarchFit, h: int, last_var: float | None = None,
                     last_resid: float | None = None) -> np.ndarray:
    """Expected conditional variances for the next h steps (per step, length h).

    out[0] = omega + alpha·last_resid² + beta·last_var; out[k] = LRV + persistence^k·(out[0] − LRV), which decays
    to the long-run variance. last_var / last_resid default to the fit's (end of its sample). Sum the first k
    entries for the variance of a k-step log return."""
    lv = fit.last_var if last_var is None else float(last_var)
    le = fit.last_resid if last_resid is None else float(last_resid)
    v1 = fit.omega + fit.alpha * le * le + fit.beta * lv
    k = np.arange(max(int(h), 0), dtype=float)
    return fit.long_run_var + fit.persistence ** k * (v1 - fit.long_run_var)


def garch11_simulate(fit: GarchFit, h: int, n_paths: int, rng: np.random.Generator,
                     last_var: float | np.ndarray | None = None,
                     last_resid: float | np.ndarray | None = None, *, max_var: float | None = None) -> np.ndarray:
    """Simulate (n_paths, h) per-step log returns forward from the end of the sample.

    Each path starts from sigma2 = omega + alpha·last_resid² + beta·last_var (scalars or per-path arrays;
    default the fit's), draws unit-variance t (or normal) shocks and feeds its own shocks back into the variance,
    so paths show volatility clustering, fat tails and decay of vol toward the long-run level. Deterministic for
    a given rng state.
    max_var: optional ceiling on the per-step conditional variance. Fits with very fat tails (nu < 3) and a large
    alpha can feed one extreme shock back into an explosive variance burst (single-day log returns > 10 seen on
    real NSE fits); a ceiling such as 10 × the sample variance keeps paths physical without touching the shocks."""
    h, n_paths = int(h), int(n_paths)
    out = np.empty((n_paths, max(h, 0)))
    lv = fit.last_var if last_var is None else last_var
    le = fit.last_resid if last_resid is None else last_resid
    v1 = fit.omega + fit.alpha * np.square(np.asarray(le, dtype=float)) + fit.beta * np.asarray(lv, dtype=float)
    var = np.broadcast_to(v1, (n_paths,)).copy()
    for t in range(h):
        if max_var is not None:
            np.minimum(var, max_var, out=var)
        e = np.sqrt(var) * t_unit_sample(rng, fit.nu, n_paths)
        out[:, t] = fit.mu + e
        var = fit.omega + fit.alpha * e * e + fit.beta * var
    return out


# --------------------------------------------------------------------------- Hurst exponent
@lru_cache(maxsize=256)
def _scales(lo: int, hi: int, k: int = 12) -> tuple[int, ...]:
    return tuple(int(v) for v in np.unique(np.round(np.geomspace(lo, hi, k)).astype(int)))


def _slope(x: list[float], y: list[float]) -> float:
    xa, ya = np.asarray(x), np.asarray(y)
    xc = xa - xa.mean()
    return float(xc @ (ya - ya.mean()) / (xc @ xc))


@lru_cache(maxsize=512)
def _expected_rs(n: int) -> float:
    """Anis–Lloyd expected R/S of n iid normals with Peters' (n − ½)/n small-sample correction."""
    i = np.arange(1, n, dtype=float)
    s = float(np.sqrt((n - i) / i).sum())
    g = (math.exp(math.lgamma((n - 1) / 2.0) - math.lgamma(n / 2.0)) / math.sqrt(math.pi) if n <= 340
         else 1.0 / math.sqrt(n * math.pi / 2.0))
    return (n - 0.5) / n * g * s


def hurst_rs(x, min_window: int = 10, max_window: int | None = None) -> float:
    """Hurst exponent of an INCREMENT series (e.g. log returns) by rescaled-range analysis, Anis–Lloyd–Peters
    corrected: H = 0.5 + slope of log(R/S)_n − log E[(R/S)_n | iid] against log n over windows n ∈
    [min_window, max_window (default len/2)], averaging non-overlapping blocks.

    ≈ 0.5 for independent increments (the correction removes classic R/S's small-sample upward bias),
    > 0.5 persistent / trending, < 0.5 anti-persistent / mean-reverting. Typical range 0.3–0.7 for returns;
    sampling sd ≈ 0.03 on 1 000 points, ≈ 0.07 on 252. Strong persistence is pulled toward 0.5 (true 0.7 reads
    ≈ 0.64 on 4 000 points) — hurst_dfa is the less biased of the two. NaN when fewer than 3 window sizes fit."""
    x = _vec(x, "x")
    x = x[np.isfinite(x)]
    n = x.size
    hi = max_window or n // 2
    if hi < 2 * min_window:
        return math.nan
    lx, ly = [], []
    for w in _scales(int(min_window), int(hi)):
        k = n // w
        seg = x[: k * w].reshape(k, w)
        dev = np.cumsum(seg - seg.mean(axis=1, keepdims=True), axis=1)
        spread = dev.max(axis=1) - np.minimum(dev.min(axis=1), 0.0)
        sd = seg.std(axis=1)
        ok = sd > 0
        if ok.any():
            lx.append(math.log(w))
            ly.append(math.log(float((spread[ok] / sd[ok]).mean())) - math.log(_expected_rs(w)))
    if len(lx) < 3:
        return math.nan
    return 0.5 + _slope(lx, ly)


def hurst_dfa(x, min_window: int = 10, max_window: int | None = None) -> float:
    """Hurst exponent of an INCREMENT series by detrended fluctuation analysis (DFA-1): F(s) = RMS residual of
    the profile cumsum(x − mean) after a linear fit in each window of s points (blocks taken from both ends),
    H = slope of log F(s) on log s for s ∈ [min_window, max_window (default len/4)].

    ≈ 0.5 independent increments, > 0.5 persistent, < 0.5 anti-persistent (recovers fractional-Gaussian-noise H
    within ≈ 0.02); values near 1.5 mean x is itself a random walk (you passed prices, not returns). Robust to
    slow trends; sampling sd ≈ 0.035 on 1 000 points, ≈ 0.08 on a 252-point window. NaN when fewer than 3 window
    sizes fit."""
    x = _vec(x, "x")
    x = x[np.isfinite(x)]
    n = x.size
    hi = max_window or n // 4
    if hi < 2 * min_window:
        return math.nan
    y = np.cumsum(x - x.mean())
    lx, ly = [], []
    for s in _scales(int(min_window), int(hi)):
        k = n // s
        segs = np.concatenate([y[: k * s].reshape(k, s), y[n - k * s:].reshape(k, s)])
        t = np.arange(s, dtype=float) - (s - 1) / 2.0
        yc = segs - segs.mean(axis=1, keepdims=True)
        ss = (yc * yc).sum(axis=1) - (yc @ t) ** 2 / float(t @ t)
        f2 = float(np.maximum(ss, 0.0).mean()) / s
        if f2 > 0:
            lx.append(math.log(s))
            ly.append(0.5 * math.log(f2))
    if len(lx) < 3:
        return math.nan
    return _slope(lx, ly)


# --------------------------------------------------------------------------- Kalman local linear trend
def _llt_run(y: list[float], ql: float, qs: float, r: float, keep: bool):
    """Forward Kalman filter for y_t = level_t + eps (var r), level_{t+1} = level_t + slope_t + eta (var ql),
    slope_{t+1} = slope_t + zeta (var qs), with exact diffuse initialisation from y0, y1 (level = y1,
    slope = y1 − y0). Returns (sum log F, sum v²/F, count, level, slope, P) — the arrays only when keep."""
    n = len(y)
    L, S = y[1], y[1] - y[0]
    p11, p12, p22 = r, r, 2.0 * r + ql + qs
    if keep:
        lev, slo, P = np.empty(n), np.empty(n), np.empty((n, 3))
        lev[0], slo[0], P[0] = y[0], math.nan, (r, math.nan, math.nan)
        lev[1], slo[1], P[1] = L, S, (p11, p12, p22)
    slf = svf = 0.0
    for t in range(2, n):
        L += S                                           # predict
        p11 += 2.0 * p12 + p22 + ql
        p12 += p22
        p22 += qs
        v = y[t] - L                                     # update
        F = max(p11 + r, 1e-300)
        k1, k2 = p11 / F, p12 / F
        L += k1 * v
        S += k2 * v
        p22 = max(p22 - k2 * p12, 0.0)
        p12 *= r / F
        p11 *= r / F
        slf += math.log(F)
        svf += v * v / F
        if keep:
            lev[t], slo[t] = L, S
            P[t, 0], P[t, 1], P[t, 2] = p11, p12, p22
    if keep:
        return slf, svf, n - 2, lev, slo, P
    return slf, svf, n - 2, None, None, None


def kalman_llt_fit(y) -> dict:
    """MLE noise variances of the local-linear-trend model for y (e.g. log prices), in y-units² per step.

    Concentrated likelihood: the level variance is profiled out and Nelder–Mead searches log(q_slope/q_level)
    and log(r/q_level) in [e⁻²⁵, e¹⁰]. Uses ALL of y — estimate on a training window when the filtered output
    feeds a prediction. On daily stock prices the MLE q_slope is often ≈ 0 (a constant drift fits as well as a
    wandering trend), which makes the filtered slope a long-memory drift estimate.
    Returns {q_level, q_slope, r, loglik, converged}."""
    y = _vec(y, "y")
    if y.size < 10 or not np.isfinite(y).all():
        raise ValueError("kalman_llt_fit needs ≥ 10 finite observations")
    yl = y.tolist()
    m = y.size - 2

    def nll(th: np.ndarray) -> float:
        qs, rr = math.exp(min(max(th[0], -25.0), 10.0)), math.exp(min(max(th[1], -25.0), 10.0))
        slf, svf, cnt, *_ = _llt_run(yl, 1.0, qs, rr, False)
        s2 = svf / cnt
        if not s2 > 0:
            return math.inf
        return 0.5 * (cnt * (_LOG_2PI + 1.0 + math.log(s2)) + slf)

    starts = [np.array([a, b]) for a in (math.log(1e-2), math.log(1e-5)) for b in (math.log(1e-2), math.log(0.5))]
    _, th0 = min(((nll(s), s) for s in starts), key=lambda t: t[0])
    th, f, _, conv = nelder_mead(nll, th0, step=np.array([2.0, 2.0]), max_iter=400, tol=1e-9)
    qs_r, r_r = math.exp(min(max(th[0], -25.0), 10.0)), math.exp(min(max(th[1], -25.0), 10.0))
    svf = _llt_run(yl, 1.0, qs_r, r_r, False)[1]
    ql = svf / m
    return {"q_level": ql, "q_slope": ql * qs_r, "r": ql * r_r, "loglik": -f, "converged": bool(conv)}


def kalman_llt(y, q_level: float | None = None, q_slope: float | None = None, r: float | None = None, *,
               trend_days: float | None = None) -> dict:
    """Forward-only (causal) Kalman filter of a local linear trend: y_t = level_t + noise(r);
    level_{t+1} = level_t + slope_t + noise(q_level); slope_{t+1} = slope_t + noise(q_slope).

    y: e.g. log prices (then slope is the trend's log return per step). Noise variances are in y-units² per step.
    - trend_days=N fills the missing ones by rule: q_level = var(Δy), q_slope = q_level·a²/(1 − a) with
      a = 2/(N + 1), r = 0 — the slope then behaves like an EWMA of returns with span ≈ N steps.
    - Any still None is estimated by MLE on the whole of y (kalman_llt_fit).
    The states at t always use y[: t + 1] only, but parameters estimated here use all of y; for strictly causal
    features pass q_level (e.g. the variance of returns on a training window) together with trend_days or q_slope.
    Returns dict: level, slope, slope_se, level_se (length n; slope NaN at t = 0 where it is not yet identified),
    P (n, 2, 2) filtered covariances, and the q_level, q_slope, r used."""
    y = _vec(y, "y")
    if y.size < 3 or not np.isfinite(y).all():
        raise ValueError("kalman_llt needs ≥ 3 finite observations")
    if trend_days is not None:
        if not trend_days > 1:
            raise ValueError("trend_days must be > 1")
        if q_level is None:
            q_level = float(np.var(np.diff(y)))
        if q_slope is None:
            a = 2.0 / (trend_days + 1.0)
            q_slope = q_level * a * a / (1.0 - a)
        if r is None:
            r = 0.0
    if q_level is None or q_slope is None or r is None:
        est = kalman_llt_fit(y)
        q_level = est["q_level"] if q_level is None else q_level
        q_slope = est["q_slope"] if q_slope is None else q_slope
        r = est["r"] if r is None else r
    q_level, q_slope, r = float(q_level), float(q_slope), float(r)
    if min(q_level, q_slope, r) < 0 or q_level + r <= 0:
        raise ValueError("noise variances must be ≥ 0 with q_level + r > 0")
    *_, lev, slo, p3 = _llt_run(y.tolist(), q_level, q_slope, r, True)
    P = np.empty((y.size, 2, 2))
    P[:, 0, 0], P[:, 0, 1], P[:, 1, 0], P[:, 1, 1] = p3[:, 0], p3[:, 1], p3[:, 1], p3[:, 2]
    return {"level": lev, "slope": slo, "slope_se": np.sqrt(p3[:, 2]), "level_se": np.sqrt(p3[:, 0]), "P": P,
            "q_level": q_level, "q_slope": q_slope, "r": r}


# --------------------------------------------------------------------------- Ornstein–Uhlenbeck
def ou_fit(x, dt: float = 1.0) -> dict:
    """Ornstein–Uhlenbeck dx = kappa(mu − x)dt + sigma dW fitted through the exact AR(1) mapping
    x[t+1] = a + b·x[t] + e:  b = exp(−kappa·dt), mu = a/(1 − b), sigma = sd(e)·√(2·kappa/(1 − b²)).

    Units: kappa per step (dt = 1), half_life = ln 2 / kappa in steps, sigma per √step, sigma_eq = sd(e)/√(1 − b²)
    the stationary sd (use (x − mu)/sigma_eq as the z-score). b ≥ 1 (no mean reversion): kappa 0, half_life inf,
    mu/sigma_eq NaN, mean_reverting False. b ≤ 0 (overshooting) is clipped to b = 1e-6 for kappa/half-life.
    Needs ≥ 10 finite points, else every value is NaN."""
    x = _vec(x, "x")
    x = x[np.isfinite(x)]
    nan = math.nan
    if x.size < 10:
        return {"kappa": nan, "mu": nan, "sigma": nan, "half_life": nan, "b": nan, "sigma_eq": nan,
                "mean_reverting": False}
    x0, x1 = x[:-1], x[1:]
    x0c = x0 - x0.mean()
    sxx = float(x0c @ x0c)
    b = float(x0c @ (x1 - x1.mean())) / sxx if sxx > 0 else 1.0
    a = float(x1.mean() - b * x0.mean())
    resid = x1 - (a + b * x0)
    sd = math.sqrt(float(resid @ resid) / max(x0.size - 2, 1))
    if b >= 1.0:
        return {"kappa": 0.0, "mu": nan, "sigma": sd / math.sqrt(dt), "half_life": math.inf, "b": b,
                "sigma_eq": nan, "mean_reverting": False}
    bc = max(b, 1e-6)
    kappa = -math.log(bc) / dt
    return {"kappa": kappa, "mu": a / (1.0 - b), "sigma": sd * math.sqrt(2.0 * kappa / (1.0 - bc * bc)),
            "half_life": math.log(2.0) / kappa, "b": b, "sigma_eq": sd / math.sqrt(1.0 - b * b),
            "mean_reverting": True}


# --------------------------------------------------------------------------- Gaussian HMM
def _scan_products(M: np.ndarray, reverse: bool) -> tuple[np.ndarray, np.ndarray]:
    """Running products of T matrices stored as (K, K, T), by recursive doubling: prefix X[..., t] = M_0…M_t,
    or suffix X[..., t] = M_t…M_{T-1}. Each product is rescaled to sum 1; the log scales come back separately.
    The (K, K, T) layout keeps every matrix entry a contiguous vector, so a product is K³ vector multiply-adds."""
    K, _, T = M.shape
    nrm = M.reshape(K * K, T).sum(axis=0)
    X = M / nrm
    logn = np.log(nrm)
    s = 1
    while s < T:
        L, R = X[:, :, :-s], X[:, :, s:]                  # earlier @ later
        Y = np.empty((K, K, T - s))
        for i in range(K):
            for k in range(K):
                acc = L[i, 0] * R[0, k]
                for j in range(1, K):
                    acc += L[i, j] * R[j, k]
                Y[i, k] = acc
        nrm = Y.reshape(K * K, T - s).sum(axis=0)
        Y /= nrm
        if reverse:
            X[:, :, :-s] = Y
            logn[:-s] = logn[:-s] + logn[s:] + np.log(nrm)
        else:
            X[:, :, s:] = Y
            logn[s:] = logn[:-s] + logn[s:] + np.log(nrm)
        s *= 2
    return X, logn


def _stationary(A: np.ndarray) -> np.ndarray:
    K = A.shape[0]
    lhs = np.vstack([A.T - np.eye(K), np.ones(K)])
    rhs = np.zeros(K + 1)
    rhs[-1] = 1.0
    pi = np.clip(np.linalg.lstsq(lhs, rhs, rcond=None)[0], 1e-12, None)
    return pi / pi.sum()


class GaussianHMM:
    """Gaussian hidden Markov model for per-step returns with 2 or 3 regimes.

    fit(r) runs Baum–Welch (EM) from several deterministic starts (value quantiles, volatility quantiles and
    seeded random ones), keeps the best likelihood and iterates it to convergence. Forward–backward runs as
    scaled matrix-product scans in numpy (no per-step Python loop), so a 5-year fit takes ~0.1 s.
    After fitting, states are sorted by mean: state 0 = lowest mean ("bear"), state K−1 = highest ("bull").
    The chain is assumed to start in its stationary distribution (startprob = stationary(trans)), so filter() on
    any window starts from the unconditional regime mix rather than the training sample's first day.
    Attributes (original return units, per step): means (K,), vars (K,), trans (K, K) with rows summing to 1,
    startprob (K,), loglik, converged, n_iter.
    filter(r) → (T, K) P(state_t | r[: t+1]) — forward only, causal. smooth(r) → P(state_t | all r).
    viterbi(r) → most likely state sequence. simulate(...) → Markov-switching paths."""

    def __init__(self, n_states: int = 2):
        if n_states not in (2, 3):
            raise ValueError("n_states must be 2 or 3")
        self.n_states = n_states
        self.means: np.ndarray | None = None
        self.vars: np.ndarray | None = None
        self.trans: np.ndarray | None = None
        self.startprob: np.ndarray | None = None
        self.loglik = math.nan
        self.converged = False
        self.n_iter = 0

    # -- forward–backward in (K, T) layout; works in whatever units the parameters are in ------------------
    @staticmethod
    def _forward(x, means, vars_, trans, start):
        """→ (filtered probs (K, T), log-likelihood, stack M[i, j, t-1] = A[i, j]·b_t(j) (scaled) or None)."""
        logb = -0.5 * (_LOG_2PI + np.log(vars_)[:, None] + (x[None, :] - means[:, None]) ** 2 / vars_[:, None])
        c = logb.max(axis=0)
        B = np.exp(logb - c)
        K, T = B.shape
        alpha = np.empty((K, T))
        alpha[:, 0] = start * B[:, 0]
        M = None
        if T > 1:
            M = trans[:, :, None] * B[None, :, 1:]       # M[i, j, t-1] = A[i, j]·b_t(j)
            X, logn = _scan_products(M, reverse=False)
            a0 = alpha[:, 0]
            alpha[:, 1:] = a0[0] * X[0]                    # explicit sum (not BLAS) keeps rows bitwise causal
            for i in range(1, K):
                alpha[:, 1:] += a0[i] * X[i]
            ll = math.log(float(alpha[:, -1].sum())) + float(logn[-1]) + float(c.sum())
        else:
            ll = math.log(float(alpha[:, 0].sum())) + float(c[0])
        alpha /= alpha.sum(axis=0)
        return alpha, ll, M

    @classmethod
    def _posterior(cls, x, means, vars_, trans, start):
        """→ (smoothed probs (K, T), expected transition counts (K, K) or None, log-likelihood)."""
        alpha, ll, M = cls._forward(x, means, vars_, trans, start)
        K, T = alpha.shape
        beta = np.ones((K, T))
        counts = None
        if T > 1:
            S, _ = _scan_products(M, reverse=True)
            beta[:, :-1] = S.sum(axis=1)
            beta[:, :-1] /= beta[:, :-1].sum(axis=0)
            xi = alpha[:, None, :-1] * M * beta[None, :, 1:]
            xi /= xi.reshape(K * K, T - 1).sum(axis=0)
            counts = xi.sum(axis=2)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=0)
        return gamma, counts, ll

    @classmethod
    def _em(cls, z, means, vars_, trans, n_iter, tol, var_floor):
        start = _stationary(trans)
        ll_prev, it, conv = -math.inf, 0, False
        for it in range(1, n_iter + 1):
            gamma, counts, ll = cls._posterior(z, means, vars_, trans, start)
            if abs(ll - ll_prev) <= tol * (1.0 + abs(ll)):
                conv = True
                break
            ll_prev = ll
            w = gamma.sum(axis=1)
            ok = w > 1e-8
            means = np.where(ok, (gamma @ z) / np.where(ok, w, 1.0), means)
            dev2 = (z[None, :] - means[:, None]) ** 2
            vars_ = np.maximum(np.where(ok, (gamma * dev2).sum(axis=1) / np.where(ok, w, 1.0), vars_), var_floor)
            if counts is not None:
                A = counts + 1e-10
                trans = A / A.sum(axis=1, keepdims=True)
            start = _stationary(trans)
        else:
            ll = cls._forward(z, means, vars_, trans, start)[1]
        return means, vars_, trans, ll, it, conv

    def fit(self, r, n_iter: int = 100, seed: int = 0, n_init: int = 6, tol: float = 1e-6, *,
            clip_z: float | None = None) -> GaussianHMM:
        """Fit to per-step returns r (finite, ≥ 30·K values, non-zero variance; ValueError otherwise).
        n_init starts each get a few EM steps, the best continues to convergence (|Δ loglik| ≤ tol·|loglik|)
        or n_iter total. Deterministic for a given seed. Returns self.
        clip_z: winsorise the training returns at median ± clip_z·(1.4826·MAD) first. A Gaussian HMM otherwise
        spends a state on a handful of outliers (corporate-action seams, crash days) instead of a regime;
        6 is a good value for raw NSE daily returns. Estimation only — filter() etc. see the data as given."""
        K = self.n_states
        r = _vec(r, "r")
        if not np.isfinite(r).all():
            raise ValueError("returns must be finite")
        if r.size < 30 * K:
            raise ValueError(f"need at least {30 * K} returns for a {K}-state HMM")
        if clip_z is not None:
            med = float(np.median(r))
            mad = 1.4826 * float(np.median(np.abs(r - med)))
            if mad > 0:
                r = np.clip(r, med - clip_z * mad, med + clip_z * mad)
        m, s = float(r.mean()), float(r.std())
        if not s > 0:
            raise ValueError("returns have zero variance")
        z = (r - m) / s
        rng = np.random.default_rng(seed)
        var_floor = 1e-2                                   # sd ≥ 10 % of the sample sd: no collapsed states
        q = np.linspace(0, 1, K + 1)
        sticky = np.full((K, K), 0.05 / (K - 1)) + np.eye(K) * (0.95 - 0.05 / (K - 1))
        inits = []
        for key in (z, np.abs(z)):                         # split by value, then by volatility
            edges = np.quantile(key, q)
            groups = [z[(key >= edges[k]) & (key <= edges[k + 1])] for k in range(K)]
            inits.append((np.array([g.mean() for g in groups]),
                          np.maximum(np.array([g.var() for g in groups]), var_floor), sticky))
        for _ in range(max(n_init - 2, 0)):
            d = rng.uniform(0.80, 0.99, K)
            inits.append((rng.normal(0.0, 0.5, K), rng.uniform(0.3, 2.0, K),
                          (1.0 - d)[:, None] / (K - 1) * (1.0 - np.eye(K)) + np.diag(d)))
        short = min(8, n_iter)
        cands = [self._em(z, mu0, v0, A0, short, tol, var_floor) for mu0, v0, A0 in inits]
        means, vars_, trans, ll, it, conv = max(cands, key=lambda c_: c_[3] if math.isfinite(c_[3]) else -math.inf)
        if not conv and n_iter > short:
            means, vars_, trans, ll, it2, conv = self._em(z, means, vars_, trans, n_iter - short, tol, var_floor)
            it += it2
        order = np.argsort(means, kind="stable")
        self.means = m + s * means[order]
        self.vars = s * s * vars_[order]
        self.trans = trans[np.ix_(order, order)]
        self.startprob = _stationary(self.trans)
        self.loglik = ll - r.size * math.log(s)
        self.converged = bool(conv)
        self.n_iter = int(it)
        return self

    def _check(self, r) -> np.ndarray:
        if self.means is None:
            raise RuntimeError("GaussianHMM is not fitted")
        r = _vec(r, "r")
        if r.size == 0 or not np.isfinite(r).all():
            raise ValueError("returns must be finite and non-empty")
        return r

    def filter(self, r) -> np.ndarray:
        """(T, K) filtered probabilities P(state_t | r[: t+1]); causal — row t never changes when data is appended."""
        r = self._check(r)
        return np.ascontiguousarray(self._forward(r, self.means, self.vars, self.trans, self.startprob)[0].T)

    def smooth(self, r) -> np.ndarray:
        """(T, K) smoothed probabilities P(state_t | all of r) — uses later data, for display only."""
        r = self._check(r)
        return np.ascontiguousarray(self._posterior(r, self.means, self.vars, self.trans, self.startprob)[0].T)

    def score(self, r) -> float:
        """Log-likelihood of r under the fitted model."""
        r = self._check(r)
        return self._forward(r, self.means, self.vars, self.trans, self.startprob)[1]

    def viterbi(self, r) -> np.ndarray:
        """Most likely state sequence (T,) of ints (uses the whole of r)."""
        r = self._check(r)
        logb = -0.5 * (_LOG_2PI + np.log(self.vars) + (r[:, None] - self.means) ** 2 / self.vars)
        logA = np.log(self.trans)
        T, K = logb.shape
        psi = np.zeros((T, K), dtype=np.intp)
        delta = np.log(self.startprob) + logb[0]
        cols = np.arange(K)
        for t in range(1, T):
            cand = delta[:, None] + logA
            psi[t] = cand.argmax(axis=0)
            delta = cand[psi[t], cols] + logb[t]
        path = np.empty(T, dtype=np.intp)
        path[-1] = int(delta.argmax())
        for t in range(T - 1, 0, -1):
            path[t - 1] = psi[t, path[t]]
        return path

    def simulate(self, h: int, n_paths: int, rng: np.random.Generator,
                 start_probs=None) -> tuple[np.ndarray, np.ndarray]:
        """Markov-switching simulation → (returns (n_paths, h), states (n_paths, h) int8).

        start_probs = probabilities of the CURRENT (last observed) state, e.g. filter(r)[-1]; default startprob.
        Each step first transitions from the previous state, then draws a normal return from the new state."""
        if self.means is None:
            raise RuntimeError("GaussianHMM is not fitted")
        h, n_paths = int(h), int(n_paths)
        p0 = self.startprob if start_probs is None else np.clip(np.asarray(start_probs, dtype=float), 0.0, None)
        p0 = p0 / p0.sum()
        cum = np.cumsum(self.trans, axis=1)
        cum[:, -1] = 1.0
        state = np.minimum(np.searchsorted(np.cumsum(p0), rng.random(n_paths), side="right"), self.n_states - 1)
        sd = np.sqrt(self.vars)
        rets = np.empty((n_paths, h))
        states = np.empty((n_paths, h), dtype=np.int8)
        for t in range(h):
            state = (rng.random(n_paths)[:, None] >= cum[state]).sum(axis=1)
            rets[:, t] = self.means[state] + sd[state] * rng.standard_normal(n_paths)
            states[:, t] = state
        return rets, states


# --------------------------------------------------------------------------- bootstrap
_BOOT_CHUNK = 1 << 20                                      # elements per chunk: bounds memory, fixed for determinism


def stationary_bootstrap_indices(n: int, length: int, n_paths: int, mean_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """Politis–Romano (1994) stationary bootstrap: (n_paths, length) indices into a series of n observations.

    Blocks start at uniform random positions and have geometric lengths with mean `mean_block` (p = 1/mean_block
    of starting a new block each step); blocks wrap around circularly. Keeps fat tails, volatility clustering and
    short-range dependence that an iid bootstrap would destroy, and the resampled series stays stationary.
    mean_block ≤ 1 gives the iid bootstrap. Deterministic for a given rng state."""
    n, length, n_paths = int(n), int(length), int(n_paths)
    if n < 1:
        raise ValueError("n must be ≥ 1")
    out = np.empty((max(n_paths, 0), max(length, 0)), dtype=np.intp)
    if length < 1 or n_paths < 1:
        return out
    p = np.float32(1.0 / max(float(mean_block), 1.0))
    small = n + length < 2 ** 31                           # int32 work arrays: ~3× faster on big draws
    dt = np.int32 if small else np.int64
    steps = np.arange(length, dtype=dt)
    rows = max(1, _BOOT_CHUNK // length)
    for r0 in range(0, n_paths, rows):
        m = min(rows, n_paths - r0)
        new = rng.random((m, length), dtype=np.float32) < p
        new[:, 0] = True
        starts = rng.integers(0, n, size=(m, length), dtype=dt)
        pos = np.where(new, steps, dt(0))                  # index of the current block's first step
        np.maximum.accumulate(pos, axis=1, out=pos)
        idx = np.take_along_axis(starts, pos, axis=1)
        idx += steps - pos
        idx %= dt(n)
        out[r0:r0 + m] = idx
    return out


# --------------------------------------------------------------------------- normal distribution
def _ret(out: np.ndarray):
    return float(out) if out.ndim == 0 else out


_erfc = np.frompyfunc(math.erfc, 1, 1)


def norm_cdf(x):
    """Standard normal CDF Φ(x) = ½·erfc(−x/√2), vectorised over math.erfc (full double precision, including
    the far tails; ~0.25 µs per element). Scalar in → float out; array in → array out."""
    x = np.asarray(x, dtype=float)
    return _ret(np.asarray(0.5 * _erfc(-x / math.sqrt(2.0)), dtype=float))


_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
      -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
      -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
      4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)


def norm_ppf(p):
    """Standard normal quantile, vectorised: Acklam's rational approximation (rel err 1.15e-9) refined by one
    Halley step on norm_cdf, giving ~machine precision. ppf(0) = −inf, ppf(1) = inf, outside [0, 1] → NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = (p > 0) & (p < 0.02425)
        hi = (p > 1 - 0.02425) & (p < 1)
        mid = (p >= 0.02425) & (p <= 1 - 0.02425)
        q = np.sqrt(-2.0 * np.log(np.where(lo, p, np.where(hi, 1.0 - p, 0.5))))
        tail = ((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
                / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0))
        qm = p - 0.5
        rr = qm * qm
        cen = (((((((_A[0] * rr + _A[1]) * rr + _A[2]) * rr + _A[3]) * rr + _A[4]) * rr + _A[5]) * qm)
               / (((((_B[0] * rr + _B[1]) * rr + _B[2]) * rr + _B[3]) * rr + _B[4]) * rr + 1.0))
        x = np.where(lo, tail, np.where(hi, -tail, cen))
        ok = lo | hi | mid
        # Halley step; for the upper tail refine −x against 1 − p to keep precision
        upper = p > 0.5
        xs = np.where(upper, -x, x)
        ps = np.where(upper, 1.0 - p, p)
        err = np.asarray(norm_cdf(xs)) - ps
        u = err * math.sqrt(2.0 * math.pi) * np.exp(0.5 * xs * xs)
        xs = xs - u / (1.0 + 0.5 * xs * u)
        x = np.where(upper, -xs, xs)
    out[ok] = x[ok]
    out[p == 0] = -np.inf
    out[p == 1] = np.inf
    return _ret(out)


# --------------------------------------------------------------------------- small statistics
def max_drawdown(paths, axis: int = 1) -> np.ndarray | float:
    """Largest peak-to-trough fall along `axis` as a positive fraction (0.25 = −25 %) of price LEVELS (not
    returns), e.g. paths of shape (n_paths, steps) → (n_paths,); a single 1-D path → float. Include the starting
    price in the path if the drawdown should count from it."""
    p = np.asarray(paths, dtype=float)
    if p.ndim == 1:
        axis = 0
    peak = np.maximum.accumulate(p, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _ret(np.asarray((1.0 - p / peak).max(axis=axis)))


def rolling_beta(r, rm, window: int) -> np.ndarray:
    """Trailing OLS beta cov(r, rm)/var(rm) over `window` aligned returns; out[t] uses [t-window+1, t]
    (causal), NaN before that, where rm has zero variance, or where the window holds a NaN."""
    r, rm = _vec(r, "r"), _vec(rm, "rm")
    if r.size != rm.size:
        raise ValueError("r and rm must have the same length")
    out = np.full(r.size, np.nan)
    if window < 2 or r.size < window:
        return out
    X = sliding_window_view(rm, window)
    Y = sliding_window_view(r, window)
    xc = X - X.mean(axis=1, keepdims=True)
    cov = (xc * (Y - Y.mean(axis=1, keepdims=True))).sum(axis=1)
    var = (xc * xc).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[window - 1:] = np.where(var > 0, cov / var, np.nan)
    return out


def rsi(close, n: int = 14) -> np.ndarray:
    """Wilder's RSI (0–100) with Wilder smoothing (alpha = 1/n) seeded by the simple average of the first n
    changes; out[:n] is NaN, out[t] uses close[: t+1]. No losses in the window → 100 (50 if also no gains)."""
    c = _vec(close, "close")
    out = np.full(c.size, np.nan)
    if n < 1 or c.size <= n:
        return out
    d = np.diff(c)
    gain, loss = np.maximum(d, 0.0), np.maximum(-d, 0.0)
    a = (n - 1) / n

    def wilder(x: np.ndarray) -> np.ndarray:
        b = np.empty(c.size - n)
        b[0] = x[:n].mean()
        b[1:] = x[n:] / n
        return _linear_recurrence(b, a)

    ag, al = wilder(gain), wilder(loss)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = 100.0 - 100.0 / (1.0 + ag / al)
    out[n:] = np.where(al > 0, val, np.where(ag > 0, 100.0, 50.0))
    return out


def autocorr(x, lag: int = 1) -> float:
    """Lag-k autocorrelation as the Pearson correlation of x[:-lag] and x[lag:] (same as pandas
    Series.autocorr); NaN if fewer than lag + 3 finite points or a constant series."""
    x = _vec(x, "x")
    x = x[np.isfinite(x)]
    if lag < 1 or x.size < lag + 3:
        return math.nan
    a, b = x[:-lag], x[lag:]
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / den if den > 0 else math.nan
