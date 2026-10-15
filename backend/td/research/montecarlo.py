"""Monte Carlo price simulation for one stock, and the calibration backtest that says how far to trust it.

No model forecasts a price; what a simulation can do honestly is describe the RANGE of outcomes implied by how
the stock has moved, and then measure — out of sample — whether ranges built that way held the real price as
often as they claimed. So every result here is a distribution (fan, terminal histogram, risk, touch odds), and
`backtest` replays the same recipe at past dates on data available then, scoring coverage, PIT and CRPS against
a naive random walk.

Models share one structure. Each produces per-step "martingale" log returns M = ε − κ, where ε is the model's
shock and κ = ln E[e^ε] its convexity (Itô) correction, so E[e^M] = 1 step by step:
- gbm: ε ~ N(0, σ²) with σ the lookback's daily vol; κ = σ²/2 (the textbook Itô term).
- garch: GARCH(1,1)-t fitted with variance targeting (long-run variance = the window's, so one crash cannot set
  the 3-year width), simulated from today's conditional variance; κ = σ²_t/2 per path and step (for unit-variance
  t shocks the higher cumulants add < 0.01 %/yr at daily vols).
- bootstrap: stationary block bootstrap (mean block 10 days) of the de-meaned window; κ = ln mean(e^ε) over the
  window, exact for the resampled marginal.
- jump: Merton jump-diffusion; jumps are days with |r − median| > 3·MAD-σ; κ = σ_d²/2 + λ(e^{μ_j + σ_j²/2} − 1),
  the compound-Poisson compensator.
- regime: 2-state Gaussian HMM, Markov-switching from today's filtered state probabilities; state variances are
  scaled so the stationary variance equals the window's (the fit winsorises outliers); κ = σ²_state/2.
- ensemble: equal mixture of garch, bootstrap, jump and regime (paths split between them).
The drift is then added identically to every model: log price = Σ (g_s + M_s) with g_s = ln(1 + D_s)/252, so
E[S_t] = S_0·Π e^{g_s} — the expected arithmetic annual return is exactly the chosen D (CAPM, shrunk history,
zero, or the quant model's tilt fading into CAPM). Historical means are far too noisy to drive a forecast (5
years of 30 % vol pin the mean only to ±13 %/yr), which is why CAPM is the default and state/regime means are
shown for context but never simulated. Models differ only in the SHAPE of risk: vol clustering, fat tails,
jumps, regimes.

Daily shocks are clipped at ±max(20 %, 1.5 × the window's largest move) (at most 50 %) in log terms: Student-t
shocks with ν < 3 otherwise produce rare one-day moves of several hundred percent, which no NSE stock can make.
Isolated spike-and-revert bars (≥ 20 % in and out, cancelling) are treated as bad prints and dropped, with a
warning. Pure functions, CPU-bound: the router runs them in a worker thread.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import settings
from ..timeutil import from_epoch
from . import tsmath as tm

log = logging.getLogger("td.research.montecarlo")

MODELS = ("ensemble", "garch", "bootstrap", "jump", "regime", "gbm")
DRIFTS = ("capm", "historical", "zero", "quant")
ENSEMBLE = ("garch", "bootstrap", "jump", "regime")

TRADING_DAYS = 252
YEAR_S = 365.25 * 86400
IST_OFFSET_S = 19800
MIN_BARS = 250                    # about one year of NSE sessions
MAX_CELLS = 20_000_000            # paths × horizon
CHUNK_CELLS = 2_000_000           # paths are generated in blocks of this many cells (bounds temporaries)
H_MIN, H_MAX = 5, 756
MIN_PATHS = 100
CHECKPOINTS = ((21, "1M"), (63, "3M"), (126, "6M"), (252, "1Y"), (504, "2Y"), (756, "3Y"))
FAN_Q = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
FAN_KEYS = ("p5", "p10", "p25", "p50", "p75", "p90", "p95")
MAX_FAN = 200
N_SAMPLE = 30
N_BINS = 60
BOOT_BLOCK = 10.0
JUMP_K = 3.0
GARCH_VAR_CAP = 16.0              # conditional variance ≤ 16 × the window's (vol ≤ 4 ×)
TAU_MAX = 0.08                    # prior spread of true expected returns around CAPM (annual, continuous)
BETA_WEEKS, MIN_WEEKS = 104, 26
QUANT_FULL, QUANT_FADE = 21, 63
COVER_LEVELS = (0.5, 0.8, 0.9)
YEAR_TOL = 0.05                   # a "5-year" lake that starts 2 weeks short still counts as 5 years


# ------------------------------------------------------------------ small helpers
def _num(x, nd: int = 4) -> float | None:
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _px(x) -> float | None:
    return _num(x, 2)


def _sig(x, digits: int = 6) -> float | None:
    x = float(x)
    return float(f"{x:.{digits}g}") if math.isfinite(x) else None


def _date(ts: int) -> str:
    return from_epoch(int(ts)).date().isoformat()


def _future_ts(last_ts: int, h: int) -> np.ndarray:
    """[last bar ts, then the next h weekdays] as IST-midnight epochs. NSE holidays are ignored: a simulated
    'trading day' is a weekday, so dates far out drift a few days early relative to the real calendar."""
    day = (int(last_ts) + IST_OFFSET_S) // 86400
    days = np.busday_offset(np.datetime64(day, "D"), np.arange(1, h + 1), roll="backward")
    out = np.empty(h + 1, dtype=np.int64)
    out[0] = int(last_ts)
    out[1:] = days.astype(np.int64) * 86400 - IST_OFFSET_S
    return out


def _cap_paths(paths: int, h: int) -> tuple[int, str | None]:
    if paths * h <= MAX_CELLS:
        return paths, None
    cut = MAX_CELLS // h
    return cut, (f"paths reduced from {paths:,} to {cut:,} (paths × horizon is capped at {MAX_CELLS:,})")


def _check_args(horizon_days, paths, model: str, drift: str, lookback_years) -> tuple[int, int, float]:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r} (use one of {', '.join(MODELS)})")
    if drift not in DRIFTS:
        raise ValueError(f"unknown drift {drift!r} (use one of {', '.join(DRIFTS)})")
    h, n = int(horizon_days), int(paths)
    if not H_MIN <= h <= H_MAX:
        raise ValueError(f"horizon must be {H_MIN}…{H_MAX} trading days, got {h}")
    if n < MIN_PATHS:
        raise ValueError(f"need at least {MIN_PATHS} paths, got {n}")
    lb = float(lookback_years)
    if not lb > 0:
        raise ValueError("lookback_years must be > 0")
    return h, n, lb


def _seed(seed) -> int:
    return int(seed or 0) % (1 << 64)


# ------------------------------------------------------------------ price series
@dataclass
class _Series:
    ts: np.ndarray
    close: np.ndarray
    removed: int = 0


def _bad_prints(close: np.ndarray) -> np.ndarray:
    """Bar j is a bad print when the move into it and the move out of it are both ≥ 20 % (log), opposite, and
    cancel to within a quarter of the smaller: an isolated spike that fully reverts next session is the
    signature of a mis-scaled print, not of a market move. Flagging bar j needs only bar j + 1, so the flags a
    backtest origin's fit relies on never need data after the origin (a spiked bar is simply never an origin)."""
    out = np.zeros(close.size, dtype=bool)
    if close.size < 3:
        return out
    r = np.diff(np.log(close))
    a, b = r[:-1], r[1:]
    out[1:-1] = ((np.abs(a) >= 0.2) & (np.abs(b) >= 0.2) & (np.sign(a) != np.sign(b))
                 & (np.abs(a + b) <= 0.25 * np.minimum(np.abs(a), np.abs(b))))
    return out


def _series(df: pd.DataFrame | None) -> _Series:
    if df is None or len(df) == 0:
        return _Series(np.empty(0, dtype=np.int64), np.empty(0))
    if not {"ts", "close"} <= set(df.columns):
        raise ValueError("daily bars need 'ts' and 'close' columns")
    ts = df["ts"].to_numpy(dtype=np.int64)
    c = df["close"].to_numpy(dtype=float)
    ok = np.isfinite(c) & (c > 0)
    ts, c = ts[ok], c[ok]
    order = np.argsort(ts, kind="stable")
    ts, c = ts[order], c[order]
    last = np.ones(ts.size, dtype=bool)
    last[:-1] = ts[1:] != ts[:-1]                 # duplicates: keep the last row
    ts, c = ts[last], c[last]
    bad = _bad_prints(c)
    return _Series(ts[~bad], c[~bad], int(bad.sum()))


def _window(s: _Series, end: int, lookback_years: float) -> tuple[int, float]:
    """Start index of the calibration window ending at bar `end`: the last `lookback_years` of calendar time,
    widened to MIN_BARS bars when a short calendar year holds fewer sessions. ValueError below MIN_BARS."""
    bars_avail = end + 1
    if bars_avail < MIN_BARS:
        raise ValueError(f"only {bars_avail} daily bars of history — the simulation needs at least {MIN_BARS} "
                         "(about one year)")
    start = int(np.searchsorted(s.ts, s.ts[end] - lookback_years * YEAR_S, side="left"))
    start = min(start, end + 1 - MIN_BARS)
    return start, float(s.ts[end] - s.ts[start]) / YEAR_S


# ------------------------------------------------------------------ beta & drift
def _weekly(ts: np.ndarray, close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Monday-based week number, last close of that week); 1970-01-01 was a Thursday, hence the +3."""
    wk = ((ts + IST_OFFSET_S) // 86400 + 3) // 7
    last = np.ones(wk.size, dtype=bool)
    last[:-1] = wk[1:] != wk[:-1]
    return wk[last], close[last]


def _beta(s: _Series, end: int, idx: _Series | None) -> tuple[float | None, int]:
    """OLS beta of weekly log returns vs the index over the 2 years up to bar `end` (data ≤ that bar only)."""
    if idx is None or idx.ts.size == 0:
        return None, 0
    end_ts = int(s.ts[end])
    lo_ts = end_ts - 2.2 * YEAR_S
    a0 = int(np.searchsorted(s.ts, lo_ts))
    b0, b1 = int(np.searchsorted(idx.ts, lo_ts)), int(np.searchsorted(idx.ts, end_ts, side="right"))
    ws, cs = _weekly(s.ts[a0:end + 1], s.close[a0:end + 1])
    wi, ci = _weekly(idx.ts[b0:b1], idx.close[b0:b1])
    common, js, ji = np.intersect1d(ws, wi, assume_unique=True, return_indices=True)
    keep = common >= ws[-1] - BETA_WEEKS
    rs, ri = np.diff(np.log(cs[js][keep])), np.diff(np.log(ci[ji][keep]))
    if rs.size < MIN_WEEKS:
        return None, int(rs.size)
    ri_c = ri - ri.mean()
    var = float(ri_c @ ri_c)
    if not var > 0:
        return None, int(rs.size)
    return float(ri_c @ (rs - rs.mean())) / var, int(rs.size)


@dataclass
class _Drift:
    g: np.ndarray                 # per-step log drift (length h): E[S_t] = S_0·exp(cumsum(g))
    kind: str                     # drift actually used
    annual: float                 # arithmetic annual drift, averaged over the horizon
    note: str
    beta: float | None            # Blume-adjusted beta (None when not measured)
    warnings: list[str] = field(default_factory=list)


def _drift(kind: str, h: int, r: np.ndarray, beta_raw: float | None, n_weeks: int, rf: float, erp: float,
           quant_drift: float | None, has_index: bool) -> _Drift:
    warns: list[str] = []
    beta = None if beta_raw is None else 0.67 * beta_raw + 0.33
    if beta_raw is None and kind != "zero":
        warns.append(("fewer than 26 weekly returns overlap with NIFTY" if has_index else "no NIFTY history")
                     + " — β could not be measured, β = 1 assumed for the CAPM drift")
    b = 1.0 if beta is None else beta
    d_capm = max(rf + b * erp, -0.95)
    c_capm = math.log1p(d_capm)
    capm_note = (f"CAPM: risk-free {rf:.1%} + β {b:.2f} × equity risk premium {erp:.1%} = {d_capm:.1%}/yr"
                 + (f" (β {beta_raw:.2f} from {n_weeks} weekly returns vs NIFTY, Blume-adjusted 0.67β + 0.33)"
                    if beta_raw is not None else " (β assumed 1)"))
    if kind == "quant" and (quant_drift is None or not math.isfinite(quant_drift) or quant_drift <= -0.95):
        warns.append("no quant-model drift available — CAPM drift used instead")
        kind = "capm"
    if kind == "capm":
        c, note = np.full(h, c_capm / TRADING_DAYS), capm_note
    elif kind == "zero":
        c, note = np.zeros(h), "zero: martingale — the expected price equals today's; a pure risk view"
    elif kind == "historical":
        n = r.size
        var = float(r.var(ddof=1))
        c_hist = TRADING_DAYS * (float(r.mean()) + 0.5 * var)
        se = TRADING_DAYS * math.sqrt(var / n)            # standard error of the annualised mean
        d = c_hist - c_capm
        tau2 = min(max(d * d - se * se, 0.0), TAU_MAX ** 2)
        w = tau2 / (tau2 + se * se) if se > 0 else 1.0
        cc = c_capm + w * d
        c = np.full(h, cc / TRADING_DAYS)
        note = (f"historical: the window's mean return {math.expm1(c_hist):.1%}/yr is known only to ±{se:.0%} "
                f"(1 s.e.) — raw historical means are extremely noisy — so it is shrunk toward CAPM "
                f"{d_capm:.1%} (James–Stein-style, {w:.0%} weight on history) → {math.expm1(cc):.1%}/yr")
    else:                                                  # quant
        c_q = math.log1p(float(quant_drift))
        s = np.arange(1, h + 1, dtype=float)
        w = np.clip((QUANT_FADE - s) / (QUANT_FADE - QUANT_FULL), 0.0, 1.0)
        c = (w * c_q + (1.0 - w) * c_capm) / TRADING_DAYS
        note = (f"quant: the quant model's 1-month view {float(quant_drift):.1%}/yr for the first {QUANT_FULL} "
                f"days, fading linearly to CAPM {d_capm:.1%}/yr by day {QUANT_FADE}")
    annual = math.expm1(float(c.mean()) * TRADING_DAYS)
    if kind == "quant":
        note += f" (average over the horizon {annual:.1%}/yr)"
    note += ". All models share this drift; they differ only in the shape of risk."
    return _Drift(c, kind, annual, note, beta, warns)


# ------------------------------------------------------------------ calibration
@dataclass
class _Fit:
    r: np.ndarray
    members: tuple[str, ...]
    sigma: float
    clip: float
    garch: tm.GarchFit | None = None
    garch_cap: float = math.inf
    garch_next_var: float = math.nan
    hmm: tm.GaussianHMM | None = None
    hmm_probs: np.ndarray | None = None
    hmm_scale: float = 1.0
    jump: dict | None = None
    boot_eps: np.ndarray | None = None
    boot_kappa: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _fit(r: np.ndarray, model: str) -> _Fit:
    sigma = float(r.std(ddof=1))
    if not sigma > 0:
        raise ValueError("the price did not move in the lookback window — nothing to simulate")
    eps = r - r.mean()
    clip = min(0.5, max(0.2, 1.5 * float(np.abs(eps).max())))
    members = ENSEMBLE if model == "ensemble" else (model,)
    f = _Fit(r=r, members=members, sigma=sigma, clip=clip)
    var = sigma * sigma
    if "garch" in members:
        g = tm.garch11_fit(r, dist="t", variance_targeting=True)
        f.garch, f.garch_cap = g, GARCH_VAR_CAP * var
        f.garch_next_var = min(g.omega + g.alpha * g.last_resid ** 2 + g.beta * g.last_var, f.garch_cap)
        if not g.converged:
            f.warnings.append(f"GARCH fit: {g.note or 'did not converge'}")
    if "regime" in members:
        try:
            hmm = tm.GaussianHMM(2).fit(r, n_iter=100, seed=0, clip_z=6.0)
            f.hmm, f.hmm_probs = hmm, hmm.filter(r)[-1]
            f.hmm_scale = var / float(hmm.startprob @ hmm.vars)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as e:
            if model == "regime":
                raise ValueError(f"the regime model could not be fitted: {e}") from e
            f.warnings.append(f"regime model dropped from the ensemble (HMM fit failed: {e})")
            f.members = tuple(m for m in members if m != "regime")
    if "jump" in members:
        med = float(np.median(r))
        dev = r - med
        rob = 1.4826 * float(np.median(np.abs(dev))) or sigma
        is_jump = np.abs(dev) > JUMP_K * rob
        rest = r[~is_jump]
        sd_d = float(rest.std(ddof=1)) if rest.size > 2 else sigma
        jumps = r[is_jump] - float(rest.mean())
        nj = int(jumps.size)
        f.jump = {"lam_step": nj / r.size, "lambda": nj / r.size * TRADING_DAYS,
                  "mu_j": float(jumps.mean()) if nj else 0.0, "sigma_j": float(jumps.std(ddof=1)) if nj >= 2 else 0.0,
                  "sigma_d": sd_d, "n": nj}
        if nj == 0:
            f.warnings.append("no jumps beyond 3 robust σ in the window — the jump model is a plain diffusion")
    if "bootstrap" in members:
        f.boot_eps = eps
        f.boot_kappa = math.log(float(np.mean(np.exp(eps))))
    return f


# ------------------------------------------------------------------ path generation
def _member(f: _Fit, name: str, h: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, h) martingale log returns M = clip(ε) − κ with E[e^M] = 1 per step (see the module docstring)."""
    L = f.clip
    if name == "gbm":
        M = rng.standard_normal((n, h))
        M *= f.sigma
        np.clip(M, -L, L, out=M)
        M -= 0.5 * f.sigma ** 2
        return M
    if name == "garch":
        g = f.garch
        M = tm.garch11_simulate(g, h, n, rng, max_var=f.garch_cap)
        M -= g.mu
        # replay the variance recursion garch11_simulate ran, for the per-path Itô term
        var = np.full(n, g.omega + g.alpha * g.last_resid ** 2 + g.beta * g.last_var)
        for t in range(h):
            np.minimum(var, f.garch_cap, out=var)
            e = M[:, t]
            nxt = g.omega + g.alpha * e * e + g.beta * var
            np.clip(e, -L, L, out=e)
            e -= 0.5 * var
            var = nxt
        return M
    if name == "bootstrap":
        idx = tm.stationary_bootstrap_indices(f.boot_eps.size, h, n, BOOT_BLOCK, rng)
        M = f.boot_eps[idx]
        M -= f.boot_kappa
        return M
    if name == "jump":
        j = f.jump
        M = rng.standard_normal((n, h))
        M *= j["sigma_d"]
        if j["n"]:
            k = rng.poisson(j["lam_step"], (n, h))
            hit = k > 0
            kk = k[hit].astype(float)
            M[hit] += kk * j["mu_j"] + np.sqrt(kk) * j["sigma_j"] * rng.standard_normal(kk.size)
        np.clip(M, -L, L, out=M)
        M -= 0.5 * j["sigma_d"] ** 2 + j["lam_step"] * math.expm1(j["mu_j"] + 0.5 * j["sigma_j"] ** 2)
        return M
    if name == "regime":
        hmm = f.hmm
        M, states = hmm.simulate(h, n, rng, start_probs=f.hmm_probs)
        M -= hmm.means[states]                             # state means are context only, never simulated
        M *= math.sqrt(f.hmm_scale)
        np.clip(M, -L, L, out=M)
        M -= (0.5 * f.hmm_scale) * hmm.vars[states]
        return M
    raise ValueError(f"unknown model {name!r}")


def _paths(f: _Fit, h: int, n: int, seed: int, key: tuple[int, ...] = ()) -> tuple[np.ndarray, list[dict] | None]:
    """(n, h) martingale log returns, ensemble members stacked in order; each member has its own RNG stream
    derived from (seed, key, member index), so results are deterministic and origins independent. Members are
    generated in fixed-size row chunks so their temporaries stay ~CHUNK_CELLS whatever the path count."""
    k = len(f.members)
    counts = [n // k + (1 if i < n % k else 0) for i in range(k)]
    out = np.empty((n, h))
    rows = max(1, CHUNK_CELLS // h)
    row = 0
    for i, (name, m) in enumerate(zip(f.members, counts)):
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(*key, i)))
        for r0 in range(0, m, rows):
            c = min(rows, m - r0)
            out[row:row + c] = _member(f, name, h, c, rng)
            row += c
    comps = None if k == 1 else [{"model": name, "weight": round(m / n, 4)} for name, m in zip(f.members, counts)]
    return out, comps


# ------------------------------------------------------------------ shared calibration step
@dataclass
class _Setup:
    start: int
    years: float
    r: np.ndarray
    fit: _Fit
    drift: _Drift
    warnings: list[str]


def _setup(s: _Series, idx: _Series | None, end: int, h: int, model: str, drift: str, lookback_years: float,
           rf: float, erp: float, quant_drift: float | None) -> _Setup:
    """Everything fitted for a forecast made at bar `end`, from bars ≤ end only."""
    start, years = _window(s, end, lookback_years)
    warns: list[str] = []
    if start == 0 and years < lookback_years - YEAR_TOL:
        warns.append(f"only {years:.1f} years of history — calibrated on all of it (asked for {lookback_years:g})")
    if years < 2.0 - YEAR_TOL:
        warns.append(f"calibrated on {years:.1f} years — less than 2 years may miss a full volatility cycle")
    r = np.diff(np.log(s.close[start:end + 1]))
    if h > r.size:
        warns.append(f"the horizon ({h} days) is longer than the calibration window (≈ {years:.1f} years) — the "
                     "far end of the fan extrapolates")
    fit = _fit(r, model)
    beta_raw, n_weeks = _beta(s, end, idx)
    dr = _drift(drift, h, r, beta_raw, n_weeks, rf, erp, quant_drift, idx is not None and idx.ts.size > 0)
    return _Setup(start, years, r, fit, dr, warns + fit.warnings + dr.warnings)


def _inputs(s: _Series, st: _Setup, components, rf: float, erp: float) -> dict:
    f, r = st.fit, st.r
    g = f.garch
    garch = None if g is None else {
        "omega": _sig(g.omega), "alpha": _num(g.alpha), "beta": _num(g.beta), "nu": _num(g.nu, 2),
        "persistence": _num(g.persistence), "long_run_vol": _num(math.sqrt(g.long_run_var * TRADING_DAYS)),
        "current_vol": _num(math.sqrt(f.garch_next_var * TRADING_DAYS))}
    regime = None
    if f.hmm is not None and "regime" in f.members:
        hm = f.hmm
        regime = {"labels": ["bear", "bull"], "mu_annual": [_num(m * TRADING_DAYS) for m in hm.means],
                  "vol_annual": [_num(math.sqrt(v * f.hmm_scale * TRADING_DAYS)) for v in hm.vars],
                  "p_stay": [_num(p) for p in np.diag(hm.trans)], "current": int(np.argmax(f.hmm_probs)),
                  "probs": [_num(p) for p in f.hmm_probs]}
    j = f.jump if "jump" in f.members else None
    end = st.start + r.size
    return {
        "bars": int(r.size + 1), "first": _date(s.ts[st.start]), "last": _date(s.ts[end]),
        "lookback_years": _num(st.years, 2),
        "mu_annual": _num(float(r.mean()) * TRADING_DAYS), "sigma_annual": _num(f.sigma * math.sqrt(TRADING_DAYS)),
        "drift_annual": _num(st.drift.annual), "drift_note": st.drift.note,
        "beta": _num(st.drift.beta, 3) if st.drift.beta is not None else None, "rf": _num(rf), "erp": _num(erp),
        "garch": garch,
        "jump": None if j is None else {"lambda": _num(j["lambda"], 2), "mu_j": _num(j["mu_j"]),
                                        "sigma_j": _num(j["sigma_j"])},
        "regime": regime,
        "bootstrap_block": int(BOOT_BLOCK) if "bootstrap" in f.members else None,
        "components": components,
    }


# ------------------------------------------------------------------ simulate
def simulate(df: pd.DataFrame, *, horizon_days: int, paths: int, model: str = "ensemble", drift: str = "capm",
             lookback_years: float = 5.0, targets=(), seed: int = 0, spot: float | None = None,
             spot_source: str = "close", index_df: pd.DataFrame | None = None, rf: float | None = None,
             erp: float | None = None, quant_drift: float | None = None) -> dict:
    """Simulate `paths` daily price paths `horizon_days` ahead from `spot` (default: the last close) → McResult
    without `symbol`. df: daily bars (ts, close, …); index_df: NIFTY daily bars for beta. Deterministic for a
    given seed. ValueError for bad arguments or fewer than 250 bars."""
    t0 = time.perf_counter()
    h, n, lookback_years = _check_args(horizon_days, paths, model, drift, lookback_years)
    rf = settings.risk_free_rate if rf is None else float(rf)
    erp = settings.equity_risk_premium if erp is None else float(erp)
    warns: list[str] = []
    n, w = _cap_paths(n, h)
    if w:
        warns.append(w)
    s = _series(df)
    if s.ts.size == 0:
        raise ValueError("no daily price history")
    if s.removed:
        warns.append(f"{s.removed} suspected bad print(s) (a ≥ 20 % spike fully reversed next session) removed")
    idx = _series(index_df) if index_df is not None else None
    end = s.ts.size - 1
    spot_v = float(s.close[end]) if spot is None else float(spot)
    if not (math.isfinite(spot_v) and spot_v > 0):
        raise ValueError("spot must be a positive price")
    st = _setup(s, idx, end, h, model, drift, lookback_years, rf, erp, quant_drift)
    warns += st.warnings

    M, components = _paths(st.fit, h, n, _seed(seed))
    M += st.drift.g
    np.cumsum(M, axis=1, out=M)                            # M is now the log price relative to spot
    X = M
    fut = _future_ts(int(s.ts[end]), h)

    steps = _fan_steps(h)
    Xs = np.zeros((n, steps.size))
    Xs[:, 1:] = X[:, steps[1:] - 1]
    q = np.quantile(Xs, FAN_Q, axis=0)
    mean = np.exp(Xs).mean(axis=0)
    fan = []
    for j, t in enumerate(steps):
        row = {"t": int(t), "ts": int(fut[t])}
        for key, qi in zip(FAN_KEYS, q[:, j]):
            row[key] = _px(spot_v * math.exp(qi))
        row["mean"] = _px(spot_v * mean[j])
        fan.append(row)
    pick = np.unique(np.round(np.linspace(0, n - 1, min(N_SAMPLE, n))).astype(int))
    sample_paths = np.round(spot_v * np.exp(Xs[pick]), 2).tolist()

    xt = X[:, -1]
    terminal = _terminal(spot_v, xt)
    risk = _risk(xt, X)
    horizons = []
    for d, label in [c for c in CHECKPOINTS if c[0] <= h] + ([] if h in dict(CHECKPOINTS) else [(h, f"{h}D")]):
        col = X[:, d - 1]
        qs = np.quantile(col, (0.05, 0.25, 0.5, 0.75, 0.95))
        horizons.append({
            "label": label, "days": int(d), "ts": int(fut[d]),
            **{k: _px(spot_v * math.exp(v)) for k, v in zip(("p5", "p25", "p50", "p75", "p95"), qs)},
            "mean": _px(spot_v * float(np.exp(col).mean())), "prob_up": _num((col > 0).mean()),
            "exp_ret": _num(float(np.exp(col).mean()) - 1.0), "median_ret": _num(math.expm1(qs[2])),
            "ann_median_ret": _num(math.expm1(qs[2] * TRADING_DAYS / d))})
    tg = _targets(targets, spot_v, X, xt, warns)

    hist0 = max(0, end + 1 - TRADING_DAYS)
    inputs = _inputs(s, st, components, rf, erp)
    elapsed = int((time.perf_counter() - t0) * 1000)
    log.debug("simulate %s/%s h=%d paths=%d: %d ms", model, st.drift.kind, h, n, elapsed)
    return {
        "asof": _date(s.ts[end]), "spot": _px(spot_v), "spot_source": "live" if spot_source == "live" else "close",
        "horizon_days": h, "paths": n, "model": model, "drift": st.drift.kind, "seed": int(seed or 0),
        "elapsed_ms": elapsed,
        "history": [[int(t), _px(c)] for t, c in zip(s.ts[hist0:], s.close[hist0:])],
        "fan": fan, "sample_paths": sample_paths, "terminal": terminal, "risk": risk, "horizons": horizons,
        "targets": tg, "inputs": inputs, "warnings": list(dict.fromkeys(warns)),
    }


def _fan_steps(h: int) -> np.ndarray:
    """≤ MAX_FAN evenly spaced steps 0…h that also hit every checkpoint exactly."""
    if h + 1 <= MAX_FAN:
        return np.arange(h + 1)
    cps = [d for d, _ in CHECKPOINTS if d <= h]
    base = np.round(np.linspace(0, h, MAX_FAN - len(cps))).astype(int)
    return np.unique(np.r_[base, cps])


def _terminal(spot: float, xt: np.ndarray) -> dict:
    st = spot * np.exp(xt)
    lo, hi = np.quantile(st, (0.005, 0.995))
    if not hi > lo:
        lo, hi = lo * 0.999, hi * 1.001 + 1e-9
    edges = np.linspace(lo, hi, N_BINS + 1)
    counts = np.histogram(st, bins=edges)[0]
    m, sd = float(st.mean()), float(st.std())
    skew = float(((st - m) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
    p = np.quantile(st, (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
    return {"edges": [round(float(e), 4) for e in edges], "counts": [int(c) for c in counts],
            "mean": _px(m), "median": _px(p[3]), "std": _px(sd), "skew": _num(skew),
            "p1": _px(p[0]), "p5": _px(p[1]), "p25": _px(p[2]), "p75": _px(p[4]), "p95": _px(p[5]), "p99": _px(p[6]),
            "prob_up": _num((xt > 0).mean()), "exp_ret": _num(m / spot - 1.0), "median_ret": _num(p[3] / spot - 1.0)}


def _risk(xt: np.ndarray, X: np.ndarray) -> dict:
    ret = np.expm1(xt)
    q5, q1 = np.quantile(ret, (0.05, 0.01))
    mdd = np.empty(X.shape[0])
    for r0 in range(0, X.shape[0], 4096):                  # chunked: the running peak would double memory
        c = X[r0:r0 + 4096]
        peak = np.maximum.accumulate(c, axis=1)
        np.maximum(peak, 0.0, out=peak)                    # the start (spot) is the first peak
        mdd[r0:r0 + 4096] = -np.expm1((c - peak).min(axis=1))
    return {"var95": _num(-q5), "var99": _num(-q1),
            "cvar95": _num(-ret[ret <= q5].mean()), "cvar99": _num(-ret[ret <= q1].mean()),
            "prob_loss_10": _num((ret <= -0.10).mean()), "prob_loss_20": _num((ret <= -0.20).mean()),
            "prob_gain_20": _num((ret >= 0.20).mean()),
            "max_dd_median": _num(np.median(mdd)), "max_dd_p90": _num(np.quantile(mdd, 0.9)),
            "prob_dd_20": _num((mdd >= 0.20).mean())}


def _targets(targets, spot: float, X: np.ndarray, xt: np.ndarray, warns: list[str]) -> list[dict]:
    """Touch = the daily CLOSE crosses the level at some step (intraday touches are somewhat more likely)."""
    levels = []
    for v in list(targets or []):
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = math.nan
        if math.isfinite(v) and v > 0:
            levels.append(v)
        else:
            warns.append(f"target {v!r} ignored (not a positive price)")
    if len(levels) > 6:
        warns.append(f"only the first 6 of {len(levels)} targets are evaluated")
        levels = levels[:6]
    if not levels:
        return []
    hi = np.maximum(X.max(axis=1), 0.0)
    lo = np.minimum(X.min(axis=1), 0.0)
    out = []
    for lvl in levels:
        x = math.log(lvl / spot)
        if lvl >= spot:
            out.append({"price": _px(lvl), "direction": "up", "prob_touch": _num((hi >= x).mean()),
                        "prob_end_beyond": _num((xt >= x).mean())})
        else:
            out.append({"price": _px(lvl), "direction": "down", "prob_touch": _num((lo <= x).mean()),
                        "prob_end_beyond": _num((xt <= x).mean())})
    return out


# ------------------------------------------------------------------ calibration backtest
def _crps_sample(x: np.ndarray, y: float) -> float:
    """Fair sample CRPS: mean|X − y| − ½·mean over distinct pairs |X − X'|."""
    xs = np.sort(x)
    m = xs.size
    i = np.arange(1, m + 1, dtype=float)
    return float(np.abs(xs - y).mean() - ((2.0 * i - m - 1.0) @ xs) / (m * (m - 1.0)))


def _crps_lognormal(mu: float, sigma: float, y: float) -> float:
    """Closed-form CRPS of LN(mu, sigma²) at y > 0 (Baran & Lerch 2015)."""
    z = (math.log(y) - mu) / sigma
    return (y * (2.0 * tm.norm_cdf(z) - 1.0)
            - 2.0 * math.exp(mu + 0.5 * sigma * sigma) * (tm.norm_cdf(z - sigma) + tm.norm_cdf(sigma / math.sqrt(2.0)) - 1.0))


def _origin_forecast(s: _Series, idx: _Series | None, end: int, h: int, model: str, drift: str,
                     lookback_years: float, paths: int, rf: float, erp: float, seed: int) -> tuple[np.ndarray, list[str]]:
    """Simulated prices `h` days after bar `end`, from data ≤ end only (the arrays are cut here, so nothing
    later can leak in). The RNG stream is keyed by the origin's timestamp."""
    cut = _Series(s.ts[:end + 1], s.close[:end + 1])
    cut_idx = None
    if idx is not None:
        k = int(np.searchsorted(idx.ts, s.ts[end], side="right"))
        cut_idx = _Series(idx.ts[:k], idx.close[:k])
    st = _setup(cut, cut_idx, end, h, model, drift, lookback_years, rf, erp, None)
    M, _ = _paths(st.fit, h, paths, seed, key=(int(s.ts[end]),))
    xt = M.sum(axis=1) + float(st.drift.g.sum())
    return s.close[end] * np.exp(xt), st.warnings


def backtest(df: pd.DataFrame, *, horizon_days: int, model: str = "ensemble", drift: str = "capm",
             lookback_years: float = 5.0, origins: int = 40, paths: int = 2000,
             index_df: pd.DataFrame | None = None, rf: float | None = None, erp: float | None = None,
             seed: int = 0) -> dict:
    """Calibration backtest → McBacktest without `symbol`. Origins are spread evenly over the stretch of history
    where each has `lookback_years` before it and `horizon_days` of realised prices after it; at each, the model
    is refitted on data ≤ origin and its simulated distribution is scored against the realised price."""
    t0 = time.perf_counter()
    h, n_paths, lookback_years = _check_args(horizon_days, paths, model, drift, lookback_years)
    rf = settings.risk_free_rate if rf is None else float(rf)
    erp = settings.equity_risk_premium if erp is None else float(erp)
    warns: list[str] = []
    n_paths, w = _cap_paths(n_paths, h)
    if w:
        warns.append(w)
    if drift == "quant":
        warns.append("quant drift cannot be replayed without look-ahead — the backtest uses CAPM drift")
        drift = "capm"
    s = _series(df)
    if s.removed:
        warns.append(f"{s.removed} suspected bad print(s) (a ≥ 20 % spike fully reversed next session) removed")
    idx = _series(index_df) if index_df is not None else None
    n = s.ts.size
    if n == 0:
        raise ValueError("no daily price history")
    i_min = max(int(np.searchsorted(s.ts, s.ts[0] + (lookback_years - YEAR_TOL) * YEAR_S)), MIN_BARS - 1)
    i_max = n - 1 - h
    if i_max < i_min:
        have = (s.ts[-1] - s.ts[0]) / YEAR_S
        raise ValueError(f"the backtest needs {lookback_years:g} years of history before each origin plus {h} "
                         f"trading days after it; only {have:.1f} years ({n} bars) are available")
    want = max(int(origins), 1)
    cand = np.unique(np.round(np.linspace(i_min, i_max, want)).astype(int))
    if cand.size < want:
        warns.append(f"only {cand.size} distinct origins fit in the available history (asked for {want})")
    seed_n = _seed(seed)

    pits, crps, crps_naive, dir_hits = [], [], [], []
    hits = {lv: [] for lv in COVER_LEVELS}
    points = []
    fit_warns: dict[str, int] = {}
    q_lv = sorted({0.05, 0.5, 0.95} | {(1 - lv) / 2 for lv in COVER_LEVELS} | {(1 + lv) / 2 for lv in COVER_LEVELS})
    for i in cand:
        sim, ws = _origin_forecast(s, idx, int(i), h, model, drift, lookback_years, n_paths, rf, erp, seed_n)
        for wmsg in ws:
            fit_warns[wmsg] = fit_warns.get(wmsg, 0) + 1
        spot, real = float(s.close[i]), float(s.close[i + h])
        qv = dict(zip(q_lv, np.quantile(sim, q_lv)))
        pits.append(float((sim < real).mean() + 0.5 * (sim == real).mean()))
        for lv in COVER_LEVELS:
            hits[lv].append(qv[(1 - lv) / 2] <= real <= qv[(1 + lv) / 2])
        crps.append(100.0 * _crps_sample(sim, real) / spot)
        r_tr = np.diff(np.log(s.close[max(0, i - TRADING_DAYS):i + 1]))
        sig = float(r_tr.std(ddof=1)) * math.sqrt(h)
        crps_naive.append(100.0 * _crps_lognormal(math.log(spot) - 0.5 * sig * sig, sig, real) / spot)
        dir_hits.append(np.sign(qv[0.5] - spot) == np.sign(real - spot))
        points.append({"ts": int(s.ts[i]), "target_ts": int(s.ts[i + h]), "spot": _px(spot), "realized": _px(real),
                       "p5": _px(qv[0.05]), "p50": _px(qv[0.5]), "p95": _px(qv[0.95])})

    k = cand.size
    for wmsg, cnt in fit_warns.items():
        warns.append(wmsg if cnt == k else f"{wmsg} (at {cnt} of {k} origins)")
    coverage = [{"level": lv, "hit_rate": _num(np.mean(hits[lv])), "expected": lv} for lv in COVER_LEVELS]
    c, cn = float(np.mean(crps)), float(np.mean(crps_naive))
    skill = 1.0 - c / cn if cn > 0 else 0.0
    eff = max(1, int((cand[-1] - cand[0] + h) // h))
    log.debug("backtest %s/%s h=%d origins=%d: hit90 %.2f skill %.3f", model, drift, h, k, np.mean(hits[0.9]), skill)
    return {
        "model": model, "drift": drift, "horizon_days": h, "origins": int(k), "paths": n_paths,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "coverage": coverage,
        "pit_hist": [_num(v) for v in np.histogram(pits, bins=10, range=(0.0, 1.0))[0] / k],
        "crps": _num(c), "crps_naive": _num(cn), "skill": _num(skill),
        "direction_hit": _num(np.mean(dir_hits)), "effective_n": eff,
        "points": points, "verdict": _verdict(float(np.mean(hits[0.9])), k, eff, skill),
        "warnings": list(dict.fromkeys(warns)),
    }


def _verdict(hit90: float, k: int, eff: int, skill: float) -> str:
    if hit90 < 0.75:
        tone = "over-confident: the bands are too narrow"
    elif hit90 < 0.85:
        tone = "slightly over-confident"
    elif hit90 <= 0.95:
        tone = "well calibrated"
    else:
        tone = "under-confident: the bands are wider than needed"
    out = (f"90% bands held the actual price {hit90:.0%} of the time (ideal 90%) over {k} origins "
           f"(≈ {eff} independent window{'s' if eff != 1 else ''}) — {tone}.")
    if eff < 5:
        out += " With so few independent windows this is only a rough check."
    if skill > 0.02:
        out += f" Its distribution scored {skill:.0%} better than a plain random walk (CRPS)."
    elif skill < -0.02:
        out += f" Its distribution scored {-skill:.0%} worse than a plain random walk (CRPS)."
    else:
        out += " Its distribution scored about the same as a plain random walk (CRPS)."
    return out
