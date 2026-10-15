"""QuantEdge — a pooled next-move model for NSE stocks, with its walk-forward accuracy attached to every answer.

Daily stock direction is close to a coin toss, so the point of this module is less the forecast than the measured
honesty around it. One L2-regularised logistic regression per horizon (1, 5 and 21 trading days) is trained on a
panel of every lake stock with ≥ 3 years of daily bars, using causal features of the classic quant families:
momentum (incl. 12-1 and vol-scaled), volatility regime, a Kalman local-linear trend, mean reversion around it
(OU z), Hurst persistence, RSI, 52-week position, moving-average gaps, volume, residual momentum vs NIFTY, the
market's own momentum/vol, autocorrelation, skew and an HMM bull-regime probability.

Honesty rules built in
- Causal features: every value at bar t uses bars ≤ t only (filters run forward; rolling and expanding windows end
  at t; per-stock standardisation uses expanding mean/sd). Appending bars never changes earlier rows.
- The 2-state HMM is fitted once per stock on returns BEFORE the evaluation window (≤ 3 years of them, ending
  2·21 trading days before the first test day) and only filtered forward after that. A stock with too little
  pre-window history is fitted on its first 2 years instead, and its rows from that fit period never serve as test
  rows. Refitting per fold would cost ~0.1 s × stocks × folds for a feature the logistic weights only lightly.
- Expanding-window walk-forward, ≥ 4 folds, purge h + embargo h trading days between train and test. The
  headline AUC is the mean of the per-fold AUCs (the pooled AUC, also kept, rewards lucky guesses of which fold
  had more up-moves — with ~3 years that is a handful of market swings); rank IC is the mean daily cross-sectional
  Spearman correlation. Accuracy, Brier and log loss use probabilities calibrated on EARLIER folds only (fold 1
  uses the raw model), i.e. exactly the deployed pipeline. The naive benchmark is the base rate of each fold's
  training window (what a forecaster could have known).
- The regularisation strength is fixed a priori (α = 0.005·h on the mean log loss: overlapping h-day labels carry
  ~1/h of the information per row), never tuned on test folds.
- Calibration is Platt scaling with the intercept tied to the model's own training base rate:
  logit p = logit(base) + b·(logit p_raw − logit(base)). Only the spread b is learnt from walk-forward
  predictions (floored at 0, so a model without measured edge collapses to its base rate instead of flipping
  signs). A free Platt intercept would carry the base-rate gap between old training windows and their test folds
  over to the final model, whose own intercept already reflects the full sample.
- The final model is refit on every labelled row and calibrated with b from all walk-forward predictions.
- A horizon has an "edge" only if its mean fold AUC ≥ 0.52, sits 2 standard errors above 0.5 (date-block
  bootstrap: stocks co-move and h-day labels overlap, so 130k test rows hold only a few dozen independent market
  swings), held ≥ 0.5 in every fold and kept a calibration spread. Without one the signal is NEUTRAL with low
  confidence and the reason travels with the result — the probability itself is still reported as estimated.

Price ranges come from a GARCH(1,1)-t forecast of the h-day variance; the expected log return is
(2·p_up − 1)·E|r_h| — a direction model says nothing about size beyond that. Pure functions do the math
(`compute_features`, `train_model`, `predict`, `score_symbol`); `QuantService` holds the model, trains it in a
worker thread (retrained when missing or older than 24 h) and serves the desk, the AI analyst and the leaderboard.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings
from ..timeutil import from_epoch
from ..universe_data import INDICES, SECTOR_OF
from . import tsmath as tm

log = logging.getLogger("td.research.quant")

MODEL_VERSION = 1
HORIZONS = (1, 5, 21)
H_LABELS = {1: "Next day", 5: "1 week", 21: "1 month"}
NO_EDGE_AUC = 0.52                 # below this walk-forward AUC the model has no reliable edge
MIN_CALIB_B = 0.1                  # calibration spread below this = probabilities shrunk to the base rate
EDGE_Z = 2.0                       # AUC − 2·SE must clear 0.5 (≈ 5 % false edges on null panels; the SE runs low)
BOOT_REPS = 200
UP_P, DOWN_P = 0.55, 0.45

TRADING_DAYS = 252
YEAR_S = 365.25 * 86400
IST_OFFSET_S = 19800
INDEX_SYMBOLS = frozenset(INDICES) | {"SENSEX", "BANKEX"}

MIN_TRAIN_YEARS = 3.0              # stocks in the training panel
MIN_PREDICT_BARS = 60              # below this result() refuses (422)
Z_MIN_PERIODS = 63                 # expanding per-stock standardisation needs this many values
Z_CLIP = 4.0
HURST_WIN, HURST_EVERY = 252, 5
HURST_START = -(-HURST_WIN // HURST_EVERY) * HURST_EVERY   # first bar on the 5-day grid with a full window
FEATURE_WARMUP = HURST_START + Z_MIN_PERIODS - 1           # first bar where every feature is standardised
MIN_TRAIN_DAYS = 252               # calendar days of rows before the first test fold
EVAL_MAX_DAYS = 4 * 252            # the walk-forward covers at most the last ~4 years
MIN_FOLDS = 4
ALPHA0 = 0.005                     # L2 strength per horizon day (see module docstring)

HMM_MIN_FIT, HMM_MAX_FIT, HMM_FIRST_FIT = 250, 756, 504
GARCH_WINDOW = 756

# Kalman local linear trend on log price, noise variances in units of the local daily return variance. Only the
# ratios set the filter (level ≈ 10-day smoother, slope responds over ≈ 40 days); fixed a priori, not fitted, so
# the filter stays causal and identical across stocks.
KF_Q_LEVEL, KF_Q_SLOPE, KF_R = 1e-3, 3e-6, 0.5

FEATURES: tuple[tuple[str, str], ...] = (
    ("ret_1", "1-day return (vol-scaled)"),
    ("mom_5", "1-week momentum"),
    ("mom_21", "1-month momentum"),
    ("mom_63", "3-month momentum"),
    ("mom_126", "6-month momentum"),
    ("mom_12_1", "12-1 month momentum"),
    ("mom_vs", "6-month risk-adjusted momentum"),
    ("vol_regime", "Vol regime (21d vs 1y)"),
    ("ewma_ratio", "EWMA vol ratio (fast/slow)"),
    ("kalman_t", "Kalman trend t-stat"),
    ("ou_z", "Distance from Kalman trend (OU z)"),
    ("hurst", "Hurst exponent (DFA, 1y)"),
    ("rsi", "RSI(14)"),
    ("dist_high", "Distance to 52-week high"),
    ("dist_low", "Distance to 52-week low"),
    ("ma_20", "Gap to 20-day MA"),
    ("ma_50", "Gap to 50-day MA"),
    ("ma_200", "Gap to 200-day MA"),
    ("volume_z", "Volume z-score (20d vs 1y)"),
    ("resid_mom", "Residual momentum vs NIFTY (3M)"),
    ("mkt_mom", "NIFTY 1-month momentum"),
    ("mkt_vol", "NIFTY 1-month volatility"),
    ("ac1", "Lag-1 autocorrelation (3M)"),
    ("ar1_signal", "AR(1) next-day signal"),
    ("skew_63", "Return skew (3M)"),
    ("bull_prob", "HMM bull-regime probability"),
)
FEATURE_KEYS = tuple(k for k, _ in FEATURES)
FEATURE_LABEL = dict(FEATURES)
FI = {k: i for i, k in enumerate(FEATURE_KEYS)}

Progress = Callable[[float, str], None]


# --------------------------------------------------------------------------- small helpers
def _f(x, nd: int = 4) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _date(ts: int) -> str:
    return from_epoch(int(ts)).date().isoformat()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _logit(p):
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


def _lag(a: np.ndarray, k: int) -> np.ndarray:
    """a shifted forward by k bars (k > 0: the value k bars ago; k < 0: k bars ahead), NaN padded."""
    out = np.full(a.size, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def future_ts(last_ts: int, h: int) -> np.ndarray:
    """[last bar ts, then the next h weekdays] as IST-midnight epochs. NSE holidays are ignored: a forecast
    'trading day' is a weekday, so the cone's dates can run a day or two early around holidays."""
    day = (int(last_ts) + IST_OFFSET_S) // 86400
    days = np.busday_offset(np.datetime64(day, "D"), np.arange(1, h + 1), roll="backward")
    out = np.empty(h + 1, dtype=np.int64)
    out[0] = int(last_ts)
    out[1:] = days.astype(np.int64) * 86400 - IST_OFFSET_S
    return out


def t_quantile(p: float, nu: float) -> float:
    """Student-t quantile (standard scale) from the normal quantile by the 1/ν expansion of Abramowitz & Stegun
    26.7.5 (four terms): error < 0.3 % for ν ≥ 3 and p ≤ 0.975, exact as ν → ∞."""
    z = float(tm.norm_ppf(p))
    if not math.isfinite(nu) or nu > 1e4:
        return z
    z2 = z * z
    g1 = (z2 + 1.0) * z / 4.0
    g2 = ((5.0 * z2 + 16.0) * z2 + 3.0) * z / 96.0
    g3 = (((3.0 * z2 + 19.0) * z2 + 17.0) * z2 - 15.0) * z / 384.0
    g4 = ((((79.0 * z2 + 776.0) * z2 + 1482.0) * z2 - 1920.0) * z2 - 945.0) * z / 92160.0
    return z + g1 / nu + g2 / nu ** 2 + g3 / nu ** 3 + g4 / nu ** 4


def unit_t_quantile(p: float, nu: float) -> float:
    """Quantile of a Student-t scaled to unit variance (ν > 2)."""
    nu = max(float(nu), 2.5)
    if nu > 1e4:
        return float(tm.norm_ppf(p))
    return t_quantile(p, nu) * math.sqrt((nu - 2.0) / nu)


def h_day_nu(nu: float, h: int) -> float:
    """Degrees of freedom of a t matching the kurtosis of an h-day sum of daily unit-t(ν) shocks: excess kurtosis
    6/(ν − 4) falls as 1/h, so ν_h = 4 + (ν − 4)·h (ν ≤ 4 has no finite kurtosis; treated as 4.5)."""
    if h <= 1 or not math.isfinite(nu):
        return nu
    return min(4.0 + (max(nu, 4.5) - 4.0) * h, 1e6)


@lru_cache(maxsize=1)
def kf_slope_null_sd() -> float:
    """Sd of the filtered slope per unit return sd when returns are white noise (a driftless random walk): the
    steady-state filter maps returns to slope with weights w_k (its response to one unit return, Σ w_k = 1), so
    sd = √Σ w_k². slope / (this · σ) is a trend t-stat against the random-walk null. The filter's own slope_se
    would assume prices really are a smooth trend plus noise and gives t-stats ~8× too large."""
    y = np.zeros(3000)
    y[1000:] = 1.0
    w = tm.kalman_llt(y, q_level=KF_Q_LEVEL, q_slope=KF_Q_SLOPE, r=KF_R)["slope"][1000:]
    return float(np.sqrt(np.sum(w * w)))


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties sharing their mean rank."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    new = np.r_[True, xs[1:] != xs[:-1]]
    grp = np.cumsum(new) - 1
    counts = np.bincount(grp)
    ends = np.cumsum(counts)
    avg = ends - (counts - 1) / 2.0
    out = np.empty(x.size)
    out[order] = avg[grp]
    return out


def auc_score(score: np.ndarray, y: np.ndarray) -> float | None:
    """ROC AUC (Mann–Whitney, ties count ½); None when only one class is present."""
    y = np.asarray(y, dtype=bool)
    npos = int(y.sum())
    nneg = y.size - npos
    if npos == 0 or nneg == 0:
        return None
    r = _rank(np.asarray(score, dtype=float))
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _weighted_auc_sorted(pos: np.ndarray, w: np.ndarray, grp: np.ndarray) -> float | None:
    """AUC of rows already sorted by score (grp = tie-group id) with row weights w."""
    wp = np.where(pos, w, 0.0)
    gp = np.bincount(grp, wp)
    gn = np.bincount(grp, w - wp)
    wpos, wneg = gp.sum(), gn.sum()
    if not (wpos > 0 and wneg > 0):
        return None
    below = np.cumsum(gn) - gn
    return float((gp * (below + 0.5 * gn)).sum() / (wpos * wneg))


def auc_block_se(score: np.ndarray, y: np.ndarray, cal: np.ndarray, fold: np.ndarray, block: int,
                 reps: int = BOOT_REPS, seed: int = 0) -> float | None:
    """Standard error of the mean per-fold AUC by a block bootstrap over test DATES: each fold's dates are cut into
    consecutive blocks of `block` days and resampled with replacement (all stocks of a date move together). That
    keeps the two dependencies that make a pooled row count misleading — stocks co-moving on the same day and
    h-day labels overlapping — so the SE reflects the handful of independent market swings a fold really holds.
    On synthetic random-walk panels it still runs ~20–30 % below the true spread (longer-range dependence, and
    the training fit's own noise is invisible to it), which is why train_model floors it at the fold spread."""
    rng = np.random.default_rng(seed)
    parts = []
    for k in np.unique(fold):
        m = fold == k
        order = np.argsort(score[m], kind="mergesort")
        sc, yy, cc = score[m][order], y[m][order] > 0.5, cal[m][order]
        grp = np.cumsum(np.r_[True, sc[1:] != sc[:-1]]) - 1
        blk = (cc - cc.min()) // max(int(block), 1)
        parts.append((yy, grp, blk, int(blk.max()) + 1))
    if not parts or reps < 2:
        return None
    out = np.full(reps, np.nan)
    for r in range(reps):
        vals = []
        for yy, grp, blk, nb in parts:
            draws = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
            a = _weighted_auc_sorted(yy, draws[blk], grp)
            if a is not None:
                vals.append(a)
        if vals:
            out[r] = float(np.mean(vals))
    out = out[np.isfinite(out)]
    return float(out.std(ddof=1)) if out.size > 1 else None


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb) / den if den > 0 else math.nan


# --------------------------------------------------------------------------- logistic regression
def logistic_fit(X: np.ndarray, y: np.ndarray, alpha: float = 0.0, max_iter: int = 50,
                 tol: float = 1e-10) -> tuple[float, np.ndarray, bool]:
    """L2-regularised logistic regression by Newton/IRLS with step halving. Minimises
    mean(log loss) + α/2·‖w‖² (intercept unpenalised). Returns (intercept, weights, converged)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    A = np.empty((n, k + 1))
    A[:, 0] = 1.0
    A[:, 1:] = X
    pen = np.full(k + 1, float(alpha))
    pen[0] = 0.0
    w = np.zeros(k + 1)
    w[0] = float(_logit(min(max(y.mean(), 1e-6), 1 - 1e-6)))

    def loss(v: np.ndarray) -> float:
        eta = A @ v
        return float((np.logaddexp(0.0, eta) - y * eta).mean() + 0.5 * (pen * v * v).sum())

    f = loss(w)
    converged = False
    for _ in range(max_iter):
        p = _sigmoid(A @ w)
        g = A.T @ (p - y) / n + pen * w
        H = (A * (p * (1.0 - p))[:, None]).T @ A / n + np.diag(pen) + 1e-10 * np.eye(k + 1)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        t = 1.0
        while True:
            w_new = w - t * step
            f_new = loss(w_new)
            if f_new <= f + 1e-15 or t < 1e-6:
                break
            t *= 0.5
        done = abs(f - f_new) <= tol * (1.0 + abs(f)) or float(np.max(np.abs(t * step))) < 1e-10
        if f_new <= f:
            w, f = w_new, f_new
        if done:
            converged = True
            break
    return float(w[0]), w[1:].copy(), converged


def calib_fit(p_raw: np.ndarray, base, y: np.ndarray, b_max: float = 3.0) -> float:
    """Spread b of the base-anchored Platt map logit p = o + b·(logit p_raw − o), o = logit(base) (scalar or per
    row), by maximum likelihood (1-D Newton), clipped to [0, b_max]: b < 1 shrinks over-confident scores toward
    the base rate, b = 0 means no usable signal."""
    o = _logit(np.broadcast_to(np.asarray(base, dtype=float), p_raw.shape))
    x = _logit(p_raw) - o
    y = np.asarray(y, dtype=float)
    b = 1.0
    for _ in range(50):
        p = _sigmoid(o + b * x)
        g = float(x @ (p - y))
        hss = float((x * x) @ (p * (1.0 - p))) + 1e-12
        step = g / hss
        b_new = min(max(b - step, 0.0), b_max)
        if abs(b_new - b) < 1e-9:
            b = b_new
            break
        b = b_new
    return float(b)


def calib_apply(p_raw: np.ndarray, base, b: float) -> np.ndarray:
    o = _logit(np.broadcast_to(np.asarray(base, dtype=float), np.shape(p_raw)))
    return _sigmoid(o + b * (_logit(p_raw) - o))


def edge_status(metrics: dict | None) -> tuple[bool, str | None]:
    """Whether a horizon's walk-forward shows a usable edge: mean fold AUC ≥ 0.52, at least 2 block-bootstrap
    standard errors above 0.5, no fold below 0.50 (the edge held in every test period) and a calibration that did
    not shrink the model to its base rate. Returns (edge, note); an edge whose probabilities still lost to the base
    rate (Brier) comes back True with a "weak edge" note."""
    m = metrics or {}
    auc = m.get("auc")
    folds = [a for a in m.get("fold_auc") or [] if a is not None]
    head = "no reliable edge at this horizon"
    if auc is None:
        return False, f"{head} (not measured)"
    if auc < NO_EDGE_AUC:
        return False, f"{head} (walk-forward AUC {auc:.3f} < {NO_EDGE_AUC})"
    se = m.get("auc_se")
    if se is not None and auc - EDGE_Z * se < 0.5:
        return False, (f"{head} (AUC {auc:.3f} ± {se:.3f}: not clearly better than a coin flip once the overlap "
                       "and co-movement of the test samples are accounted for)")
    if folds and min(folds) < 0.5:
        return False, (f"{head} (AUC {auc:.3f} on average but {min(folds):.3f} in its worst walk-forward fold — "
                       "the edge did not hold in every period)")
    if m.get("calib_b") is not None and m["calib_b"] < MIN_CALIB_B:
        return False, f"{head} (walk-forward calibration shrank the model to its base rate)"
    if m.get("brier") is not None and m.get("brier_base") is not None and m["brier"] > m["brier_base"]:
        return True, (f"weak edge: ranks stocks better than chance (AUC {auc:.3f}) but its probabilities did not beat "
                      f"the base rate out of sample (Brier {m['brier']:.4f} vs {m['brier_base']:.4f})")
    return True, None


# --------------------------------------------------------------------------- features
@dataclass
class Features:
    """Causal per-bar features of one stock. raw/z are (n, F) in FEATURE_KEYS order; z is standardised with the
    stock's own expanding mean/sd, clipped to ±4, NaN while warming up."""
    ts: np.ndarray
    close: np.ndarray
    raw: np.ndarray
    z: np.ndarray
    fwd: dict[int, np.ndarray]
    r: np.ndarray                     # daily log returns aligned to bars (r[0] = NaN)
    level: np.ndarray                 # Kalman level of log price
    band_sd: np.ndarray               # trailing-year sd of log price around the level (the trend channel)
    slope: np.ndarray                 # per day, log units
    sig_fast: np.ndarray              # EWMA(0.94) daily vol through each bar
    rv21: np.ndarray
    bull_prob: np.ndarray             # NaN without an HMM
    hmm: tm.GaussianHMM | None
    hmm_fit_end_ts: int | None
    idx_lc: np.ndarray                # NIFTY log close aligned to the stock's bars (NaN when missing)
    notes: list[str] = field(default_factory=list)


def _expanding_z(X: np.ndarray, min_periods: int = Z_MIN_PERIODS, clip: float = Z_CLIP) -> np.ndarray:
    """(x_t − mean(x_≤t)) / sd(x_≤t) per column over finite values, NaN before min_periods; causal by
    construction (cumulative sums). Each column is shifted by its first finite value to limit cancellation."""
    ok = np.isfinite(X)
    first = np.argmax(ok, axis=0)
    ref = np.where(ok.any(axis=0), X[first, np.arange(X.shape[1])], 0.0)
    d = np.where(ok, X - ref, 0.0)
    cnt = np.cumsum(ok, axis=0)
    s1 = np.cumsum(d, axis=0)
    s2 = np.cumsum(d * d, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var = (s2 - cnt * mean * mean) / (cnt - 1)
        z = (d - mean) / np.sqrt(np.maximum(var, 0.0))
    z[~ok | (cnt < min_periods) | ~(var > 1e-18)] = np.nan
    return np.clip(z, -clip, clip)


def _align_index(ts: np.ndarray, index_df: pd.DataFrame | None) -> np.ndarray:
    """NIFTY log close on the stock's bar dates: the last index bar on or before each date, at most 5 days old."""
    out = np.full(ts.size, np.nan)
    if index_df is None or len(index_df) == 0:
        return out
    its = index_df["ts"].to_numpy(dtype=np.int64)
    ic = index_df["close"].to_numpy(dtype=float)
    good = np.isfinite(ic) & (ic > 0)
    its, ic = its[good], ic[good]
    if its.size == 0:
        return out
    order = np.argsort(its, kind="stable")
    its, ic = its[order], ic[order]
    j = np.searchsorted(its, ts, side="right") - 1
    ok = (j >= 0)
    jj = np.where(ok, j, 0)
    ok &= (ts - its[jj]) <= 5 * 86400
    out[ok] = np.log(ic[jj[ok]])
    return out


def hmm_params(h: tm.GaussianHMM, fit_end_ts: int | None) -> dict:
    return {"means": [float(x) for x in h.means], "vars": [float(x) for x in h.vars],
            "trans": [[float(x) for x in row] for row in h.trans], "fit_end_ts": fit_end_ts}


def hmm_from_params(p: dict) -> tm.GaussianHMM:
    h = tm.GaussianHMM(len(p["means"]))
    h.means = np.asarray(p["means"], dtype=float)
    h.vars = np.asarray(p["vars"], dtype=float)
    h.trans = np.asarray(p["trans"], dtype=float)
    h.startprob = tm._stationary(h.trans)
    return h


def _fit_hmm(r_ts: np.ndarray, r: np.ndarray, hmm_end_ts: int | None) -> tuple[tm.GaussianHMM | None, int | None]:
    """The stock's 2-state HMM: on its last ≤ 756 returns up to hmm_end_ts when it has ≥ 250 of them, else on its
    first 2 years (the caller keeps that fit period out of testing). Returns (model, ts of the last fit return)."""
    idx = None
    if hmm_end_ts is not None:
        before = np.flatnonzero(r_ts <= hmm_end_ts)
        if before.size >= HMM_MIN_FIT:
            idx = before[-HMM_MAX_FIT:]
    if idx is None:
        idx = np.arange(min(HMM_FIRST_FIT, r.size))
    if idx.size < 60:
        return None, None
    try:
        h = tm.GaussianHMM(2).fit(r[idx], n_iter=100, seed=0, clip_z=6.0)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
        log.debug("hmm fit failed: %s", e)
        return None, None
    return h, int(r_ts[idx[-1]])


def compute_features(df: pd.DataFrame, index_df: pd.DataFrame | None = None, *, hmm: dict | None = None,
                     hmm_end_ts: int | None = None) -> Features:
    """Feature matrix for one stock's daily bars (columns ts, open, high, low, close, volume; sorted, close > 0).

    hmm: stored HMM parameters to filter with (the trained model keeps one per panel stock); otherwise one is
    fitted by the rule in `_fit_hmm`. Every column at bar t depends on bars ≤ t only."""
    ts = df["ts"].to_numpy(dtype=np.int64)
    c = df["close"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float) if "high" in df else c
    lo = df["low"].to_numpy(dtype=float) if "low" in df else c
    vol = df["volume"].to_numpy(dtype=float) if "volume" in df else np.zeros(c.size)
    n = c.size
    notes: list[str] = []
    lc = np.log(c)
    r = np.full(n, np.nan)
    r[1:] = np.diff(lc)
    rs = pd.Series(r)

    sig94 = np.full(n, np.nan)
    sig99 = np.full(n, np.nan)
    if n > 1:
        sig94[1:] = tm.ewma_vol(r[1:], 0.94)
        sig99[1:] = tm.ewma_vol(r[1:], 0.99)
    sig94[sig94 <= 0] = np.nan
    sig99[sig99 <= 0] = np.nan

    def mom(k: int) -> np.ndarray:
        return lc - _lag(lc, k)

    def roll_std(k: int) -> np.ndarray:
        return rs.rolling(k).std().to_numpy(dtype=float)

    rv21, rv126, rv252 = roll_std(21), roll_std(126), roll_std(252)
    F = np.full((n, len(FEATURE_KEYS)), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_1 = r / _lag(sig94, 1)
        F[:, FI["ret_1"]] = ret_1
        for k in (5, 21, 63, 126):
            F[:, FI[f"mom_{k}"]] = mom(k)
        F[:, FI["mom_12_1"]] = _lag(lc, 21) - _lag(lc, 252)
        F[:, FI["mom_vs"]] = mom(126) / (rv126 * math.sqrt(126))
        F[:, FI["vol_regime"]] = np.log(rv21 / rv252)
        F[:, FI["ewma_ratio"]] = np.log(sig94 / sig99)

        # Kalman local linear trend; slope t-stat against the random-walk null, scaled by the slow causal EWMA vol
        if n >= 3:
            kf = tm.kalman_llt(lc, q_level=KF_Q_LEVEL, q_slope=KF_Q_SLOPE, r=KF_R)
            level, slope = kf["level"], kf["slope"]
            F[:, FI["kalman_t"]] = slope / (kf_slope_null_sd() * sig99)
        else:
            level, slope = lc.copy(), np.full(n, np.nan)
        dev = pd.Series(lc - level)
        band_sd = dev.rolling(252, min_periods=63).std().to_numpy(dtype=float)
        F[:, FI["ou_z"]] = ((dev - dev.rolling(252, min_periods=126).mean())
                            / dev.rolling(252, min_periods=126).std()).to_numpy(dtype=float)

        H = np.full(n, np.nan)
        for t in range(HURST_START, n, HURST_EVERY):
            H[t] = tm.hurst_dfa(r[t - HURST_WIN + 1:t + 1])
        F[:, FI["hurst"]] = pd.Series(H).ffill().to_numpy(dtype=float)

        F[:, FI["rsi"]] = tm.rsi(c, 14)
        F[:, FI["dist_high"]] = lc - np.log(pd.Series(hi).rolling(252).max().to_numpy(dtype=float))
        F[:, FI["dist_low"]] = lc - np.log(pd.Series(lo).rolling(252).min().to_numpy(dtype=float))
        cs = pd.Series(c)
        for k in (20, 50, 200):
            F[:, FI[f"ma_{k}"]] = lc - np.log(cs.rolling(k).mean().to_numpy(dtype=float))
        lv = pd.Series(np.log1p(np.maximum(vol, 0.0)))
        F[:, FI["volume_z"]] = ((lv.rolling(20).mean() - lv.rolling(252).mean())
                                / lv.rolling(252).std()).to_numpy(dtype=float)

        idx_lc = _align_index(ts, index_df)
        if not np.isfinite(idx_lc).any():
            notes.append("NIFTY history missing — market and relative-strength features unavailable")
        rm = np.full(n, np.nan)
        rm[1:] = np.diff(idx_lc)
        beta = np.full(n, np.nan)
        if n > 1:
            beta[1:] = tm.rolling_beta(r[1:], rm[1:], 252)
        F[:, FI["resid_mom"]] = mom(63) - beta * (idx_lc - _lag(idx_lc, 63))
        F[:, FI["mkt_mom"]] = idx_lc - _lag(idx_lc, 21)
        F[:, FI["mkt_vol"]] = np.log(pd.Series(rm).rolling(21).std().to_numpy(dtype=float) * math.sqrt(TRADING_DAYS))

        ac1 = rs.rolling(63).corr(rs.shift(1)).to_numpy(dtype=float)
        F[:, FI["ac1"]] = ac1
        F[:, FI["ar1_signal"]] = ac1 * ret_1
        F[:, FI["skew_63"]] = rs.rolling(63).skew().to_numpy(dtype=float)

    bull = np.full(n, np.nan)
    model, fit_end = None, None
    if n > 61:
        if hmm is not None:
            model, fit_end = hmm_from_params(hmm), hmm.get("fit_end_ts")
        else:
            model, fit_end = _fit_hmm(ts[1:], r[1:], hmm_end_ts)
        if model is not None:
            bull[1:] = model.filter(r[1:])[:, -1]
    if model is None:
        notes.append("HMM regime unavailable (too little history)")
    F[:, FI["bull_prob"]] = bull
    F[~np.isfinite(F)] = np.nan

    fwd = {h: _lag(lc, -h) - lc for h in HORIZONS}
    return Features(ts=ts, close=c, raw=F, z=_expanding_z(F), fwd=fwd, r=r, level=level, band_sd=band_sd,
                    slope=slope, sig_fast=sig94, rv21=rv21, bull_prob=bull, hmm=model, hmm_fit_end_ts=fit_end,
                    idx_lc=idx_lc, notes=notes)


# --------------------------------------------------------------------------- model
@dataclass
class HorizonModel:
    h: int
    intercept: float
    coef: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    base: float                       # training base rate (calibration anchor)
    calib_b: float                    # calibration spread from the walk-forward
    alpha: float
    metrics: dict

    def p_raw(self, z: np.ndarray) -> np.ndarray:
        x = (np.nan_to_num(z, nan=0.0) - self.mean) / self.sd
        x = np.where(np.isfinite(z), x, 0.0)            # missing feature → scaler mean (contributes nothing)
        return _sigmoid(self.intercept + x @ self.coef)

    def p_up(self, z: np.ndarray) -> np.ndarray:
        return calib_apply(self.p_raw(z), self.base, self.calib_b)

    def to_json(self) -> dict:
        return {"h": self.h, "intercept": self.intercept, "coef": self.coef.tolist(), "mean": self.mean.tolist(),
                "sd": self.sd.tolist(), "calibration": {"base": self.base, "b": self.calib_b}, "alpha": self.alpha,
                "metrics": self.metrics}

    @classmethod
    def from_json(cls, d: dict) -> HorizonModel:
        return cls(h=int(d["h"]), intercept=float(d["intercept"]), coef=np.asarray(d["coef"], dtype=float),
                   mean=np.asarray(d["mean"], dtype=float), sd=np.asarray(d["sd"], dtype=float),
                   base=float(d["calibration"]["base"]), calib_b=float(d["calibration"]["b"]),
                   alpha=float(d["alpha"]), metrics=dict(d["metrics"]))


@dataclass
class QuantModel:
    trained_at: float
    data_source: str
    features: list[str]
    horizons: dict[int, HorizonModel]
    symbols: list[str]
    rows: int
    hmm: dict[str, dict]
    hmm_end_ts: int | None
    stock_metrics: dict[str, list[dict]]
    evaluation: dict
    fingerprint: str
    elapsed_s: float = 0.0
    version: int = MODEL_VERSION

    @property
    def metrics(self) -> list[dict]:
        return [self.horizons[h].metrics for h in sorted(self.horizons)]

    def to_json(self) -> dict:
        return {"model": "QuantEdge", "version": self.version, "trained_at": self.trained_at,
                "data_source": self.data_source, "features": self.features,
                "horizons": [self.horizons[h].to_json() for h in sorted(self.horizons)],
                "symbols": self.symbols, "rows": self.rows, "hmm": self.hmm, "hmm_end_ts": self.hmm_end_ts,
                "stock_metrics": self.stock_metrics, "evaluation": self.evaluation,
                "lake_fingerprint": self.fingerprint, "elapsed_s": self.elapsed_s}

    @classmethod
    def from_json(cls, d: dict) -> QuantModel:
        if d.get("model") != "QuantEdge" or int(d.get("version", 0)) != MODEL_VERSION:
            raise ValueError("not a QuantEdge model of this version")
        if list(d.get("features") or []) != list(FEATURE_KEYS):
            raise ValueError("model was trained on a different feature set")
        hz = {int(x["h"]): HorizonModel.from_json(x) for x in d["horizons"]}
        return cls(trained_at=float(d["trained_at"]), data_source=str(d.get("data_source", "")),
                   features=list(d["features"]), horizons=hz, symbols=list(d["symbols"]), rows=int(d["rows"]),
                   hmm=dict(d.get("hmm") or {}), hmm_end_ts=d.get("hmm_end_ts"),
                   stock_metrics=dict(d.get("stock_metrics") or {}), evaluation=dict(d.get("evaluation") or {}),
                   fingerprint=str(d.get("lake_fingerprint", "")), elapsed_s=float(d.get("elapsed_s", 0.0)))


def save_model(model: QuantModel, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(model.to_json(), separators=(",", ":"), allow_nan=False))
    tmp.replace(path)


def load_model(path: Path) -> QuantModel:
    return QuantModel.from_json(json.loads(Path(path).read_text()))


def fingerprint(frames: dict[str, pd.DataFrame]) -> str:
    h = hashlib.sha1()
    for sym in sorted(frames):
        df = frames[sym]
        last = int(df["ts"].iat[-1]) if len(df) else 0
        h.update(f"{sym}:{len(df)}:{last};".encode())
    return h.hexdigest()[:16]


def is_index(symbol: str) -> bool:
    return symbol.upper() in INDEX_SYMBOLS


@dataclass
class _Panel:
    X: np.ndarray            # (rows, F) per-stock z features
    fwd: np.ndarray          # (rows, len(HORIZONS)) forward log returns (NaN = no label yet)
    ts: np.ndarray
    cal: np.ndarray          # calendar index of each row
    sym: np.ndarray          # symbol id
    hmm_cal: np.ndarray      # calendar index of the last return the stock's HMM was fitted on


def _metrics(h: int, folds: int, p_raw, p_cal, y, fwd, cal, base_train, fold_auc: list[float],
             auc_se: float | None) -> dict:
    eps = 1e-12
    brier = float(np.mean((p_cal - y) ** 2))
    brier_base = float(np.mean((base_train - y) ** 2))
    ll = float(-np.mean(y * np.log(np.clip(p_cal, eps, 1)) + (1 - y) * np.log(np.clip(1 - p_cal, eps, 1))))
    ll_base = float(-np.mean(y * np.log(np.clip(base_train, eps, 1))
                             + (1 - y) * np.log(np.clip(1 - base_train, eps, 1))))
    ics = []
    order = np.argsort(cal, kind="stable")
    cs, ps, fs = cal[order], p_raw[order], fwd[order]
    bounds = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1], True])
    for a, b in pairwise(bounds):
        if b - a >= 5:
            v = _spearman(ps[a:b], fs[a:b])
            if math.isfinite(v):
                ics.append(v)
    pooled = auc_score(p_raw, y)
    fa = np.asarray(fold_auc, dtype=float)
    return {"h": h, "auc": _f(fa.mean()) if fa.size else None, "acc": _f(np.mean((p_cal >= 0.5) == (y > 0.5))),
            "brier": _f(brier, 5), "brier_base": _f(brier_base, 5), "logloss": _f(ll, 5),
            "logloss_base": _f(ll_base, 5), "base_rate": _f(np.mean(y)), "ic": _f(np.mean(ics) if ics else math.nan),
            "n_test": int(y.size), "folds": folds, "fold_auc": [_f(a) for a in fa],
            "auc_sd": _f(fa.std(ddof=1)) if fa.size > 1 else None, "auc_se": _f(auc_se),
            "auc_pooled": _f(pooled) if pooled is not None else None}


def _fold_bounds(n_cal: int, first_row_idx: int) -> tuple[int, list[tuple[int, int]]]:
    test_start = max(first_row_idx + MIN_TRAIN_DAYS, n_cal - EVAL_MAX_DAYS)
    span = n_cal - test_start
    if span < MIN_FOLDS * 2 * max(HORIZONS):
        raise ValueError(f"not enough history for a walk-forward ({span} test days; need "
                         f"{MIN_FOLDS * 2 * max(HORIZONS)} after {MIN_TRAIN_DAYS} training days)")
    k = max(MIN_FOLDS, round(span / TRADING_DAYS))
    edges = np.linspace(test_start, n_cal, k + 1).round().astype(int)
    return test_start, [(int(a), int(b)) for a, b in pairwise(edges)]


def train_model(frames: dict[str, pd.DataFrame], index_df: pd.DataFrame | None, *,
                progress: Progress | None = None, now: float | None = None,
                data_source: str | None = None) -> tuple[QuantModel, dict[str, Features]]:
    """Fit QuantEdge on a panel of daily bar frames {symbol: df}. Index symbols and stocks with < 3 years of bars
    are skipped. Returns the model and the per-stock features (reused to score the leaderboard)."""
    t0 = time.perf_counter()
    say = progress or (lambda frac, msg: None)
    panel = {s: df for s, df in frames.items()
             if not is_index(s) and len(df) > FEATURE_WARMUP + max(HORIZONS)
             and (df["ts"].iat[-1] - df["ts"].iat[0]) / YEAR_S >= MIN_TRAIN_YEARS}
    if len(panel) < 3:
        raise ValueError(f"only {len(panel)} stocks have ≥ {MIN_TRAIN_YEARS:g} years of daily bars "
                         "(the quant model needs at least 3)")
    if index_df is None or len(index_df) < 300:
        raise ValueError("NIFTY daily history is missing from the lake — the quant model needs it")
    syms = sorted(panel)
    cal = np.unique(np.concatenate([panel[s]["ts"].to_numpy(dtype=np.int64) for s in syms]))
    first_row_ts = min(int(panel[s]["ts"].iat[FEATURE_WARMUP]) for s in syms)
    test_start, folds = _fold_bounds(cal.size, int(np.searchsorted(cal, first_row_ts)))
    hmm_end_ts = int(cal[max(test_start - 2 * max(HORIZONS) - 1, 0)])

    feats: dict[str, Features] = {}
    parts = []
    for i, s in enumerate(syms):
        if i % 10 == 0:
            say(0.05 + 0.6 * i / len(syms), f"features {i}/{len(syms)}")
        fe = compute_features(panel[s], index_df, hmm_end_ts=hmm_end_ts)
        feats[s] = fe
        ok = np.isfinite(fe.z).all(axis=1)
        if not ok.any():
            continue
        fit_end = fe.hmm_fit_end_ts if fe.hmm_fit_end_ts is not None else hmm_end_ts
        m = int(ok.sum())
        parts.append((fe.z[ok], np.column_stack([fe.fwd[h][ok] for h in HORIZONS]), fe.ts[ok],
                      np.searchsorted(cal, fe.ts[ok]), np.full(m, i),
                      np.full(m, int(np.searchsorted(cal, fit_end, side="right")) - 1)))
    if not parts:
        raise ValueError("no stock has a complete feature row")
    P = _Panel(*(np.concatenate([p[j] for p in parts]) for j in range(6)))

    hz_models: dict[int, HorizonModel] = {}
    stock_rows: dict[str, list[dict]] = {s: [] for s in syms}
    for hi, h in enumerate(HORIZONS):
        alpha = ALPHA0 * h
        fwd = P.fwd[:, hi]
        labelled = np.isfinite(fwd)
        y_all = (fwd > 0).astype(float)
        oos = []                                    # (rows idx, p_raw, fold, base_train)
        for k, (s, e) in enumerate(folds):
            say(0.65 + 0.25 * (hi * len(folds) + k) / (len(HORIZONS) * len(folds)),
                f"walk-forward {H_LABELS[h].lower()} · fold {k + 1}/{len(folds)}")
            # train rows end 2h days before the test fold (purge h + embargo h) and only use stocks whose HMM was
            # fitted before that cutoff; test rows lie after their stock's HMM fit period
            cut = s - 2 * h - 1
            tr = labelled & (P.cal <= cut) & (P.hmm_cal <= cut)
            te = labelled & (P.cal >= s) & (P.cal < e) & (P.cal > P.hmm_cal)
            if tr.sum() < 500 or te.sum() < 50:
                continue
            mu, sd = P.X[tr].mean(axis=0), P.X[tr].std(axis=0)
            sd[sd < 1e-9] = 1.0
            b0, w, _ = logistic_fit((P.X[tr] - mu) / sd, y_all[tr], alpha)
            idx = np.flatnonzero(te)
            oos.append((idx, _sigmoid(b0 + ((P.X[idx] - mu) / sd) @ w), k, float(y_all[tr].mean())))
        if len(oos) < 2:
            raise ValueError(f"walk-forward produced too few test folds for h={h}")
        # cross-fitted calibration: fold k uses a spread b fitted on the folds before it
        p_cal_parts = []
        for j, (idx, pr, _, base) in enumerate(oos):
            if j == 0:
                p_cal_parts.append(pr)
            else:
                b = calib_fit(np.concatenate([o[1] for o in oos[:j]]),
                              np.concatenate([np.full(o[0].size, o[3]) for o in oos[:j]]),
                              np.concatenate([y_all[o[0]] for o in oos[:j]]))
                p_cal_parts.append(calib_apply(pr, base, b))
        idx_all = np.concatenate([o[0] for o in oos])
        p_raw = np.concatenate([o[1] for o in oos])
        p_cal = np.concatenate(p_cal_parts)
        base_train = np.concatenate([np.full(o[0].size, o[3]) for o in oos])
        y = y_all[idx_all]
        fold_auc = [auc_score(o[1], y_all[o[0]]) for o in oos]
        fold_id = np.concatenate([np.full(o[0].size, o[2]) for o in oos])
        se = auc_block_se(p_raw, y, P.cal[idx_all], fold_id, block=max(2 * h, 10))
        good = [a for a in fold_auc if a is not None]
        if se is not None and len(good) > 1:
            # the date bootstrap cannot see the noise of the training fit itself; the spread across folds can
            se = max(se, float(np.std(good, ddof=1)) / math.sqrt(len(good)))
        metrics = _metrics(h, len(oos), p_raw, p_cal, y, fwd[idx_all], P.cal[idx_all], base_train,
                           [a for a in fold_auc if a is not None], se)
        for si in np.unique(P.sym[idx_all]):
            m = P.sym[idx_all] == si
            a = auc_score(p_raw[m], y[m])
            stock_rows[syms[si]].append({"h": h, "auc": _f(a) if a is not None else None,
                                         "acc": _f(np.mean((p_cal[m] >= 0.5) == (y[m] > 0.5))), "n": int(m.sum())})
        cb = calib_fit(p_raw, base_train, y)
        metrics["calib_b"] = round(cb, 4)
        metrics["edge"], metrics["edge_note"] = edge_status(metrics)

        say(0.9 + 0.03 * hi, f"final fit {H_LABELS[h].lower()}")
        mu, sd = P.X[labelled].mean(axis=0), P.X[labelled].std(axis=0)
        sd[sd < 1e-9] = 1.0
        b0, w, conv = logistic_fit((P.X[labelled] - mu) / sd, y_all[labelled], alpha)
        if not conv:
            log.warning("quant h=%d final fit did not converge", h)
        hz_models[h] = HorizonModel(h=h, intercept=b0, coef=w, mean=mu, sd=sd,
                                    base=float(y_all[labelled].mean()), calib_b=cb, alpha=alpha, metrics=metrics)
        log.info("quant h=%d: OOS AUC %s ± %s (folds %s, pooled %s) acc %s brier %s/%s IC %s n=%d calib b=%.3f %s",
                 h, metrics["auc"], metrics["auc_se"], metrics["fold_auc"], metrics["auc_pooled"], metrics["acc"],
                 metrics["brier"], metrics["brier_base"], metrics["ic"], metrics["n_test"], cb,
                 "edge" if metrics["edge"] else "no edge")

    hmm = {s: hmm_params(fe.hmm, fe.hmm_fit_end_ts) for s, fe in feats.items() if fe.hmm is not None}
    evaluation = {"test_start": _date(cal[folds[0][0]]), "test_end": _date(cal[folds[-1][1] - 1]),
                  "folds": [[_date(cal[a]), _date(cal[b - 1])] for a, b in folds],
                  "purge_embargo_days": "2·h", "hmm_fit_end": _date(hmm_end_ts),
                  "alpha": {str(h): ALPHA0 * h for h in HORIZONS}}
    model = QuantModel(trained_at=float(now if now is not None else time.time()),
                       data_source=data_source if data_source is not None else settings.data_source,
                       features=list(FEATURE_KEYS), horizons=hz_models, symbols=syms, rows=int(P.X.shape[0]),
                       hmm=hmm, hmm_end_ts=hmm_end_ts, stock_metrics={s: v for s, v in stock_rows.items() if v},
                       evaluation=evaluation, fingerprint=fingerprint({**panel, "NIFTY": index_df}),
                       elapsed_s=round(time.perf_counter() - t0, 2))
    return model, feats


# --------------------------------------------------------------------------- prediction
@dataclass
class _Vol:
    fit: tm.GarchFit
    fc: np.ndarray                 # next 21 daily variances
    cond: np.ndarray               # filtered conditional variances over the fit window


def _garch(r: np.ndarray) -> _Vol | None:
    x = r[np.isfinite(r)][-GARCH_WINDOW:]
    if x.size < 30:
        return None
    fit = tm.garch11_fit(x, dist="t", variance_targeting=True)
    return _Vol(fit=fit, fc=tm.garch11_forecast(fit, max(HORIZONS)), cond=tm.garch11_filter(fit, x))


def _horizon(h: int, p: float, vol: _Vol, spot: float, hm: HorizonModel | None) -> dict:
    sd = math.sqrt(float(vol.fc[:h].sum()))
    # E|r_h| ≈ σ_h·√(2/π) (normal approximation of the mean absolute h-day log return), and with the size of the
    # move independent of its sign, E[r_h] ≈ (2·p_up − 1)·E|r_h|. The distribution is taken symmetric around it,
    # so this is also the median log return.
    m = (2.0 * p - 1.0) * sd * math.sqrt(2.0 / math.pi)
    nu = h_day_nu(vol.fit.nu, h)
    q68, q95 = unit_t_quantile(0.8413447460685429, nu), unit_t_quantile(0.975, nu)
    met = hm.metrics if hm else {}
    auc = met.get("auc")
    edge, note = edge_status(met)
    # without a measured edge the probability is still the honest estimate, but it is no call on the stock
    signal = ("UP" if p >= UP_P else "DOWN" if p <= DOWN_P else "NEUTRAL") if edge else "NEUTRAL"
    tilt = abs(p - 0.5)
    if not edge:
        conf = "low"
    elif tilt >= 0.10 and auc >= 0.55:
        conf = "high"
    elif tilt >= 0.05:
        conf = "medium"
    else:
        conf = "low"
    return {"h": h, "label": H_LABELS[h], "p_up": round(p, 4), "exp_ret": round(m, 5), "ret_sd": round(sd, 5),
            "range68": [round(spot * math.exp(m - q68 * sd), 2), round(spot * math.exp(m + q68 * sd), 2)],
            "range95": [round(spot * math.exp(m - q95 * sd), 2), round(spot * math.exp(m + q95 * sd), 2)],
            "price_median": round(spot * math.exp(m), 2), "signal": signal, "confidence": conf,
            "oos_auc": auc, "oos_acc": met.get("acc"), "base_rate": met.get("base_rate"),
            "edge": edge, "edge_note": note}


def _model_inputs(model: QuantModel | None, symbol: str, df: pd.DataFrame,
                  index_df: pd.DataFrame | None) -> Features:
    hmm = model.hmm.get(symbol) if model is not None else None
    return compute_features(df, index_df, hmm=hmm, hmm_end_ts=model.hmm_end_ts if model is not None else None)


def score_symbol(model: QuantModel, symbol: str, df: pd.DataFrame, index_df: pd.DataFrame | None,
                 feats: Features | None = None) -> dict | None:
    """Leaderboard row: p_up / exp_ret / signal per horizon at the last bar, or None when the stock lacks the
    history for a full feature row."""
    fe = feats if feats is not None else _model_inputs(model, symbol, df, index_df)
    z = fe.z[-1]
    if np.isnan(z).sum() > 2 or not np.isfinite(z[[FI["mom_12_1"], FI["ma_200"]]]).all():
        return None
    vol = _garch(fe.r)
    if vol is None:
        return None
    spot = float(fe.close[-1])
    out = {"symbol": symbol, "sector": SECTOR_OF.get(symbol, "OTHER"), "close": round(spot, 2),
           "ts": int(fe.ts[-1]), "h": {}}
    for h, hm in model.horizons.items():
        hz = _horizon(h, float(hm.p_up(z[None, :])[0]), vol, spot, hm)
        out["h"][h] = {"p_up": hz["p_up"], "exp_ret": hz["exp_ret"], "signal": hz["signal"],
                       "score": float(hm.p_raw(z[None, :])[0])}
    return out


def _tone(v: float, up: float, down: float) -> str:
    return "up" if v >= up else "down" if v <= down else "neutral"


def _pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:+.{nd}f} %"


def _components(fe: Features, vol: _Vol | None, symbol: str = "") -> list[dict]:
    """The model's main ingredients as readable gauges for the desk (latest bar)."""
    zrow = fe.z[-1]
    raw = fe.raw[-1]
    out: list[dict] = []

    def add(key: str, label: str, value, display: str, z, reading: str, tone: str) -> None:
        v = _f(value, 4)
        if v is None:
            return
        out.append({"key": key, "label": label, "value": v, "display": display, "z": _f(z, 2),
                    "reading": reading, "tone": tone})

    kt, slope = raw[FI["kalman_t"]], fe.slope[-1]
    if math.isfinite(kt) and math.isfinite(slope):
        ann = math.expm1(slope * TRADING_DAYS)
        word = ("rising" if kt >= 2 else "mildly rising" if kt >= 1 else "falling" if kt <= -2
                else "mildly falling" if kt <= -1 else "flat (no clear trend)")
        add("trend", "Kalman trend", kt, f"t = {kt:+.2f} · {_pct(ann)}/yr", zrow[FI["kalman_t"]],
            f"The filtered price trend is {word} (slope {_pct(ann)} a year, t-stat {kt:+.1f}).", _tone(kt, 1, -1))

    m12 = raw[FI["mom_12_1"]]
    if math.isfinite(m12):
        g = math.expm1(m12)
        add("momentum", "12-1 momentum", g, _pct(g), zrow[FI["mom_12_1"]],
            f"Return from 12 months to 1 month ago was {_pct(g)} (the classic momentum window skips the last "
            "month, which tends to reverse).", _tone(g, 0.05, -0.05))

    dev = (np.log(fe.close) - fe.level)[-252:]
    ou = tm.ou_fit(dev) if np.isfinite(dev).sum() >= 60 else None
    if ou and ou["mean_reverting"] and ou["sigma_eq"] > 0:
        z_ou = (dev[-1] - ou["mu"]) / ou["sigma_eq"]
        hl = ou["half_life"]
        side = "above" if z_ou > 0 else "below"
        add("mean_reversion", "Mean reversion (OU)", z_ou, f"z = {z_ou:+.2f} · half-life {hl:.0f} d", z_ou,
            f"Price is {abs(z_ou):.1f} sd {side} its Kalman trend; such gaps have halved in about {hl:.0f} "
            "trading days over the last year.", _tone(-z_ou, 1, -1))

    bp = fe.bull_prob[-1]
    if math.isfinite(bp):
        state = "bull" if bp >= 0.65 else "bear" if bp <= 0.35 else "sideways: neither regime dominates"
        add("regime", "HMM regime", bp, f"{bp * 100:.0f} % bull", None,
            f"A 2-state hidden Markov model of daily returns puts {bp * 100:.0f} % on the higher-return regime "
            f"({state}).", _tone(bp, 0.65, 0.35))

    if vol is not None:
        now_v = math.sqrt(float(vol.fc[0]) * TRADING_DAYS)
        lr = math.sqrt(vol.fit.long_run_var * TRADING_DAYS)
        cond = np.sqrt(vol.cond * TRADING_DAYS)
        pct = float(np.mean(cond <= now_v)) if cond.size else math.nan
        ratio = now_v / lr if lr > 0 else math.nan
        word = "elevated" if ratio > 1.15 else "subdued" if ratio < 0.85 else "normal"
        add("vol_regime", "Volatility regime", ratio,
            f"{now_v * 100:.0f} % vs {lr * 100:.0f} % long-run"
            + (f" · {_ordinal(round(pct * 100))} pct" if math.isfinite(pct) else ""),
            zrow[FI["vol_regime"]],
            f"GARCH volatility is {word}: {now_v * 100:.0f} % annualised against a long-run {lr * 100:.0f} %.",
            "neutral")

    hu = raw[FI["hurst"]]
    if math.isfinite(hu):
        kind = ("persistent (trending)" if hu > 0.55 else "anti-persistent (mean-reverting)" if hu < 0.45
                else "close to a random walk")
        add("hurst", "Hurst exponent", hu, f"H = {hu:.2f}", zrow[FI["hurst"]],
            f"DFA Hurst exponent of the last year's returns is {hu:.2f}: {kind} (sampling error ≈ ±0.08).",
            "neutral")

    rv = raw[FI["rsi"]]
    if math.isfinite(rv):
        word = ("overbought (≥ 70) — stretched after a strong run" if rv >= 70
                else "oversold (≤ 30) — stretched after a sharp fall" if rv <= 30 else "in its neutral band")
        add("rsi", "RSI(14)", rv, f"{rv:.0f}", zrow[FI["rsi"]], f"RSI is {rv:.0f}, {word}.", _tone(-rv, -30, -70))

    hi52, lo52 = raw[FI["dist_high"]], raw[FI["dist_low"]]
    if math.isfinite(hi52) and math.isfinite(lo52):
        h_, l_ = math.exp(-hi52), math.exp(-lo52)          # 52w high / low relative to the close
        pos = (1.0 - l_) / (h_ - l_) if h_ > l_ else 0.5
        add("week52", "52-week position", pos, f"{pos * 100:.0f} % of range · {_pct(math.expm1(hi52))} from high",
            zrow[FI["dist_high"]],
            f"The close sits {pos * 100:.0f} % of the way from the 52-week low to the high "
            f"({_pct(math.expm1(hi52))} below the high).", _tone(pos, 0.8, 0.2))

    il = fe.idx_lc
    if not is_index(symbol) and fe.close.size > 252 and math.isfinite(il[-1]) and math.isfinite(il[-253]):
        ex = (math.log(fe.close[-1] / fe.close[-253])) - (il[-1] - il[-253])
        add("relative_strength", "Relative strength vs NIFTY", math.expm1(ex),
            f"{_pct(math.expm1(ex))} vs NIFTY (1Y)", zrow[FI["resid_mom"]],
            f"Over the last year the stock {'beat' if ex > 0 else 'lagged'} NIFTY by {abs(ex) * 100:.1f} % in log "
            "terms.", _tone(ex, 0.02, -0.02))
    return out


def predict(model: QuantModel | None, symbol: str, df: pd.DataFrame, index_df: pd.DataFrame | None) -> dict:
    """QuantResult (minus `status` / `model`, which the service adds) for one stock at its last bar."""
    sym = symbol.upper()
    n = len(df)
    if n < MIN_PREDICT_BARS:
        raise ValueError(f"{sym} has only {n} daily bars — the quant model needs at least {MIN_PREDICT_BARS}")
    fe = _model_inputs(model, sym, df, index_df)
    warnings = list(fe.notes)
    spot = float(fe.close[-1])
    vol = _garch(fe.r)
    if vol is not None and not vol.fit.converged and vol.fit.note:
        warnings.append(f"GARCH: {vol.fit.note}")
    if is_index(sym):
        warnings.append(f"{sym} is an index, not part of the stock panel the model was trained on")

    zrow = fe.z[-1]
    missing = [FEATURE_LABEL[k] for k, ok in zip(FEATURE_KEYS, np.isfinite(zrow)) if not ok]
    horizons: list[dict] = []
    contributions: list[dict] = []
    stock_metrics: list[dict] = []
    usable = len(missing) <= 2 and np.isfinite(zrow[[FI["mom_12_1"], FI["ma_200"]]]).all()
    if model is not None and vol is not None and usable:
        if missing:
            warnings.append("missing inputs set to neutral: " + ", ".join(missing))
        for h in sorted(model.horizons):
            hm = model.horizons[h]
            hz = _horizon(h, float(hm.p_up(zrow[None, :])[0]), vol, spot, hm)
            if hz["edge_note"]:
                warnings.append(f"{H_LABELS[h]}: {hz['edge_note']}")
            horizons.append(hz)
        hm = model.horizons.get(21)
        if hm is not None:
            x = np.where(np.isfinite(zrow), (np.nan_to_num(zrow) - hm.mean) / hm.sd, 0.0)
            contrib = hm.coef * x                          # raw model log-odds (before the calibration spread)
            for j in np.argsort(-np.abs(contrib))[:10]:
                contributions.append({"feature": FEATURE_KEYS[j], "label": FEATURE_LABEL[FEATURE_KEYS[j]],
                                      "value": round(float(x[j]), 3), "contribution": round(float(contrib[j]), 4)})
        stock_metrics = list(model.stock_metrics.get(sym, []))
        if not stock_metrics:
            warnings.append(f"{sym} was not in the training panel (needs ≥ {MIN_TRAIN_YEARS:g} years of bars) — "
                            "no walk-forward record on this stock itself")
    elif model is not None and not usable:
        warnings.append(f"{sym} needs about {FEATURE_WARMUP + 1} daily bars (≈ 1.3 years) for the model's "
                        f"standardised features; it has {n}")

    regime = None
    if fe.hmm is not None and math.isfinite(fe.bull_prob[-1]):
        hmm = fe.hmm
        bp = float(fe.bull_prob[-1])
        probs = (1.0 - bp, bp)
        state = "bull" if bp >= 0.65 else "bear" if bp <= 0.35 else "sideways"
        regime = {"state": state, "labels": ["bear", "bull"], "probs": [round(float(p), 4) for p in probs],
                  "mu_annual": [round(float(m) * TRADING_DAYS, 4) for m in hmm.means],
                  "vol_annual": [round(math.sqrt(float(v) * TRADING_DAYS), 4) for v in hmm.vars],
                  "expected_duration": [round(1.0 / max(1.0 - float(hmm.trans[i, i]), 1e-6), 1)
                                        for i in range(hmm.n_states)]}

    vol_out = None
    if vol is not None:
        cond = np.sqrt(vol.cond)
        now_sd = math.sqrt(float(vol.fc[0]))
        ew, r21 = fe.sig_fast[-1], fe.rv21[-1]
        vol_out = {"garch_now": _f(now_sd * math.sqrt(TRADING_DAYS)),
                   "garch_21d": _f(math.sqrt(float(vol.fc.mean()) * TRADING_DAYS)),
                   "long_run": _f(math.sqrt(vol.fit.long_run_var * TRADING_DAYS)),
                   "ewma": _f(ew * math.sqrt(TRADING_DAYS)), "realized_21": _f(r21 * math.sqrt(TRADING_DAYS)),
                   "percentile": _f(np.mean(cond <= now_sd)) if cond.size else None}

    chart = None
    tail = slice(max(n - TRADING_DAYS, 1), n)
    lv, lse = fe.level[tail], fe.band_sd[tail]
    ok = np.isfinite(lv) & np.isfinite(lse)
    if ok.any():
        tsl = fe.ts[tail][ok]
        bpl = fe.bull_prob[tail][ok]
        chart = {"ts": [int(t) for t in tsl], "close": [round(float(x), 2) for x in fe.close[tail][ok]],
                 "trend": [round(float(x), 2) for x in np.exp(lv[ok])],
                 "upper": [round(float(x), 2) for x in np.exp(lv[ok] + 2 * lse[ok])],
                 "lower": [round(float(x), 2) for x in np.exp(lv[ok] - 2 * lse[ok])],
                 "bull_prob": [round(float(x), 4) for x in bpl] if np.isfinite(bpl).all() else [],
                 "forecast": []}
        if vol is not None:
            h21 = next((x for x in horizons if x["h"] == 21), None)
            m21 = h21["exp_ret"] if h21 else 0.0
            if h21 is None:
                warnings.append("forecast cone shows volatility only (no direction model available)")
            fts = future_ts(int(fe.ts[-1]), max(HORIZONS))
            cum = np.concatenate([[0.0], np.cumsum(vol.fc)])
            for t in range(max(HORIZONS) + 1):
                m, sd = m21 * t / max(HORIZONS), math.sqrt(cum[t])
                nu = h_day_nu(vol.fit.nu, max(t, 1))
                q68, q95 = unit_t_quantile(0.8413447460685429, nu), unit_t_quantile(0.975, nu)
                chart["forecast"].append({"t": t, "ts": int(fts[t]), "p50": round(spot * math.exp(m), 2),
                                          "lo68": round(spot * math.exp(m - q68 * sd), 2),
                                          "hi68": round(spot * math.exp(m + q68 * sd), 2),
                                          "lo95": round(spot * math.exp(m - q95 * sd), 2),
                                          "hi95": round(spot * math.exp(m + q95 * sd), 2)})
    return {"symbol": sym, "asof": _date(fe.ts[-1]), "spot": round(spot, 2), "horizons": horizons,
            "components": _components(fe, vol, sym), "contributions": contributions, "regime": regime,
            "vol": vol_out, "chart": chart, "stock_metrics": stock_metrics, "warnings": warnings}


# --------------------------------------------------------------------------- service
class QuantService:
    """Holds the QuantEdge model for the gateway: loads it from disk, retrains it in a worker thread when it is
    missing, incompatible or older than `max_age_h`, and serves results / the leaderboard without ever waiting for
    training (results say status "training" until the first model exists)."""

    BOARD_TTL = 600.0
    RESULT_TTL = 600.0
    INDEX_TTL = 300.0
    RETRY_AFTER_ERROR = 900.0

    def __init__(self, data, *, model_path: Path | str | None = None, max_age_h: float = 24.0, lake=None):
        self.data = data
        self.path = Path(model_path) if model_path else Path(settings.data_dir) / "research" / "quant_model.json"
        self.max_age_s = max_age_h * 3600.0
        self._lake = lake
        self._model: QuantModel | None = None
        self._loaded = False
        self._task: asyncio.Task | None = None
        self._progress, self._message = 0.0, ""
        self._error: tuple[float, str] | None = None
        self._index: tuple[float, pd.DataFrame] | None = None
        self._results: dict[str, tuple[float, tuple, dict]] = {}
        self._scores: dict[str, tuple[tuple, dict | None]] = {}
        self._board: tuple[float, float, dict] | None = None     # (built at, model trained_at, table)
        self._board_task: asyncio.Task | None = None

    # ------------------------------------------------------------ model lifecycle
    @property
    def training(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _ensure_loaded(self) -> None:
        """Load the model from disk once (a concurrent second load is harmless) and warm the leaderboard."""
        if self._loaded:
            return
        m = None
        if self.path.exists():
            try:
                m = await asyncio.to_thread(load_model, self.path)
                if m.data_source != settings.data_source:
                    log.info("quant model on disk was trained on %s data — retraining", m.data_source)
                    m = None
            except (OSError, ValueError, KeyError, TypeError) as e:
                log.warning("quant model %s unusable (%s) — retraining", self.path, e)
        if self._loaded:
            return
        self._model, self._loaded = m, True
        if m is not None and not self._stale():
            self._board_task = asyncio.create_task(self._refresh_board(m), name="quant-board")

    def _stale(self) -> bool:
        m = self._model
        return m is None or time.time() - m.trained_at > self.max_age_s

    def _kick(self) -> None:
        """Start background training when the model is missing or stale (not again right after a failure)."""
        if self.training or not self._stale():
            return
        if self._error and time.time() - self._error[0] < self.RETRY_AFTER_ERROR:
            return
        self._task = asyncio.create_task(self._train_job(), name="quant-train")

    def _say(self, frac: float, msg: str) -> None:
        self._progress, self._message = round(min(max(frac, 0.0), 1.0), 3), msg

    async def train(self, force: bool = True) -> dict:
        await self._ensure_loaded()
        if not self.training and (force or self._stale()):
            self._error = None
            self._task = asyncio.create_task(self._train_job(), name="quant-train")
        if self._task is not None:
            await asyncio.shield(self._task)
        return await self.info()

    async def _symbols(self) -> list[str]:
        lake = self._lake or getattr(self.data, "lake", None)
        if lake is None:
            from ..lake import Lake
            lake = Lake(settings.lake_dir)
        return await asyncio.to_thread(lake.symbols, "D")

    async def _nifty(self) -> pd.DataFrame:
        now = time.monotonic()
        if self._index and now - self._index[0] < self.INDEX_TTL:
            return self._index[1]
        try:
            df = await self.data.index_daily("NIFTY")
        except Exception as e:                                   # missing index → features say so
            log.warning("NIFTY history unavailable: %s", e)
            df = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        self._index = (now, df)
        return df

    async def _load_frames(self, syms: list[str], lo: float, hi: float) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for i, s in enumerate(syms):
            if i % 20 == 0:
                self._say(lo + (hi - lo) * i / max(len(syms), 1), f"loading daily bars {i}/{len(syms)}")
            try:
                df, _ = await self.data.daily(s, fetch=False)
            except KeyError:
                continue
            except Exception as e:
                log.warning("quant: %s history unreadable: %s", s, e)
                continue
            if len(df):
                frames[s] = df
        return frames

    async def _train_job(self) -> None:
        t0 = time.perf_counter()
        self._say(0.0, "loading daily bars")
        try:
            syms = [s for s in await self._symbols() if not is_index(s)]
            frames = await self._load_frames(syms, 0.0, 0.05)
            index_df = await self._nifty()
            model, feats = await asyncio.to_thread(train_model, frames, index_df, progress=self._say)
            self._say(0.95, "scoring the universe")
            await asyncio.to_thread(self._score_all, model, frames, index_df, feats)
            await asyncio.to_thread(save_model, model, self.path)
            self._model = model
            self._results.clear()
            self._error = None
            self._say(1.0, "ready")
            log.info("quant model trained on %d stocks, %d rows in %.0f s", len(model.symbols), model.rows,
                     time.perf_counter() - t0)
        except Exception as e:
            self._error = (time.time(), f"{type(e).__name__}: {e}")
            self._say(0.0, f"training failed: {e}")
            log.exception("quant training failed")

    def _score_all(self, model: QuantModel, frames: dict[str, pd.DataFrame], index_df: pd.DataFrame,
                   feats: dict[str, Features] | None = None) -> None:
        rows = []
        for s, df in frames.items():
            if is_index(s) or len(df) < MIN_PREDICT_BARS:
                continue
            key = (len(df), int(df["ts"].iat[-1]), float(df["close"].iat[-1]), model.trained_at)
            cached = self._scores.get(s)
            if cached and cached[0] == key:
                row = cached[1]
            else:
                try:
                    row = score_symbol(model, s, df, index_df, (feats or {}).get(s))
                except Exception as e:
                    log.warning("quant score %s failed: %s", s, e)
                    row = None
                self._scores[s] = (key, row)
            if row:
                rows.append(row)
        asof = _date(max(r["ts"] for r in rows)) if rows else ""
        self._board = (time.monotonic(), model.trained_at, {"asof": asof, "rows": rows})

    # ------------------------------------------------------------ public API
    async def info(self) -> dict:
        await self._ensure_loaded()
        self._kick()
        m = self._model
        if self.training:
            status, progress, message = "training", self._progress, self._message or "training"
            if m is not None:
                message += " (serving the previous model meanwhile)"
        elif m is not None:
            ev = m.evaluation
            status, progress = "ready", 1.0
            message = (f"trained {from_epoch(m.trained_at).strftime('%Y-%m-%d %H:%M')} on {len(m.symbols)} stocks, "
                       f"{m.rows:,} rows · walk-forward {ev.get('test_start')} → {ev.get('test_end')} "
                       f"({len(ev.get('folds') or [])} folds)")
            if self._error:
                message += f" · last retrain failed: {self._error[1]}"
        elif self._error:
            status, progress, message = "error", 0.0, self._error[1]
        else:
            status, progress, message = "missing", 0.0, "no model yet — it trains on first use (about a minute)"
        return {"status": status, "progress": progress, "message": message,
                "trained_at": round(m.trained_at) if m else None, "symbols": len(m.symbols) if m else 0,
                "rows": m.rows if m else 0, "features": list(FEATURE_KEYS), "metrics": m.metrics if m else []}

    async def result(self, symbol: str) -> dict:
        await self._ensure_loaded()
        self._kick()
        sym = symbol.strip().upper()
        df, meta = await self.data.daily(sym, fetch=False)
        if df is None or len(df) == 0:
            raise ValueError(f"no daily history for {sym}" + (f" ({meta.get('warning')})" if meta.get("warning")
                                                               else ""))
        index_df = await self._nifty()
        model = self._model
        key = (len(df), int(df["ts"].iat[-1]), float(df["close"].iat[-1]), model.trained_at if model else None,
               len(index_df))
        hit = self._results.get(sym)
        if hit and hit[1] == key and time.monotonic() - hit[0] < self.RESULT_TTL:
            out = dict(hit[2])
        else:
            out = await asyncio.to_thread(predict, model, sym, df, index_df)
            self._results[sym] = (time.monotonic(), key, out)
            if len(self._results) > 256:
                oldest = min(self._results, key=lambda k: self._results[k][0])
                self._results.pop(oldest, None)
            out = dict(out)
        info = await self.info()
        warnings = [w for w in (meta.get("warnings") or ([meta["warning"]] if meta.get("warning") else []))]
        if model is None:
            out["status"] = info["status"] if info["status"] != "ready" else "missing"
            warnings.append(f"quant model {info['status']}: {info['message']}")
        else:
            out["status"] = "ready"
        out["model"] = info
        out["warnings"] = list(dict.fromkeys(warnings + list(out.get("warnings") or [])))
        return out

    async def leaderboard(self, h: int = 21, n: int = 15) -> dict:
        if h not in HORIZONS:
            raise ValueError(f"h must be one of {', '.join(map(str, HORIZONS))}")
        await self._ensure_loaded()
        self._kick()
        info = await self.info()
        model = self._model
        if model is None:
            return {"h": h, "asof": "", "up": [], "down": [], "model": info}
        board = self._board
        fresh = board is not None and board[1] == model.trained_at
        if not fresh:
            if self._board_task is not None and not self._board_task.done():
                await asyncio.shield(self._board_task)
            if self._board is None or self._board[1] != model.trained_at:
                await self._refresh_board(model)
        elif time.monotonic() - board[0] > self.BOARD_TTL and (self._board_task is None or self._board_task.done()):
            self._board_task = asyncio.create_task(self._refresh_board(model))     # serve stale, refresh behind
        table = self._board[2] if self._board else {"asof": "", "rows": []}
        rows = [r for r in table["rows"] if h in r["h"]]
        # ties in the calibrated p_up (a horizon shrunk to its base rate) are ordered by the raw model score
        rows.sort(key=lambda r: (-r["h"][h]["p_up"], -r["h"][h]["score"], r["symbol"]))
        lead = [{"symbol": r["symbol"], "sector": r["sector"], "close": r["close"], "p_up": r["h"][h]["p_up"],
                 "exp_ret": r["h"][h]["exp_ret"], "signal": r["h"][h]["signal"]} for r in rows]
        n = max(1, int(n))
        out = {"h": h, "asof": table["asof"], "up": lead[:n], "down": lead[::-1][:n], "model": info}
        met = model.horizons[h].metrics if h in model.horizons else None
        out["edge"], out["edge_note"] = edge_status(met)
        return out

    async def _refresh_board(self, model: QuantModel) -> None:
        """Re-score every lake stock whose bars changed since its cached score (runs in the background too)."""
        try:
            syms = [s for s in await self._symbols() if not is_index(s)]
            frames: dict[str, pd.DataFrame] = {}
            for s in syms:
                try:
                    df, _ = await self.data.daily(s, fetch=False)
                except Exception as e:
                    log.debug("quant board: %s skipped (%s)", s, e)
                    continue
                if len(df):
                    frames[s] = df
            index_df = await self._nifty()
            await asyncio.to_thread(self._score_all, model, frames, index_df)
        except Exception:
            log.exception("quant leaderboard refresh failed")


__all__ = ["QuantService", "QuantModel", "HorizonModel", "Features", "compute_features", "train_model", "predict",
           "score_symbol", "logistic_fit", "calib_fit", "calib_apply", "edge_status", "auc_score", "t_quantile",
           "unit_t_quantile", "save_model", "load_model", "FEATURES", "FEATURE_KEYS", "HORIZONS"]
