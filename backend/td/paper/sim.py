"""Minute-by-minute replay of intraday trades: stops, targets, break-even, trailing stops and a time exit.

Vectorised across trades (one numpy step per minute for all of them), so a backtest of every symbol-day of
five years runs in seconds. Fill rules are deliberately pessimistic:
  * entry at the OPEN of the entry minute (the decision used only the bars before it);
  * a stop and a target touched in the same minute count as the stop;
  * a bar that opens through the stop fills at that open (gap through the stop), not at the stop;
  * trailing / break-even stops move only after a bar has closed, never inside the bar that triggered them.
Prices here are raw; slippage and charges are applied by the caller (costs.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .daydata import C, H, L, O

STOP, TARGET, TIME, TRAIL = 0, 1, 2, 3
REASONS = {STOP: "stop", TARGET: "target", TIME: "time", TRAIL: "trail"}


@dataclass(frozen=True)
class ExitRules:
    target_r: float = 0.0          # take profit at this multiple of the initial risk (0 = none)
    breakeven_r: float = 0.0       # move the stop to entry once the trade is this many R in profit (0 = never)
    trail_atr: float = 0.0         # chandelier: stop follows the best price minus this many ATR (0 = off)
    trail_after_r: float = 0.0     # ... switched on only once the trade is this many R in profit
    exit_minute: int = 360         # 15:15 — square off before the broker's auto square-off


def simulate(bars: np.ndarray, entry_min: np.ndarray, side: np.ndarray, stop: np.ndarray, atr: np.ndarray,
             rules: ExitRules) -> dict[str, np.ndarray]:
    """Replay trades. bars: [n, 375, 5] (the trade's day), entry_min/side(+1/-1)/stop/atr: [n].

    Returns entry, exit price, exit minute, reason code, R multiple (gross), MFE and MAE in R.
    """
    n = len(entry_min)
    rows = np.arange(n)
    entry = bars[rows, entry_min, O].astype(np.float64)
    side = side.astype(np.float64)
    stop = stop.astype(np.float64).copy()
    risk = np.abs(entry - stop)
    risk = np.where(risk > 0, risk, np.nan)
    target = np.where(rules.target_r > 0, entry + side * rules.target_r * risk, np.nan)
    best = entry.copy()
    worst = entry.copy()
    exit_px = np.full(n, np.nan)
    exit_min = np.full(n, -1, np.int32)
    reason = np.full(n, -1, np.int8)
    trailing = np.zeros(n, bool)
    open_ = np.isfinite(risk)
    t0 = int(entry_min.min()) if n else 0
    last = rules.exit_minute
    for t in range(t0, last + 1):
        live = open_ & (entry_min <= t)
        if not live.any():
            if not (open_ & (entry_min > t)).any():
                break
            continue
        o = bars[:, t, O].astype(np.float64)
        h = bars[:, t, H].astype(np.float64)
        lo = bars[:, t, L].astype(np.float64)
        if t == last:                                              # time exit at the open of 15:15
            m = live
            exit_px[m], exit_min[m], reason[m] = o[m], t, TIME
            open_ &= ~m
            break
        fav = np.where(side > 0, h, -lo)                          # favourable extreme (signed)
        adv = np.where(side > 0, lo, -h)                          # adverse extreme (signed)
        sstop = side * stop
        # stop first (pessimistic); a bar opening through the stop fills at its open
        hit = live & (adv <= sstop)
        if hit.any():
            px = np.where(side * o <= sstop, o, stop)
            exit_px[hit] = px[hit]
            exit_min[hit] = t
            reason[hit] = np.where(trailing[hit], TRAIL, STOP)
            open_ &= ~hit
            live &= ~hit
        if rules.target_r > 0:
            tg = live & (fav >= side * target)
            if tg.any():
                px = np.where(side * o >= side * target, o, target)
                exit_px[tg], exit_min[tg], reason[tg] = px[tg], t, TARGET
                open_ &= ~tg
                live &= ~tg
        # after the bar: excursions, break-even and trailing stops (they act from the next bar on)
        best = np.where(live, np.where(side > 0, np.maximum(best, h), np.minimum(best, lo)), best)
        worst = np.where(live, np.where(side > 0, np.minimum(worst, lo), np.maximum(worst, h)), worst)
        prog = side * (best - entry) / risk
        if rules.breakeven_r > 0:
            be = live & (prog >= rules.breakeven_r)
            stop = np.where(be & (side * stop < side * entry), entry, stop)
        if rules.trail_atr > 0:
            on = live & (prog >= rules.trail_after_r)
            trailing |= on
            cand = best - side * rules.trail_atr * atr
            stop = np.where(on & (side * cand > side * stop), cand, stop)
    still = open_ & (exit_min < 0)
    if still.any():                                                # data ended early (half day): exit at last close
        lastc = bars[:, last, C].astype(np.float64)
        exit_px[still], exit_min[still], reason[still] = lastc[still], last, TIME
    r_mult = side * (exit_px - entry) / risk
    return {"entry": entry, "exit": exit_px, "exit_min": exit_min, "reason": reason, "risk": risk,
            "r": r_mult, "mfe_r": side * (best - entry) / risk, "mae_r": side * (worst - entry) / risk}


# ============================================================================ swing (daily bars)
@dataclass(frozen=True)
class SwingExits:
    stop_atr: float = 2.0          # initial stop below the entry, in ATR
    trail_atr: float = 3.0         # chandelier: highest close minus this many ATR (0 = off)
    target_atr: float = 0.0        # take profit (0 = none)
    max_days: int = 20             # time stop: exit at the close of this many sessions after entry
    breakeven_atr: float = 0.0     # move the stop to entry once the close is this many ATR above it (0 = off)


def simulate_daily(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, atr: np.ndarray,
                   entry_idx: np.ndarray, rules: SwingExits) -> dict[str, np.ndarray]:
    """Long trades entered at the OPEN of day entry_idx (decided on the previous close).

    The ATR used for stops is the one known at the decision (day entry_idx - 1). Stops are checked against each
    day's low (a gap below the stop fills at the open); the trailing stop is raised after each close and acts
    from the next day. Returns entry, exit, exit index, reason code, holding days and the return.
    """
    k = len(entry_idx)
    n = len(c)
    e = entry_idx
    entry = o[e].astype(np.float64)
    a = atr[np.maximum(e - 1, 0)].astype(np.float64)
    stop = entry - rules.stop_atr * a
    target = entry + rules.target_atr * a if rules.target_atr > 0 else np.full(k, np.inf)
    best_close = entry.copy()
    exit_px = np.full(k, np.nan)
    exit_i = np.full(k, -1, np.int64)
    reason = np.full(k, -1, np.int8)
    open_ = (e < n) & np.isfinite(a) & (a > 0)
    for d in range(rules.max_days + 1):
        i = e + d
        live = open_ & (i < n)
        if not live.any():
            break
        ii = np.minimum(i, n - 1)
        oo, hh, ll, cc = o[ii], h[ii], lo[ii], c[ii]
        hit = live & (ll <= stop)
        px = np.where(oo <= stop, oo, stop)
        exit_px[hit], exit_i[hit], reason[hit] = px[hit], ii[hit], STOP
        open_ &= ~hit
        live &= ~hit
        tg = live & (hh >= target)
        px = np.where(oo >= target, oo, target)
        exit_px[tg], exit_i[tg], reason[tg] = px[tg], ii[tg], TARGET
        open_ &= ~tg
        live &= ~tg
        last = live & ((d == rules.max_days) | (i == n - 1))
        exit_px[last], exit_i[last], reason[last] = cc[last], ii[last], TIME
        open_ &= ~last
        live &= ~last
        best_close = np.where(live, np.maximum(best_close, cc), best_close)
        if rules.trail_atr > 0:
            cand = best_close - rules.trail_atr * a
            stop = np.where(live & (cand > stop), cand, stop)
        if rules.breakeven_atr > 0:
            be = live & (cc >= entry + rules.breakeven_atr * a)
            stop = np.where(be & (stop < entry), entry, stop)
    ret = exit_px / entry - 1
    return {"entry": entry, "exit": exit_px, "exit_i": exit_i, "reason": reason, "days": exit_i - e,
            "ret": ret, "risk": rules.stop_atr * a / entry}
