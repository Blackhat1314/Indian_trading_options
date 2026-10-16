"""Point-in-time features for the paper trader, shared by live trading and backtests.

Everything here is computed from data available at the decision time and nothing later:
  * daily context (ATR, average volume, beta, prior-day levels, momentum, volatility) uses bars up to the
    previous session, so it is fixed before the open;
  * intraday features at decision minute m use minute bars 0 .. m-1 (09:15 .. the minute before the decision).
The same functions run on one day (live) and on every day of a symbol's history at once (backtests), so a
backtest trades exactly what the live engine would have seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .daydata import C, H, L, O, SESSION, V

EPS = 1e-12
RVOL_DAYS = 14          # relative volume vs the same minute of the last 14 sessions
FULL_DAY = 300          # a session with fewer printed minutes (Muhurat, outages) is not a normal trading day


# ============================================================================ daily context
def _wilder(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def daily_context(d: pd.DataFrame, index_close: pd.Series | None = None, next_date: int | None = None) -> pd.DataFrame:
    """Per-session context from daily bars, each row using sessions BEFORE that row's date only.

    `d`: daily bars with ts/open/high/low/close/volume. `index_close`: NIFTY close indexed by date (yyyymmdd).
    Returns a frame indexed by date (yyyymmdd int) with: pdc pdh pdl pdo atr atr_pct adv beta ret1 ret5 ret20
    ret60 rv20 hh20 ll20 hh252 clv nr7 e20 e50 e200 (prior-day values).
    """
    d = d.sort_values("ts").reset_index(drop=True)
    date = pd.to_datetime(d["ts"] + 19800, unit="s").dt.strftime("%Y%m%d").astype(np.int32)
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    atr = _wilder(tr, 14)
    r = np.r_[np.nan, np.log(c[1:] / c[:-1])]
    s = pd.Series
    ctx = pd.DataFrame({
        "pdc": c, "pdh": h, "pdl": l, "pdo": o, "atr": atr, "atr_pct": atr / c,
        "adv": s(v).rolling(20, min_periods=10).mean().to_numpy(),
        "ret1": s(c).pct_change(1).to_numpy(), "ret5": s(c).pct_change(5).to_numpy(),
        "ret20": s(c).pct_change(20).to_numpy(), "ret60": s(c).pct_change(60).to_numpy(),
        "rv20": s(r).rolling(20, min_periods=10).std().to_numpy() * np.sqrt(252),
        "hh20": s(h).rolling(20, min_periods=10).max().to_numpy(),
        "ll20": s(l).rolling(20, min_periods=10).min().to_numpy(),
        "hh252": s(h).rolling(252, min_periods=60).max().to_numpy(),
        "clv": np.where(h > l, (c - l) / np.maximum(h - l, EPS), 0.5),
        "nr7": (s(h - l).rolling(7, min_periods=7).min().to_numpy() >= (h - l) - EPS).astype(float),
        "e20": s(c).ewm(span=20, adjust=False).mean().to_numpy(),
        "e50": s(c).ewm(span=50, adjust=False).mean().to_numpy(),
        "e200": s(c).ewm(span=200, adjust=False, min_periods=100).mean().to_numpy(),
    }, index=date.to_numpy())
    if index_close is not None and len(index_close):
        ir = np.log(index_close / index_close.shift(1)).reindex(ctx.index)
        sr = pd.Series(r, index=ctx.index)
        cov = sr.rolling(60, min_periods=30).cov(ir)
        var = ir.rolling(60, min_periods=30).var()
        ctx["beta"] = (cov / var).clip(-1, 3).fillna(1.0).to_numpy()
    else:
        ctx["beta"] = 1.0
    # shift: the row for date t carries what was known at the close of t-1
    out = ctx.shift(1)
    if next_date is not None and len(ctx) and next_date > int(ctx.index.max()):
        out.loc[next_date] = ctx.iloc[-1].to_numpy()        # today, before its daily bar exists (live)
    out.index.name = "date"
    return out


# ============================================================================ intraday features
FEATURES = [
    # opening auction / gap
    "gap_atr", "gap_pct",
    # move since the open and since yesterday's close (ATR units), and vs the index
    "ret_open_atr", "ret_pc_atr", "resid_pc", "resid_open",
    # opening range and session range
    "or_range_atr", "or_pos", "body_or", "day_pos", "above_orh_atr", "below_orl_atr",
    # volume / participation
    "lrvol", "turnover_cr",
    # VWAP control
    "vwap_atr", "vwap_slope_atr", "frac_above_vwap",
    # path quality
    "eff_ratio", "new_high_frac", "new_low_frac", "last15_atr", "ivol_ratio",
    # prior-day context
    "pdh_atr", "pdl_atr", "clv_prev", "ret5", "ret20", "rv20", "atr_pct", "nr7", "trend_d",
    "hh20_atr", "ll20_atr",
]


def intraday_features(bars: np.ndarray, m: int, ctx: pd.DataFrame | None, dates: np.ndarray,
                      printed: np.ndarray | None = None, idx_bars: np.ndarray | None = None, idx_ctx: pd.DataFrame | None = None,
                      idx_dates: np.ndarray | None = None) -> pd.DataFrame:
    """Features at decision minute m (bars 0 .. m-1 complete) for every day in `bars` [n, 375, 5].

    `ctx`: daily_context of this symbol (indexed by date); `idx_*`: the same for NIFTY (for residual moves).
    Rows for dates without context get NaN context features (callers drop them).
    """
    n = len(dates)
    b = bars[:, :m, :].astype(np.float64)
    o0 = b[:, 0, O]
    cm = b[:, -1, C]
    hi = b[:, :, H].max(1)
    lo = b[:, :, L].min(1)
    k = min(m, 15)
    orh = b[:, :k, H].max(1)
    orl = b[:, :k, L].min(1)
    vol = b[:, :, V].sum(1)
    tp = (b[:, :, H] + b[:, :, L] + b[:, :, C]) / 3
    cumv = np.cumsum(b[:, :, V], 1)
    cumpv = np.cumsum(tp * b[:, :, V], 1)
    vwap_path = np.where(cumv > 0, cumpv / np.maximum(cumv, EPS), tp)
    vwap = vwap_path[:, -1]
    vwap_prev = vwap_path[:, max(0, m - 6)]
    closes = np.concatenate([o0[:, None], b[:, :, C]], 1)
    path = np.abs(np.diff(closes, axis=1)).sum(1)
    run_hi = np.maximum.accumulate(b[:, :, H], 1)
    run_lo = np.minimum.accumulate(b[:, :, L], 1)
    new_hi = (b[:, 1:, H] >= run_hi[:, :-1] - EPS).mean(1) if m > 1 else np.zeros(n)
    new_lo = (b[:, 1:, L] <= run_lo[:, :-1] + EPS).mean(1) if m > 1 else np.zeros(n)
    lr = np.diff(np.log(np.maximum(closes, EPS)), axis=1)
    ivol = lr.std(1) * np.sqrt(SESSION)                    # intraday realised vol, scaled to a session
    c15 = b[:, max(0, m - 16), C] if m > 15 else o0

    # relative volume vs the same minute over the previous RVOL_DAYS full sessions (point in time)
    full = (printed >= FULL_DAY) if printed is not None else bars[:, :, V].sum(1) > 0
    cv = pd.Series(np.where(full, vol, np.nan))
    base = cv.rolling(RVOL_DAYS, min_periods=5).mean().shift(1).to_numpy()
    rvol = np.where(base > 0, vol / np.maximum(base, EPS), np.nan)

    cx = ctx.reindex(dates) if ctx is not None else pd.DataFrame(index=dates)

    def col(name: str, default=np.nan) -> np.ndarray:
        return cx[name].to_numpy(float) if name in cx else np.full(n, default)

    pdc, pdh, pdl, atr = col("pdc"), col("pdh"), col("pdl"), col("atr")
    atr = np.where(atr > 0, atr, np.nan)
    beta = np.nan_to_num(col("beta", 1.0), nan=1.0)

    # index move to the same minute (NIFTY), aligned by date
    idx_pc = np.full(n, 0.0)
    idx_open = np.full(n, 0.0)
    if idx_bars is not None and idx_dates is not None and len(idx_dates):
        pos = pd.Series(np.arange(len(idx_dates)), index=idx_dates).reindex(dates).to_numpy()
        have = ~np.isnan(pos)
        p = pos[have].astype(int)
        ic = idx_bars[p, m - 1, C].astype(float)
        io = idx_bars[p, 0, O].astype(float)
        ipc = idx_ctx["pdc"].reindex(idx_dates[p]).to_numpy(float) if idx_ctx is not None else np.full(len(p), np.nan)
        idx_pc[have] = np.where(ipc > 0, ic / ipc - 1, 0.0)
        idx_open[have] = np.where(io > 0, ic / io - 1, 0.0)

    rng = np.maximum(orh - orl, EPS)
    drange = np.maximum(hi - lo, EPS)
    f = pd.DataFrame({
        "date": dates, "m": m, "open": o0, "price": cm, "hi": hi, "lo": lo, "orh": orh, "orl": orl,
        "vwap": vwap, "atr": atr, "pdc": pdc,
        "gap_atr": (o0 - pdc) / atr, "gap_pct": o0 / pdc - 1,
        "ret_open_atr": (cm - o0) / atr, "ret_pc_atr": (cm - pdc) / atr,
        "resid_pc": (cm / pdc - 1) - beta * idx_pc, "resid_open": (cm / o0 - 1) - beta * idx_open,
        "or_range_atr": (orh - orl) / atr, "or_pos": (cm - orl) / rng,
        "body_or": (b[:, k - 1, C] - o0) / rng, "day_pos": (cm - lo) / drange,
        "above_orh_atr": (cm - orh) / atr, "below_orl_atr": (orl - cm) / atr,
        "lrvol": np.log(np.clip(rvol, 0.05, 50)), "turnover_cr": vwap * vol / 1e7,
        "vwap_atr": (cm - vwap) / atr, "vwap_slope_atr": (vwap - vwap_prev) / atr,
        "frac_above_vwap": (b[:, :, C] > vwap_path).mean(1),
        "eff_ratio": np.abs(cm - o0) / np.maximum(path, EPS),
        "new_high_frac": new_hi, "new_low_frac": new_lo, "last15_atr": (cm - c15) / atr,
        "ivol_ratio": ivol / np.maximum(col("rv20") / np.sqrt(252), EPS),
        "pdh_atr": (cm - pdh) / atr, "pdl_atr": (cm - pdl) / atr, "clv_prev": col("clv"),
        "ret5": col("ret5"), "ret20": col("ret20"), "rv20": col("rv20"), "atr_pct": col("atr_pct"),
        "nr7": col("nr7"), "trend_d": np.sign(pdc - col("e20")) + np.sign(col("e20") - col("e50"))
                                     + np.sign(col("e50") - col("e200")),
        "hh20_atr": (cm - col("hh20")) / atr, "ll20_atr": (cm - col("ll20")) / atr,
        "idx_pc": idx_pc, "idx_open": idx_open, "beta": beta,
        "full_day": full,
    })
    return f


# ============================================================================ swing (daily) features
SWING_FEATURES = [
    "r5", "r20", "r60", "r120", "r250x20", "resid60", "e20_atr", "e50_pct", "e200_pct", "e50_slope", "adj_slope90",
    "hh252_pct", "hh55_pct", "hh20_pct", "rsi2", "rsi14", "atr_pct", "rv20", "vol_ratio", "bbw_pct", "nr7",
    "vr", "accum20", "clv", "up_days20",
]


def _rsi(c: np.ndarray, n: int) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50).to_numpy()


def swing_features(d: pd.DataFrame, index_close: pd.Series | None = None) -> pd.DataFrame:
    """Features at the CLOSE of each session (decide after the close, enter at the next open).

    `d`: daily bars (ts/open/high/low/close/volume); `index_close`: NIFTY closes indexed by yyyymmdd.
    """
    d = d.sort_values("ts").reset_index(drop=True)
    date = pd.to_datetime(d["ts"] + 19800, unit="s").dt.strftime("%Y%m%d").astype(np.int32).to_numpy()
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    v = np.where(v > 0, v, np.nan) if not (v > 0).any() else v
    s = pd.Series
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    atr = _wilder(tr, 14)
    r = np.r_[np.nan, np.log(c[1:] / c[:-1])]
    e20 = s(c).ewm(span=20, adjust=False).mean().to_numpy()
    e50 = s(c).ewm(span=50, adjust=False).mean().to_numpy()
    e200 = s(c).ewm(span=200, adjust=False, min_periods=100).mean().to_numpy()
    lc = np.log(c)
    x = s(np.arange(len(c), dtype=float))
    cov = x.rolling(90, min_periods=60).cov(s(lc))
    var = x.rolling(90, min_periods=60).var()
    corr = x.rolling(90, min_periods=60).corr(s(lc))
    slope = (cov / var).to_numpy()                          # log-price per day
    rv20 = s(r).rolling(20, min_periods=10).std().to_numpy() * np.sqrt(252)
    rv60 = s(r).rolling(60, min_periods=30).std().to_numpy() * np.sqrt(252)
    std20 = s(c).rolling(20, min_periods=15).std()
    bbw = (4 * std20 / s(c).rolling(20, min_periods=15).mean())
    sv = np.sign(np.r_[0, np.diff(c)]) * v
    f = pd.DataFrame({
        "date": date, "close": c, "open": o, "high": h, "low": l, "atr": atr, "turnover_cr": c * v / 1e7,
        "r5": s(c).pct_change(5).to_numpy(), "r20": s(c).pct_change(20).to_numpy(),
        "r60": s(c).pct_change(60).to_numpy(), "r120": s(c).pct_change(120).to_numpy(),
        "r250x20": s(c).shift(20).to_numpy() / s(c).shift(250).to_numpy() - 1,
        "e20_atr": (c - e20) / atr, "e50_pct": c / e50 - 1, "e200_pct": c / e200 - 1,
        "e50_slope": e50 / s(e50).shift(10).to_numpy() - 1,
        "adj_slope90": (np.exp(slope * 252) - 1) * (corr.to_numpy() ** 2),
        "hh252_pct": c / s(h).rolling(252, min_periods=120).max().to_numpy() - 1,
        "hh55_pct": c / s(h).shift(1).rolling(55, min_periods=40).max().to_numpy() - 1,
        "hh20_pct": c / s(h).shift(1).rolling(20, min_periods=15).max().to_numpy() - 1,
        "rsi2": _rsi(c, 2), "rsi14": _rsi(c, 14), "atr_pct": atr / c, "rv20": rv20,
        "vol_ratio": rv20 / rv60,
        "bbw_pct": bbw.rolling(252, min_periods=120).rank(pct=True).to_numpy(),
        "nr7": (s(h - l).rolling(7, min_periods=7).min().to_numpy() >= (h - l) - EPS).astype(float),
        "vr": v / s(v).shift(1).rolling(20, min_periods=10).mean().to_numpy(),
        "accum20": s(sv).rolling(20, min_periods=10).sum().to_numpy() / s(v).rolling(20, min_periods=10).sum().to_numpy(),
        "clv": np.where(h > l, (c - l) / np.maximum(h - l, EPS), 0.5),
        "up_days20": s((np.r_[0, np.diff(c)] > 0).astype(float)).rolling(20, min_periods=10).mean().to_numpy(),
    })
    if index_close is not None and len(index_close):
        ic = index_close.reindex(date).to_numpy(float)
        ir = np.r_[np.nan, np.log(ic[1:] / ic[:-1])]
        b = s(r).rolling(120, min_periods=60).cov(s(ir)) / s(ir).rolling(120, min_periods=60).var()
        f["beta"] = b.clip(-1, 3).fillna(1.0).to_numpy()
        f["resid60"] = f["r60"] - f["beta"] * (s(ic).pct_change(60).to_numpy())
    else:
        f["beta"], f["resid60"] = 1.0, f["r60"]
    return f


def market_regime(index_d: pd.DataFrame, vix_d: pd.DataFrame | None = None) -> pd.DataFrame:
    """Index-level regime features at each close: trend, momentum, volatility, VIX."""
    f = swing_features(index_d)
    out = pd.DataFrame({"date": f["date"], "m_r20": f["r20"], "m_r60": f["r60"], "m_e200": f["e200_pct"],
                        "m_e50": f["e50_pct"], "m_rv20": f["rv20"], "m_slope": f["adj_slope90"]})
    if vix_d is not None and len(vix_d):
        v = vix_d.sort_values("ts")
        vd = pd.to_datetime(v["ts"] + 19800, unit="s").dt.strftime("%Y%m%d").astype(np.int32).to_numpy()
        vc = pd.Series(v["close"].to_numpy(float), index=vd)
        out["vix"] = vc.reindex(out["date"]).ffill().to_numpy()
        out["vix_chg5"] = pd.Series(out["vix"]).pct_change(5).to_numpy()
        out["vix_pct"] = pd.Series(out["vix"]).rolling(252, min_periods=60).rank(pct=True).to_numpy()
    return out
