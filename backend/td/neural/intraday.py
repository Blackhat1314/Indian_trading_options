"""The 09:30 → close panel: one row per stock and trading day with 1-minute data.

What a row knows (all of it available at 09:30, nothing later):
  daily context  the stock's daily-model inputs as of the PREVIOUS close (td.neural.dataset: 40-day technical
                 sequence, cross-sectional ranks, point-in-time fundamentals, news, yesterday's 1-minute summaries)
  bars           the first 15 one-minute bars (09:15 … 09:29): open / high / low / close vs the previous close in
                 ATR units, volume vs the usual volume of that minute, a "bar exists" flag, and NIFTY's path
  intra          at 09:30: the live TOP5 ranker's price/volume factors (computed by td.analytics.intraday.rank, the
                 live code), the stock's gap / move / VWAP distance / opening-range position in ATR units, their
                 cross-sectional ranks, the market (NIFTY gap and move, breadth) and overnight news
Labels (fwd): the log return from the 09:30 price to the close (> 0 = "up") and the same minus the day's median
stock (> 0 = "beat"). The 09:30 price is the close of the 09:29 bar, as in research/evaluate_intraday.py.

Not inputs: the ranker's OI, news and order-book factors — the history has none of them (they are zero in every
training row), so a network could not learn what they mean and would only be confused by them live.

`build(...)` makes the training panel (walk-forward folds over trading days, the daily context with an HMM cutoff
before the first test day); `day_features(...)` is the per-day feature code, shared by training and by the live
09:30 scoring in the analytics worker (td.neural.live), so the two cannot drift apart.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..analytics import intraday as rk
from ..analytics.context import default_curve
from ..analytics.state import HistoryContext, InstState
from ..research import quant as qm
from ..universe_data import SECTOR_OF
from . import dataset as ds
from . import minute as mn

log = logging.getLogger("td.neural.intraday")

LOCK_MIN = mn.FIRST                 # 09:30 = session minute 15
MIN_PRE, MIN_POST = 7, 30           # bars needed before / after the lock (evaluate_intraday's snapshot rule)
MAX_GAP = 0.25                      # |open / previous close − 1| above this: a split or bonus the bars don't adjust for
MAX_STALE_DAYS = 7                  # the previous daily bar may be at most this many calendar days old
MIN_STOCKS = 20                     # a day needs this many stocks
PURGE = 2                           # trading days between the last training row and a test fold (labels are same-day)
COST_PCT = 0.06                     # round trip in % (evaluate_intraday's default: 6 bps)
TOP_N = 5
MAX_PER_SECTOR = 2
CLOSE_S = 15 * 3600 + 30 * 60
LOCK_S = 9 * 3600 + 30 * 60
PV = ("rvol", "resid", "vwap", "orb", "sector")
BAR_KEYS: tuple[str, ...] = ("o", "h", "l", "c", "v", "on", "mkt")
INTRA_KEYS: tuple[str, ...] = (
    "z_rvol", "z_resid", "z_vwap", "z_orb", "z_sector", "ranker_pv",
    "gap_atr", "chg_atr", "vwap_atr", "or_pos", "range_atr", "beta",
    "cs_gap", "cs_chg", "cs_resid", "cs_rvol", "cs_turnover", "cs_atr_pct", "cs_range",
    "mkt_gap", "mkt_chg", "mkt_range", "breadth", "uni_chg",
    "on_news_n", "on_news_sent", "on_news_on",
)
HEADS: tuple[str, ...] = ("up", "beat")
HEAD_LABEL = {"up": "Up by the close", "beat": "Beats the day's median stock"}


# --------------------------------------------------------------------------- daily context per bar
def daily_context(df: pd.DataFrame, ref: pd.DataFrame | None) -> dict[str, np.ndarray]:
    """ATR(14), ADV(20) and 60-day beta to NIFTY as of each daily bar (inclusive) — load_context's formulas, which
    the live ranker uses, so a training row's context equals what the live worker would have loaded that morning."""
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    v = df["volume"].to_numpy(dtype=float)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()
    adv = pd.Series(v).rolling(20, min_periods=1).mean().to_numpy()
    beta = np.ones(c.size)
    if ref is not None and len(ref):
        r = pd.Series(c, index=df["ts"].to_numpy()).pct_change()
        rr = pd.Series(ref["close"].to_numpy(dtype=float), index=ref["ts"].to_numpy()).pct_change()
        j = pd.concat([r, rr], axis=1, join="inner").dropna()
        if len(j):
            cov = j.iloc[:, 0].rolling(60, min_periods=21).cov(j.iloc[:, 1])
            var = j.iloc[:, 1].rolling(60, min_periods=21).var()
            b = (cov / var.where(var > 0)).clip(0.2, 3.0)
            beta = b.reindex(df["ts"].to_numpy(), method="ffill").fillna(1.0).to_numpy()
    return {"day": mn.ist_day(df["ts"].to_numpy()), "close": c, "high": h, "low": l, "atr": atr, "adv": adv,
            "beta": beta}


def base_from_curve(base_cum: list[float] | None, adv: float) -> np.ndarray:
    """Usual volume of each of the first 15 bars from a cumulative curve (the live context's base_cum)."""
    cum = np.asarray(base_cum if base_cum and len(base_cum) > LOCK_MIN else default_curve(adv or 0.0), dtype=float)
    return np.diff(np.r_[0.0, cum[:LOCK_MIN]])


# --------------------------------------------------------------------------- one day's features
@dataclass
class DayInputs:
    """Everything one 09:30 decision needs, one entry per stock."""
    syms: list[str]
    bars: np.ndarray            # (n, 15, 5) o h l c v of the first 15 bars, NaN where a bar is missing
    base: np.ndarray            # (n, 15) usual volume of those bars
    pdc: np.ndarray
    atr: np.ndarray
    adv: np.ndarray
    beta: np.ndarray
    nifty: np.ndarray           # (15, 5)
    nifty_pdc: float
    nifty_atr: float
    news: np.ndarray            # (n, 3) overnight news: log count, tone, archive covers it
    vol: np.ndarray | None = None   # total volume at 09:30 when known better than the bars' sum (live feed)


def _ffill_price(bars: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Missing bars carry the previous close (or `start` before the first bar). Returns (filled o/h/l/c, on)."""
    on = np.isfinite(bars[..., 3])
    out = bars[..., :4].copy()
    last = np.asarray(start, dtype=float).copy()
    for k in range(bars.shape[1]):
        miss = ~on[:, k]
        out[miss, k, :] = last[miss, None]
        last = np.where(on[:, k], out[:, k, 3], last)
    return out, on


def _states(di: DayInputs) -> tuple[list[InstState], np.ndarray]:
    """InstStates as they stood at 09:30, built from the bars like evaluate_intraday.snapshot."""
    states, ok = [], np.zeros(len(di.syms), dtype=bool)
    for i, s in enumerate(di.syms):
        b = di.bars[i]
        on = np.isfinite(b[:, 3])
        if on.sum() < MIN_PRE or not (di.pdc[i] > 0):
            continue
        a = b[on]
        st = InstState(i, "NSE_EQ", s, "EQ", s, SECTOR_OF.get(s, "OTHER"))
        ctx = HistoryContext()
        ctx.atr = float(di.atr[i])
        ctx.atr_pct = ctx.atr / di.pdc[i] * 100
        ctx.adv = float(di.adv[i])
        ctx.beta = float(di.beta[i])
        ctx.pdc = float(di.pdc[i])
        cum = np.cumsum(np.nan_to_num(di.base[i]))
        ctx.base_cum = [float(x) for x in cum] + [float(cum[-1])]
        st.ctx = ctx
        st.pc = ctx.pdc
        st.open = float(a[0, 0])
        st.high = float(a[:, 1].max())
        st.low = float(a[:, 2].min())
        st.ltp = float(a[-1, 3])
        vsum = float(np.nansum(a[:, 4]))
        st.vol = int(max(vsum, float(di.vol[i]) if di.vol is not None else 0.0))
        typ = (a[:, 1] + a[:, 2] + a[:, 3]) / 3
        st.atp = float((typ * a[:, 4]).sum() / vsum) if vsum > 0 else st.ltp
        st.orh, st.orl = st.high, st.low
        if abs(st.open / st.pc - 1) > MAX_GAP:
            continue
        states.append(st)
        ok[i] = True
    return states, ok


def _rank_pm1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros(x.size)
    f = np.isfinite(x)
    if f.sum() > 1:
        r = pd.Series(x[f]).rank(pct=True, method="average").to_numpy()
        out[f] = 2 * r - 1
    return out


def day_features(di: DayInputs, cfg: rk.RankConfig) -> dict:
    """Features of one day's stocks at 09:30. Returns {"ok": bool (n,), and for the ok stocks in order:
    "bars" (k, 15, 7), "intra" (k, I), "ranker" [score, score_s, eligible, picked long, picked short] (k, 5),
    "liquid" (k,), "ltp" (k,), "sym" list}."""
    states, ok = _states(di)
    idx = np.flatnonzero(ok)
    k = idx.size
    empty = {"ok": ok, "bars": np.zeros((0, LOCK_MIN, len(BAR_KEYS)), np.float32),
             "intra": np.zeros((0, len(INTRA_KEYS)), np.float32), "ranker": np.zeros((0, 5)),
             "liquid": np.zeros(0, bool), "ltp": np.zeros(0), "sym": []}
    if k < 5:
        return empty | {"ok": np.zeros_like(ok)}
    n_on = np.isfinite(di.nifty[:, 3])
    if not n_on.any() or not (di.nifty_pdc > 0):
        return empty | {"ok": np.zeros_like(ok)}
    na = di.nifty[n_on]
    n_ltp = float(na[-1, 3])
    nifty_chg = (n_ltp / di.nifty_pdc - 1) * 100
    rows = rk.rank(states, {}, nifty_chg, float(LOCK_MIN), {}, cfg, LOCK_MIN)
    by = {r["symbol"]: r for r in rows}
    longs = {r["symbol"] for r in rk.pick(rows, cfg, "long")}
    shorts = {r["symbol"] for r in rk.pick(rows, cfg, "short")}

    pdc, atr = di.pdc[idx], di.atr[idx]
    af = np.where(atr > 0, atr / pdc, 0.02)
    px, on = _ffill_price(di.bars[idx], pdc)
    feats = np.zeros((k, LOCK_MIN, len(BAR_KEYS)), np.float32)
    for j in range(4):
        feats[:, :, j] = np.clip(np.log(px[:, :, j] / pdc[:, None]) / af[:, None], -10, 10)
    vol = np.nan_to_num(di.bars[idx, :, 4])
    vfe = np.log1p(vol) - np.log1p(np.nan_to_num(di.base[idx]))
    feats[:, :, 4] = np.where(on, np.clip(vfe, -6, 6), 0.0)
    feats[:, :, 5] = on
    naf = (di.nifty_atr / di.nifty_pdc) if di.nifty_atr > 0 else 0.01
    npx, _ = _ffill_price(di.nifty[None], np.array([di.nifty_pdc]))
    feats[:, :, 6] = np.clip(np.log(npx[0, :, 3] / di.nifty_pdc) / naf, -10, 10)[None, :]

    st = [s for s in states]
    w = cfg.weights
    r_ = [by[s.symbol] for s in st]
    f = {key: np.array([r["f"][key] for r in r_]) for key in PV}
    pv = sum(w[key] * f[key] for key in PV)
    ltp = np.array([s.ltp for s in st])
    opn = np.array([s.open for s in st])
    hi = np.array([s.high for s in st])
    lo = np.array([s.low for s in st])
    atp = np.array([s.atp for s in st])
    chg = np.log(ltp / pdc)
    gap = np.log(opn / pdc)
    rng = hi - lo
    beta = di.beta[idx]
    resid = chg - beta * math.log(n_ltp / di.nifty_pdc)
    rvol = np.array([r["rvol"] for r in r_])
    turnover = np.array([r["turnover"] for r in r_])
    chg_atr = chg / af
    intra = {
        **{f"z_{key}": f[key] / 3.0 for key in PV}, "ranker_pv": pv,
        "gap_atr": np.clip(gap / af, -10, 10), "chg_atr": np.clip(chg_atr, -10, 10),
        "vwap_atr": np.clip((ltp - atp) / np.where(atr > 0, atr, pdc * 0.02), -5, 5),
        "or_pos": np.where(rng > 0, 2 * (ltp - lo) / np.where(rng > 0, rng, 1) - 1, 0.0),
        "range_atr": np.clip(rng / np.where(atr > 0, atr, pdc * 0.02), 0, 10), "beta": beta - 1.0,
        "cs_gap": _rank_pm1(gap), "cs_chg": _rank_pm1(chg), "cs_resid": _rank_pm1(resid),
        "cs_rvol": _rank_pm1(np.log(np.clip(rvol, 0.05, 50))), "cs_turnover": _rank_pm1(np.log1p(turnover)),
        "cs_atr_pct": _rank_pm1(af), "cs_range": _rank_pm1(rng / np.where(atr > 0, atr, 1)),
        "mkt_gap": np.full(k, np.clip(math.log(float(na[0, 0]) / di.nifty_pdc) / naf, -10, 10)),
        "mkt_chg": np.full(k, np.clip(math.log(n_ltp / di.nifty_pdc) / naf, -10, 10)),
        "mkt_range": np.full(k, np.clip((float(na[:, 1].max()) - float(na[:, 2].min())) / di.nifty_pdc / naf, 0, 10)),
        "breadth": np.full(k, float(np.mean(np.sign(chg)))),
        "uni_chg": np.full(k, float(np.clip(np.median(chg_atr), -10, 10))),
        "on_news_n": di.news[idx, 0], "on_news_sent": di.news[idx, 1], "on_news_on": di.news[idx, 2],
    }
    X = np.column_stack([intra[key] for key in INTRA_KEYS]).astype(np.float32)
    ranker = np.column_stack([[r["score"] for r in r_], [r["score_s"] for r in r_],
                              [1.0 if r["eligible"] else 0.0 for r in r_],
                              [1.0 if s.symbol in longs else 0.0 for s in st],
                              [1.0 if s.symbol in shorts else 0.0 for s in st]])
    liquid = turnover >= cfg.min_turnover_cr
    return {"ok": ok, "bars": feats, "intra": np.nan_to_num(X, nan=0.0), "ranker": ranker, "liquid": liquid,
            "ltp": ltp, "sym": [s.symbol for s in st]}


# --------------------------------------------------------------------------- overnight news
def overnight_news(df: pd.DataFrame | None, symbol: str, day: np.ndarray, prev_day: np.ndarray,
                   start: float | None) -> np.ndarray:
    """(n, 3): headlines that arrived after the previous session's close and by 09:30 — log count, mean tone,
    and whether the archive covered the window."""
    out = np.zeros((day.size, 3))
    if start is None or day.size == 0:
        return out
    lo = prev_day.astype(np.int64) * 86400 - mn.IST_OFFSET + CLOSE_S
    hi = day.astype(np.int64) * 86400 - mn.IST_OFFSET + LOCK_S
    out[:, 2] = (lo >= start).astype(float)
    if df is None or df.empty:
        return out
    mine = df[df["symbol"] == symbol.upper()]
    if mine.empty:
        return out
    rt = mine["rt"].to_numpy(dtype=float)
    cs = np.r_[0.0, np.cumsum(mine["sign"].to_numpy(dtype=float))]
    a = np.searchsorted(rt, lo, side="right")
    b = np.searchsorted(rt, hi, side="right")
    cnt = (b - a).astype(float)
    sgn = cs[b] - cs[a]
    on = out[:, 2] > 0
    out[:, 0] = np.where(on, np.log1p(cnt), 0.0)
    out[:, 1] = np.where(on, sgn / (cnt + 1.0), 0.0)
    return out


# --------------------------------------------------------------------------- the training panel
@dataclass
class IPanel:
    daily: ds.Panel
    drow: np.ndarray                # daily-panel row of the previous close, per row
    syms: list[str]
    row_sym: np.ndarray             # index into syms
    row_day: np.ndarray             # IST day number
    row_cal: np.ndarray             # index into cal_days
    row_hmm_cal: np.ndarray         # last intraday-calendar index the daily HMM saw
    cal_days: np.ndarray
    bars: np.ndarray                # (rows, 15, len(BAR_KEYS)) float32
    intra: np.ndarray               # (rows, len(INTRA_KEYS)) float32
    fwd: np.ndarray                 # (rows, 2): log return 09:30 → close, and minus the day's median
    ret_pct: np.ndarray             # simple % return 09:30 → close
    ranker: np.ndarray              # (rows, 5) score, score_s, eligible, picked long, picked short
    liquid: np.ndarray
    sector: np.ndarray              # sector name per row
    folds: list[tuple[int, int]] = field(default_factory=list)
    test_start: int | None = None
    coverage: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    weights: dict = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return int(self.row_sym.size)

    def date(self, cal_idx: int) -> str:
        return ds.date_of(mn.day_of_epoch(int(self.cal_days[cal_idx])))


def _valid(t: pd.DataFrame) -> np.ndarray:
    return ((t["pre"].to_numpy() >= MIN_PRE) & (t["post"].to_numpy() >= MIN_POST)
            & np.isfinite(t["px_lock"].to_numpy(dtype=float)) & np.isfinite(t["close"].to_numpy(dtype=float)))


def build(frames: dict[str, pd.DataFrame], index_df: pd.DataFrame | None, tables: dict[str, pd.DataFrame], *,
          fundamentals: dict | None = None, news: pd.DataFrame | None = None, news_start: float | None = None,
          weights: dict | None = None, min_turnover_cr: float = 5.0,
          progress: Callable[[float, str], None] | None = None) -> IPanel:
    say = progress or (lambda f, m: None)
    if index_df is None or len(index_df) < 300:
        raise ValueError("NIFTY daily history is missing — the 09:30 model needs it")
    nt = tables.get("NIFTY")
    if nt is None or nt.empty:
        raise ValueError("NIFTY 1-minute history is missing (intraday/NIFTY.parquet) — the 09:30 model needs it")
    cfg = rk.RankConfig(min_turnover_cr=min_turnover_cr)
    if weights:
        cfg.weights.update({k: float(v) for k, v in weights.items() if k in cfg.weights})
    syms = sorted(s for s, df in frames.items() if not qm.is_index(s) and s in tables
                  and len(df) > qm.FEATURE_WARMUP + ds.SEQ_LEN)
    if len(syms) < MIN_STOCKS:
        raise ValueError(f"only {len(syms)} stocks have both daily history and 1-minute bars (need ≥ {MIN_STOCKS})")

    # provisional calendar: days with enough stocks whose daily features have warmed up
    n_days = nt["day"].to_numpy(dtype=np.int64)[_valid(nt) | (nt["pre"].to_numpy() >= 1)]
    counts: dict[int, int] = {}
    warm: dict[str, int] = {}
    for s in syms:
        dd = mn.ist_day(frames[s]["ts"].to_numpy())
        warm[s] = int(dd[min(qm.FEATURE_WARMUP + ds.SEQ_LEN, dd.size - 1)])
        t = tables[s]
        d = t["day"].to_numpy(dtype=np.int64)[_valid(t)]
        for x in d[d > warm[s]]:
            counts[int(x)] = counts.get(int(x), 0) + 1
    cal_days = np.array(sorted(d for d, c in counts.items() if c >= MIN_STOCKS and d in set(n_days.tolist())),
                        dtype=np.int64)
    if cal_days.size == 0:
        raise ValueError("no trading day has 1-minute bars for enough stocks and NIFTY")
    test_start, folds = qm._fold_bounds(cal_days.size, 0)
    # the daily context's HMM must not see the test periods: cut it 2·21 trading days before the first test day
    idx_days = mn.ist_day(index_df["ts"].to_numpy())
    j0 = int(np.searchsorted(idx_days, cal_days[test_start]))
    hmm_end_ts = int(index_df["ts"].iat[max(j0 - 2 * max(ds.HORIZONS) - 1, 0)])
    say(0.02, f"daily context for {len(syms)} stocks")
    P = ds.build({s: frames[s] for s in syms}, index_df, fundamentals=fundamentals, news=news,
                 news_start=news_start, minute={s: tables[s] for s in syms}, walk_forward=False,
                 hmm_end_ts=hmm_end_ts, progress=lambda f, m: say(0.02 + 0.4 * f, f"daily context: {m}"))
    psym = {s: i for i, s in enumerate(P.syms)}
    prow: dict[tuple[int, int], int] = {}
    pday = mn.ist_day(P.row_ts)
    for r in range(P.rows):
        prow[(int(P.row_sym[r]), int(pday[r]))] = r

    nctx = daily_context(index_df, None)
    n_by_day = {int(d): i for i, d in enumerate(nt["day"].to_numpy(dtype=np.int64))}
    n_bars = mn.first_bars(nt)
    cal_set = {int(d): i for i, d in enumerate(cal_days)}

    # gather per-stock rows: (day, stock) with a daily-context row for the previous close
    per_day: dict[int, list] = {}
    for si, s in enumerate(syms):
        if si % 10 == 0:
            say(0.45 + 0.15 * si / len(syms), f"1-minute rows {si}/{len(syms)}")
        if s not in psym:
            continue
        t = tables[s]
        ok = _valid(t)
        d = t["day"].to_numpy(dtype=np.int64)
        keep = np.flatnonzero(ok & np.isin(d, cal_days))
        if keep.size == 0:
            continue
        ctx = daily_context(frames[s], index_df)
        j = np.searchsorted(ctx["day"], d[keep]) - 1
        good = (j >= 0) & (d[keep] - ctx["day"][np.maximum(j, 0)] <= MAX_STALE_DAYS)
        keep, j = keep[good], j[good]
        bars = mn.first_bars(t)
        base = t[mn.BASE_COLS].to_numpy(dtype=float)
        px = t["px_lock"].to_numpy(dtype=float)
        cl = t["close"].to_numpy(dtype=float)
        nn_ = overnight_news(news, s, d[keep], ctx["day"][j], news_start)
        for q, (ti, jj) in enumerate(zip(keep, j)):
            r = prow.get((psym[s], int(ctx["day"][jj])))
            if r is None:
                continue
            b = base[ti]
            if not np.isfinite(b).all():
                b = base_from_curve(None, ctx["adv"][jj])
            per_day.setdefault(int(d[ti]), []).append(
                (s, r, bars[ti], b, ctx["close"][jj], ctx["atr"][jj], ctx["adv"][jj], ctx["beta"][jj], nn_[q],
                 px[ti], cl[ti]))

    out: dict[str, list] = {k: [] for k in ("sym", "drow", "day", "bars", "intra", "ranker", "liquid", "ret", "ltp")}
    days_sorted = sorted(per_day)
    for qi, dday in enumerate(days_sorted):
        if qi % 50 == 0:
            say(0.60 + 0.35 * qi / max(len(days_sorted), 1), f"09:30 snapshots {qi}/{len(days_sorted)} days")
        items = per_day[dday]
        if len(items) < MIN_STOCKS or dday not in n_by_day:
            continue
        nj = int(np.searchsorted(nctx["day"], dday)) - 1
        if nj < 0:
            continue
        di = DayInputs(syms=[x[0] for x in items], bars=np.stack([x[2] for x in items]),
                       base=np.stack([x[3] for x in items]), pdc=np.array([x[4] for x in items]),
                       atr=np.array([x[5] for x in items]), adv=np.array([x[6] for x in items]),
                       beta=np.array([x[7] for x in items]), nifty=n_bars[n_by_day[dday]],
                       nifty_pdc=float(nctx["close"][nj]), nifty_atr=float(nctx["atr"][nj]),
                       news=np.stack([x[8] for x in items]))
        res = day_features(di, cfg)
        okk = np.flatnonzero(res["ok"])
        if okk.size < MIN_STOCKS:
            continue
        px = np.array([items[i][9] for i in okk])
        cl = np.array([items[i][10] for i in okk])
        out["sym"] += [items[i][0] for i in okk]
        out["drow"] += [items[i][1] for i in okk]
        out["day"] += [dday] * okk.size
        out["bars"].append(res["bars"])
        out["intra"].append(res["intra"])
        out["ranker"].append(res["ranker"])
        out["liquid"].append(res["liquid"])
        out["ret"].append(np.log(cl / px))
        out["ltp"].append(px)
    if not out["sym"]:
        raise ValueError("no usable 09:30 rows")
    sym_idx = {s: i for i, s in enumerate(syms)}
    row_sym = np.array([sym_idx[s] for s in out["sym"]])
    row_day = np.array(out["day"], dtype=np.int64)
    row_cal = np.array([cal_set[int(x)] for x in row_day])
    ret = np.concatenate(out["ret"])
    med = pd.Series(ret).groupby(row_cal).transform("median").to_numpy()
    fwd = np.column_stack([ret, ret - med])
    drow = np.array(out["drow"])
    hmm_day = mn.ist_day(P.cal[P.row_hmm_cal[drow]])
    row_hmm_cal = np.searchsorted(cal_days, hmm_day, side="right") - 1
    liquid = np.concatenate(out["liquid"])
    cover = {"stocks": len(set(out["sym"])), "rows": int(row_sym.size), "days": int(np.unique(row_cal).size),
             "first_day": ds.date_of(mn.day_of_epoch(int(cal_days[0]))),
             "last_day": ds.date_of(mn.day_of_epoch(int(cal_days[-1]))),
             "liquid_rows": round(float(liquid.mean()), 3), **{f"daily_{k}": v for k, v in P.coverage.items()}}
    notes = list(P.notes)
    say(1.0, f"{row_sym.size:,} rows over {cover['days']} days")
    return IPanel(daily=P, drow=drow, syms=syms, row_sym=row_sym, row_day=row_day, row_cal=row_cal,
                  row_hmm_cal=row_hmm_cal, cal_days=cal_days, bars=np.concatenate(out["bars"]),
                  intra=np.concatenate(out["intra"]), fwd=fwd, ret_pct=(np.exp(ret) - 1) * 100,
                  ranker=np.concatenate(out["ranker"]), liquid=liquid,
                  sector=np.array([SECTOR_OF.get(s, "OTHER") for s in out["sym"]]), folds=folds,
                  test_start=test_start, coverage=cover, notes=notes, weights=dict(cfg.weights))


# --------------------------------------------------------------------------- one row of numbers (trees, linear)
def flat_inputs(seq: np.ndarray, static: np.ndarray, bars: np.ndarray, intra: np.ndarray) -> np.ndarray:
    """The network's four inputs as one row of numbers per stock — what the linear and tree models read. Built
    from the same arrays in training and live, so both see identical numbers."""
    return np.hstack([seq[:, -1, :], static, intra, bars[:, -1, :], bars[:, :, 3], bars[:, :, 4]]).astype(np.float32)


# --------------------------------------------------------------------------- picks from scores
def top_bottom(score: np.ndarray, liquid: np.ndarray, sector: np.ndarray, n: int = TOP_N,
               per_sector: int = MAX_PER_SECTOR) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the n best and n worst liquid stocks by score, at most `per_sector` per sector on each side."""
    def take(order):
        got, per = [], {}
        for i in order:
            if not liquid[i]:
                continue
            if per.get(sector[i], 0) >= per_sector:
                continue
            per[sector[i]] = per.get(sector[i], 0) + 1
            got.append(i)
            if len(got) >= n:
                break
        return np.array(got, dtype=int)
    order = np.argsort(-score, kind="stable")
    longs = take(order)
    shorts = take([i for i in order[::-1] if i not in set(longs.tolist())])
    return longs, shorts
