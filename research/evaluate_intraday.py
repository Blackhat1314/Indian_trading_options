#!/usr/bin/env python3
"""Walk-forward evaluation of the 09:30 intraday ranker on the Parquet lake.

It replays every day that has 1-minute bars, rebuilds the exact pre-open context the live
analytics would have had (ATR, beta, ADV and the 14-session volume curve — all from data
*before* that day), snapshots the universe at the lock time, runs the *same* ranker code
(td.analytics.intraday.rank / pick) and scores the picks against lock→close returns.

What it reports (per day and summarised):
  IC            Spearman rank correlation, score vs lock→close return, whole universe
  hits          long picks that finished in the day's top-5 / top-20 (lock→close)
  gainers       long picks that were in the day's top-5 gainers by full-day % change
  long / short  average pick return net of round-trip costs, % green, excess vs universe
  factor IC     the same IC for each factor on its own (which inputs actually carry signal)

History only has prices and volume, so the OI, news and order-book factors are zero here;
the live system has them. Treat this as a test of the price/volume core, not of everything.

  python research/evaluate_intraday.py                         # lake from .env (sim or dhan)
  python research/evaluate_intraday.py --lake data/lake/dhan --days 250 --cost-bps 8
  python research/evaluate_intraday.py --fit                   # IC-weighted weights, train/test split
  python research/evaluate_intraday.py --fit --write-weights   # save them for the live ranker

The summary is also written to data/research/intraday_eval.json (shown on the SYSTEM desk).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from td.analytics import intraday  # noqa: E402
from td.analytics.context import default_curve  # noqa: E402
from td.analytics.state import SESSION, HistoryContext, InstState, build_base_curve  # noqa: E402
from td.config import settings  # noqa: E402
from td.lake import Lake  # noqa: E402
from td.timeutil import parse_hms  # noqa: E402
from td.universe_data import FNO_STOCKS, SECTOR_OF  # noqa: E402

IST_OFFSET = 19800
PRICE_VOLUME = ["rvol", "resid", "vwap", "orb", "sector"]       # factors that exist in history
ALL_FACTORS = list(intraday.DEFAULT_WEIGHTS)


# --------------------------------------------------------------------------- loading
@dataclass
class SymData:
    symbol: str
    sector: str
    d_dates: np.ndarray            # daily bar dates (datetime64[D])
    d_ohlcv: np.ndarray            # (n, 5)
    m_days: dict                   # date -> (minutes int array, ohlcv (k,5))


def _ist_date(ts: np.ndarray) -> np.ndarray:
    return ((ts + IST_OFFSET) // 86400).astype("datetime64[D]")


def _session_minute(ts: np.ndarray) -> np.ndarray:
    sod = (ts + IST_OFFSET) % 86400
    return ((sod - (9 * 3600 + 15 * 60)) // 60).astype(int)


def load_symbol(lake: Lake, sym: str, minute_rows: int) -> SymData | None:
    d = lake.read("D", sym)
    m = lake.read("1m", sym, last_n=minute_rows)
    if d.empty or m.empty:
        return None
    d = d.sort_values("ts")
    m = m.sort_values("ts")
    d_dates = _ist_date(d["ts"].to_numpy())
    mts = m["ts"].to_numpy()
    mdates = _ist_date(mts)
    mins = _session_minute(mts)
    ohlcv = m[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    keep = (mins >= 0) & (mins < SESSION)
    mdates, mins, ohlcv = mdates[keep], mins[keep], ohlcv[keep]
    days: dict = {}
    if len(mdates):
        cuts = np.flatnonzero(np.diff(mdates.astype(np.int64))) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(mdates)]):
            days[mdates[a]] = (mins[a:b], ohlcv[a:b])
    return SymData(sym, SECTOR_OF.get(sym, "OTHER"), d_dates, d[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float), days)


# --------------------------------------------------------------------------- context
def context_for(s: SymData, day: np.datetime64, ref: SymData | None) -> HistoryContext | None:
    """Pre-open facts for `day`, using only bars strictly before it (no look-ahead)."""
    i = int(np.searchsorted(s.d_dates, day))
    if i < 21:
        return None
    o, h, l, c, v = (s.d_ohlcv[:i, k] for k in range(5))
    ctx = HistoryContext()
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    ctx.atr = float(tr[-14:].mean())
    ctx.atr_pct = ctx.atr / c[-1] * 100 if c[-1] else 0.0
    ctx.adv = float(v[-20:].mean())
    ctx.pdh, ctx.pdl, ctx.pdc = float(h[-1]), float(l[-1]), float(c[-1])
    ctx.hh20, ctx.ll20 = float(h[-20:].max()), float(l[-20:].min())
    if ref is not None:
        j = int(np.searchsorted(ref.d_dates, day))
        a = pd.Series(c, index=s.d_dates[:i]).pct_change()
        b = pd.Series(ref.d_ohlcv[:j, 3], index=ref.d_dates[:j]).pct_change()
        jb = pd.concat([a, b], axis=1, join="inner").dropna().tail(60)
        if len(jb) > 20 and jb.iloc[:, 1].var() > 0:
            ctx.beta = float(np.clip(jb.cov().iloc[0, 1] / jb.iloc[:, 1].var(), 0.2, 3.0))
    prior = sorted(k for k in s.m_days if k < day)[-14:]
    vols = []
    for k in prior:
        mins, arr = s.m_days[k]
        full = np.zeros(SESSION)
        full[mins] = arr[:, 4]
        if len(mins) >= SESSION * 0.9:
            vols.append(full.tolist())
    ctx.base_cum = build_base_curve(vols) if vols else default_curve(ctx.adv)
    return ctx


def snapshot(s: SymData, day: np.datetime64, ctx: HistoryContext, lock_m: int, or_m: int):
    """InstState as it stood at the lock, plus what happened afterwards."""
    if day not in s.m_days:
        return None
    mins, arr = s.m_days[day]
    pre = mins < lock_m
    if pre.sum() < max(3, lock_m // 2) or (~pre).sum() < 30:
        return None
    a = arr[pre]
    st = InstState(0, "NSE_EQ", s.symbol, "EQ", s.symbol, s.sector)
    st.ctx = ctx
    st.pc = ctx.pdc
    st.open = float(a[0, 0])
    st.high = float(a[:, 1].max())
    st.low = float(a[:, 2].min())
    st.ltp = float(a[-1, 3])
    st.vol = int(a[:, 4].sum())
    typical = (a[:, 1] + a[:, 2] + a[:, 3]) / 3
    st.atp = float((typical * a[:, 4]).sum() / a[:, 4].sum()) if a[:, 4].sum() > 0 else st.ltp
    orw = mins[pre] < or_m
    if orw.any():
        st.orh = float(a[orw, 1].max())
        st.orl = float(a[orw, 2].min())
    post = arr[~pre]
    close = float(arr[-1, 3])
    return st, {
        "fwd": (close / st.ltp - 1) * 100,                       # lock -> close
        "day": (close / st.pc - 1) * 100 if st.pc else 0.0,      # prev close -> close
        "mfe": (post[:, 1].max() / st.ltp - 1) * 100,
        "mae": (post[:, 2].min() / st.ltp - 1) * 100,
    }


# --------------------------------------------------------------------------- scoring
def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def rescore(rows: list[dict], weights: dict) -> None:
    """Recompute long/short scores from stored factor z-scores (same formula as the ranker)."""
    for r in rows:
        f = r["f"]
        direction = sum(weights[k] * f[k] for k in ALL_FACTORS if k != "rvol")
        r["score"] = direction + weights["rvol"] * f["rvol"]
        r["score_s"] = -direction + weights["rvol"] * f["rvol"]


def evaluate_day(rows: list[dict], out: dict, cfg: intraday.RankConfig, cost: float) -> dict:
    fwd = {r["symbol"]: out[r["symbol"]]["fwd"] for r in rows}
    dayret = {r["symbol"]: out[r["symbol"]]["day"] for r in rows}
    syms = list(fwd)
    f_arr = np.array([fwd[s] for s in syms])
    order = np.argsort(-f_arr)
    top5 = {syms[i] for i in order[:5]}
    top20 = {syms[i] for i in order[:20]}
    gainers = {s for s, _ in sorted(dayret.items(), key=lambda kv: -kv[1])[:5]}
    longs = intraday.pick(rows, cfg, "long")
    shorts = intraday.pick(rows, cfg, "short")
    lr = [fwd[r["symbol"]] for r in longs]
    sr = [-fwd[r["symbol"]] for r in shorts]
    uni = float(f_arr.mean())
    return {
        "n": len(rows),
        "ic": spearman([r["score"] for r in rows], [fwd[r["symbol"]] for r in rows]),
        "factor_ic": {k: spearman([r["f"][k] for r in rows], [fwd[r["symbol"]] for r in rows]) for k in ALL_FACTORS},
        "longs": [r["symbol"] for r in longs],
        "shorts": [r["symbol"] for r in shorts],
        "long_ret": float(np.mean(lr)) - cost if lr else None,
        "short_ret": float(np.mean(sr)) - cost if sr else None,
        "long_green": sum(x > cost for x in lr),
        "short_green": sum(x > cost for x in sr),
        "universe": uni,
        "hits_top5": sum(r["symbol"] in top5 for r in longs),
        "hits_top20": sum(r["symbol"] in top20 for r in longs),
        "hits_gainers": sum(r["symbol"] in gainers for r in longs),
        "long_mfe": float(np.mean([out[r["symbol"]]["mfe"] for r in longs])) if longs else None,
    }


def _nanmean(xs) -> float:
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def summarise(days: list[dict], top_n: int) -> dict:
    if not days:
        return {}
    ics = np.array([d["ic"] for d in days if math.isfinite(d["ic"])])
    lr = [d["long_ret"] for d in days if d["long_ret"] is not None]
    sr = [d["short_ret"] for d in days if d["short_ret"] is not None]
    n_long = sum(len(d["longs"]) for d in days)
    n_short = sum(len(d["shorts"]) for d in days)
    avg_n = float(np.mean([d["n"] for d in days]))
    fic = {k: _nanmean([d["factor_ic"][k] for d in days]) for k in ALL_FACTORS}
    return {
        "days": len(days),
        "universe_size": round(avg_n),
        "mean_ic": float(ics.mean()) if len(ics) else None,
        "ic_t": float(ics.mean() / ics.std(ddof=1) * math.sqrt(len(ics))) if len(ics) > 2 and ics.std(ddof=1) > 0 else None,
        "ic_positive_pct": float((ics > 0).mean() * 100) if len(ics) else None,
        "long_days": len(lr), "short_days": len(sr),
        "abstain_long_pct": float(sum(1 for d in days if not d["longs"]) / len(days) * 100),
        "long_avg_net": float(np.mean(lr)) if lr else None,
        "short_avg_net": float(np.mean(sr)) if sr else None,
        "long_green_pct": float(sum(d["long_green"] for d in days) / n_long * 100) if n_long else None,
        "short_green_pct": float(sum(d["short_green"] for d in days) / n_short * 100) if n_short else None,
        "universe_avg": float(np.mean([d["universe"] for d in days])),
        "top5_hits_per_day": float(np.mean([d["hits_top5"] for d in days])),
        "top20_hit_rate": float(sum(d["hits_top20"] for d in days) / n_long * 100) if n_long else None,
        "gainer_hits_per_day": float(np.mean([d["hits_gainers"] for d in days])),
        "random_top5_hits_per_day": top_n * 5 / avg_n if avg_n else None,
        "random_top20_rate": 20 / avg_n * 100 if avg_n else None,
        "long_mfe_avg": _nanmean([d["long_mfe"] for d in days]) if n_long else None,
        "factor_ic": fic,
    }


def ic_weights(factor_ic: dict, base: dict) -> dict:
    """Weights proportional to positive mean IC for the price/volume factors; others unchanged."""
    budget = sum(base[k] for k in PRICE_VOLUME)
    pos = {k: max(0.0, factor_ic.get(k) or 0.0) for k in PRICE_VOLUME}
    tot = sum(pos.values())
    w = dict(base)
    if tot <= 0:
        return w
    for k in PRICE_VOLUME:
        w[k] = round(budget * pos[k] / tot, 4)
    return w


# --------------------------------------------------------------------------- main
def clean(o):
    """JSON-safe: NaN/inf -> null (browsers reject NaN), numpy scalars -> Python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def fmt(v, d=2, pct=False, sign=False):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s + ("%" if pct else "")


def print_summary(title: str, s: dict, cost_bps: float) -> None:
    print(f"\n== {title} ==")
    if not s:
        print("  no days evaluated")
        return
    print(f"  days {s['days']}  ·  universe ≈ {s['universe_size']} stocks  ·  costs {cost_bps:.0f} bps round trip")
    print(f"  mean rank IC        {fmt(s['mean_ic'], 3, sign=True)}   t={fmt(s['ic_t'], 2)}   positive on {fmt(s['ic_positive_pct'], 0)}% of days")
    print(f"  long picks          avg {fmt(s['long_avg_net'], 2, True, True)} net   green {fmt(s['long_green_pct'], 0)}%   abstained {fmt(s['abstain_long_pct'], 0)}% of days")
    print(f"  short picks         avg {fmt(s['short_avg_net'], 2, True, True)} net   green {fmt(s['short_green_pct'], 0)}%")
    print(f"  universe avg        {fmt(s['universe_avg'], 2, True, True)} (lock → close)")
    print(f"  top-5 hits / day    {fmt(s['top5_hits_per_day'], 2)}   (random ≈ {fmt(s['random_top5_hits_per_day'], 2)})")
    print(f"  in top-20           {fmt(s['top20_hit_rate'], 0)}% of picks   (random ≈ {fmt(s['random_top20_rate'], 0)}%)")
    print(f"  day's top-5 gainers {fmt(s['gainer_hits_per_day'], 2)} picks / day")
    print("  factor IC           " + "  ".join(f"{k} {fmt(v, 3, sign=True)}" for k, v in s["factor_ic"].items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lake", type=Path, default=None, help="lake root (default: from .env DATA_SOURCE)")
    ap.add_argument("--days", type=int, default=120, help="most recent N days with 1-minute data")
    ap.add_argument("--lock", default=settings.lock_time, help="lock time HH:MM[:SS] (default from .env)")
    ap.add_argument("--or-minutes", type=int, default=settings.or_minutes)
    ap.add_argument("--top", type=int, default=settings.top_n)
    ap.add_argument("--min-turnover", type=float, default=settings.min_turnover_cr)
    ap.add_argument("--cost-bps", type=float, default=6.0, help="round-trip cost in basis points (brokerage+STT+slippage)")
    ap.add_argument("--weights", type=Path, default=Path(settings.data_dir) / "config" / "ranker_weights.json")
    ap.add_argument("--fit", action="store_true", help="fit IC-weighted weights on the first 60%% of days, test on the rest")
    ap.add_argument("--write-weights", action="store_true", help="with --fit: save fitted weights for the live ranker")
    ap.add_argument("--out", type=Path, default=Path(settings.data_dir) / "research" / "intraday_eval.json")
    ap.add_argument("--quiet", action="store_true", help="no per-day lines")
    a = ap.parse_args()

    lake = Lake(a.lake) if a.lake else Lake()
    lock = parse_hms(a.lock)
    lock_m = (lock.hour * 60 + lock.minute) - (9 * 60 + 15)
    cost = a.cost_bps / 100.0
    t0 = time.time()

    syms_1m = set(lake.symbols("1m"))
    universe = [s for s, *_ in FNO_STOCKS if s in syms_1m]
    if not universe:
        print(f"No 1-minute data in {lake.root}. Run: python backfill/backfill.py run --intervals D,1 --years 1"
              "  (or `backfill.py sim` for a synthetic lake)")
        return 1
    rows_needed = (a.days + 20) * SESSION
    ref = load_symbol(lake, "NIFTY", rows_needed)
    data = [x for x in (load_symbol(lake, s, rows_needed) for s in universe) if x]
    all_days = sorted({d for x in data for d in x.m_days})
    ref_days = set(ref.m_days) if ref else set(all_days)
    days = [d for d in all_days if d in ref_days][-a.days:]
    print(f"lake {lake.root}  ·  {len(data)} stocks with 1m+daily bars  ·  {len(days)} days "
          f"{days[0] if days else ''} → {days[-1] if days else ''}  ·  lock {a.lock}  (loaded in {time.time() - t0:.1f}s)")

    weights = dict(intraday.DEFAULT_WEIGHTS)
    if a.weights.exists():
        try:
            weights.update({k: float(v) for k, v in json.loads(a.weights.read_text()).items() if k in weights})
            print(f"weights from {a.weights}")
        except (ValueError, OSError):
            print(f"ignoring unreadable weights file {a.weights}")
    cfg = intraday.RankConfig(weights=dict(weights), top_n=a.top, min_turnover_cr=a.min_turnover)

    per_day: list[tuple[str, list[dict], dict]] = []
    records: list[dict] = []
    for day in days:
        states, outcomes = [], {}
        for x in data:
            ctx = context_for(x, day, ref)
            if ctx is None:
                continue
            snap = snapshot(x, day, ctx, lock_m, a.or_minutes)
            if snap is None:
                continue
            st, out = snap
            states.append(st)
            outcomes[x.symbol] = out
        if len(states) < 20:
            continue
        nifty_chg = 0.0
        if ref is not None:
            rctx = context_for(ref, day, None)
            rs = snapshot(ref, day, rctx, lock_m, a.or_minutes) if rctx else None
            if rs:
                nifty_chg = rs[0].chg
        rows = intraday.rank(states, {}, nifty_chg, float(lock_m), {}, cfg, a.or_minutes)
        res = evaluate_day(rows, outcomes, cfg, cost)
        res["date"] = str(day)
        records.append(res)
        per_day.append((str(day), rows, outcomes))
        if not a.quiet:
            print(f"  {day}  n={res['n']:3d}  IC {fmt(res['ic'], 3, sign=True):>7}  long {fmt(res['long_ret'], 2, True, True):>7} "
                  f"short {fmt(res['short_ret'], 2, True, True):>7}  uni {fmt(res['universe'], 2, True, True):>7}  "
                  f"top5 {res['hits_top5']}  top20 {res['hits_top20']}  gainers {res['hits_gainers']}  "
                  f"{' '.join(res['longs'])}")

    summary = summarise(records, a.top)
    print_summary("RESULT (current weights)", summary, a.cost_bps)

    fit = None
    if a.fit and len(per_day) >= 20:
        split = int(len(per_day) * 0.6)
        train = summarise(records[:split], a.top)
        fitted = ic_weights(train["factor_ic"], weights)
        test_base, test_fit = [], []
        for (d, rows, outc) in per_day[split:]:
            rescore(rows, weights)
            test_base.append(evaluate_day(rows, outc, cfg, cost) | {"date": d})
            cfg_f = intraday.RankConfig(weights=fitted, top_n=a.top, min_turnover_cr=a.min_turnover)
            rescore(rows, fitted)
            test_fit.append(evaluate_day(rows, outc, cfg_f, cost) | {"date": d})
        sb, sf = summarise(test_base, a.top), summarise(test_fit, a.top)
        print(f"\nfitted on {split} days (IC-weighted, price/volume factors only):")
        print("  " + "  ".join(f"{k} {weights[k]:.2f}→{fitted[k]:.2f}" for k in PRICE_VOLUME))
        print_summary(f"TEST {len(test_base)} days · current weights", sb, a.cost_bps)
        print_summary(f"TEST {len(test_fit)} days · fitted weights", sf, a.cost_bps)
        fit = {"weights": fitted, "train_days": split, "test_current": sb, "test_fitted": sf}
        better = (sf.get("mean_ic") or -1) > (sb.get("mean_ic") or -1)
        if a.write_weights:
            if better:
                a.weights.parent.mkdir(parents=True, exist_ok=True)
                a.weights.write_text(json.dumps(fitted, indent=2))
                print(f"\nwrote {a.weights} — restart the analytics worker to use it")
            else:
                print("\nfitted weights did not beat the current ones out of sample — not written")
    elif a.fit:
        print("\n--fit needs at least 20 evaluated days")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(clean({
        "generated": time.time(), "lake": str(lake.root), "lock": a.lock, "cost_bps": a.cost_bps,
        "weights": weights, "factors_available": PRICE_VOLUME, "summary": summary, "fit": fit,
        "days": records[-250:],
    }), indent=1))
    print(f"\nsaved {a.out}  ({time.time() - t0:.1f}s)")
    print("Note: OI, news and order-book factors are not in the history and score zero here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
