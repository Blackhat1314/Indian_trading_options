"""Backtests of the paper trader on the data lake — the evidence the PAPER desk shows.

    python -m td.paper.backtest                        # both strategies, default capital / leverage
    python -m td.paper.backtest --intraday --capital 500000 --leverage 3
    python -m td.paper.backtest --swing --from 2023-01-01

Intraday is walk-forward: the models are re-fitted every quarter on everything before that quarter and only
then used to trade it, so no trade is chosen by a model that saw its outcome. Each day replays the same rules
as the live engine (decide at 09:30 and 10:00, edge-over-cost filter, risk-based size, leverage cap, daily loss
limit) with costs charged on the actual order sizes. Swing replays its fixed factor rules (no fitting).
Reports go to data/paper/backtest/<kind>.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings
from ..lake import Lake
from . import intraday as I
from .costs import charges
from .sim import REASONS, STOP

log = logging.getLogger("td.paper.backtest")
STOP_EXTRA_BPS = 4.0


def report_dir() -> Path:
    return Path(settings.data_dir) / "paper" / "backtest"


def _stats(eq: pd.Series, trades: pd.DataFrame, capital: float) -> dict:
    if eq.empty:
        return {}
    v = eq.to_numpy(float)
    r = np.diff(np.r_[capital, v]) / np.r_[capital, v[:-1]]
    yrs = max(len(v) / 252, 1e-9)
    dd = v / np.maximum.accumulate(np.r_[capital, v])[1:] - 1
    out = {"start": int(eq.index[0]), "end": int(eq.index[-1]), "days": int(len(v)),
           "return": float(v[-1] / capital - 1), "cagr": float((v[-1] / capital) ** (1 / yrs) - 1),
           "sharpe": float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0,
           "max_dd": float(dd.min()), "trades": int(len(trades))}
    if len(trades):
        pnl = trades["pnl"]
        out.update(win_rate=float((pnl > 0).mean()), avg_pnl=float(pnl.mean()),
                   avg_ret=float(trades["ret_net"].mean()) if "ret_net" in trades else None,
                   profit_factor=float(pnl[pnl > 0].sum() / max(-pnl[pnl < 0].sum(), 1e-9)),
                   charges=float(trades["charges"].sum()) if "charges" in trades else None)
    return out


# ============================================================================ intraday
def walk_forward(ds: pd.DataFrame, start: int, every_months: int = 3) -> pd.DataFrame:
    """Out-of-sample predictions pred_L / pred_S for every row dated >= start."""
    from .train import train
    ds = ds.copy()
    ds["pred_L"] = np.nan
    ds["pred_S"] = np.nan
    month = (ds["date"] // 100).to_numpy()
    months = sorted(set(month[month >= start // 100]))
    for i in range(0, len(months), every_months):
        block = months[i:i + every_months]
        first = block[0] * 100 + 1
        prior = np.sort(ds.loc[ds["date"] < first, "date"].unique())
        if len(prior) < 200:
            continue
        t = time.time()
        models = train(ds, until=int(prior[-2]))                    # one-day embargo
        te = np.isin(month, block)
        for m in I.DECISIONS:
            sel = te & (ds["m"].to_numpy() == m)
            sub = ds[sel]
            for key in ("L", "S"):
                ds.loc[sel, f"pred_{key}"] = models[f"m{m}_{key}"].predict(sub)
        log.info("walk-forward %s..%s fitted in %.0fs", block[0], block[-1], time.time() - t)
    return ds


def replay_intraday(ds: pd.DataFrame, capital: float, leverage: float, start: int, end: int | None = None,
                    monitor_n: int | None = None, monitors_out: dict | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Day-by-day book with compounding equity, using pred_L / pred_S already in ds.

    monitor_n: the edge monitor (I.EdgeMonitor) — every signal is shadow-traded at a reference size; money goes on
    the signals only while the shadow record of the last `monitor_n` signals made money after costs."""
    monitors = {m: I.EdgeMonitor(I.MONITOR_N if monitor_n is None else monitor_n) for m in I.DECISIONS}
    d = ds[(ds["date"] >= start) & np.isfinite(ds["pred_L"])]
    if end:
        d = d[d["date"] < end]
    d = d[I.liquid(d)]
    equity, curve, book = capital, {}, []
    price45 = d[d["m"] == 45].set_index(["date", "symbol"])["price"]
    for day, g in d.groupby("date"):
        day_pnl, open_gross, taken = 0.0, [], set()
        trading_on = {m: mon.allows() for m, mon in monitors.items()}
        shadows: dict[int, list[float]] = {m: [] for m in I.DECISIONS}
        for m in I.DECISIONS:
            gm = g[g["m"] == m]
            if gm.empty:
                continue
            if m != min(I.DECISIONS):
                # daily loss limit: realised + marked-to-market at the decision minute
                mtm = 0.0
                for t in book:
                    if t["date"] != day:
                        continue
                    if t["exit_min"] < m:
                        mtm += t["pnl"]
                    else:
                        px = price45.get((day, t["symbol"]))
                        if px is not None and np.isfinite(px):
                            mtm += t["side"] * (px - t["entry"]) * t["qty"]
                if mtm <= -I.DAILY_LOSS_LIMIT * equity:
                    continue
                open_gross = [t for t in open_gross if t[1] > m]
            budget = leverage * equity - sum(x[0] for x in open_gross)
            props = []
            for side, key in ((1, "L"), (-1, "S")):
                price = gm["entry"].to_numpy(float)
                risk_pct = gm[f"risk_{key}"].to_numpy(float)
                pred = gm[f"pred_{key}"].to_numpy(float)
                rows = []
                for i in range(len(gm)):
                    sym = gm["symbol"].iat[i]
                    if sym in taken or not (np.isfinite(pred[i]) and np.isfinite(risk_pct[i]) and risk_pct[i] > 0):
                        continue
                    notional = min(I.RISK_PER_TRADE * equity / risk_pct[i], I.MAX_POSITION_FRAC * leverage * equity)
                    edge = pred[i] - I.cost_estimate(notional)
                    if edge > I.MIN_EDGE[m]:
                        rows.append((edge, i, notional))
                rows.sort(reverse=True)
                props += [(e, i, n, side, key) for e, i, n in rows[:I.MAX_PER_SIDE]]
            props.sort(reverse=True)
            for e, i, n, side, key in props:                       # shadow record at a reference ₹1 lakh size
                y = float(gm.iloc[i][f"y_{key}"])
                shadows[m].append(y - I.cost_estimate(100_000.0))
            if not trading_on[m]:
                continue
            want = sum(p[2] for p in props)
            scale = min(1.0, budget / want) if want > 0 else 0.0
            for edge, i, notional, side, key in props:
                r = gm.iloc[i]
                if r["symbol"] in taken:
                    continue
                px = float(r["entry"])
                qty = int(notional * scale / px)
                if qty <= 0:
                    continue
                y = float(r[f"y_{key}"])
                exit_px = px * (1 + side * y)
                reason = int(r[f"reason_{key}"])
                slip = (2 * I.SLIP_BPS + (STOP_EXTRA_BPS if reason == STOP else 0)) / 1e4 * qty * px
                buy, sell = (qty * px, qty * exit_px) if side > 0 else (qty * exit_px, qty * px)
                ch = charges(buy, sell, "intraday")["total"]
                pnl = side * (exit_px - px) * qty - ch - slip
                taken.add(r["symbol"])
                open_gross.append((qty * px, int(r[f"exit_min_{key}"])))
                book.append({"date": int(day), "symbol": r["symbol"], "side": side, "m": m, "decision": I.DECISIONS[m],
                             "qty": qty, "entry": px, "exit": exit_px, "exit_min": int(r[f"exit_min_{key}"]),
                             "why_exit": REASONS.get(reason, "?"), "pred": float(r[f"pred_{key}"]), "edge": float(edge),
                             "risk_pct": float(r[f"risk_{key}"]), "charges": ch, "slippage": slip, "pnl": pnl,
                             "ret_net": pnl / (qty * px), "setup": I.setup_label(r, side)})
                day_pnl += pnl
        equity += day_pnl
        curve[int(day)] = equity
        for m, mon in monitors.items():
            mon.extend(shadows[m])                                # outcomes are known after the close
    if monitors_out is not None:
        monitors_out.update(monitors)
    return pd.Series(curve, dtype=float), pd.DataFrame(book)


def run_intraday(capital: float, leverage: float, start: int, holdout: int | None, workers: int = 4,
                 replay: bool = False) -> dict:
    from .train import build_dataset
    t = time.time()
    saved = report_dir() / "intraday_predictions.parquet"
    if replay and saved.exists():
        ds = pd.read_parquet(saved)                    # re-run the book on the last walk-forward's predictions
        mons: dict = {}
        eq, book = replay_intraday(ds, capital, leverage, start, monitors_out=mons)
        return _intraday_report(eq, book, mons, capital, leverage, holdout, t)
    ds = build_dataset(workers=workers)
    ds = walk_forward(ds, start)
    keep = ["date", "m", "symbol", "entry", "price", "turnover_cr", "atr", "pred_L", "pred_S", "orl", "orh", "lo", "hi"] + \
        [f"{c}_{k}" for c in ("y", "risk", "exit_min", "reason") for k in ("L", "S")] + \
        ["gap_atr", "ret_open_atr", "ret5", "vwap_atr", "below_orl_atr", "above_orh_atr", "idx_open"]
    report_dir().mkdir(parents=True, exist_ok=True)
    ds.loc[np.isfinite(ds["pred_L"]), keep].to_parquet(report_dir() / "intraday_predictions.parquet")
    mons: dict = {}
    eq, book = replay_intraday(ds, capital, leverage, start, monitors_out=mons)
    return _intraday_report(eq, book, mons, capital, leverage, holdout, t)


def _intraday_report(eq: pd.Series, book: pd.DataFrame, mons: dict, capital: float, leverage: float,
                     holdout: int | None, t: float) -> dict:
    rep = {"kind": "intraday", "monitor": {str(m): mon.hist for m, mon in mons.items()},
           "monitor_window": I.MONITOR_N, "generated_at": time.time(), "capital": capital, "leverage": leverage,
           "method": "walk-forward, quarterly refit, 1-day embargo; costs on actual order sizes",
           "all": _stats(eq, book, capital), "secs": round(time.time() - t)}
    if holdout:
        pre, post = eq[eq.index < holdout], eq[eq.index >= holdout]
        rep["before_holdout"] = _stats(pre, book[book["date"] < holdout], capital)
        rep["holdout"] = _stats(post / (pre.iloc[-1] / capital) if len(pre) else post, book[book["date"] >= holdout], capital)
        rep["holdout_from"] = holdout
    rep["equity"] = [[int(k), round(float(v), 2)] for k, v in eq.items()]
    if len(book):
        book["month"] = book["date"] // 100
        rep["monthly"] = [{"month": int(k), "pnl": round(float(v), 2)} for k, v in book.groupby("month")["pnl"].sum().items()]
        rep["by_setup"] = [{"setup": k, "n": int(len(g)), "win": float((g.pnl > 0).mean()), "pnl": float(g.pnl.sum())}
                           for k, g in book.groupby("setup")]
        rep["by_side"] = {("long" if k > 0 else "short"): {"n": int(len(g)), "pnl": float(g.pnl.sum())}
                          for k, g in book.groupby("side")}
        rep["trades"] = book.tail(300).round(4).to_dict("records")
    return rep


# ============================================================================ swing
def run_swing(capital: float, start: int, holdout: int | None) -> dict:
    from . import swing as S
    from .swingbook import SwingParams, run
    lake = Lake()
    syms = [s for s in lake.symbols("D") if s not in ("NIFTY", "BANKNIFTY", "FINNIFTY", "INDIAVIX")]
    panel, reg = S.build_panel(lake, syms)
    sc = S.score(panel)
    sc = sc[sc["date"] >= start]
    C = S.CONFIG
    p = SwingParams(max_positions=C.max_positions, risk_per_trade=C.risk_per_trade, max_weight=C.max_weight,
                    stop_atr=C.stop_atr, trail_atr=C.trail_atr, max_days=C.max_days, slip_bps=C.slip_bps, regime=False)
    curve, tr = run(sc, "score", p, capital, risk_scale=reg["risk_scale"])
    eq = curve.set_index("date")["equity"]
    if len(tr):
        tr["ret_net"] = tr["pnl"] / (tr["qty"] * tr["entry"])
    rep = {"kind": "swing", "generated_at": time.time(), "capital": capital,
           "method": "fixed factor rules (no fitting), regime-scaled risk, delivery charges", "all": _stats(eq, tr, capital)}
    nd = lake.read("D", "NIFTY").sort_values("ts")
    nd["date"] = pd.to_datetime(nd["ts"] + 19800, unit="s").dt.strftime("%Y%m%d").astype(int)
    bench = nd.set_index("date")["close"].reindex(eq.index).ffill()
    rep["benchmark"] = {"name": "NIFTY 50 buy & hold", **_stats(bench / bench.iloc[0] * capital, pd.DataFrame(), capital)}
    if holdout:
        pre, post = eq[eq.index < holdout], eq[eq.index >= holdout]
        rep["before_holdout"] = _stats(pre, tr[tr["entry_date"] < holdout] if len(tr) else tr, capital)
        if len(pre):
            rep["holdout"] = _stats(post / (pre.iloc[-1] / capital), tr[tr["entry_date"] >= holdout] if len(tr) else tr, capital)
            b = bench[bench.index >= holdout]
            rep["holdout_benchmark"] = _stats(b / b.iloc[0] * capital, pd.DataFrame(), capital)
        rep["holdout_from"] = holdout
    rep["equity"] = [[int(k), round(float(v), 2)] for k, v in eq.items()]
    rep["benchmark_equity"] = [[int(k), round(float(v / bench.iloc[0] * capital), 2)] for k, v in bench.items()]
    if len(tr):
        tr["month"] = tr["exit_date"] // 100
        rep["monthly"] = [{"month": int(k), "pnl": round(float(v), 2)} for k, v in tr.groupby("month")["pnl"].sum().items()]
        rep["trades"] = tr.tail(300).round(4).to_dict("records")
    return rep


def save(rep: dict) -> Path:
    d = report_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{rep['kind']}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rep, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(p)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intraday", action="store_true")
    ap.add_argument("--swing", action="store_true")
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--leverage", type=float, default=5.0)
    ap.add_argument("--from", dest="start", default="2022-10-01")
    ap.add_argument("--holdout", default="2025-10-01", help="report results from this date separately ('' = none)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--replay", action="store_true",
                    help="intraday: re-run the book on the saved walk-forward predictions (no refitting)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    both = not (a.intraday or a.swing)
    start = int(a.start.replace("-", ""))
    hold = int(a.holdout.replace("-", "")) if a.holdout else None
    for kind in ("swing", "intraday"):
        if not (both or getattr(a, kind)):
            continue
        rep = run_swing(a.capital, start, hold) if kind == "swing" else \
            run_intraday(a.capital, min(5.0, max(1.0, a.leverage)), start, hold, a.workers, a.replay)
        path = save(rep)
        s = rep["all"]
        print(f"{kind}: CAGR {100 * s.get('cagr', 0):.1f}%  Sharpe {s.get('sharpe', 0):.2f}  max DD "
              f"{100 * s.get('max_dd', 0):.1f}%  trades {s.get('trades', 0)}  -> {path}")
        if "holdout" in rep:
            h = rep["holdout"]
            print(f"   holdout from {hold}: return {100 * h.get('return', 0):.1f}%  Sharpe {h.get('sharpe', 0):.2f}  "
                  f"max DD {100 * h.get('max_dd', 0):.1f}%  trades {h.get('trades', 0)}")


if __name__ == "__main__":
    main()
