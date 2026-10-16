"""Long-only swing portfolio simulation on daily bars (the backtest twin of the live swing engine).

Each session: entries decided at the previous close fill at today's open (with slippage), stops are checked
against today's low (gap-downs fill at the open), trailing stops ratchet up at the close, time stops exit at
the close, and new candidates are ranked for tomorrow. Delivery charges (STT both sides, stamp duty, DP fee)
come from costs.py. Sizing is risk-based: each position risks `risk_per_trade` of equity at its initial stop,
capped at `max_weight` of equity, with at most `max_positions` open.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import charges, fill_price


@dataclass
class SwingParams:
    max_positions: int = 8
    risk_per_trade: float = 0.01
    max_weight: float = 0.20
    stop_atr: float = 2.5
    trail_atr: float = 4.0
    max_days: int = 40
    slip_bps: float = 6.0
    min_score: float = -np.inf          # only candidates scoring above this
    regime: bool = True                 # trade only when the market regime allows new longs


@dataclass
class Pos:
    symbol: str
    qty: int
    entry: float
    stop: float
    atr: float
    best: float
    day0: int
    entry_date: int
    reason: dict = field(default_factory=dict)


def run(panel: pd.DataFrame, score_col: str, p: SwingParams, capital: float = 1_000_000.0,
        regime_ok: pd.Series | None = None, risk_scale: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """panel: one row per (date, symbol) with open/high/low/close/atr and `score_col` (higher = better).

    Returns (equity curve by date, trade list).
    """
    dates = np.sort(panel["date"].unique())
    by_date = {d: g.set_index("symbol") for d, g in panel.groupby("date")}
    cash, pos, trades, curve = capital, {}, [], []
    pending: list[tuple[str, float, dict]] = []
    for di, d in enumerate(dates):
        today = by_date[d]
        # 1) fills at the open for yesterday's decisions
        for sym, sc, why in pending:
            if sym in pos or sym not in today.index or len(pos) >= p.max_positions:
                continue
            row = today.loc[sym]
            px = fill_price(+1, float(row["open"]), p.slip_bps)
            a = float(why["atr"])
            stop = px - p.stop_atr * a
            equity = cash + sum(q.qty * float(by_date[d].loc[q.symbol, "open"]) if q.symbol in today.index else q.qty * q.entry
                                for q in pos.values())
            rs = float(why.get("risk_scale", 1.0))
            qty = int(min(rs * p.risk_per_trade * equity / max(px - stop, 1e-9), p.max_weight * equity / px, cash / px))
            if qty <= 0:
                continue
            fee = charges(qty * px, 0, "delivery")["total"]
            cash -= qty * px + fee
            pos[sym] = Pos(sym, qty, px, stop, a, px, di, int(d), dict(why, score=sc, buy_fee=fee))
        pending = []
        # 2) stops during the day, then trailing / time exits at the close
        for sym in list(pos):
            q = pos[sym]
            if sym not in today.index:
                continue
            row = today.loc[sym]
            o, lo, c = float(row["open"]), float(row["low"]), float(row["close"])
            exit_px, why = None, None
            if lo <= q.stop:
                exit_px, why = fill_price(-1, min(o, q.stop), p.slip_bps * 1.5), ("trail" if q.stop > q.entry - p.stop_atr * q.atr + 1e-9 else "stop")
            elif di - q.day0 >= p.max_days:
                exit_px, why = fill_price(-1, c, p.slip_bps), "time"
            if exit_px is not None:
                fee = charges(0, q.qty * exit_px, "delivery")["total"]
                cash += q.qty * exit_px - fee
                pnl = q.qty * (exit_px - q.entry) - fee - q.reason.get("buy_fee", 0)
                trades.append({"symbol": sym, "entry_date": q.entry_date, "exit_date": int(d), "entry": q.entry,
                               "exit": exit_px, "qty": q.qty, "pnl": pnl, "ret": exit_px / q.entry - 1,
                               "days": di - q.day0, "why_exit": why, **{k: v for k, v in q.reason.items() if k != "buy_fee"}})
                del pos[sym]
                continue
            q.best = max(q.best, c)
            if p.trail_atr > 0:
                q.stop = max(q.stop, q.best - p.trail_atr * q.atr)
        equity = cash + sum(q.qty * float(today.loc[q.symbol, "close"]) if q.symbol in today.index else q.qty * q.entry
                            for q in pos.values())
        curve.append({"date": int(d), "equity": equity, "cash": cash, "n": len(pos)})
        # 3) tomorrow's entries
        if regime_ok is not None and p.regime and not bool(regime_ok.get(d, False)):
            continue
        free = p.max_positions - len(pos)
        if free <= 0:
            continue
        cand = today[np.isfinite(today[score_col]) & (today[score_col] > p.min_score)
                     & ~today.index.isin(list(pos))].sort_values(score_col, ascending=False).head(free)
        rs = float(risk_scale.get(d, 1.0)) if risk_scale is not None else 1.0
        if rs <= 0:
            continue
        pending = [(s, float(r[score_col]), {"atr": float(r["atr"]), "risk_scale": rs}) for s, r in cand.iterrows()]
    return pd.DataFrame(curve), pd.DataFrame(trades)


def summary(curve: pd.DataFrame, trades: pd.DataFrame) -> dict:
    eq = curve["equity"].to_numpy(float)
    r = np.diff(eq) / eq[:-1]
    years = len(eq) / 252
    dd = eq / np.maximum.accumulate(eq) - 1
    out = {"cagr": (eq[-1] / eq[0]) ** (1 / max(years, 1e-9)) - 1 if len(eq) > 1 else 0.0,
           "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 2 and r.std() > 0 else 0.0,
           "max_dd": float(dd.min()) if len(dd) else 0.0, "trades": int(len(trades)),
           "exposure": float((curve["n"] > 0).mean()) if len(curve) else 0.0}
    if len(trades):
        out.update(win=float((trades["pnl"] > 0).mean()), avg_ret=float(trades["ret"].mean()),
                   avg_days=float(trades["days"].mean()),
                   profit_factor=float(trades.loc[trades.pnl > 0, "pnl"].sum() / max(-trades.loc[trades.pnl < 0, "pnl"].sum(), 1e-9)))
    return out
