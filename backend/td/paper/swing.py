"""Swing strategy (long only, days to weeks): a multi-factor momentum book with regime-scaled risk.

Score (robust z-scores across the liquid F&O universe at each close, higher = better):
  resid60      60-day return net of beta x NIFTY (stock-specific momentum, CAPM residual)
  r60          60-day return (time-series momentum)
  accum20      volume on up days minus volume on down days over 20 sessions (accumulation / distribution)
  adj_slope90  annualised slope of a 90-day log-price regression x its R^2 (trend strength AND smoothness)
  vol_ratio    20-day vs 60-day realised volatility (volatility expanding with the move)
  hh252_pct    distance from the 52-week high (half weight; leaders trade near their highs)
Book: up to 15 positions, each risking 0.6% of the swing capital at a 2.5 ATR stop (max 10% of capital in
one stock); a chandelier stop trails the highest close by 4 ATR; a time stop exits after 40 sessions.
Regime: count the warning lights — NIFTY below its 200-day EMA, below its 50-day EMA, fewer than 40% of
stocks above their 50-day EMA — and scale new-position risk 100% / 60% / 30% / 0% for 0 / 1 / 2 / 3 lights.
In tests on 2022-2025 data this cut the worst drawdown from ~23% to ~14% for a small loss of return.
Entries are decided after the close and filled at the next open; stops can trigger any time in the session.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import market_regime, swing_features

WEIGHTS = {"resid60": 1.0, "r60": 1.0, "accum20": 1.0, "adj_slope90": 1.0, "vol_ratio": 1.0, "hh252_pct": 0.5}
LABELS = {"resid60": "60-day strength vs NIFTY", "r60": "60-day momentum", "accum20": "volume accumulation",
          "adj_slope90": "trend strength x smoothness", "vol_ratio": "volatility expanding", "hh252_pct": "near 52-week high"}
LIQ_PCT = 0.20
REGIME_SCALE = {0: 1.0, 1: 0.6, 2: 0.3, 3: 0.0}


@dataclass(frozen=True)
class SwingConfig:
    max_positions: int = 15
    risk_per_trade: float = 0.006
    max_weight: float = 0.10
    stop_atr: float = 2.5
    trail_atr: float = 4.0
    max_days: int = 40
    slip_bps: float = 6.0


CONFIG = SwingConfig()


def _robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    return ((s - med) / (mad if mad > 1e-12 else (s.std() or 1.0))).clip(-3, 3)


def score(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `score` and per-factor z columns (z_<factor>) by date; drops the least liquid 20% per date."""
    p = panel.copy()
    q = p.groupby("date")["turnover_cr"].transform(lambda s: s.quantile(LIQ_PCT))
    p = p[p["turnover_cr"] >= q].copy()
    g = p.groupby("date")
    tot = 0.0
    for k, w in WEIGHTS.items():
        p[f"z_{k}"] = g[k].transform(_robust_z)
        tot = tot + w * p[f"z_{k}"].fillna(0)
    p["score"] = tot / sum(WEIGHTS.values())
    return p


def regime_table(index_daily: pd.DataFrame, panel: pd.DataFrame, vix_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per date: the three warning lights, the risk scale, and context (NIFTY trend, breadth, VIX)."""
    reg = market_regime(index_daily, vix_daily).set_index("date")
    breadth = panel.groupby("date")["e50_pct"].apply(lambda s: float((s > 0).mean()))
    reg["breadth50"] = breadth.reindex(reg.index)
    lights = (reg["m_e200"] < 0).astype(int) + (reg["m_e50"] < 0).astype(int) + (reg["breadth50"] < 0.40).astype(int)
    reg["lights"] = lights
    reg["risk_scale"] = lights.map(REGIME_SCALE).astype(float)
    return reg


def build_panel(lake, symbols: list[str], index_symbol: str = "NIFTY") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Swing features for every symbol and day + the regime table, from the lake's daily bars."""
    nd = lake.read("D", index_symbol)
    if nd is None or nd.empty:
        return pd.DataFrame(), pd.DataFrame()
    nd = nd.sort_values("ts")
    iclose = pd.Series(nd["close"].to_numpy(float),
                       index=pd.to_datetime(nd["ts"] + 19800, unit="s").dt.strftime("%Y%m%d").astype(int).to_numpy())
    parts = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for sym in symbols:
            d = lake.read("D", sym)
            if d is None or len(d) < 260:
                continue
            f = swing_features(d, iclose)
            f["symbol"] = sym
            parts.append(f)
        if not parts:
            return pd.DataFrame(), pd.DataFrame()
        panel = pd.concat(parts, ignore_index=True)
        vix = lake.read("D", "INDIAVIX")
        reg = regime_table(nd, panel, vix if vix is not None and len(vix) else None)
    return panel, reg


def reasons(row: pd.Series) -> list[dict]:
    """The factor z-scores behind a pick, strongest first, in words."""
    out = []
    for k, w in WEIGHTS.items():
        z = row.get(f"z_{k}")
        if z is not None and np.isfinite(z):
            out.append({"feature": k, "label": LABELS[k], "z": round(float(z), 2), "weight": w})
    return sorted(out, key=lambda r: -abs(r["z"] * r["weight"]))


def candidates(scored_day: pd.DataFrame, held: set[str], free_slots: int) -> pd.DataFrame:
    c = scored_day[np.isfinite(scored_day["score"]) & ~scored_day["symbol"].isin(list(held))]
    return c.sort_values("score", ascending=False).head(max(0, free_slots))
