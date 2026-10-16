"""Intraday strategy: decide at 09:30 (and 10:00), long or short, flat by 15:15.

How a trade is chosen (each decision time, over the whole liquid F&O universe):
  1. Features from the first 15 (or 45) minutes + the prior-day context (features.py), plus their
     cross-sectional ranks (how a stock compares with every other stock at the same minute) and the
     market's state (breadth above VWAP, dispersion, NIFTY's move).
  2. Two models per decision time estimate the return of a long and of a short trade held from the decision
     to 15:15 with a structural stop (the far side of the opening range, or the day's extreme at 10:00):
     an ensemble of ridge regression and gradient-boosted trees, trained walk-forward on years of minute data.
  3. A trade is taken only when the predicted return beats its own estimated cost (charges for that order size
     + slippage) by MIN_EDGE; at most MAX_PER_SIDE longs and shorts per decision, best edge first.
  4. Size: each trade risks RISK_PER_TRADE of the intraday capital at its stop; total exposure is capped at
     the user's leverage x capital.
Every trade stores the features, the prediction, the cost estimate, the sizing arithmetic and a plain-language
reason, so the book shows exactly why it was opened and why it was closed.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings
from .costs import charges
from .features import FEATURES
from .models import Ensemble

DECISIONS = {15: "09:30", 45: "10:00"}
XS = ["ret_open_atr", "gap_atr", "lrvol", "resid_pc", "vwap_atr", "ret5", "ret20", "atr_pct"]
EXTRA = ["idx_pc", "idx_open", "breadth_vwap", "xs_disp", "lturn"]
MODEL_FEATURES = [f for f in FEATURES if f not in ("gap_pct", "turnover_cr")] + [c + "_xr" for c in XS] + EXTRA

MIN_EDGE = {15: 0.0010, 45: 0.0015}   # predicted return must beat the trade's cost by this (stricter at 10:00)
MAX_PER_SIDE = 3
RISK_PER_TRADE = 0.005       # 0.5% of intraday capital at the stop
MAX_POSITION_FRAC = 0.40     # one position uses at most 40% of the buying power
LIQ_PCT = 0.20               # skip the least liquid 20% of the universe (turnover so far today)
STOP_MIN_ATR, STOP_MAX_ATR = 0.15, 1.0
SLIP_BPS = 4.0               # per side, for the cost estimate at decision time
EXIT_MINUTE = 360            # 15:15
DAILY_LOSS_LIMIT = 0.02      # no new trades once the day is down 2% of intraday capital


MONITOR_N = 100              # edge monitor window (signals); 30-150 all helped in tests, 100 best before the holdout


class EdgeMonitor:
    """Trade a strategy only while it is working: every signal is shadow-traded (taken or not), and money goes
    on new signals only while the last `n` shadow trades averaged a profit after costs. Shadow trading goes on
    while it is switched off, so it switches back on by itself once the edge returns. Uses past outcomes only."""

    def __init__(self, n: int = MONITOR_N, history: list[float] | None = None):
        self.n = n
        self.hist: list[float] = list(history or [])[-n:] if n else []

    def extend(self, outcomes) -> None:
        if self.n:
            self.hist = (self.hist + [float(x) for x in outcomes if x == x])[-self.n:]

    def allows(self) -> bool:
        if not self.n or len(self.hist) < self.n // 2:
            return True                                          # not enough record yet: trust the backtest
        return sum(self.hist) / len(self.hist) > 0

    def state(self) -> dict:
        avg = sum(self.hist) / len(self.hist) if self.hist else None
        return {"window": self.n, "signals": len(self.hist), "avg_net": avg, "trading": self.allows()}


def model_dir() -> Path:
    return Path(settings.data_dir) / "paper" / "models"


PRETRAINED = Path(__file__).resolve().parent / "pretrained"


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the cross-sectional and market-wide columns to one decision's universe snapshot (or many)."""
    f = frame.copy()
    keys = ["date", "m"] if "date" in f and "m" in f else None
    g = f.groupby(keys) if keys else None
    for c in XS:
        f[c + "_xr"] = g[c].rank(pct=True) if g is not None else f[c].rank(pct=True)
    if g is not None:
        f["breadth_vwap"] = g["vwap_atr"].transform(lambda s: (s > 0).mean())
        f["xs_disp"] = g["ret_open_atr"].transform("std")
    else:
        f["breadth_vwap"] = (f["vwap_atr"] > 0).mean()
        f["xs_disp"] = f["ret_open_atr"].std()
    f["lturn"] = np.log1p(f["turnover_cr"].clip(lower=0))
    return f


def liquid(frame: pd.DataFrame) -> pd.Series:
    keys = ["date", "m"] if "date" in frame and "m" in frame else None
    q = frame.groupby(keys)["turnover_cr"].transform(lambda s: s.quantile(LIQ_PCT)) if keys \
        else frame["turnover_cr"].quantile(LIQ_PCT)
    return frame["turnover_cr"] >= q


def stop_distance(frame: pd.DataFrame, side: int, entry: np.ndarray, m: int) -> np.ndarray:
    """Structural stop distance (price units): beyond the opening range at 09:30, the day's extreme at 10:00,
    kept between STOP_MIN_ATR and STOP_MAX_ATR."""
    atr = frame["atr"].to_numpy(float)
    if side > 0:
        lvl = frame["orl"].to_numpy(float) if m == 15 else frame["lo"].to_numpy(float)
        dist = entry - lvl
    else:
        lvl = frame["orh"].to_numpy(float) if m == 15 else frame["hi"].to_numpy(float)
        dist = lvl - entry
    return np.clip(dist, STOP_MIN_ATR * atr, STOP_MAX_ATR * atr)


# ============================================================================ models on disk
@dataclass
class SideModel:
    model: Ensemble
    lo: np.ndarray
    hi: np.ndarray
    meta: dict

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame.reindex(columns=MODEL_FEATURES).astype(float).to_numpy()
        X = np.where(np.isfinite(X), X, np.nan)
        X = np.clip(X, self.lo, self.hi)
        return np.where(np.isnan(X), 0.0, X)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.matrix(frame))

    def drivers(self, frame: pd.DataFrame, i: int, k: int = 4) -> list[tuple[str, float]]:
        """Largest linear contributions (fraction of return) to row i's prediction: the 'why' in feature terms."""
        x = self.matrix(frame.iloc[[i]])[0]
        c = self.model.ridge.contributions(x) * self.model.w
        order = np.argsort(-np.abs(c))[:k]
        return [(MODEL_FEATURES[j], float(c[j])) for j in order]

    def to_json(self) -> dict:
        return {"features": MODEL_FEATURES, "lo": self.lo.tolist(), "hi": self.hi.tolist(), "meta": self.meta,
                "model": self.model.to_json()}

    @classmethod
    def from_json(cls, d: dict) -> "SideModel":
        if d.get("features") != MODEL_FEATURES:
            raise ValueError("model was trained on a different feature set")
        return cls(Ensemble.from_json(d["model"]), np.asarray(d["lo"], float), np.asarray(d["hi"], float), d["meta"])


def fit_side(frame: pd.DataFrame, y: np.ndarray, meta: dict) -> SideModel:
    X = frame.reindex(columns=MODEL_FEATURES).astype(float).to_numpy()
    X = np.where(np.isfinite(X), X, np.nan)
    lo, hi = np.nanquantile(X, 0.005, axis=0), np.nanquantile(X, 0.995, axis=0)
    lo, hi = np.where(np.isfinite(lo), lo, -1e9), np.where(np.isfinite(hi), hi, 1e9)
    Xc = np.where(np.isnan(X), 0.0, np.clip(X, lo, hi))
    ok = np.isfinite(y)
    model = Ensemble().fit(Xc[ok], y[ok])
    return SideModel(model, lo, hi, dict(meta, n=int(ok.sum()), trained_at=time.time()))


def save_models(models: dict[str, SideModel], directory: Path | None = None) -> Path:
    directory = directory or model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    for key, sm in models.items():
        tmp = directory / f"intraday_{key}.json.tmp"
        tmp.write_text(json.dumps(sm.to_json()))
        tmp.replace(directory / f"intraday_{key}.json")
    return directory


def load_models() -> dict[str, SideModel]:
    """Models trained on this install's own history if present, else the ones shipped with the app."""
    out: dict[str, SideModel] = {}
    for m in DECISIONS:
        for side in ("L", "S"):
            key = f"m{m}_{side}"
            for d in (model_dir(), PRETRAINED):
                p = d / f"intraday_{key}.json"
                if p.exists():
                    try:
                        out[key] = SideModel.from_json(json.loads(p.read_text()))
                        out[key].meta["source"] = "local" if d == model_dir() else "shipped"
                        break
                    except Exception:
                        continue
    return out


# ============================================================================ the decision
REASON_TEXT = {
    "gap_atr": "opening gap", "ret_open_atr": "move since the open", "ret_pc_atr": "move since yesterday's close",
    "resid_pc": "move vs NIFTY (beta-adjusted)", "resid_open": "move vs NIFTY since the open",
    "or_range_atr": "opening-range width", "or_pos": "position in the opening range", "body_or": "first candle body",
    "day_pos": "position in the day's range", "above_orh_atr": "distance above the opening-range high",
    "below_orl_atr": "distance below the opening-range low", "lrvol": "relative volume",
    "vwap_atr": "distance from VWAP", "vwap_slope_atr": "VWAP slope", "frac_above_vwap": "time spent above VWAP",
    "eff_ratio": "trend efficiency", "new_high_frac": "new-high persistence", "new_low_frac": "new-low persistence",
    "last15_atr": "last 15-minute move", "ivol_ratio": "intraday volatility vs normal",
    "pdh_atr": "distance from yesterday's high", "pdl_atr": "distance from yesterday's low",
    "clv_prev": "yesterday's close location", "ret5": "5-day return", "ret20": "20-day return",
    "rv20": "20-day volatility", "atr_pct": "volatility (ATR %)", "nr7": "narrow-range day before",
    "trend_d": "daily trend (EMA stack)", "hh20_atr": "distance from 20-day high", "ll20_atr": "distance from 20-day low",
    "idx_pc": "NIFTY since yesterday", "idx_open": "NIFTY since the open", "breadth_vwap": "market breadth above VWAP",
    "xs_disp": "cross-market dispersion", "lturn": "turnover so far",
}
for _c in XS:
    REASON_TEXT[_c + "_xr"] = REASON_TEXT.get(_c, _c) + " (rank vs all stocks)"


def setup_label(r: pd.Series, side: int) -> str:
    """A short human name for the pattern the model is acting on (descriptive; the model decides)."""
    if side < 0:
        if r.get("gap_atr", 0) > 0.4 and r.get("ret_open_atr", 0) < 0:
            return "Gap-up failure"
        if r.get("ret5", 0) > 0.03 and r.get("vwap_atr", 0) < 0:
            return "Overextended fade"
        if r.get("below_orl_atr", 0) > 0:
            return "Opening-range breakdown"
        if r.get("vwap_atr", 0) < 0:
            return "Below VWAP, relative weakness"
        return "Short: model edge"
    if r.get("gap_atr", 0) < -0.4 and r.get("ret_open_atr", 0) > 0:
        return "Gap-down reclaim"
    if r.get("above_orh_atr", 0) > 0 and r.get("vwap_atr", 0) > 0:
        return "Opening-range breakout"
    if r.get("vwap_atr", 0) > 0 and r.get("idx_open", 0) < 0:
        return "VWAP strength vs a weak market"
    if r.get("ret5", 0) < -0.03 and r.get("vwap_atr", 0) > 0:
        return "Oversold bounce above VWAP"
    return "Long: model edge"


def facts(r: pd.Series) -> dict:
    """The key numbers at the decision, labelled and formatted for people."""
    def num(k):
        v = r.get(k)
        return float(v) if v is not None and np.isfinite(v) else None

    def fmt(v, f):
        return None if v is None else f(v)
    rv = num("lrvol")
    return {
        "Opening gap": fmt(num("gap_pct"), lambda v: f"{v:+.2%}"),
        "Move since the open": fmt(num("ret_open_atr"), lambda v: f"{v:+.2f} ATR"),
        "Relative volume": fmt(rv, lambda v: f"{math.exp(v):.1f}x normal"),
        "Distance from VWAP": fmt(num("vwap_atr"), lambda v: f"{v:+.2f} ATR"),
        "Move vs NIFTY (beta-adjusted)": fmt(num("resid_pc"), lambda v: f"{v:+.2%}"),
        "5-day return": fmt(num("ret5"), lambda v: f"{v:+.2%}"),
        "Opening range": fmt(num("or_range_atr"), lambda v: f"{v:.2f} ATR"),
        "NIFTY since the open": fmt(num("idx_open"), lambda v: f"{v:+.2%}"),
        "Turnover so far": fmt(num("turnover_cr"), lambda v: f"₹{v:,.1f} cr"),
        "Daily ATR": fmt(num("atr"), lambda v: f"₹{v:,.2f}"),
    }


def cost_estimate(notional: float) -> float:
    if notional <= 0:
        return 1.0
    return charges(notional, notional, "intraday")["total"] / notional + 2 * SLIP_BPS / 1e4


def decide(frame: pd.DataFrame, m: int, models: dict[str, SideModel], capital: float,
           exclude: set[str] | None = None) -> list[dict]:
    """Candidates for one decision. `frame`: prepared features for the universe at minute m (one row per
    symbol, with `price` = the decision price). Returns trade proposals for a reference capital; the broker
    re-sizes them per account."""
    exclude = exclude or set()
    f = frame[liquid(frame) & np.isfinite(frame["atr"]) & (frame["atr"] > 0)].reset_index(drop=True)
    if f.empty:
        return []
    out = []
    for side, key in ((1, "L"), (-1, "S")):
        sm = models.get(f"m{m}_{key}")
        if sm is None:
            continue
        pred = sm.predict(f)
        price = f["price"].to_numpy(float)
        dist = stop_distance(f, side, price, m)
        risk_pct = dist / price
        rows = []
        for i in range(len(f)):
            sym = f.at[i, "symbol"]
            if sym in exclude or not np.isfinite(pred[i]) or not np.isfinite(risk_pct[i]) or risk_pct[i] <= 0:
                continue
            notional = min(RISK_PER_TRADE * capital / risk_pct[i], MAX_POSITION_FRAC * capital * 5)
            cost = cost_estimate(notional)
            edge = pred[i] - cost
            if edge > MIN_EDGE[m]:
                rows.append((edge, i, cost, notional))
        rows.sort(reverse=True)
        for edge, i, cost, notional in rows[:MAX_PER_SIDE]:
            r = f.iloc[i]
            drivers = sm.drivers(f, i)
            out.append({
                "symbol": r["symbol"], "side": side, "m": m, "decision": DECISIONS[m], "price": float(r["price"]),
                "stop_dist": float(dist[i]), "risk_pct": float(risk_pct[i]), "pred": float(pred[i]),
                "cost": float(cost), "edge": float(edge), "atr": float(r["atr"]), "setup": setup_label(r, side),
                "drivers": [{"feature": k, "label": REASON_TEXT.get(k, k), "contrib_pct": round(v, 6)} for k, v in drivers],
                "facts": facts(r),
                "model": {"source": sm.meta.get("source"), "trained_to": sm.meta.get("train_to"), "n": sm.meta.get("n")},
            })
            exclude = exclude | {r["symbol"]}
    out.sort(key=lambda t: -t["edge"])
    return out
