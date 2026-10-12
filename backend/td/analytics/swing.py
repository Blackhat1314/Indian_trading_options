"""Swing scanner (3–10 day horizon) on daily bars + today's live provisional bar.

Each setup is defined by explicit rules and comes with its own 'receipts': the historical
hit-rate of exactly that setup across the whole universe in the data lake
(hit = +TARGET% reached before -STOP% within HORIZON days; same-day conflicts count as a loss).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import smc

TARGET = 0.04
STOP = 0.02
HORIZON = 10


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).to_numpy()


def atr(h, l, c, n=14) -> np.ndarray:
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def features(df: pd.DataFrame, nifty_close: np.ndarray | None = None) -> pd.DataFrame:
    """Vectorised daily features for one symbol (no look-ahead: everything uses data up to t)."""
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    out = pd.DataFrame({"ts": df["ts"].to_numpy(), "close": c, "high": h, "low": l})
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    a14 = atr(h, l, c, 14)
    a10, a50 = atr(h, l, c, 10), atr(h, l, c, 50)
    out["e20"], out["e50"], out["e200"], out["atr"] = e20, e50, e200, a14
    out["atr_pct"] = a14 / c * 100
    out["rsi"] = rsi(c)
    out["trend"] = (c > e20).astype(int) + (e20 > e50).astype(int) + (e50 > e200).astype(int)
    hh20 = pd.Series(h).rolling(20).max().shift(1).to_numpy()
    hh55 = pd.Series(h).rolling(55).max().shift(1).to_numpy()
    hh252 = pd.Series(h).rolling(252, min_periods=60).max().shift(1).to_numpy()
    ll10 = pd.Series(l).rolling(10).min().shift(1).to_numpy()
    av20 = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    ok = av20 > 0
    out["vol_ratio"] = np.divide(v, av20, out=np.full(len(v), np.nan), where=ok)
    out["hh20"], out["hh55"], out["hh252"], out["ll10"] = hh20, hh55, hh252, ll10
    out["dist_hh55"] = (c / hh55 - 1) * 100
    out["contraction"] = a10 / a50
    ret63 = pd.Series(c).pct_change(63).to_numpy() * 100
    if nifty_close is not None and len(nifty_close) == len(c):
        nret = pd.Series(nifty_close).pct_change(63).to_numpy() * 100
        out["rs63"] = ret63 - nret
    else:
        out["rs63"] = ret63
    # setups (mutually exclusive, priority order)
    breakout = (c > hh20) & (out["vol_ratio"].to_numpy() >= 1.5) & (out["trend"].to_numpy() >= 2)
    vcp = (out["contraction"].to_numpy() < 0.75) & (c >= hh55 * 0.95) & (out["trend"].to_numpy() >= 2) & ~breakout
    pullback = (out["trend"].to_numpy() == 3) & (l <= e20 * 1.01) & (c > e20) & \
        (out["rsi"].to_numpy() >= 40) & (out["rsi"].to_numpy() <= 60) & ~breakout & ~vcp
    breakdown = (c < pd.Series(l).rolling(20).min().shift(1).to_numpy()) & \
        (out["vol_ratio"].to_numpy() >= 1.5) & (out["trend"].to_numpy() <= 1)
    momentum = (out["trend"].to_numpy() == 3) & (out["rsi"].to_numpy() > 60) & ~breakout & ~vcp & ~pullback
    setup = np.full(len(c), "", dtype=object)
    setup[momentum] = "MOMENTUM"
    setup[pullback] = "PULLBACK"
    setup[vcp] = "VCP"
    setup[breakout] = "BREAKOUT"
    setup[breakdown] = "BREAKDOWN"
    out["setup"] = setup
    return out


def outcomes(f: pd.DataFrame, target: float = TARGET, stop: float = STOP, horizon: int = HORIZON) -> np.ndarray:
    """+1 hit target first, -1 stop first (or both same day), 0 neither within horizon, nan = not enough future."""
    c, h, l = f["close"].to_numpy(), f["high"].to_numpy(), f["low"].to_numpy()
    n = len(c)
    res = np.full(n, np.nan)
    for i in range(n - 1):
        if not f["setup"].iat[i]:
            continue
        if i + horizon >= n:
            continue
        short = f["setup"].iat[i] == "BREAKDOWN"
        entry = c[i]
        tp = entry * (1 - target) if short else entry * (1 + target)
        sl = entry * (1 + stop) if short else entry * (1 - stop)
        r = 0.0
        for j in range(i + 1, i + horizon + 1):
            hit_sl = h[j] >= sl if short else l[j] <= sl
            hit_tp = l[j] <= tp if short else h[j] >= tp
            if hit_sl:
                r = -1.0
                break
            if hit_tp:
                r = 1.0
                break
        res[i] = r
    return res


def setup_stats(all_feats: dict[str, pd.DataFrame]) -> dict:
    agg: dict[str, dict] = {}
    for sym, f in all_feats.items():
        if f is None or f.empty:
            continue
        res = outcomes(f)
        c = f["close"].to_numpy()
        fwd = pd.Series(c).shift(-HORIZON).to_numpy() / c - 1
        for i, s in enumerate(f["setup"].to_numpy()):
            if not s or math.isnan(res[i]):
                continue
            a = agg.setdefault(s, {"n": 0, "win": 0, "loss": 0, "fwd": []})
            a["n"] += 1
            a["win"] += res[i] == 1
            a["loss"] += res[i] == -1
            if np.isfinite(fwd[i]):
                a["fwd"].append(fwd[i] * (-1 if s == "BREAKDOWN" else 1))
    out = {}
    for s, a in agg.items():
        fw = np.array(a["fwd"]) if a["fwd"] else np.array([0.0])
        out[s] = {"n": a["n"], "hit_rate": round(a["win"] / a["n"] * 100, 1) if a["n"] else 0,
                  "stop_rate": round(a["loss"] / a["n"] * 100, 1) if a["n"] else 0,
                  "avg_fwd": round(float(fw.mean()) * 100, 2), "median_fwd": round(float(np.median(fw)) * 100, 2),
                  "target_pct": TARGET * 100, "stop_pct": STOP * 100, "horizon": HORIZON}
    return out


def scan(daily: dict[str, pd.DataFrame], sector_of: dict[str, str], nifty: pd.DataFrame | None,
         live: dict[str, dict] | None = None) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Return (scanner rows sorted by score, features per symbol). `live` adds today's provisional
    bar per symbol: {symbol: {open, high, low, close, volume_proj, ts}}."""
    rows, feats = [], {}
    nifty_close_by_ts = dict(zip(nifty["ts"], nifty["close"])) if nifty is not None and not nifty.empty else {}
    for sym, df in daily.items():
        if df is None or len(df) < 60:
            continue
        df = df.copy()
        lv = (live or {}).get(sym)
        if lv and lv.get("close"):
            today = {"ts": lv["ts"], "open": lv["open"], "high": lv["high"], "low": lv["low"],
                     "close": lv["close"], "volume": lv.get("volume_proj", 0), "oi": 0}
            if df["ts"].iat[-1] >= lv["ts"]:
                df = df.iloc[:-1]
            df = pd.concat([df, pd.DataFrame([today])], ignore_index=True)
        nc = np.array([nifty_close_by_ts.get(t, np.nan) for t in df["ts"]]) if nifty_close_by_ts else None
        if nc is not None and np.isnan(nc).mean() > 0.2:
            nc = None
        elif nc is not None:
            nc = pd.Series(nc).ffill().bfill().to_numpy()
        f = features(df, nc)
        feats[sym] = f
        last = f.iloc[-1]
        sm = smc.analyze(df.tail(160).reset_index(drop=True), k=3, max_items=20)
        rows.append({"symbol": sym, "sector": sector_of.get(sym, "OTHER"), "close": round(float(last.close), 2),
                     "setup": last.setup, "trend": int(last.trend), "rsi": round(float(last.rsi), 1),
                     "atr_pct": round(float(last.atr_pct), 2), "vol_ratio": _r(last.vol_ratio),
                     "rs63": _r(last.rs63), "dist_hh55": _r(last.dist_hh55), "contraction": _r(last.contraction),
                     "e20": _r(last.e20), "e50": _r(last.e50), "e200": _r(last.e200), "atr": _r(last.atr),
                     "ll10": _r(last.ll10), "hh20": _r(last.hh20),
                     "smc_trend": sm["trend"], "smc_event": (sm["last_event"] or {}).get("kind"),
                     "smc_event_dir": (sm["last_event"] or {}).get("dir"), "smc_age": sm["bars_since_event"],
                     "bull_fvg": sm["nearest_bull_fvg"], "bear_fvg": sm["nearest_bear_fvg"],
                     "live": bool(lv)})
    if not rows:
        return [], feats
    rs = np.array([r["rs63"] if r["rs63"] is not None else np.nan for r in rows], float)
    rs_rank = pd.Series(rs).rank(pct=True).fillna(0.5).to_numpy()
    for r, rr in zip(rows, rs_rank):
        r["rs_rank"] = round(float(rr) * 100, 1)
        bonus = {"BREAKOUT": 25, "VCP": 20, "PULLBACK": 18, "MOMENTUM": 10, "BREAKDOWN": 0, "": 0}[r["setup"]]
        smc_b = 10 if (r["smc_event_dir"] == 1 and (r["smc_age"] or 99) <= 10) else \
            -10 if (r["smc_event_dir"] == -1 and (r["smc_age"] or 99) <= 10) else 0
        volb = min(15.0, max(0.0, ((r["vol_ratio"] or 1) - 1) * 10))
        r["score"] = round(r["trend"] * 10 + rr * 25 + bonus + smc_b + volb, 1)
        if r["setup"] == "BREAKDOWN":
            r["score"] = round((3 - r["trend"]) * 10 + (1 - rr) * 25 + 25 + (10 if r["smc_event_dir"] == -1 else 0)
                               + volb, 1)
        # informational levels: stop below the 10-day low / 20 EMA, target = 2R
        if r["setup"] and r["setup"] != "BREAKDOWN" and r["atr"]:
            cands = [x for x in (r["ll10"], (r["e20"] - 0.5 * r["atr"]) if r["e20"] else None) if x]
            stop = max(cands) if cands else r["close"] - r["atr"]
            stop = min(stop, r["close"] - 0.5 * r["atr"])
            r["stop"] = round(stop, 2)
            r["target"] = round(r["close"] + 2 * (r["close"] - stop), 2)
        elif r["setup"] == "BREAKDOWN" and r["atr"]:
            r["stop"] = round(r["close"] + 1.0 * r["atr"], 2)
            r["target"] = round(r["close"] - 2.0 * r["atr"], 2)
    rows.sort(key=lambda r: (r["setup"] != "", r["score"]), reverse=True)
    return rows, feats


def _r(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(x) or math.isinf(x)) else round(x, nd)
