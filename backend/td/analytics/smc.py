"""Smart Money Concepts — objective, NON-repainting definitions.

Every object is stamped with the bar on which it became *knowable*:
  swing high/low   fractal pivot with `k` bars each side, confirmed k bars later
  BOS / CHoCH      close beyond the last confirmed swing (BOS = with trend, CHoCH = against)
  FVG              3-candle imbalance: low[i] > high[i-2] (bull) / high[i] < low[i-2] (bear)
  order block      last opposite-colour candle before the impulse that produced a BOS
  liquidity sweep  wick through a confirmed swing that closes back inside

Because nothing looks ahead, the same output can be backtested honestly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Swing:
    idx: int
    confirm_idx: int
    ts: int
    price: float
    kind: str           # "H" | "L"


def _atr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n, min_periods=1).mean().to_numpy()


def analyze(df: pd.DataFrame, k: int = 2, max_items: int = 40) -> dict:
    if df is None or len(df) < 2 * k + 3:
        return {"trend": 0, "swings": [], "events": [], "fvgs": [], "obs": [], "sweeps": []}
    ts = df["ts"].to_numpy().astype("int64")
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    atr = _atr(h, l, c)

    # ---- swings (fractals), keyed by confirmation bar
    confirm_at: dict[int, list[Swing]] = {}
    swings: list[Swing] = []
    for i in range(k, n - k):
        # pivot: strictly beyond the k bars on the left, at least equal to the k bars on the right
        # (equal highs/lows — double tops/bottoms — keep the first bar as the pivot)
        if h[i] > h[i - k:i].max() and h[i] >= h[i + 1:i + k + 1].max():
            s = Swing(i, i + k, int(ts[i]), float(h[i]), "H")
            swings.append(s)
            confirm_at.setdefault(i + k, []).append(s)
        if l[i] < l[i - k:i].min() and l[i] <= l[i + 1:i + k + 1].min():
            s = Swing(i, i + k, int(ts[i]), float(l[i]), "L")
            swings.append(s)
            confirm_at.setdefault(i + k, []).append(s)

    trend = 0
    last_h: Swing | None = None
    last_l: Swing | None = None
    h_broken = l_broken = True
    events, sweeps, obs = [], [], []
    for j in range(n):
        for s in confirm_at.get(j, []):
            if s.kind == "H":
                last_h, h_broken = s, False
            else:
                last_l, l_broken = s, False
        # liquidity sweeps
        if last_h and not h_broken and h[j] > last_h.price and c[j] < last_h.price:
            sweeps.append({"ts": int(ts[j]), "price": last_h.price, "dir": -1, "idx": j})
        if last_l and not l_broken and l[j] < last_l.price and c[j] > last_l.price:
            sweeps.append({"ts": int(ts[j]), "price": last_l.price, "dir": 1, "idx": j})
        # structure breaks
        if last_h and not h_broken and c[j] > last_h.price:
            kind = "BOS" if trend >= 0 else "CHoCH"
            trend, h_broken = 1, True
            events.append({"ts": int(ts[j]), "price": last_h.price, "dir": 1, "kind": kind, "idx": j,
                           "from_ts": last_h.ts})
            # order block: last bearish candle between the swing low that started the leg and j
            start = last_l.idx if last_l else max(0, j - 20)
            for b in range(j - 1, max(start - 1, j - 30), -1):
                if c[b] < o[b]:
                    obs.append({"ts": int(ts[b]), "top": float(h[b]), "bottom": float(l[b]), "dir": 1,
                                "idx": b, "mitigated": False})
                    break
        if last_l and not l_broken and c[j] < last_l.price:
            kind = "BOS" if trend <= 0 else "CHoCH"
            trend, l_broken = -1, True
            events.append({"ts": int(ts[j]), "price": last_l.price, "dir": -1, "kind": kind, "idx": j,
                           "from_ts": last_l.ts})
            start = last_h.idx if last_h else max(0, j - 20)
            for b in range(j - 1, max(start - 1, j - 30), -1):
                if c[b] > o[b]:
                    obs.append({"ts": int(ts[b]), "top": float(h[b]), "bottom": float(l[b]), "dir": -1,
                                "idx": b, "mitigated": False})
                    break

    # ---- fair value gaps with fill tracking
    fvgs = []
    for j in range(2, n):
        if l[j] > h[j - 2] and (l[j] - h[j - 2]) > 0.05 * atr[j]:
            fvgs.append({"ts": int(ts[j - 1]), "top": float(l[j]), "bottom": float(h[j - 2]), "dir": 1,
                         "idx": j, "filled": False, "fill_pct": 0.0})
        elif h[j] < l[j - 2] and (l[j - 2] - h[j]) > 0.05 * atr[j]:
            fvgs.append({"ts": int(ts[j - 1]), "top": float(l[j - 2]), "bottom": float(h[j]), "dir": -1,
                         "idx": j, "filled": False, "fill_pct": 0.0})
    for g in fvgs:
        size = g["top"] - g["bottom"]
        after = slice(g["idx"] + 1, n)
        if g["dir"] == 1:
            lowest = l[after].min() if g["idx"] + 1 < n else np.inf
            if lowest < g["top"]:
                g["fill_pct"] = float(min(1.0, (g["top"] - lowest) / size)) if size > 0 else 1.0
        else:
            highest = h[after].max() if g["idx"] + 1 < n else -np.inf
            if highest > g["bottom"]:
                g["fill_pct"] = float(min(1.0, (highest - g["bottom"]) / size)) if size > 0 else 1.0
        g["filled"] = g["fill_pct"] >= 0.999
    for ob in obs:
        after = slice(ob["idx"] + 3, n)
        if ob["idx"] + 3 < n:
            if ob["dir"] == 1 and l[after].min() <= ob["top"]:
                ob["mitigated"] = True
            if ob["dir"] == -1 and h[after].max() >= ob["bottom"]:
                ob["mitigated"] = True

    def tail(xs):
        return xs[-max_items:]

    last_close = float(c[-1])
    open_fvgs = [g for g in fvgs if not g["filled"]]
    nearest_bull = min((g for g in open_fvgs if g["dir"] == 1 and g["top"] <= last_close),
                       key=lambda g: last_close - g["top"], default=None)
    nearest_bear = min((g for g in open_fvgs if g["dir"] == -1 and g["bottom"] >= last_close),
                       key=lambda g: g["bottom"] - last_close, default=None)
    return {
        "trend": trend,
        "swings": tail([asdict(s) for s in swings]),
        "events": tail(events),
        "fvgs": tail(open_fvgs) + tail([g for g in fvgs if g["filled"]])[-10:],
        "obs": tail([ob for ob in obs if not ob["mitigated"]]),
        "sweeps": tail(sweeps),
        "last_event": events[-1] if events else None,
        "nearest_bull_fvg": nearest_bull,
        "nearest_bear_fvg": nearest_bear,
        "bars_since_event": (n - 1 - events[-1]["idx"]) if events else None,
    }
