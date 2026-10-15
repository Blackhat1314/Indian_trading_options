"""Market-wide views: breadth, sector heatmap, movers, F&O OI build-up board."""
from __future__ import annotations

import numpy as np

from ..universe_data import SECTOR_NAMES
from .intraday import buildup_tag
from .state import InstState, safe


HIST_EDGES = [-5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5]
HIST_BINS = ["<-3", "-3..-2", "-2..-1", "-1..-.5", "-.5..0", "0..+.5", "+.5..1", "1..2", "2..3", ">3"]


def breadth(eqs: list[InstState]) -> dict:
    live = [s for s in eqs if s.ltp and s.pc]
    if not live:
        # no prices yet (feed down, token expired, before the open): same shape, n = 0 says "no reading"
        return {"n": 0, "adv": 0, "dec": 0, "unch": 0, "ad_ratio": 0.0, "above_vwap": 0, "above_vwap_pct": 0.0,
                "near_high": 0, "near_low": 0, "above_pdh": 0, "below_pdl": 0, "up_turnover": 0.0,
                "down_turnover": 0.0, "median_chg": 0.0, "hist": [0] * len(HIST_BINS), "hist_bins": HIST_BINS}
    chg = np.array([s.chg for s in live])
    adv = int((chg > 0.05).sum())
    dec = int((chg < -0.05).sum())
    above_vwap = int(sum(1 for s in live if s.ltp > s.vwap))
    near_high = int(sum(1 for s in live if s.high and s.ltp >= s.high * 0.998))
    near_low = int(sum(1 for s in live if s.low and s.ltp <= s.low * 1.002))
    above_pdh = int(sum(1 for s in live if s.ctx.pdh and s.ltp > s.ctx.pdh))
    below_pdl = int(sum(1 for s in live if s.ctx.pdl and s.ltp < s.ctx.pdl))
    up_turn = float(sum(s.turnover_cr for s in live if s.chg > 0))
    dn_turn = float(sum(s.turnover_cr for s in live if s.chg < 0))
    hist = np.histogram(np.clip(chg, -5, 5), bins=HIST_EDGES)[0].tolist()
    return {
        "n": len(live), "adv": adv, "dec": dec, "unch": len(live) - adv - dec,
        "ad_ratio": safe(adv / dec if dec else adv, 2),
        "above_vwap": above_vwap, "above_vwap_pct": safe(above_vwap / len(live) * 100, 1),
        "near_high": near_high, "near_low": near_low, "above_pdh": above_pdh, "below_pdl": below_pdl,
        "up_turnover": safe(up_turn, 1), "down_turnover": safe(dn_turn, 1),
        "median_chg": safe(float(np.median(chg))), "hist": hist,
        "hist_bins": HIST_BINS,
    }


def sectors(eqs: list[InstState]) -> list[dict]:
    by: dict[str, list[InstState]] = {}
    for s in eqs:
        if s.ltp and s.pc:
            by.setdefault(s.sector, []).append(s)
    out = []
    for sec, items in by.items():
        chg = [x.chg for x in items]
        best = max(items, key=lambda x: x.chg)
        worst = min(items, key=lambda x: x.chg)
        out.append({
            "sector": sec, "name": SECTOR_NAMES.get(sec, sec), "n": len(items),
            "chg": safe(float(np.mean(chg))), "median": safe(float(np.median(chg))),
            "adv": sum(1 for c in chg if c > 0.05), "dec": sum(1 for c in chg if c < -0.05),
            "turnover": safe(sum(x.turnover_cr for x in items), 1),
            "best": {"symbol": best.symbol, "chg": safe(best.chg)},
            "worst": {"symbol": worst.symbol, "chg": safe(worst.chg)},
            "members": sorted([{"symbol": x.symbol, "chg": safe(x.chg), "sid": x.sid,
                                "turnover": safe(x.turnover_cr, 1)} for x in items],
                              key=lambda m: -m["turnover"]),
        })
    out.sort(key=lambda r: r["chg"], reverse=True)
    return out


def movers(rows: list[dict], n: int = 10) -> dict:
    if not rows:
        return {"gainers": [], "losers": [], "active": [], "rvol": []}
    pick = ("sid", "symbol", "ltp", "chg", "rvol", "turnover", "sector")

    def slim(r):
        return {k: r[k] for k in pick}
    by_chg = sorted(rows, key=lambda r: r["chg"])
    return {
        "gainers": [slim(r) for r in by_chg[::-1][:n]],
        "losers": [slim(r) for r in by_chg[:n]],
        "active": [slim(r) for r in sorted(rows, key=lambda r: -r["turnover"])[:n]],
        "rvol": [slim(r) for r in sorted(rows, key=lambda r: -r["rvol"])[:n]],
    }


def fno_board(futs: list[InstState], eq_by_symbol: dict[str, InstState], n: int = 15) -> dict:
    rows = []
    for f in futs:
        if not (f.ltp and f.pc and f.poi and f.oi):
            continue
        pchg = (f.ltp / f.pc - 1) * 100
        ochg = f.oi_chg_pct
        eq = eq_by_symbol.get(f.underlying)
        basis = (f.ltp / eq.ltp - 1) * 100 if eq and eq.ltp else 0.0
        rows.append({
            "sid": f.sid, "eq_sid": eq.sid if eq else None, "symbol": f.underlying, "sector": f.sector,
            "ltp": safe(f.ltp), "chg": safe(pchg), "oi": int(f.oi), "oi_chg": safe(ochg),
            "oi_chg_abs": int(f.oi - f.poi), "tag": buildup_tag(pchg, ochg), "basis": safe(basis, 3),
            "oi_value_cr": safe(f.oi * f.ltp / 1e7, 1), "vol": int(f.vol), "lot": f.lot,
            "oih": int(f.oih), "oil": int(f.oil),
        })
    groups = {t: sorted([r for r in rows if r["tag"] == t], key=lambda r: -abs(r["oi_chg"]))[:n]
              for t in ("LB", "SB", "SC", "LU")}
    counts = {t: sum(1 for r in rows if r["tag"] == t) for t in ("LB", "SB", "SC", "LU")}
    spurts = sorted(rows, key=lambda r: -r["oi_chg"])[:n]
    return {"groups": groups, "counts": counts, "spurts": spurts, "all": rows}
