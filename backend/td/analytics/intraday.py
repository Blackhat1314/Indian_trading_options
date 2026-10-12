"""Intraday ranker: transparent cross-sectional factor model for the 09:30 top-N.

Target = return from the lock time (09:30) to the close — the part you can actually trade.
Every factor is z-scored across the universe each refresh, so scores are relative
("which stocks are most in play right now"), never absolute predictions.

Factors (long side; the short side mirrors the directional ones):
  rvol      log relative volume vs the same minute over the last 14 sessions   (stocks in play)
  resid     return since prev close minus beta x NIFTY return                   (stock-specific strength)
  vwap      distance from VWAP in ATR units                                     (who is in control)
  orb       opening-range position / breakout                                   (intraday momentum)
  oi        futures OI build-up (long build-up > short covering > ...)          (positioning)
  news      time-decayed catalyst score from headlines                          (the reason)
  obi       resting order-book imbalance (weak, small weight)
  sector    stock vs its sector median (avoid buying a whole sector as one bet)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .state import InstState, safe

DEFAULT_WEIGHTS = {"rvol": 0.22, "resid": 0.18, "vwap": 0.14, "orb": 0.12, "oi": 0.10, "news": 0.14,
                   "sector": 0.06, "obi": 0.04}


@dataclass
class RankConfig:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    min_turnover_cr: float = 5.0
    min_rvol: float = 1.2
    min_score: float = 0.35
    max_per_sector: int = 2
    top_n: int = 5


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 3:
        return np.zeros_like(x)
    med = np.median(x[finite])
    mad = np.median(np.abs(x[finite] - med)) * 1.4826
    sd = mad if mad > 1e-9 else (np.std(x[finite]) or 1.0)
    z = (x - med) / sd
    z[~finite] = 0.0
    return np.clip(z, -3.0, 3.0)


def buildup_tag(price_chg: float, oi_chg: float) -> str:
    if abs(oi_chg) < 0.25 and abs(price_chg) < 0.1:
        return ""
    if price_chg >= 0 and oi_chg >= 0:
        return "LB"          # long build-up
    if price_chg < 0 and oi_chg >= 0:
        return "SB"          # short build-up
    if price_chg >= 0 and oi_chg < 0:
        return "SC"          # short covering
    return "LU"              # long unwinding


def _oi_signal(price_chg: float, oi_chg: float) -> float:
    tag = buildup_tag(price_chg, oi_chg)
    mag = min(1.0, abs(oi_chg) / 6.0)
    return {"LB": 1.0 + mag, "SC": 0.6, "SB": -(1.0 + mag), "LU": -0.6}.get(tag, 0.0)


def rank(eqs: list[InstState], futs: dict[str, InstState], nifty_chg: float, minute: float,
         news_scores: dict[str, tuple[float, dict | None]], cfg: RankConfig, or_minutes: int = 15) -> list[dict]:
    live = [s for s in eqs if s.ltp and s.pc]
    if len(live) < 5:
        return []
    n = len(live)
    chg = np.array([s.chg for s in live])
    beta = np.array([s.ctx.beta or 1.0 for s in live])
    atr = np.array([s.ctx.atr or (s.pc * 0.02) for s in live])
    resid = chg - beta * nifty_chg
    rvol = np.array([s.rvol(minute) for s in live])
    lrvol = np.log(np.clip(rvol, 0.05, 50))
    vwap_d = np.array([(s.ltp - s.vwap) / a if a else 0 for s, a in zip(live, atr)])
    sectors = [s.sector for s in live]
    sec_med = {}
    for sec in set(sectors):
        vals = [c for c, sc in zip(chg, sectors) if sc == sec]
        sec_med[sec] = float(np.median(vals)) if vals else 0.0
    sec_rel = np.array([c - sec_med[sc] for c, sc in zip(chg, sectors)])
    orb, orpos = [], []
    for s in live:
        rng = (s.orh - s.orl) if s.orh and s.orl else 0
        pos = (s.ltp - s.orl) / rng if rng > 0 else 0.5
        orpos.append(pos)
        if minute >= or_minutes and rng > 0:
            orb.append(1.0 + min(1.0, (s.ltp - s.orh) / rng) if s.ltp > s.orh else
                       -1.0 - min(1.0, (s.orl - s.ltp) / rng) if s.ltp < s.orl else (pos - 0.5))
        else:
            orb.append(pos - 0.5)
    orb = np.array(orb)
    oi_sig, oi_tags, oi_chgs = [], [], []
    for s in live:
        f = futs.get(s.symbol)
        if f and f.poi and f.oi and f.pc:
            pc_ = (f.ltp / f.pc - 1) * 100 if f.ltp else 0
            oc = f.oi_chg_pct
            oi_sig.append(_oi_signal(pc_, oc))
            oi_tags.append(buildup_tag(pc_, oc))
            oi_chgs.append(oc)
        else:
            oi_sig.append(0.0)
            oi_tags.append("")
            oi_chgs.append(0.0)
    oi_sig = np.array(oi_sig)
    news = np.array([news_scores.get(s.symbol, (0.0, None))[0] for s in live])
    obi = np.array([((s.tbq - s.tsq) / (s.tbq + s.tsq)) if (s.tbq + s.tsq) else 0 for s in live])

    w = cfg.weights
    zr, zres, zv, zo, zoi, zn, zs, zb = (_z(lrvol), _z(resid), _z(vwap_d), _z(orb), _z(oi_sig),
                                         _z(news), _z(sec_rel), _z(obi))
    long_dir = w["resid"] * zres + w["vwap"] * zv + w["orb"] * zo + w["oi"] * zoi + w["news"] * zn \
        + w["sector"] * zs + w["obi"] * zb
    vol_term = w["rvol"] * zr
    score_long = long_dir + vol_term
    score_short = -long_dir + vol_term

    rows = []
    for i, s in enumerate(live):
        turnover = s.turnover_cr
        ns, nev = news_scores.get(s.symbol, (0.0, None))
        rows.append({
            "sid": s.sid, "symbol": s.symbol, "sector": s.sector,
            "ltp": safe(s.ltp), "chg": safe(chg[i]), "gap": safe(s.gap), "pc": safe(s.pc),
            "open": safe(s.open), "high": safe(s.high), "low": safe(s.low), "vwap": safe(s.vwap),
            "rvol": safe(rvol[i]), "resid": safe(resid[i]), "vwap_atr": safe(vwap_d[i]),
            "or_pos": safe(orpos[i]), "orh": safe(s.orh), "orl": safe(s.orl), "orb": safe(orb[i]),
            "oi_tag": oi_tags[i], "oi_chg": safe(oi_chgs[i]), "news": safe(ns), "obi": safe(obi[i]),
            "headline": nev["title"] if nev else None, "news_type": nev["type"] if nev else None,
            "turnover": safe(turnover), "atr": safe(atr[i]), "atr_pct": safe(s.ctx.atr_pct),
            "beta": safe(beta[i]), "pdh": safe(s.ctx.pdh), "pdl": safe(s.ctx.pdl),
            "score": safe(score_long[i], 3), "score_s": safe(score_short[i], 3),
            "f": {"rvol": safe(zr[i]), "resid": safe(zres[i]), "vwap": safe(zv[i]), "orb": safe(zo[i]),
                  "oi": safe(zoi[i]), "news": safe(zn[i]), "sector": safe(zs[i]), "obi": safe(zb[i])},
            "eligible": bool(turnover >= cfg.min_turnover_cr * min(1.0, max(0.15, (minute + 1) / 15.0))
                             and rvol[i] >= cfg.min_rvol),
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    for k, r in enumerate(rows):
        r["rank"] = k + 1
    return rows


def pick(rows: list[dict], cfg: RankConfig, side: str = "long") -> list[dict]:
    """Top-N with eligibility, direction and sector caps. May return fewer than N (abstain)."""
    key = "score" if side == "long" else "score_s"
    cands = sorted(rows, key=lambda r: r[key], reverse=True)
    out, per_sector = [], {}
    for r in cands:
        if len(out) >= cfg.top_n:
            break
        if not r["eligible"] or r[key] < cfg.min_score:
            continue
        if side == "long" and (r["vwap_atr"] < 0 or r["resid"] <= 0):
            continue
        if side == "short" and (r["vwap_atr"] > 0 or r["resid"] >= 0):
            continue
        if per_sector.get(r["sector"], 0) >= cfg.max_per_sector:
            continue
        per_sector[r["sector"]] = per_sector.get(r["sector"], 0) + 1
        out.append(r)
    return out


def factor_chips(r: dict, side: str = "long") -> list[str]:
    sgn = 1 if side == "long" else -1
    chips = []
    if r["rvol"] >= 2:
        chips.append(f"RVOL {r['rvol']:.1f}x")
    if abs(r["gap"]) >= 1:
        chips.append(f"GAP {r['gap']:+.1f}%")
    if sgn * r["orb"] >= 1:
        chips.append("ORB" if side == "long" else "ORBD")
    if sgn * r["vwap_atr"] > 0.15:
        chips.append("VWAP+" if side == "long" else "VWAP-")
    if r["oi_tag"]:
        chips.append(r["oi_tag"])
    if sgn * r["news"] > 0.15:
        chips.append(r["news_type"] or "NEWS")
    if sgn * r["resid"] > 1:
        chips.append("RS")
    return chips


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 5:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    c = np.corrcoef(ra, rb)[0, 1]
    return float(c) if math.isfinite(c) else float("nan")
