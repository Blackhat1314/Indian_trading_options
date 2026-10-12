"""Option math: Black–Scholes price/greeks, implied volatility, and chain analytics
(PCR, max pain, GEX by strike, walls, expected move, skew)."""
from __future__ import annotations

import math

_SQRT2 = math.sqrt(2.0)


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _d1d2(S, K, T, r, sigma):
    sigma = max(sigma, 1e-6)
    T = max(T, 1e-6)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, cp: str = "call") -> float:
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if cp == "call":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, cp: str = "call") -> dict:
    d1, d2 = _d1d2(S, K, T, r, sigma)
    pdf = _npdf(d1)
    sqT = math.sqrt(max(T, 1e-6))
    gamma = pdf / (S * max(sigma, 1e-6) * sqT)
    vega = S * pdf * sqT / 100.0                     # per 1 vol point
    if cp == "call":
        delta = _ncdf(d1)
        theta = (-S * pdf * sigma / (2 * sqT) - r * K * math.exp(-r * T) * _ncdf(d2)) / 365.0
    else:
        delta = _ncdf(d1) - 1
        theta = (-S * pdf * sigma / (2 * sqT) + r * K * math.exp(-r * T) * _ncdf(-d2)) / 365.0
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def implied_vol(price: float, S: float, K: float, T: float, r: float = 0.065, cp: str = "call") -> float | None:
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max(0.0, S - K) if cp == "call" else max(0.0, K - S)
    if price < intrinsic - 1e-6:
        return None
    lo, hi = 1e-4, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, mid, cp) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-5:
            break
    return 0.5 * (lo + hi)


def normalize_chain(raw: dict) -> tuple[float, list[dict]]:
    """Dhan option-chain payload -> (spot, rows sorted by strike)."""
    spot = float(raw.get("last_price") or 0)
    rows = []
    for k, v in (raw.get("oc") or {}).items():
        try:
            K = float(k)
        except ValueError:
            continue
        ce, pe = v.get("ce") or {}, v.get("pe") or {}

        def side(d, p):
            g = d.get("greeks") or {}
            oi = int(d.get("oi") or 0)
            poi = int(d.get("previous_oi") or 0)
            return {
                f"{p}_ltp": float(d.get("last_price") or 0), f"{p}_oi": oi, f"{p}_oi_chg": oi - poi if poi else 0,
                f"{p}_vol": int(d.get("volume") or 0), f"{p}_iv": float(d.get("implied_volatility") or 0),
                f"{p}_delta": float(g.get("delta") or 0), f"{p}_gamma": float(g.get("gamma") or 0),
                f"{p}_theta": float(g.get("theta") or 0), f"{p}_vega": float(g.get("vega") or 0),
                f"{p}_bid": float(d.get("top_bid_price") or 0), f"{p}_ask": float(d.get("top_ask_price") or 0),
                f"{p}_prev": float(d.get("previous_close_price") or 0),
            }
        row = {"strike": K}
        row.update(side(ce, "ce"))
        row.update(side(pe, "pe"))
        rows.append(row)
    rows.sort(key=lambda r: r["strike"])
    return spot, rows


def max_pain(rows: list[dict]) -> float | None:
    if not rows:
        return None
    strikes = [r["strike"] for r in rows]
    best, best_k = None, None
    for k in strikes:
        pay = 0.0
        for r in rows:
            K = r["strike"]
            pay += r["ce_oi"] * max(0.0, k - K) + r["pe_oi"] * max(0.0, K - k)
        if best is None or pay < best:
            best, best_k = pay, k
    return best_k


def years_to_expiry(expiry: str | None, now: float | None = None) -> float | None:
    """Expiry date (YYYY-MM-DD, 15:30 IST) -> years remaining, floored at one trading hour."""
    if not expiry:
        return None
    import time as _t
    from datetime import datetime, timedelta, timezone
    try:
        exp = datetime.fromisoformat(expiry[:10]).replace(hour=15, minute=30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    except ValueError:
        return None
    secs = exp.timestamp() - (now or _t.time())
    return max(secs, 3600.0) / (365.0 * 86400)


def fill_missing_greeks(rows: list[dict], spot: float, T: float | None, r: float = 0.065) -> None:
    """Dhan returns IV/greeks of 0 for strikes that barely trade (deep ITM, far OTM). Solve them from the
    last price where possible; when the price is stale (below intrinsic) show no IV and the intrinsic delta."""
    if not T or spot <= 0:
        return
    for row in rows:
        K = row["strike"]
        for p, cp in (("ce", "call"), ("pe", "put")):
            if row[f"{p}_iv"] and row[f"{p}_delta"]:
                continue
            price = row[f"{p}_ltp"]
            iv = implied_vol(price, spot, K, T, r, cp) if price > 0 else None
            if iv and 0.02 < iv < 3.0:
                g = bs_greeks(spot, K, T, r, iv, cp)
                row[f"{p}_iv"] = round(iv * 100, 2)
                for k in ("delta", "gamma", "theta", "vega"):
                    row[f"{p}_{k}"] = g[k]
            else:
                itm = (spot > K) if cp == "call" else (spot < K)
                row[f"{p}_iv"] = None
                if not row[f"{p}_delta"]:
                    row[f"{p}_delta"] = (1.0 if cp == "call" else -1.0) if itm else 0.0


def chain_analytics(raw: dict, lot: int = 1, name: str = "", expiry: str | None = None) -> dict:
    spot, rows = normalize_chain(raw)
    if not rows or spot <= 0:
        return {"name": name, "spot": spot, "rows": []}
    fill_missing_greeks(rows, spot, years_to_expiry(expiry))
    ce_oi = sum(r["ce_oi"] for r in rows)
    pe_oi = sum(r["pe_oi"] for r in rows)
    ce_vol = sum(r["ce_vol"] for r in rows)
    pe_vol = sum(r["pe_vol"] for r in rows)
    atm = min(rows, key=lambda r: abs(r["strike"] - spot))
    atm_iv = (atm["ce_iv"] + atm["pe_iv"]) / 2 if atm["ce_iv"] and atm["pe_iv"] else atm["ce_iv"] or atm["pe_iv"]
    straddle = atm["ce_ltp"] + atm["pe_ltp"]
    # dealer-gamma convention: dealers long calls / short puts sold to customers is unknown,
    # so we use the common retail convention GEX = (call_gamma*call_OI - put_gamma*put_OI) * S^2 * 1%
    gex_rows = []
    total_gex = 0.0
    for r in rows:
        g = (r["ce_gamma"] * r["ce_oi"] - r["pe_gamma"] * r["pe_oi"]) * spot * spot * 0.01
        total_gex += g
        gex_rows.append({"strike": r["strike"], "gex": g})
    # gamma flip: strike where cumulative GEX changes sign (nearest to spot)
    flip = None
    cum = 0.0
    prev = None
    for gr in gex_rows:
        cum += gr["gex"]
        if prev is not None and (prev < 0) != (cum < 0):
            if flip is None or abs(gr["strike"] - spot) < abs(flip - spot):
                flip = gr["strike"]
        prev = cum
    call_wall = max(rows, key=lambda r: r["ce_oi"])["strike"]
    put_wall = max(rows, key=lambda r: r["pe_oi"])["strike"]
    # 25-delta skew approximation
    def iv_at_delta(target: float, side: str):
        cand = [r for r in rows if r[f"{side}_delta"] and r[f"{side}_iv"]]
        if not cand:
            return None
        return min(cand, key=lambda r: abs(r[f"{side}_delta"] - target))[f"{side}_iv"]
    put25, call25 = iv_at_delta(-0.25, "pe"), iv_at_delta(0.25, "ce")
    skew = (put25 - call25) if put25 and call25 else None
    near = [r for r in rows if abs(r["strike"] - spot) / spot < 0.06] or rows
    return {
        "name": name,
        "spot": round(spot, 2),
        "atm": atm["strike"],
        "pcr_oi": round(pe_oi / ce_oi, 3) if ce_oi else None,
        "pcr_vol": round(pe_vol / ce_vol, 3) if ce_vol else None,
        "max_pain": max_pain(rows),
        "atm_iv": round(atm_iv, 2) if atm_iv else None,
        "straddle": round(straddle, 2),
        "expected_move_pct": round(straddle / spot * 100, 2) if spot else None,
        "skew_25d": round(skew, 2) if skew is not None else None,
        "total_gex": total_gex,
        "gamma_flip": flip,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "ce_oi_total": ce_oi,
        "pe_oi_total": pe_oi,
        "ce_oi_chg_total": sum(r["ce_oi_chg"] for r in rows),
        "pe_oi_chg_total": sum(r["pe_oi_chg"] for r in rows),
        "rows": near,
        "gex": [g for g in gex_rows if abs(g["strike"] - spot) / spot < 0.06],
        "lot": lot,
    }
