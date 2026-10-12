"""Derived GEO views: chokepoint status, fires at energy sites, airspace monitor,
the live geo-risk index and the market-impact map. Pure functions over normalized inputs,
shared by the live and simulated modes (and unit-tested).
"""
from __future__ import annotations

import math
import time
from statistics import median

from .sites import FNO, haversine_km, region_of

# ------------------------------------------------------------------ chokepoints
NEAR_KM = 150          # conflict this close to a strait counts against it


def chokepoint_view(chokepoints: list[dict], ships: dict[int, dict], transits: dict[str, dict[int, float]],
                    portwatch: dict[str, dict], events: list[dict], now: float) -> list[dict]:
    """Live AIS counts + PortWatch transit trend + nearby conflict -> status per chokepoint."""
    out = []
    for cp in chokepoints:
        inside = [s for s in ships.values()
                  if haversine_km(cp["lat"], cp["lon"], s["lat"], s["lon"]) <= cp["r_km"]]
        tankers = [s for s in inside if s.get("cls") == "TANKER"]
        speeds = [s["sog"] for s in inside if s.get("sog") is not None and s["sog"] > 0.5]
        log24 = {m: t for m, t in transits.get(cp["id"], {}).items() if now - t < 86400}
        near = [e for e in events if now - e["ts"] < 72 * 3600
                and haversine_km(cp["lat"], cp["lon"], e["lat"], e["lon"]) <= NEAR_KM]
        conflict = round(sum(e["sev"] for e in near), 1)
        pw = portwatch.get(cp["id"]) or {}
        ratio = pw.get("ratio")
        status, reasons = "NORMAL", []
        if ratio is not None and ratio < 0.5:
            status = "DISRUPTED"
            reasons.append(f"transits {ratio:.0%} of normal")
        elif ratio is not None and ratio < 0.8:
            status = "STRAINED"
            reasons.append(f"transits {ratio:.0%} of normal")
        if conflict >= 60:
            status = "DISRUPTED"
            reasons.append(f"{len(near)} conflict events within {NEAR_KM} km (72 h)")
        elif conflict >= 20 and status == "NORMAL":
            status = "WATCH"
            reasons.append(f"{len(near)} conflict events within {NEAR_KM} km (72 h)")
        out.append({
            "id": cp["id"], "name": cp["name"], "lat": cp["lat"], "lon": cp["lon"], "r_km": cp["r_km"],
            "india": cp["india"], "why": cp.get("why", ""), "status": status, "reasons": reasons,
            "ships_now": len(inside), "tankers_now": len(tankers),
            "avg_speed": round(sum(speeds) / len(speeds), 1) if speeds else None,
            "entered_24h": len(log24), "conflict": conflict, "conflict_n": len(near),
            "pw": {k: pw.get(k) for k in ("as_of", "last7_total", "last7_tanker", "base_total", "base_tanker",
                                          "ratio", "tanker_ratio")} if pw else None,
            "series": pw.get("series", [])[-60:] if pw else [],
        })
    return out


def update_transits(chokepoints: list[dict], ships: dict[int, dict], transits: dict[str, dict[int, float]],
                    now: float) -> None:
    """Record when each vessel is first seen inside a chokepoint circle (kept 24 h)."""
    for cp in chokepoints:
        log = transits.setdefault(cp["id"], {})
        for m, s in ships.items():
            if m not in log and haversine_km(cp["lat"], cp["lon"], s["lat"], s["lon"]) <= cp["r_km"]:
                log[m] = now
        for m in [m for m, t in log.items() if now - t > 86400]:
            del log[m]


# ------------------------------------------------------------------ fires at energy sites
def fire_sites(sites: list[dict], hotspots: list[dict], history: dict[str, list[int]], now: float) -> list[dict]:
    """Match satellite hotspots (last 36 h) to energy sites and flag unusual activity.

    history[site_id] = daily hotspot counts for previous days (oldest first), used as the
    site's own baseline — refineries and gas fields flare every day, so only a jump counts.
    """
    recent = [h for h in hotspots if now - h["ts"] < 36 * 3600]
    out = []
    for s in sites:
        r = s.get("r_km", 8)
        near = [h for h in recent if abs(h["lat"] - s["lat"]) < 1 and haversine_km(s["lat"], s["lon"], h["lat"],
                                                                                    h["lon"]) <= r]
        n = len(near)
        frp = round(sum(h["frp"] for h in near), 1)
        hist = history.get(s["id"], [])
        base = median(hist) if len(hist) >= 5 else None
        if n == 0:
            status = "QUIET"
        elif base is None:
            status = "ANOMALY" if frp >= 150 or (not s.get("flare") and n >= 4) else "LEARNING"
        elif n >= max(4, 3 * base + 2) or (frp >= 150 and n > base + 1):
            status = "ANOMALY"
        else:
            status = "FLARING" if s.get("flare") or base > 0 else "ACTIVE"
        out.append({"id": s["id"], "name": s["name"], "kind": s["kind"], "lat": s["lat"], "lon": s["lon"],
                    "country": s.get("country", ""), "owner": s.get("owner", ""), "symbols": s.get("symbols", {}),
                    "weight": s.get("weight", 1), "hotspots": n, "frp": frp,
                    "baseline": None if base is None else round(base, 1), "status": status,
                    "last": max((h["ts"] for h in near), default=None)})
    out.sort(key=lambda x: ({"ANOMALY": 0, "ACTIVE": 1, "LEARNING": 2, "FLARING": 3, "QUIET": 4}[x["status"]],
                            -x["frp"]))
    return out


# ------------------------------------------------------------------ airspace
def airspace(regions: list[dict], counts: dict[str, int], baselines: dict[str, list[int]]) -> list[dict]:
    out = []
    for r in regions:
        n = counts.get(r["id"])
        hist = baselines.get(r["id"], [])
        base = median(hist) if len(hist) >= 3 else None
        ratio = round(n / base, 2) if (n is not None and base) else None
        if n is None:
            status = "NO DATA"
        elif base is None:
            status = "LEARNING"
        elif base >= 12 and ratio is not None and ratio < 0.3:
            status = "CLOSED?"
        elif base >= 8 and ratio is not None and ratio < 0.6:
            status = "REDUCED"
        else:
            status = "NORMAL"
        out.append({"id": r["id"], "name": r["name"], "lat": r["lat"], "lon": r["lon"], "nm": r["nm"],
                    "india": r["india"], "count": n, "baseline": base, "ratio": ratio, "status": status})
    return out


# ------------------------------------------------------------------ conflict summary
def conflict_summary(events: list[dict], now: float) -> list[dict]:
    """Severity by region over the last 24 h, India-weighted."""
    agg: dict[str, dict] = {}
    for e in events:
        if now - e["ts"] > 86400:
            continue
        reg = region_of(e["lat"], e["lon"])
        a = agg.setdefault(reg["id"], {"id": reg["id"], "name": reg["name"], "india": reg["india"], "n": 0,
                                       "sev": 0.0, "top": None})
        a["n"] += 1
        a["sev"] += e["sev"]
        if a["top"] is None or e["sev"] > a["top"]["sev"]:
            a["top"] = {"label": e["label"], "place": e["place"], "sev": e["sev"], "url": e["url"]}
    rows = list(agg.values())
    for a in rows:
        a["sev"] = round(a["sev"], 1)
        a["weighted"] = round(a["sev"] * a["india"], 1)
    rows.sort(key=lambda x: -x["weighted"])
    return rows


# ------------------------------------------------------------------ risk index
def _squash(x: float, half: float) -> float:
    """0..100, reaching 50 at `half`."""
    return 100.0 * (1 - math.exp(-math.log(2) * max(0.0, x) / half)) if half > 0 else 0.0


def risk_index(regions: list[dict], chokes: list[dict], fires: list[dict], air: list[dict], macro: dict,
               conflict_base: float | None = None) -> dict:
    """Live geo-risk 0–100 for an Indian-market trader (our own proxy, not the academic GPR).

    conflict   35%  India-weighted conflict severity (last 24 h), vs its own trailing median once a day of
                    history exists (fixed scale before that)
    shipping   25%  chokepoint disruption (transits vs normal, conflict near the strait)
    energy     15%  unusual fire activity at refineries / terminals / gas fields
    airspace   10%  watched airspaces far below their usual traffic
    markets    15%  crude and rupee moves, India VIX
    """
    w_conflict = sum(r["weighted"] for r in regions)
    if conflict_base:
        c_score = _squash(w_conflict / conflict_base, 1.3)     # a normal day for this feed scores ~40
    else:
        c_score = _squash(w_conflict, 1500)
    stress = {"NORMAL": 0, "WATCH": 35, "STRAINED": 60, "DISRUPTED": 100}
    ship_parts = [stress.get(c["status"], 0) * c["india"] for c in chokes]
    s_score = min(100.0, max(ship_parts, default=0) + 0.25 * sum(sorted(ship_parts)[:-1]))
    e_score = min(100.0, sum(20 * f["weight"] for f in fires if f["status"] == "ANOMALY"))
    a_parts = [(100 if a["status"] == "CLOSED?" else 50 if a["status"] == "REDUCED" else 0) * a["india"]
               for a in air]
    a_score = min(100.0, max(a_parts, default=0) + 0.3 * sum(sorted(a_parts)[:-1]))
    crude = abs(macro.get("crude_chg") or 0.0)
    inr = abs(macro.get("usdinr_chg") or 0.0)
    vix = macro.get("india_vix") or 0.0
    m_score = min(100.0, _squash(crude, 2.0) * 0.5 + _squash(inr, 0.4) * 0.25 + _squash(max(0.0, vix - 12), 6) * 0.25)
    comps = {"conflict": c_score, "shipping": s_score, "energy": e_score, "airspace": a_score, "markets": m_score}
    weights = {"conflict": 0.35, "shipping": 0.25, "energy": 0.15, "airspace": 0.10, "markets": 0.15}
    value = sum(comps[k] * weights[k] for k in comps)
    drivers = sorted(comps, key=lambda k: -comps[k] * weights[k])
    level = "LOW" if value < 25 else "ELEVATED" if value < 45 else "HIGH" if value < 65 else "SEVERE"
    return {"value": round(value, 1), "level": level, "components": {k: round(v, 1) for k, v in comps.items()},
            "weights": weights, "drivers": drivers[:3], "conflict_raw": round(w_conflict, 1)}


# ------------------------------------------------------------------ market impact
OMC = ["BPCL", "HINDPETRO", "IOC"]
UPSTREAM = ["ONGC", "OIL"]
CRUDE_USERS = ["ASIANPAINT", "BERGEPAINT", "PIDILITIND", "INDIGO", "APOLLOTYRE", "MRF", "BALKRISIND"]
IT = ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "PERSISTENT", "COFORGE"]
DEFENCE = ["HAL", "BEL", "BDL", "MAZDOCK", "BHARATFORG", "SOLARINDS"]
GOLD_LENDERS = ["MUTHOOTFIN", "MANAPPURAM"]
GAS = ["PETRONET", "GAIL", "IGL"]


def _syms(names: list[str], sign: int) -> list[dict]:
    return [{"s": n, "dir": sign} for n in names if n in FNO]


def impact(chokes: list[dict], fires: list[dict], regions: list[dict], air: list[dict], macro: dict,
           now: float | None = None) -> list[dict]:
    """Rule-based map from geo signals to NSE names. Transparent heuristics — every rule says why."""
    now = now or time.time()
    out: list[dict] = []

    def add(sid, title, sev, syms, why):
        out.append({"id": sid, "title": title, "sev": sev, "symbols": syms, "why": why, "ts": now})

    crude = macro.get("crude_chg")
    if crude is not None and abs(crude) >= 1.5:
        up = crude > 0
        add("crude", f"Crude {'up' if up else 'down'} {crude:+.1f}%", 3 if abs(crude) >= 3 else 2,
            _syms(OMC, -1 if up else 1) + _syms(UPSTREAM, 1 if up else -1) + _syms(CRUDE_USERS, -1 if up else 1),
            "Oil marketers and crude-input users lose margin when crude rises; upstream producers gain.")
    inr = macro.get("usdinr_chg")
    if inr is not None and abs(inr) >= 0.3:
        weak = inr > 0
        add("inr", f"Rupee {'weaker' if weak else 'stronger'} ({inr:+.2f}% USD/INR)", 2,
            _syms(IT, 1 if weak else -1) + _syms(OMC, -1 if weak else 1),
            "A weaker rupee lifts dollar-earning IT services and raises import costs for oil marketers.")
    gold = macro.get("gold_chg")
    if gold is not None and gold >= 1.5:
        add("gold", f"Gold up {gold:+.1f}% (risk-off)", 1, _syms(GOLD_LENDERS, 1),
            "Higher gold raises collateral values for gold-loan companies.")
    gulf = [c for c in chokes if c["id"] in ("hormuz", "bab", "suez") and c["status"] in ("STRAINED", "DISRUPTED")]
    if gulf:
        names = ", ".join(c["name"] for c in gulf)
        add("chokepoints", f"Shipping stress: {names}", 3 if any(c["status"] == "DISRUPTED" for c in gulf) else 2,
            _syms(OMC, -1) + _syms(UPSTREAM, 1) + _syms(GAS, -1),
            "Gulf/Red Sea disruption lifts freight, insurance and crude/LNG prices for an import-dependent India.")
    for f in fires:
        if f["status"] == "ANOMALY" and f.get("symbols"):
            add(f"fire:{f['id']}", f"Unusual fire activity at {f['name']}", 3 if f["weight"] >= 2 else 2,
                [{"s": s, "dir": d} for s, d in f["symbols"].items()],
                f"{f['hotspots']} satellite hotspots ({f['frp']:.0f} MW) vs a usual {f['baseline'] or 0:g}/day. "
                "Could be flaring or maintenance — check imagery before acting.")
        elif f["status"] == "ANOMALY" and f["country"] in ("SA", "AE", "QA", "IR", "IQ", "KW", "RU"):
            add(f"fire:{f['id']}", f"Unusual fire activity at {f['name']} ({f['country']})", 2,
                _syms(OMC, -1) + _syms(UPSTREAM, 1),
                "A hit on a major exporter's facility tightens crude supply; India imports most of its crude.")
    ip = next((r for r in regions if r["id"] == "india_pak"), None)
    if ip and ip["sev"] >= 60:
        add("india_pak", "India–Pakistan conflict activity elevated", 3 if ip["sev"] >= 150 else 2,
            _syms(DEFENCE, 1) + _syms(["INDIGO"], -1),
            f"{ip['n']} events in 24 h. Defence names tend to rally on escalation; airlines suffer airspace closures.")
    ic = next((r for r in regions if r["id"] == "china_india"), None)
    if ic and ic["sev"] >= 40:
        add("china_india", "India–China border activity elevated", 2, _syms(DEFENCE, 1),
            f"{ic['n']} events in 24 h on the northern border.")
    closed = [a for a in air if a["status"] in ("CLOSED?", "REDUCED") and a["india"] >= 0.7]
    if closed:
        add("airspace", "Airspace avoidance: " + ", ".join(a["name"] for a in closed), 2, _syms(["INDIGO"], -1),
            "Closed or avoided airspace means longer routes and higher fuel burn for Indian carriers.")
    out.sort(key=lambda x: -x["sev"])
    return out
