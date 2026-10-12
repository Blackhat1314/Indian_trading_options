"""User alerts evaluated against the live ranking rows every refresh.

Alert definition (stored in Redis hash td:alerts):
  {"id", "user", "symbol", "type", "value", "note", "repeat": bool, "active": bool, "created"}
Types:
  PRICE_ABOVE / PRICE_BELOW    value = price
  CHG_ABOVE / CHG_BELOW        value = % change vs prev close
  RVOL_ABOVE                   value = relative volume multiple
  VWAP_CROSS_UP / _DOWN        no value
  ORB_UP / ORB_DOWN            opening-range breakout / breakdown
  OI_BUILDUP                   value = LB | SB | SC | LU
  NEWS                         any new catalyst headline for the symbol
  TOP_PICK                     symbol enters the top-N intraday ranking
"""
from __future__ import annotations

import time
import uuid

TYPES = ["PRICE_ABOVE", "PRICE_BELOW", "CHG_ABOVE", "CHG_BELOW", "RVOL_ABOVE", "VWAP_CROSS_UP",
         "VWAP_CROSS_DOWN", "ORB_UP", "ORB_DOWN", "OI_BUILDUP", "NEWS", "TOP_PICK"]


def new_alert(user: str, symbol: str, typ: str, value=None, note: str = "", repeat: bool = False) -> dict:
    if typ not in TYPES:
        raise ValueError(f"unknown alert type {typ}")
    return {"id": uuid.uuid4().hex[:10], "user": user, "symbol": symbol.upper(), "type": typ, "value": value,
            "note": note, "repeat": repeat, "active": True, "created": time.time(), "fired": 0}


class AlertEngine:
    def __init__(self):
        self.prev: dict[str, dict] = {}            # symbol -> previous row (for crosses)
        self.cooldown: dict[str, float] = {}

    def evaluate(self, alerts: list[dict], rows_by_symbol: dict[str, dict], top_symbols: set[str],
                 news_symbols: set[str], now: float) -> list[dict]:
        fired = []
        for a in alerts:
            if not a.get("active"):
                continue
            r = rows_by_symbol.get(a["symbol"])
            if not r:
                continue
            p = self.prev.get(a["symbol"], r)
            t, v = a["type"], a.get("value")
            hit = False
            try:
                if t == "PRICE_ABOVE":
                    hit = r["ltp"] >= float(v) > p["ltp"] or (p is r and r["ltp"] >= float(v))
                elif t == "PRICE_BELOW":
                    hit = r["ltp"] <= float(v) < p["ltp"] or (p is r and r["ltp"] <= float(v))
                elif t == "CHG_ABOVE":
                    hit = r["chg"] >= float(v)
                elif t == "CHG_BELOW":
                    hit = r["chg"] <= float(v)
                elif t == "RVOL_ABOVE":
                    hit = r["rvol"] >= float(v)
                elif t == "VWAP_CROSS_UP":
                    hit = p["vwap_atr"] <= 0 < r["vwap_atr"]
                elif t == "VWAP_CROSS_DOWN":
                    hit = p["vwap_atr"] >= 0 > r["vwap_atr"]
                elif t == "ORB_UP":
                    hit = r["orb"] >= 1 and p["orb"] < 1
                elif t == "ORB_DOWN":
                    hit = r["orb"] <= -1 and p["orb"] > -1
                elif t == "OI_BUILDUP":
                    hit = r["oi_tag"] == (v or "LB") and p.get("oi_tag") != r["oi_tag"]
                elif t == "NEWS":
                    hit = a["symbol"] in news_symbols
                elif t == "TOP_PICK":
                    hit = a["symbol"] in top_symbols
            except (TypeError, ValueError):
                hit = False
            if hit and now - self.cooldown.get(a["id"], 0) > 60:
                self.cooldown[a["id"]] = now
                a["fired"] = a.get("fired", 0) + 1
                if not a.get("repeat"):
                    a["active"] = False
                fired.append({"alert": a, "ts": now, "symbol": a["symbol"], "ltp": r["ltp"], "chg": r["chg"],
                              "msg": describe(a, r)})
        self.prev = rows_by_symbol
        return fired


def describe(a: dict, r: dict) -> str:
    t, v = a["type"], a.get("value")
    m = {
        "PRICE_ABOVE": f"{a['symbol']} crossed above {v}",
        "PRICE_BELOW": f"{a['symbol']} fell below {v}",
        "CHG_ABOVE": f"{a['symbol']} up {r['chg']:+.2f}% (≥ {v}%)",
        "CHG_BELOW": f"{a['symbol']} down {r['chg']:+.2f}% (≤ {v}%)",
        "RVOL_ABOVE": f"{a['symbol']} relative volume {r['rvol']:.1f}x (≥ {v}x)",
        "VWAP_CROSS_UP": f"{a['symbol']} reclaimed VWAP",
        "VWAP_CROSS_DOWN": f"{a['symbol']} lost VWAP",
        "ORB_UP": f"{a['symbol']} broke above opening range {r['orh']}",
        "ORB_DOWN": f"{a['symbol']} broke below opening range {r['orl']}",
        "OI_BUILDUP": f"{a['symbol']} futures {r['oi_tag']} (OI {r['oi_chg']:+.1f}%)",
        "NEWS": f"{a['symbol']} news: {r.get('headline') or 'new catalyst'}",
        "TOP_PICK": f"{a['symbol']} entered the intraday top list",
    }
    return m.get(t, f"{a['symbol']} {t}")
