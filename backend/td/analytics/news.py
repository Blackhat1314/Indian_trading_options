"""News → catalysts.

Turns headlines into structured events (type, direction, impact, novelty) and keeps a
time-decayed catalyst score per stock that feeds the intraday ranker. Rules are explicit
and testable; an LLM classifier can be plugged in via `set_llm_classifier`.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from collections import deque
from collections.abc import Callable

# event type -> (regex, base impact weight)
EVENT_RULES: list[tuple[str, re.Pattern, float]] = [
    ("RESULTS", re.compile(r"\b(q[1-4]|quarter|results?|net profit|pat|revenue|ebitda|earnings|beats?|miss(es)?)\b", re.I), 1.0),
    ("ORDER_WIN", re.compile(r"\b(bags?|wins?|secures?|order|contract|loa|letter of award)\b", re.I), 0.9),
    ("BLOCK_DEAL", re.compile(r"\b(block deal|bulk deal|offload|stake sale|ofs)\b", re.I), 0.7),
    ("M&A", re.compile(r"\b(acquir\w*|merger|amalgamation|stake|buyout|takeover|demerg\w*)\b", re.I), 0.8),
    ("RATING", re.compile(r"\b(upgrade[sd]?|downgrade[sd]?|target price|overweight|underweight|initiates? coverage)\b", re.I), 0.6),
    ("REGULATORY", re.compile(r"\b(usfda|sebi|rbi|cci|nclt|penalty|ban|approval|observations?|warning letter|probe|raid)\b", re.I), 0.8),
    ("MANAGEMENT", re.compile(r"\b(ceo|cfo|md|chairman|resigns?|appoint\w*|steps down)\b", re.I), 0.6),
    ("POLICY", re.compile(r"\b(government|pli|scheme|policy|duty|tariff|subsidy|gst|budget)\b", re.I), 0.6),
    ("GEOPOLITICAL", re.compile(r"\b(war|missile|strike|sanctions?|hormuz|red sea|border|ceasefire|attack|conflict)\b", re.I), 0.7),
    ("MACRO", re.compile(r"\b(repo rate|inflation|cpi|wpi|gdp|fiis?|diis?|fpis?|crude|rupee|fed|yields?|gift nifty|sgx nifty|net (sold|bought)|weak start|strong start|flat start)\b", re.I), 0.3),
]

POS = re.compile(r"\b(jumps?|surges?|soars?|beats?|rises?|gains?|record|upgrade[sd]?|bags?|wins?|approval|"
                 r"raises?|strong|robust|buys?|expands?|incentive|ceasefire)\b", re.I)
NEG = re.compile(r"\b(falls?|drops?|plunges?|slumps?|miss(es)?|downgrade[sd]?|resigns?|penalty|ban|probe|"
                 r"observations?|warning|weak|cuts?|sells?|loss|decline[sd]?|raid|default|war|attack)\b", re.I)

# headline keyword -> sector impacts (for macro / geopolitical news without a stock tag)
SECTOR_RULES: list[tuple[re.Pattern, dict[str, int]]] = [
    (re.compile(r"\b(crude|oil prices?|brent|hormuz|opec)\b.*\b(rise|jump|surge|spike|up|soar)", re.I),
     {"ENERGY_UPSTREAM": 1, "ENERGY_OMC": -1, "PAINTS": -1, "AVIATION": -1}),
    (re.compile(r"\b(crude|oil prices?|brent)\b.*\b(fall|drop|slump|plunge|down|ease)", re.I),
     {"ENERGY_UPSTREAM": -1, "ENERGY_OMC": 1, "PAINTS": 1, "AVIATION": 1}),
    (re.compile(r"\b(defen[cs]e|missile|war|border|military)\b", re.I), {"DEFENCE": 1}),
    (re.compile(r"\b(rate cut|cuts? repo)\b", re.I), {"BANK": 1, "REALTY": 1, "AUTO": 1}),
    (re.compile(r"\b(rate hike|hikes? repo)\b", re.I), {"BANK": -1, "REALTY": -1}),
]
GROUPS = {
    "ENERGY_UPSTREAM": ["ONGC", "OIL", "RELIANCE"],
    "ENERGY_OMC": ["IOC", "BPCL", "HINDPETRO"],
    "PAINTS": ["ASIANPAINT", "BERGEPAINT"],
    "AVIATION": ["INDIGO"],
}

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "in", "to", "for", "on", "and", "by", "at", "its", "with", "from", "as", "is", "rs", "cr"}


def _tokens(s: str) -> set[str]:
    return {t for t in _TOKEN.findall(s.lower()) if t not in _STOP and len(t) > 1}


def classify(title: str, sentiment: str | None = None) -> dict:
    etype, weight = "OTHER", 0.4
    for name, rx, w in EVENT_RULES:
        if rx.search(title):
            etype, weight = name, w
            break
    if sentiment in ("positive", "negative", "neutral"):
        sign = {"positive": 1, "negative": -1, "neutral": 0}[sentiment]
    else:
        p, n = len(POS.findall(title)), len(NEG.findall(title))
        sign = 1 if p > n else -1 if n > p else 0
    return {"type": etype, "weight": weight, "sign": sign}


class NewsEngine:
    def __init__(self, sector_of: dict[str, str], half_life_min: float = 75.0, max_items: int = 400):
        self.sector_of = sector_of
        self.symbols_by_sector: dict[str, list[str]] = {}
        for s, sec in sector_of.items():
            self.symbols_by_sector.setdefault(sec, []).append(s)
        self.items: deque[dict] = deque(maxlen=max_items)
        self.seen: set[str] = set()
        self.by_symbol: dict[str, list[dict]] = {}
        self.tau = half_life_min / math.log(2)
        self._llm: Callable[[str], dict] | None = None

    def set_llm_classifier(self, fn: Callable[[str], dict]) -> None:
        """Optional: fn(title) -> {"type","sign","weight","symbols"} to override rules."""
        self._llm = fn

    def _novelty(self, symbol: str, toks: set[str], ts: float) -> float:
        best = 0.0
        for prev in self.by_symbol.get(symbol, [])[-20:]:
            if ts - prev["ts"] > 86400:
                continue
            inter = len(toks & prev["_tok"])
            union = len(toks | prev["_tok"]) or 1
            best = max(best, inter / union)
        return 1.0 if best < 0.5 else 0.3            # repeated story → 30% weight

    def add(self, item: dict) -> dict | None:
        title = (item.get("title") or "").strip()
        if not title:
            return None
        key = item.get("id") or hashlib.md5(title.lower().encode()).hexdigest()
        if key in self.seen:
            return None
        self.seen.add(key)
        cls = classify(title, item.get("sentiment"))
        if self._llm:
            try:
                cls.update({k: v for k, v in self._llm(title).items() if v is not None})
            except Exception:
                pass
        ts = float(item.get("ts") or time.time())
        toks = _tokens(title)
        symbols = list(dict.fromkeys(item.get("symbols") or []))
        impacts: dict[str, int] = {s: cls["sign"] for s in symbols}
        if not symbols:
            for rx, rules in SECTOR_RULES:
                if rx.search(title):
                    for grp, sign in rules.items():
                        names = GROUPS.get(grp) or self.symbols_by_sector.get(grp, [])
                        for s in names:
                            if s in self.sector_of:
                                impacts[s] = sign
        novelty = min((self._novelty(s, toks, ts) for s in impacts), default=1.0)
        ev = {
            "id": key, "ts": ts, "title": title, "text": item.get("text", ""), "source": item.get("source", "DHAN"),
            "category": item.get("category", ""), "type": cls["type"], "sign": cls["sign"],
            "weight": cls["weight"] * novelty, "novel": novelty == 1.0, "symbols": list(impacts),
            "impacts": impacts, "sentiment": item.get("sentiment") or ("positive" if cls["sign"] > 0 else
                                                                      "negative" if cls["sign"] < 0 else "neutral"),
            "_tok": toks,
        }
        self.items.appendleft(ev)
        for s in impacts:
            self.by_symbol.setdefault(s, []).append(ev)
        return ev

    def score(self, symbol: str, now: float, session_open: float | None = None) -> tuple[float, dict | None]:
        """Signed, time-decayed catalyst score and the strongest contributing headline.
        Overnight / pre-market news is treated as fresh at the open (decay starts at 09:15)."""
        total, top, top_v = 0.0, None, 0.0
        for ev in self.by_symbol.get(symbol, [])[-15:]:
            if now - ev["ts"] > 24 * 3600:
                continue
            start = max(ev["ts"], session_open) if session_open else ev["ts"]
            age_min = max(0.0, (now - start) / 60.0)
            v = ev["impacts"].get(symbol, 0) * ev["weight"] * math.exp(-age_min / self.tau)
            total += v
            if abs(v) > abs(top_v):
                top_v, top = v, ev
        return total, top

    def public(self, ev: dict) -> dict:
        return {k: v for k, v in ev.items() if not k.startswith("_")}

    def latest(self, n: int = 150) -> list[dict]:
        return [self.public(e) for e in list(self.items)[:n]]
