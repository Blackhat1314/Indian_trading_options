"""Synthetic NSE market — lets the full system run (and be tested) without Dhan credentials.

It produces *byte-identical Dhan binary packets* (quote / full / ticker / prev-close / CAS
/ 20-level depth) that go through the very same parsers as the live feed, plus Dhan-shaped
option-chain and news payloads.

The day is generated with a factor model so the analytics have something real to find:
  r_stock = beta * market + sector + catalyst drift + idiosyncratic noise
A handful of stocks get news catalysts (gap + elevated volume + continuation or fade),
a few move with no news ("stealth" moves), the rest are noise.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from ..dhan import constants as C
from ..dhan import parser as P
from ..dhan.instruments import Universe
from ..timeutil import IST, MARKET_CLOSE, MARKET_OPEN, at, previous_weekday, trading_days_back
from ..universe_data import FNO_STOCKS, INDICES, SECTORS

SESSION_SECS = 6 * 3600 + 15 * 60     # 09:15 -> 15:30
_REF = {s: (sec, px, vol) for s, sec, px, vol in FNO_STOCKS}


def ucurve(minute: np.ndarray | float) -> np.ndarray | float:
    """Intraday volume/volatility profile (minute 0 = 09:15). Mean ≈ 1."""
    m = np.asarray(minute, dtype=float)
    open_part = 2.6 * np.exp(-m / 18.0)
    close_part = 1.3 * np.exp(-(375 - m) / 25.0)
    return 0.62 + open_part + close_part


def _round_tick(px: np.ndarray | float, tick: float = 0.05):
    return np.round(np.asarray(px) / tick) * tick


@dataclass
class Catalyst:
    idx: int
    kind: str
    sentiment: int        # +1 / -1
    gap: float            # overnight gap (log return)
    drift: float          # intraday drift after open (log return)
    rvol: float           # peak relative volume multiplier
    news_minute: int      # minute of session the news lands (negative = pre-market)


NEWS_TEMPLATES = {
    "RESULTS": [("{n} Q2 net profit jumps {x}% YoY, beats street estimates", 1),
                ("{n} Q2 profit falls {x}% YoY, misses estimates on weak margins", -1)],
    "ORDER_WIN": [("{n} bags ₹{y} crore order from {c}", 1)],
    "M&A": [("{n} to acquire {x}% stake in {t} for ₹{y} crore", 1)],
    "RATING": [("{b} upgrades {n} to Buy, raises target price by {x}%", 1),
               ("{b} downgrades {n} to Sell citing valuation concerns", -1)],
    "REGULATORY": [("{n} receives final USFDA approval for generic drug", 1),
                   ("USFDA issues {x} observations at {n} manufacturing facility", -1)],
    "BLOCK_DEAL": [("{f} buys {x} lakh shares of {n} in block deal", 1),
                   ("Promoter entity sells {x}% stake in {n} via block deal", -1)],
    "MANAGEMENT": [("{n} CFO resigns with immediate effect", -1)],
    "POLICY": [("Government approves ₹{y} crore incentive scheme benefiting {n}", 1)],
}
CLIENTS = ["NHAI", "Indian Railways", "Ministry of Defence", "NTPC", "a Middle East utility", "Coal India"]
TARGETS = ["a battery-tech startup", "its JV partner", "a European engineering firm", "a specialty chemicals unit"]
BROKERS = ["Morgan Stanley", "Jefferies", "CLSA", "Nomura", "Kotak Institutional Equities", "Goldman Sachs"]
FUNDS = ["Goldman Sachs Fund", "SBI Mutual Fund", "Societe Generale", "GQG Partners", "Nippon India MF"]


class MarketSimulator:
    def __init__(self, universe: Universe, sim_date: date, prev_close: dict[str, float],
                 adv: dict[str, float], seed: int = 0, start: str = "09:13:30"):
        self.u = universe
        self.date = sim_date
        self.rng = np.random.default_rng(seed or None)
        self.py = random.Random(seed or None)
        eqs = universe.equities()
        futs = {f.underlying: f for f in universe.futures()}
        self.eqs = eqs
        self.futs = [futs.get(e.symbol) for e in eqs]
        n = len(eqs)
        self.n = n
        self.syms = [e.symbol for e in eqs]
        self.sector_idx = np.array([SECTORS.index(e.sector) if e.sector in SECTORS else 0 for e in eqs])
        self.vol_d = np.array([_REF.get(s, ("", 1000, 1.8))[2] / 100.0 for s in self.syms])
        self.beta = self.rng.uniform(0.7, 1.35, n)
        self.prev_close = np.array([prev_close.get(s, _REF.get(s, ("", 1000, 2))[1]) for s in self.syms], float)
        self.adv = np.array([adv.get(s, 2e6) for s in self.syms], float)

        # ---- scenario for the day -------------------------------------
        self.mkt_gap = self.rng.normal(0, 0.0035)
        self.mkt_drift = self.rng.normal(0, 0.0045)
        self.sec_gap = self.rng.normal(0, 0.003, len(SECTORS))
        self.sec_drift = self.rng.normal(0, 0.004, len(SECTORS))
        idio_gap = self.rng.normal(0, 0.25, n) * self.vol_d
        self.gap = self.beta * self.mkt_gap + self.sec_gap[self.sector_idx] + idio_gap
        self.drift = np.zeros(n)
        self.rvol_peak = self.rng.uniform(0.75, 1.3, n)
        self.catalysts: list[Catalyst] = []
        order = self.rng.permutation(n)
        n_cat = max(4, n // 20)
        for i in order[:n_cat]:
            sent = 1 if self.rng.random() < 0.6 else -1
            kind = self.py.choice(list(NEWS_TEMPLATES))
            tmpls = [t for t in NEWS_TEMPLATES[kind] if t[1] == sent] or NEWS_TEMPLATES[kind]
            sent = tmpls[0][1]
            premarket = self.rng.random() < 0.7
            gap = sent * self.rng.uniform(0.012, 0.05) if premarket else 0.0
            cont = self.rng.random() < 0.65
            drift = sent * self.rng.uniform(0.006, 0.03) * (1 if cont else -0.6)
            news_min = -int(self.rng.integers(5, 300)) if premarket else int(self.rng.integers(20, 300))
            self.catalysts.append(Catalyst(int(i), kind, sent, gap, drift, float(self.rng.uniform(2.5, 6.0)),
                                           news_min))
            self.gap[i] += gap
            self.drift[i] += drift
            self.rvol_peak[i] = self.catalysts[-1].rvol
        for i in order[n_cat:n_cat + max(3, n // 35)]:          # stealth movers
            self.drift[i] += self.rng.choice([-1, 1]) * self.rng.uniform(0.01, 0.028)
            self.rvol_peak[i] = self.rng.uniform(1.6, 3.0)
        self.cat_by_idx = {c.idx: c for c in self.catalysts}

        # ---- state -----------------------------------------------------
        self.open = self.prev_close * np.exp(self.gap)
        self.px = self.open.copy()
        self.hi = self.open.copy()
        self.lo = self.open.copy()
        self.cumvol = np.zeros(n)
        self.cumval = np.zeros(n)
        self.emitted_vol = np.zeros(n)
        self.basis = self.rng.uniform(0.002, 0.008, n)
        self.lot = np.array([f.lot if f else 1 for f in self.futs])
        self.fut_vol = np.zeros(n)
        oi_base = self.adv * self.rng.uniform(0.8, 2.5, n)
        self.oi_prev = np.round(oi_base / self.lot) * self.lot
        # OI direction tied to the day's move with noise (long/short build-up vs covering/unwinding)
        dir_ = np.sign(self.gap + self.drift + self.rng.normal(0, 0.004, n))
        oi_up = self.rng.random(n) < 0.62
        self.oi_target = self.oi_prev * (1 + np.where(oi_up, 1, -1) * self.rng.uniform(0.01, 0.12, n)
                                         * (1 + 2 * (np.abs(self.gap + self.drift) > 0.015)))
        self.oi = self.oi_prev.copy()
        self.oih = self.oi.copy()
        self.oil = self.oi.copy()
        self.book_bias = dir_ * self.rng.uniform(0.0, 0.25, n)
        self.started = False
        self.closed = False
        self.sim_t = at(sim_date, datetime.strptime(start, "%H:%M:%S").time())
        self.open_dt = at(sim_date, MARKET_OPEN)
        self.close_dt = at(sim_date, MARKET_CLOSE)

        # indices
        self.idx_names = [i.symbol for i in universe.indices()]
        self.idx_sid = {i.symbol: i.sid for i in universe.indices()}
        self.idx_prev = {name: float(prev_close.get(name, INDICES[name][1])) for name in self.idx_names if name in INDICES}
        self.idx_level = {k: v * math.exp(self.mkt_gap * (1 if k != "INDIAVIX" else -6)) for k, v in self.idx_prev.items()}
        self.mkt_path = 0.0
        self.news_out: list[dict] = []
        self._news_emitted: set[int] = set()
        self._news_id = 0
        self._mk_premarket_news()

    # ------------------------------------------------------------------ news
    def _headline(self, c: Catalyst) -> str:
        sym = self.syms[c.idx]
        tmpls = [t for t in NEWS_TEMPLATES[c.kind] if t[1] == c.sentiment] or NEWS_TEMPLATES[c.kind]
        t = self.py.choice(tmpls)[0]
        return t.format(n=sym.title() if len(sym) > 4 else sym, x=self.py.randint(8, 65),
                        y=self.py.choice([450, 820, 1250, 2300, 3100, 5400]), c=self.py.choice(CLIENTS),
                        t=self.py.choice(TARGETS), b=self.py.choice(BROKERS), f=self.py.choice(FUNDS))

    def _news_item(self, c: Catalyst, when: datetime) -> dict:
        self._news_id += 1
        e = self.eqs[c.idx]
        title = self._headline(c)
        return {
            "id": f"sim-{self.date.isoformat()}-{self._news_id}",
            "ts": when.timestamp(),
            "title": title,
            "text": title + ". (simulated headline)",
            "sentiment": "positive" if c.sentiment > 0 else "negative",
            "category": "COMPANIES",
            "subCategory": c.kind,
            "symbols": [e.symbol],
            "sids": [e.sid],
            "source": "SIM",
        }

    def _mk_premarket_news(self):
        for c in self.catalysts:
            if c.news_minute < 0:
                when = self.open_dt + timedelta(minutes=c.news_minute)
                self.news_out.append(self._news_item(c, when))
                self._news_emitted.add(c.idx)
        # a couple of macro / noise headlines
        macro = [
            ("Crude oil rises 2% in Asian trade on Middle East supply worries", "neutral", ["ONGC", "BPCL", "IOC"]),
            ("GIFT Nifty indicates a {} start for domestic markets".format("positive" if self.mkt_gap > 0 else "weak"),
             "positive" if self.mkt_gap > 0 else "negative", []),
            ("FIIs net {} ₹{} crore in cash segment on Friday".format(
                "bought" if self.mkt_gap > 0 else "sold", self.py.randint(800, 4200)),
             "positive" if self.mkt_gap > 0 else "negative", []),
        ]
        for title, sent, syms in macro:
            self._news_id += 1
            self.news_out.append({"id": f"sim-{self.date.isoformat()}-{self._news_id}",
                                  "ts": (self.open_dt - timedelta(minutes=self.py.randint(20, 90))).timestamp(),
                                  "title": title, "text": title, "sentiment": sent, "category": "EQUITY_MARKETS",
                                  "subCategory": "MACRO", "symbols": [s for s in syms if s in self.syms],
                                  "sids": [], "source": "SIM"})

    def drain_news(self) -> list[dict]:
        out, self.news_out = self.news_out, []
        return out

    # ------------------------------------------------------------------ clock
    def advance(self, seconds: float) -> list[tuple[str, bytes]]:
        """Advance simulated time; returns [(channel, binary message)] where channel is
        "feed" (market feed packets) or "cas" (closing auction packets)."""
        msgs: list[tuple[str, bytes]] = []
        target = self.sim_t + timedelta(seconds=seconds)
        while self.sim_t < target:
            self.sim_t += timedelta(seconds=1)
            for b in self._step():
                msgs.append(("cas", b) if len(b) == 128 else ("feed", b))
        return msgs

    def _session_minute(self) -> float:
        return (self.sim_t - self.open_dt).total_seconds() / 60.0

    def _step(self) -> list[bytes]:
        t = self.sim_t
        if t < self.open_dt:
            return []
        if not self.started:
            self.started = True
            return self._prev_close_packets() + self._emit(all_=True)
        if t >= self.close_dt:
            if not self.closed:
                self.closed = True
                return self._emit(all_=True, final=True)
            if t < self.close_dt + timedelta(minutes=10):
                return self._cas_packets() if int(t.timestamp()) % 5 == 0 else []
            return []
        m = self._session_minute()
        uc = float(ucurve(m))
        n = self.n
        sec_vol = 1.0 / math.sqrt(SESSION_SECS)
        mkt = self.mkt_drift / SESSION_SECS + 0.0075 * sec_vol * math.sqrt(uc) * self.rng.normal()
        sec = self.sec_drift / SESSION_SECS + 0.006 * sec_vol * math.sqrt(uc) * self.rng.normal(0, 1, len(SECTORS))
        # catalyst/stealth drift is front-loaded (decays over ~2h)
        w = math.exp(-m / 120.0) / (120.0 * 60.0 * (1 - math.exp(-375 / 120.0)))
        drift = self.drift * w
        noise = self.vol_d * 0.85 * sec_vol * math.sqrt(uc) * self.rng.normal(0, 1, n)
        ret = self.beta * mkt + sec[self.sector_idx] + drift + noise
        # intraday news arrival => jump
        for c in self.catalysts:
            if c.news_minute >= 0 and c.idx not in self._news_emitted and m >= c.news_minute:
                self._news_emitted.add(c.idx)
                ret[c.idx] += c.sentiment * self.rng.uniform(0.008, 0.025)
                self.rvol_peak[c.idx] = max(self.rvol_peak[c.idx], c.rvol)
                self.news_out.append(self._news_item(c, t))
        self.px *= np.exp(ret)
        self.hi = np.maximum(self.hi, self.px)
        self.lo = np.minimum(self.lo, self.px)
        self.mkt_path += mkt
        rv = 1 + (self.rvol_peak - 1) * np.exp(-m / 100.0)
        v = self.adv / SESSION_SECS * (uc / 0.832) * rv * self.rng.lognormal(-0.12, 0.5, n)
        self.cumvol += v
        self.cumval += v * self.px
        self.fut_vol += v * 0.35
        frac = min(1.0, m / 300.0)
        self.oi = self.oi_prev + (self.oi_target - self.oi_prev) * frac + self.rng.normal(0, 0.002, n) * self.oi_prev
        self.oi = np.round(self.oi / self.lot) * self.lot
        self.oih = np.maximum(self.oih, self.oi)
        self.oil = np.minimum(self.oil, self.oi)
        for name in self.idx_level:
            if name == "INDIAVIX":
                self.idx_level[name] *= math.exp(-6 * mkt + 0.0004 * self.rng.normal())
            elif name == "BANKNIFTY":
                self.idx_level[name] *= math.exp(1.1 * mkt + 0.6 * sec[SECTORS.index("BANK")])
            elif name == "FINNIFTY":
                self.idx_level[name] *= math.exp(1.05 * mkt + 0.5 * sec[SECTORS.index("FIN")])
            elif name == "MIDCPNIFTY":
                self.idx_level[name] *= math.exp(1.25 * mkt + 0.0002 * self.rng.normal())
            else:
                self.idx_level[name] *= math.exp(mkt)
        return self._emit()

    # ---------------------------------------------------------------- packets
    def _prev_close_packets(self) -> list[bytes]:
        out = []
        for i, e in enumerate(self.eqs):
            out.append(P.encode_prev(1, e.sid, float(self.prev_close[i]), 0))
            f = self.futs[i]
            if f:
                out.append(P.encode_prev(2, f.sid, float(self.prev_close[i] * (1 + self.basis[i])), int(self.oi_prev[i])))
        for name, lvl in self.idx_prev.items():
            out.append(P.encode_prev(0, self.idx_sid[name], float(lvl), 0))
        return out

    def _depth5(self, i: int, px: float) -> list[tuple]:
        tick = 0.05 if px < 2000 else 0.1 if px < 10000 else 0.5
        bias = self.book_bias[i]
        rows = []
        base = max(50, int(self.adv[i] / 4000))
        for k in range(5):
            bq = int(base * self.rng.uniform(0.5, 1.5) * (1 + bias))
            aq = int(base * self.rng.uniform(0.5, 1.5) * (1 - bias))
            rows.append((max(1, bq), max(1, aq), int(self.rng.integers(1, 20)), int(self.rng.integers(1, 20)),
                         float(round(px - tick * (k + 1), 2)), float(round(px + tick * k, 2))))
        return rows

    def _emit(self, all_: bool = False, final: bool = False) -> list[bytes]:
        out: list[bytes] = []
        ltt = int(self.sim_t.timestamp())
        p = 1.0 if all_ else 0.35
        mask = self.rng.random(self.n) < p
        pxs = _round_tick(self.px)
        for i in np.nonzero(mask)[0]:
            e = self.eqs[i]
            px = float(pxs[i])
            vol = int(self.cumvol[i])
            ltq = max(1, vol - int(self.emitted_vol[i]))
            self.emitted_vol[i] = vol
            atp = float(self.cumval[i] / self.cumvol[i]) if self.cumvol[i] > 0 else px
            pend = int(self.adv[i] * 0.02)
            tbq = int(pend * (1 + self.book_bias[i]) * self.rng.uniform(0.8, 1.2))
            tsq = int(pend * (1 - self.book_bias[i]) * self.rng.uniform(0.8, 1.2))
            dc = px if final else 0.0
            out.append(P.encode_quote(1, e.sid, px, min(ltq, 65535), ltt, atp, vol, tsq, tbq,
                                      float(self.open[i]), dc, float(self.hi[i]), float(self.lo[i])))
            f = self.futs[i]
            if f and (all_ or self.rng.random() < 0.6):
                fpx = float(_round_tick(self.px[i] * (1 + self.basis[i])))
                fvol = int(self.fut_vol[i])
                out.append(P.encode_full(2, f.sid, fpx, int(self.lot[i]), ltt, fpx, fvol, tsq // 3, tbq // 3,
                                         int(self.oi[i]), int(self.oih[i]), int(self.oil[i]),
                                         float(self.open[i] * (1 + self.basis[i])), fpx if final else 0.0,
                                         float(self.hi[i] * (1 + self.basis[i])),
                                         float(self.lo[i] * (1 + self.basis[i])),
                                         self._depth5(i, fpx)))
        for name, lvl in self.idx_level.items():
            if all_ or self.rng.random() < 0.8:
                out.append(P.encode_ticker(0, self.idx_sid[name], float(round(lvl, 2)), ltt))
        return out

    def _cas_packets(self) -> list[bytes]:
        out = []
        for i in range(self.n):
            e = self.eqs[i]
            px = float(self.px[i])
            bq = int(self.adv[i] * self.rng.uniform(0.002, 0.01))
            sq = int(self.adv[i] * self.rng.uniform(0.002, 0.01))
            iep = round(px * (1 + (bq - sq) / max(1, bq + sq) * 0.002), 2)
            out.append(P.encode_cas(1, e.sid, iep, min(bq, sq), min(bq, sq), self._depth5(i, px), bq - sq, round(px, 2)))
        return out

    def depth20(self, sid: int) -> list[bytes]:
        """20-level depth for one instrument (bid + ask packets)."""
        idx = next((i for i, e in enumerate(self.eqs) if e.sid == sid), None)
        seg = 1
        if idx is None:
            idx = next((i for i, f in enumerate(self.futs) if f and f.sid == sid), None)
            seg = 2
        if idx is None:
            return []
        px = float(self.px[idx] * (1 + (self.basis[idx] if seg == 2 else 0)))
        tick = 0.05 if px < 2000 else 0.1 if px < 10000 else 0.5
        base = max(50, int(self.adv[idx] / 3000))
        bias = self.book_bias[idx]
        bids, asks = [], []
        for k in range(20):
            wall = 4.0 if self.rng.random() < 0.08 else 1.0
            bids.append((round(px - tick * (k + 1), 2), int(base * wall * self.rng.uniform(0.4, 1.6) * (1 + bias)),
                         int(self.rng.integers(1, 40))))
            wall = 4.0 if self.rng.random() < 0.08 else 1.0
            asks.append((round(px + tick * k, 2), int(base * wall * self.rng.uniform(0.4, 1.6) * (1 - bias)),
                         int(self.rng.integers(1, 40))))
        return [P.encode_depth20(C.RESP_DEPTH_BID, seg, sid, bids) + P.encode_depth20(C.RESP_DEPTH_ASK, seg, sid, asks)]

    # ---------------------------------------------------------- option chains
    def spot_of(self, name: str) -> float | None:
        if name in self.idx_level:
            return self.idx_level[name]
        if name in self.syms:
            return float(self.px[self.syms.index(name)])
        return None

    def option_chain(self, name: str, expiry: str) -> dict:
        from ..analytics.optmath import bs_greeks, bs_price

        spot = self.spot_of(name)
        if spot is None:
            return {}
        exp_d = datetime.strptime(expiry, "%Y-%m-%d").date()
        T = max(0.5, (exp_d - self.date).days + 0.6) / 365.0
        step = 50 if name in ("NIFTY", "FINNIFTY") else 100 if name == "BANKNIFTY" else 25 if name == "MIDCPNIFTY" \
            else max(1.0, round(spot * 0.01 / 5) * 5) if spot > 500 else 2.5
        atm = round(spot / step) * step
        base_iv = (0.115 if name in INDICES else 0.28) * (1 + 0.15 * (self.idx_level.get("INDIAVIX", 12.4) / 12.4 - 1))
        seed = abs(hash((name, expiry, self.date.isoformat()))) % (2 ** 32)
        rng = np.random.default_rng(seed)
        minute = max(0.0, self._session_minute())
        oc = {}
        for k in range(-20, 21):
            K = atm + k * step
            if K <= 0:
                continue
            mny = math.log(K / spot)
            iv = base_iv * (1 + 1.8 * mny * mny * 40 - 0.9 * mny)       # smile + put skew
            iv = max(0.05, iv)
            ce_oi_base = 9e5 * math.exp(-((k - 3) ** 2) / 60.0) * rng.uniform(0.6, 1.4)
            pe_oi_base = 9e5 * math.exp(-((k + 3) ** 2) / 60.0) * rng.uniform(0.6, 1.4)
            if name not in INDICES:
                ce_oi_base /= 40
                pe_oi_base /= 40
            drift_dir = math.copysign(1, self.mkt_path)
            ce_chg = (rng.uniform(-0.1, 0.25) - 0.08 * drift_dir) * (minute / 375 + 0.1)
            pe_chg = (rng.uniform(-0.1, 0.25) + 0.08 * drift_dir) * (minute / 375 + 0.1)
            row = {}
            for side, oi0, chg, cp in (("ce", ce_oi_base, ce_chg, "call"), ("pe", pe_oi_base, pe_chg, "put")):
                price = bs_price(spot, K, T, 0.065, iv, cp)
                g = bs_greeks(spot, K, T, 0.065, iv, cp)
                oi_now = int(oi0 * (1 + chg))
                row[side] = {
                    "last_price": round(max(0.05, price), 2), "average_price": round(max(0.05, price * 0.98), 2),
                    "oi": oi_now, "previous_oi": int(oi0), "volume": int(oi0 * rng.uniform(0.5, 3.0) * (minute / 375 + 0.05)),
                    "previous_volume": int(oi0 * 1.5), "previous_close_price": round(max(0.05, price * rng.uniform(0.85, 1.15)), 2),
                    "implied_volatility": round(iv * 100, 2),
                    "greeks": {"delta": round(g["delta"], 5), "gamma": round(g["gamma"], 6),
                               "theta": round(g["theta"], 4), "vega": round(g["vega"], 4)},
                    "top_bid_price": round(max(0.05, price * 0.998), 2), "top_ask_price": round(max(0.05, price * 1.002), 2),
                    "top_bid_quantity": int(rng.integers(50, 2000)), "top_ask_quantity": int(rng.integers(50, 2000)),
                    "security_id": int(300000 + abs(hash((name, K, side))) % 600000),
                }
            oc[f"{K:.6f}"] = row
        return {"last_price": round(spot, 2), "oc": oc}

    def expiries(self, name: str) -> list[str]:
        d = self.date
        out = []
        weekly = name in ("NIFTY",)
        # NSE expiries on Tuesday (from Sep 2025); monthly = last Tuesday
        cur = d
        while len(out) < (4 if weekly else 3):
            cur += timedelta(days=1)
            if cur.weekday() == 1:
                nxt = cur + timedelta(days=7)
                if weekly or nxt.month != cur.month:
                    out.append(cur.isoformat())
        return out


# --------------------------------------------------------------------------
# synthetic history for the data lake (used in sim mode and by tests)
# --------------------------------------------------------------------------
def generate_history(universe: Universe, end: date, days: int = 320, minute_days: int = 25,
                     seed: int = 7) -> dict[str, dict[str, pd.DataFrame]]:
    """Return {interval: {symbol: DataFrame}} with daily bars for `days` and 1-min bars for the
    last `minute_days` sessions ending at `end` (inclusive)."""
    rng = np.random.default_rng(seed)
    dates = trading_days_back(end, days)
    out: dict[str, dict[str, pd.DataFrame]] = {"D": {}, "1m": {}}
    n_days = len(dates)
    mkt = rng.normal(0.0004, 0.0085, n_days)
    sec = {s: rng.normal(0.0002, 0.006, n_days) for s in SECTORS}
    minute_dates = dates[-minute_days:]
    m_idx = np.arange(375)
    uc = ucurve(m_idx)
    uc = uc / uc.sum()

    def make(sym: str, px0: float, vol: float, beta: float, sector: str | None, adv: float, index: bool):
        ret = beta * mkt + (sec[sector] if sector in sec else 0) + rng.normal(0, vol * 0.8, n_days)
        if index:
            ret = mkt * beta + rng.normal(0, 0.002, n_days)
        closes = px0 * np.exp(np.cumsum(ret) - ret.sum())     # end near reference price
        opens = np.r_[closes[0], closes[:-1]] * np.exp(rng.normal(0, vol * 0.25, n_days))
        rng_hi = np.abs(rng.normal(0, vol * 0.6, n_days))
        rng_lo = np.abs(rng.normal(0, vol * 0.6, n_days))
        highs = np.maximum(opens, closes) * (1 + rng_hi)
        lows = np.minimum(opens, closes) * (1 - rng_lo)
        vols = (adv * rng.lognormal(0, 0.35, n_days) * (1 + 8 * np.abs(ret))).astype("int64") if not index \
            else np.zeros(n_days, "int64")
        ts = np.array([int(at(d, MARKET_OPEN).replace(hour=0, minute=0).timestamp()) for d in dates])
        daily = pd.DataFrame({"ts": ts, "open": opens, "high": highs, "low": lows, "close": closes,
                              "volume": vols, "oi": 0})
        rows = []
        for j, d in enumerate(dates):
            if d not in minute_dates:
                continue
            o, c = opens[j], closes[j]
            steps = rng.normal(0, 1, 375) * np.sqrt(ucurve(m_idx) / ucurve(m_idx).mean())
            path = np.cumsum(steps)
            path = path - np.linspace(0, path[-1], 375)            # bridge
            span = max(abs(math.log(c / o)), vol * 0.5)
            path = path / (np.abs(path).max() + 1e-9) * span * 0.6
            closes_m = o * np.exp(np.linspace(0, math.log(c / o), 375) + path)
            opens_m = np.r_[o, closes_m[:-1]]
            wig = np.abs(rng.normal(0, vol * 0.05, 375))
            hi_m = np.maximum(opens_m, closes_m) * (1 + wig)
            lo_m = np.minimum(opens_m, closes_m) * (1 - wig)
            v_m = (vols[j] * uc * rng.lognormal(0, 0.4, 375)).astype("int64") if not index else np.zeros(375, "int64")
            t0 = int(at(d, MARKET_OPEN).timestamp())
            rows.append(pd.DataFrame({"ts": t0 + m_idx * 60, "open": opens_m, "high": hi_m, "low": lo_m,
                                      "close": closes_m, "volume": v_m, "oi": 0}))
            daily.loc[j, ["high", "low"]] = [max(hi_m.max(), highs[j]), min(lo_m.min(), lows[j])]
            daily.loc[j, "volume"] = int(v_m.sum()) if not index else 0
        out["D"][sym] = daily
        if rows:
            out["1m"][sym] = pd.concat(rows, ignore_index=True)

    for e in universe.equities():
        sector, px, vol = _REF.get(e.symbol, ("OTHER", 1000.0, 1.8))
        turnover_cr = float(rng.uniform(40, 1400))
        adv = turnover_cr * 1e7 / px
        make(e.symbol, px, vol / 100.0, float(rng.uniform(0.7, 1.35)), sector, adv, False)
    for name, (_, lvl, vol) in INDICES.items():
        if name == "INDIAVIX":
            make(name, lvl, 0.04, -4.0, None, 0, True)
        else:
            make(name, lvl, vol / 100.0, 1.0 if name == "NIFTY" else 1.1, None, 0, True)
    return out


def sim_trading_date(today: date | None = None) -> date:
    from ..timeutil import now_ist

    d = today or now_ist().date()
    while d.weekday() >= 5:
        d = d - timedelta(days=1)
    return d


def history_end_for(sim_date: date) -> date:
    return previous_weekday(sim_date)


__all__ = ["MarketSimulator", "generate_history", "sim_trading_date", "history_end_for", "IST"]
