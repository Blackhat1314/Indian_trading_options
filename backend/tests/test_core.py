import math
import struct

import numpy as np
import pandas as pd

from td.analytics import intraday, optmath, smc, swing
from td.analytics.news import NewsEngine, classify
from td.analytics.state import InstState
from td.dhan import constants as C
from td.dhan import parser as P


# ------------------------------------------------------------------ parser
def test_quote_roundtrip():
    b = P.encode_quote(1, 2885, 1420.55, 12, 1700000000, 1418.2, 123456, 5000, 6000, 1410, 0, 1425, 1405)
    assert len(b) == 50
    (pkt,) = P.parse_market_feed(b)
    assert pkt["t"] == "quote" and pkt["sid"] == 2885 and pkt["vol"] == 123456
    assert pkt["ltp"] == 1420.55 and pkt["tbq"] == 6000 and pkt["tsq"] == 5000
    assert pkt["o"] == 1410 and pkt["h"] == 1425 and pkt["l"] == 1405


def test_full_packet_with_depth_and_stacking():
    depth = [(100, 120, 2, 3, 99.95, 100.05)] * 5
    b = P.encode_full(2, 49081, 100.0, 50, 1700000000, 99.9, 1000, 300, 400, 125000, 130000, 120000,
                      98, 0, 101, 97, depth)
    assert len(b) == 162
    t = P.encode_ticker(0, 13, 25450.5, 1700000000)
    pv = P.encode_prev(2, 49081, 98.5, 120000)
    pkts = P.parse_market_feed(b + t + pv)
    assert [p["t"] for p in pkts] == ["full", "tick", "prev"]
    assert pkts[0]["oi"] == 125000 and pkts[0]["depth"][0]["bq"] == 100
    assert pkts[2]["poi"] == 120000


def test_layout_matches_official_sdk_struct():
    # dhanhq-py uses '<BHBIfHIfIIIIIIffff100s' for full packets
    assert struct.calcsize("<BHBIfHIfIIIIIIffff100s") == 162
    assert struct.calcsize("<BHBIfHIfIIIffff") == 50


def test_depth20_roundtrip():
    bids = [(100 - i * 0.05, 10 + i, 1) for i in range(20)]
    asks = [(100 + i * 0.05, 20 + i, 2) for i in range(20)]
    msg = P.encode_depth20(C.RESP_DEPTH_BID, 1, 2885, bids) + P.encode_depth20(C.RESP_DEPTH_ASK, 1, 2885, asks)
    out = P.parse_depth(msg)
    assert [o["side"] for o in out] == ["bid", "ask"]
    assert len(out[0]["levels"]) == 20 and out[1]["levels"][0][1] == 20


def test_cas_packet():
    b = P.encode_cas(1, 2885, 1421.0, 500, 400, [(1, 1, 1, 1, 1.0, 1.1)] * 5, 100, 1420.0)
    assert len(b) == 128
    (pkt,) = P.parse_cas(b)
    assert pkt["t"] == "cas" and pkt["imb"] == 100 and pkt["iep"] == 1421.0


# --------------------------------------------------------------- options
def test_bs_put_call_parity_and_iv():
    S, K, T, r, v = 100, 100, 0.25, 0.05, 0.2
    c = optmath.bs_price(S, K, T, r, v, "call")
    p = optmath.bs_price(S, K, T, r, v, "put")
    assert abs((c - p) - (S - K * math.exp(-r * T))) < 1e-9
    iv = optmath.implied_vol(c, S, K, T, r, "call")
    assert abs(iv - v) < 1e-3


def test_max_pain():
    rows = [{"strike": k, "ce_oi": 100 if k >= 110 else 10, "pe_oi": 100 if k <= 90 else 10} for k in range(80, 121, 10)]
    assert optmath.max_pain(rows) == 100


# ---------------------------------------------------------------- smc
def _bars(prices):
    rows = []
    for i, p in enumerate(prices):
        rows.append({"ts": i * 60, "open": p - 0.2, "high": p + 0.5, "low": p - 0.5, "close": p})
    return pd.DataFrame(rows)


def test_smc_bos_is_not_repainting():
    up = list(np.linspace(100, 110, 20)) + list(np.linspace(110, 104, 8)) + list(np.linspace(104, 118, 20))
    df = _bars(up)
    full = smc.analyze(df, k=2)
    assert full["trend"] == 1
    assert any(e["kind"] in ("BOS", "CHoCH") and e["dir"] == 1 for e in full["events"])
    # events detected on a prefix must be identical when more data arrives (no look-ahead)
    part = smc.analyze(df.iloc[:40].reset_index(drop=True), k=2)
    for e in part["events"]:
        assert any(f["ts"] == e["ts"] and f["kind"] == e["kind"] for f in full["events"])


def test_fvg_detection():
    df = pd.DataFrame([
        {"ts": 0, "open": 100, "high": 101, "low": 99, "close": 100.5},
        {"ts": 60, "open": 100.5, "high": 105, "low": 100.4, "close": 104.8},
        {"ts": 120, "open": 104.8, "high": 106, "low": 102, "close": 105.5},
        {"ts": 180, "open": 105.5, "high": 107, "low": 104, "close": 106},
        {"ts": 240, "open": 106, "high": 107.5, "low": 105, "close": 107},
        {"ts": 300, "open": 107, "high": 108, "low": 106, "close": 107.5},
        {"ts": 360, "open": 107.5, "high": 108.5, "low": 107, "close": 108},
    ])
    out = smc.analyze(df, k=1)
    assert any(g["dir"] == 1 and g["bottom"] == 101 and g["top"] == 102 for g in out["fvgs"])


# ------------------------------------------------------------ intraday
def _state(sym, sector, pc, ltp, vol, adv, atp=None, sid=1):
    s = InstState(sid, "NSE_EQ", sym, "EQ", sym, sector)
    s.pc = pc
    s.ltp = ltp
    s.open = pc
    s.high = max(pc, ltp)
    s.low = min(pc, ltp)
    s.vol = vol
    s.atp = atp or (pc + ltp) / 2
    s.orh, s.orl = max(pc, ltp) * 0.999, min(pc, ltp)
    s.ctx.adv = adv
    s.ctx.atr = pc * 0.02
    s.ctx.base_cum = list(np.cumsum(np.full(375, adv / 375)))
    return s


def test_ranker_prefers_in_play_leader():
    eqs = [_state(f"S{i}", "IT" if i % 2 else "BANK", 100, 100 + (i % 3 - 1) * 0.2, 4000, 100000, sid=i)
           for i in range(30)]
    leader = _state("LEADER", "AUTO", 100, 103, 40000, 100000, atp=101.5, sid=99)
    eqs.append(leader)
    rows = intraday.rank(eqs, {}, 0.0, 20, {"LEADER": (1.0, None)}, intraday.RankConfig(min_turnover_cr=0), 15)
    assert rows[0]["symbol"] == "LEADER"
    picks = intraday.pick(rows, intraday.RankConfig(min_turnover_cr=0, min_rvol=1.0), "long")
    assert picks and picks[0]["symbol"] == "LEADER"


def test_buildup_tags():
    assert intraday.buildup_tag(1.0, 5.0) == "LB"
    assert intraday.buildup_tag(-1.0, 5.0) == "SB"
    assert intraday.buildup_tag(1.0, -5.0) == "SC"
    assert intraday.buildup_tag(-1.0, -5.0) == "LU"


def test_spearman():
    assert intraday.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) > 0.99


# ---------------------------------------------------------------- news
def test_news_classify_and_novelty():
    assert classify("Tata Power bags ₹1,250 crore order from NTPC")["type"] == "ORDER_WIN"
    c = classify("Sun Pharma Q2 profit falls 20%, misses estimates")
    assert c["type"] == "RESULTS" and c["sign"] == -1
    ne = NewsEngine({"TATAPOWER": "POWER"})
    a = ne.add({"id": "1", "ts": 1000, "title": "Tata Power bags ₹1,250 crore order from NTPC",
                "symbols": ["TATAPOWER"], "sentiment": "positive"})
    b = ne.add({"id": "2", "ts": 1100, "title": "Tata Power bags ₹1,250 crore order from NTPC: report",
                "symbols": ["TATAPOWER"], "sentiment": "positive"})
    assert a["novel"] and not b["novel"]
    s, top = ne.score("TATAPOWER", 1200)
    assert s > 0 and top["id"] == "1"


# --------------------------------------------------------------- swing
def test_swing_features_and_stats():
    rng = np.random.default_rng(1)
    n = 300
    c = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.015, n)))
    df = pd.DataFrame({"ts": np.arange(n) * 86400, "open": c * 0.998, "high": c * 1.01, "low": c * 0.99,
                       "close": c, "volume": rng.integers(1e5, 3e5, n)})
    f = swing.features(df)
    assert {"trend", "rsi", "setup", "rs63"} <= set(f.columns)
    stats = swing.setup_stats({"X": f})
    for s in stats.values():
        assert 0 <= s["hit_rate"] <= 100


# ---------------------------------------------------------------- lake
def test_lake_symbol_names_round_trip(tmp_path):
    from td.lake import Lake
    lake = Lake(tmp_path)
    df = pd.DataFrame({"ts": [1, 2], "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0],
                       "volume": [10, 20]})
    for sym in ("M&M", "BANDHANBNK", "COROMANDEL"):
        lake.write("D", sym, df)
    assert lake.symbols("D") == ["BANDHANBNK", "COROMANDEL", "M&M"]
    assert len(lake.read("D", "M&M")) == 2


def test_news_macro_flows():
    assert classify("FIIs net sold ₹1,014 crore in cash segment")["type"] == "MACRO"
    assert classify("GIFT Nifty indicates a weak start")["type"] == "MACRO"


def test_news_block_deal_before_mna():
    assert classify("Promoter entity sells 50% stake in MCX via block deal")["type"] == "BLOCK_DEAL"
    assert classify("Federal Bank to acquire 39% stake in a startup")["type"] == "M&A"
