"""Pure binary parsers for Dhan v2 websocket feeds (little-endian).

Layouts follow docs.dhanhq.co (Live Market Feed, Full Market Depth, CAS Live Feed)
and are cross-checked against the official dhanhq-py SDK struct formats.

Every parser returns plain dicts with short keys so they can be JSON-encoded cheaply:
  sid  security id        seg  segment code     ltp  last price      ltq  last qty
  ltt  last trade epoch   atp  avg trade price  vol  day volume      tbq/tsq total buy/sell qty
  o/h/l/c day open/high/low/close   oi/oih/oil  open interest       pc prev close  poi prev OI
"""
from __future__ import annotations

import struct

from . import constants as C

_HDR = struct.Struct("<BHBI")                       # code, length, segment, security id (8 bytes)
_TICKER = struct.Struct("<BHBIfI")                  # 16
_QUOTE = struct.Struct("<BHBIfHIfIIIffff")          # 50
_OI = struct.Struct("<BHBII")                       # 12
_PREV = struct.Struct("<BHBIfI")                    # 16
_FULL = struct.Struct("<BHBIfHIfIIIIIIffff")        # 62 + 100 depth
_DEPTH5 = struct.Struct("<IIHHff")                  # 20 bytes x 5
_DISC = struct.Struct("<BHBIH")                     # 10

_D20_HDR = struct.Struct("<hBBiI")                  # length, code, segment, sid, seq/rows (12)
_D20_ROW = struct.Struct("<dII")                    # price(float64), qty, orders (16)

_CAS = struct.Struct("<BHBIfii")                    # header + IEP, buy@IEP, sell@IEP (20)
_CAS_TAIL = struct.Struct("<if")                    # imbalance qty, reference price
_CAS_INDEX = struct.Struct("<hBB21sd")              # 33 bytes


def _r2(x: float) -> float:
    return round(float(x), 2)


def _depth5(buf: bytes, off: int) -> list[dict]:
    rows = []
    for i in range(5):
        bq, aq, bo, ao, bp, ap = _DEPTH5.unpack_from(buf, off + i * 20)
        rows.append({"bq": bq, "aq": aq, "bo": bo, "ao": ao, "bp": _r2(bp), "ap": _r2(ap)})
    return rows


def parse_packet(buf: bytes, off: int = 0) -> tuple[dict | None, int]:
    """Parse one market-feed packet at `off`. Returns (packet|None, bytes consumed)."""
    code = buf[off]
    size = C.PACKET_SIZE.get(code)
    if size is None:
        # unknown packet: trust the header length if sane, else stop
        if off + 8 <= len(buf):
            _, ln, _, _ = _HDR.unpack_from(buf, off)
            if 8 <= ln <= len(buf) - off:
                return None, ln
        return None, len(buf) - off
    if off + size > len(buf):
        return None, len(buf) - off

    if code in (C.RESP_TICKER, C.RESP_INDEX):
        _, _, seg, sid, ltp, ltt = _TICKER.unpack_from(buf, off)
        return {"t": "tick", "seg": seg, "sid": sid, "ltp": _r2(ltp), "ltt": ltt}, size
    if code == C.RESP_QUOTE:
        (_, _, seg, sid, ltp, ltq, ltt, atp, vol, tsq, tbq, o, c, h, l) = _QUOTE.unpack_from(buf, off)
        return {"t": "quote", "seg": seg, "sid": sid, "ltp": _r2(ltp), "ltq": ltq, "ltt": ltt,
                "atp": _r2(atp), "vol": vol, "tsq": tsq, "tbq": tbq,
                "o": _r2(o), "dc": _r2(c), "h": _r2(h), "l": _r2(l)}, size
    if code == C.RESP_OI:
        _, _, seg, sid, oi = _OI.unpack_from(buf, off)
        return {"t": "oi", "seg": seg, "sid": sid, "oi": oi}, size
    if code == C.RESP_PREV_CLOSE:
        _, _, seg, sid, pc, poi = _PREV.unpack_from(buf, off)
        return {"t": "prev", "seg": seg, "sid": sid, "pc": _r2(pc), "poi": poi}, size
    if code == C.RESP_FULL:
        (_, _, seg, sid, ltp, ltq, ltt, atp, vol, tsq, tbq, oi, oih, oil, o, c, h, l) = _FULL.unpack_from(buf, off)
        return {"t": "full", "seg": seg, "sid": sid, "ltp": _r2(ltp), "ltq": ltq, "ltt": ltt,
                "atp": _r2(atp), "vol": vol, "tsq": tsq, "tbq": tbq, "oi": oi, "oih": oih, "oil": oil,
                "o": _r2(o), "dc": _r2(c), "h": _r2(h), "l": _r2(l),
                "depth": _depth5(buf, off + 62)}, size
    if code == C.RESP_MARKET_STATUS:
        _, _, seg, sid = _HDR.unpack_from(buf, off)
        return {"t": "status", "seg": seg, "sid": sid}, size
    if code == C.RESP_DISCONNECT:
        _, _, seg, sid, reason = _DISC.unpack_from(buf, off)
        return {"t": "disconnect", "reason": reason,
                "msg": C.DATA_API_ERRORS.get(reason, "unknown")}, size
    return None, size


def parse_market_feed(buf: bytes) -> list[dict]:
    """Parse a websocket binary message that may contain one or more stacked packets."""
    out: list[dict] = []
    off = 0
    n = len(buf)
    while off < n:
        pkt, used = parse_packet(buf, off)
        if used <= 0:
            break
        if pkt is not None:
            out.append(pkt)
        off += used
    return out


def parse_depth(buf: bytes, levels: int = 20) -> list[dict]:
    """Parse 20-level (or 200-level) depth messages. Bid (41) and ask (51) sides arrive as
    separate packets; several instruments may be stacked in one message."""
    out: list[dict] = []
    off = 0
    n = len(buf)
    while off + 12 <= n:
        ln, code, seg, sid, extra = _D20_HDR.unpack_from(buf, off)
        rows = extra if levels > 20 else 20
        body = rows * 16
        if code in (C.RESP_DEPTH_BID, C.RESP_DEPTH_ASK) and off + 12 + body <= n:
            levels_out = []
            for i in range(rows):
                price, qty, orders = _D20_ROW.unpack_from(buf, off + 12 + i * 16)
                if qty or price:
                    levels_out.append((round(price, 2), qty, orders))
            out.append({"t": "depth", "side": "bid" if code == C.RESP_DEPTH_BID else "ask",
                        "seg": seg, "sid": sid, "levels": levels_out})
            off += 12 + body
        elif code == C.RESP_DISCONNECT:
            out.append({"t": "disconnect", "sid": sid})
            off += max(ln, 14)
        else:
            off += max(ln, 12) if ln > 0 else n
    return out


def parse_cas(buf: bytes) -> list[dict]:
    """Closing Auction Session feed (Pro plan): 128-byte instrument packets and
    33-byte indicative index packets."""
    out: list[dict] = []
    off = 0
    n = len(buf)
    while off < n:
        # index packet: length (int16) comes first and equals 33
        if n - off >= 33 and struct.unpack_from("<h", buf, off)[0] == 33:
            ln, code, seg, name, val = _CAS_INDEX.unpack_from(buf, off)
            out.append({"t": "cas_index", "seg": seg,
                        "name": name.split(b"\x00", 1)[0].decode(errors="ignore").strip(),
                        "close": _r2(val)})
            off += 33
        elif n - off >= 128 and _HDR.unpack_from(buf, off)[1] == 128:
            code, ln, seg, sid, iep, bq, sq = _CAS.unpack_from(buf, off)
            imb, ref = _CAS_TAIL.unpack_from(buf, off + 120)
            out.append({"t": "cas", "seg": seg, "sid": sid, "iep": _r2(iep), "bq": bq, "sq": sq,
                        "imb": imb, "ref": _r2(ref), "depth": _depth5(buf, off + 20)})
            off += 128
        else:
            pkt, used = parse_packet(buf, off)
            if pkt:
                out.append(pkt)
            if used <= 0:
                break
            off += used
    return out


def encode_cas(seg, sid, iep, bq, sq, depth, imb, ref, code=60) -> bytes:
    head = _CAS.pack(code, 128, seg, sid, iep, bq, sq)
    body = b"".join(_DEPTH5.pack(*row) for row in depth[:5])
    body += b"\x00" * (100 - len(body))
    return head + body + _CAS_TAIL.pack(imb, ref)


# ---------------------------------------------------------------------------
# encoders — used by the simulator and the unit tests to produce byte-identical
# packets, so the same parsing path is exercised in sim and live mode.
# ---------------------------------------------------------------------------
def encode_quote(seg, sid, ltp, ltq, ltt, atp, vol, tsq, tbq, o, c, h, l) -> bytes:
    return _QUOTE.pack(C.RESP_QUOTE, 50, seg, sid, ltp, ltq, ltt, atp, vol, tsq, tbq, o, c, h, l)


def encode_full(seg, sid, ltp, ltq, ltt, atp, vol, tsq, tbq, oi, oih, oil, o, c, h, l, depth) -> bytes:
    head = _FULL.pack(C.RESP_FULL, 162, seg, sid, ltp, ltq, ltt, atp, vol, tsq, tbq, oi, oih, oil, o, c, h, l)
    body = b"".join(_DEPTH5.pack(*row) for row in depth[:5])
    body += b"\x00" * (100 - len(body))
    return head + body


def encode_ticker(seg, sid, ltp, ltt, code=C.RESP_TICKER) -> bytes:
    return _TICKER.pack(code, 16, seg, sid, ltp, ltt)


def encode_prev(seg, sid, pc, poi) -> bytes:
    return _PREV.pack(C.RESP_PREV_CLOSE, 16, seg, sid, pc, poi)


def encode_depth20(side_code, seg, sid, levels: list[tuple[float, int, int]]) -> bytes:
    rows = list(levels[:20]) + [(0.0, 0, 0)] * (20 - min(20, len(levels)))
    return _D20_HDR.pack(332, side_code, seg, sid, 0) + b"".join(_D20_ROW.pack(*r) for r in rows)
