#!/usr/bin/env python3
"""A local stand-in for the Dhan Data API — to test the *live* code path (DATA_SOURCE=dhan) without an
account or market hours. It serves the scrip master, the market-feed and 20-depth websockets, and the
historical / option-chain / news REST endpoints from the built-in simulator, with Dhan-like quirks:

  * security ids overlap across segments (NIFTY index = 13, an NSE equity = 13)
  * previous-close packets are sent only for indices (equities and futures get none)

    python scripts/mock_dhan.py --port 9100 --days=300
    # then start TradeDesk with:
    DATA_SOURCE=dhan DHAN_CLIENT_ID=mock DHAN_ACCESS_TOKEN=mock \
    DHAN_BASE_URL=http://127.0.0.1:9100/v2 DHAN_FEED_URL=ws://127.0.0.1:9100/feed \
    DHAN_DEPTH20_URL=ws://127.0.0.1:9100/twentydepth DHAN_SCRIP_MASTER_URL=http://127.0.0.1:9100/scrip-master.csv \
    python run.py
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import PlainTextResponse  # noqa: E402

from td.dhan import parser as P  # noqa: E402
from td.dhan.instruments import sim_universe  # noqa: E402
from td.sim.simulator import MarketSimulator, generate_history  # noqa: E402
from td.timeutil import IST, now_ist, previous_weekday  # noqa: E402
from td.universe_data import INDICES  # noqa: E402

UNI = sim_universe()
TODAY = now_ist().date()
EXPIRY = (TODAY + timedelta(days=(3 - TODAY.weekday()) % 7 + 14)).isoformat()   # a Thursday ~2-3 weeks out
print("building synthetic history ...", flush=True)
DAYS = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--days=")), "80"))
HIST = generate_history(UNI, previous_weekday(TODAY), days=DAYS, minute_days=12, seed=5)
PREV = {s: float(df["close"].iat[-1]) for s, df in HIST["D"].items() if not df.empty}
ADV = {s: float(df["volume"].tail(20).mean()) for s, df in HIST["D"].items() if not df.empty}
SIM = MarketSimulator(UNI, TODAY, PREV, ADV, seed=5, start="09:13:30")
SEG_NAME = {0: "IDX_I", 1: "NSE_EQ", 2: "NSE_FNO"}
BY_SEG_SID = {(i.seg, i.sid): i for i in UNI.instruments}
app = FastAPI()
subscribers: list[tuple[WebSocket, set]] = []


class TodayBars:
    """Today's 1-minute OHLCV per equity, recorded from the simulator (served by /charts/intraday)."""

    def __init__(self):
        self.bars: dict[str, list[list]] = {s: [] for s in SIM.syms}
        self.minute = None

    def sample(self):
        t = SIM.sim_t
        if t < SIM.open_dt or t >= SIM.close_dt:
            return
        m = int(t.replace(second=0, microsecond=0).timestamp())
        px, vol = SIM.px, SIM.cumvol
        if m != self.minute:
            self.minute, self.o, self.h, self.l, self.v0 = m, px.copy(), px.copy(), px.copy(), vol.copy()
            for i, s in enumerate(SIM.syms):
                self.bars[s].append([m, 0, 0, 0, 0, 0])
        self.h = np.maximum(self.h, px)
        self.l = np.minimum(self.l, px)
        for i, s in enumerate(SIM.syms):
            self.bars[s][-1][1:] = [round(float(self.o[i]), 2), round(float(self.h[i]), 2), round(float(self.l[i]), 2),
                                    round(float(px[i]), 2), int(vol[i] - self.v0[i])]


TODAY_BARS = TodayBars()


def fast_forward():
    """Replay the session from the open to now, so a mid-session start looks like Dhan's: full-day volume,
    high/low and today's 1-minute bars are already there."""
    now = now_ist()
    if now <= SIM.open_dt:
        return
    end = min(now, SIM.close_dt)
    emit = SIM._emit
    SIM._emit = lambda *a, **k: []                      # skip packet encoding while replaying
    while SIM.sim_t < end:
        SIM.advance(1.0)
        TODAY_BARS.sample()
    SIM._emit = emit
    SIM.drain_news()


print("replaying today's session up to now ...", flush=True)
fast_forward()


def scrip_master() -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["EXCH_ID", "SEGMENT", "SECURITY_ID", "ISIN", "INSTRUMENT", "UNDERLYING_SECURITY_ID", "UNDERLYING_SYMBOL",
                "SYMBOL_NAME", "DISPLAY_NAME", "INSTRUMENT_TYPE", "SERIES", "LOT_SIZE", "SM_EXPIRY_DATE", "STRIKE_PRICE",
                "OPTION_TYPE", "TICK_SIZE", "EXPIRY_FLAG"])
    for i in UNI.instruments:
        if i.kind == "IDX":
            disp = "INDIA VIX" if i.symbol == "INDIAVIX" else i.symbol
            w.writerow(["NSE", "I", i.sid, "NA", "INDEX", i.sid, i.symbol, disp, disp, "INDEX", "", 1, "", "", "", 0.05, ""])
        elif i.kind == "EQ":
            w.writerow(["NSE", "E", i.sid, f"INE{i.sid:06d}01", "EQUITY", "", i.symbol, i.symbol, i.symbol, "ES", "EQ", 1,
                        "", "", "", 0.05, ""])
        elif i.kind == "FUT":
            w.writerow(["NSE", "D", i.sid, "NA", "FUTSTK", "", i.underlying, f"{i.underlying}-FUT", f"{i.underlying} FUT",
                        "FUT", "", i.lot, f"{EXPIRY} 14:30:00", "", "", 0.05, "M"])
    return out.getvalue()


@app.get("/scrip-master.csv", response_class=PlainTextResponse)
async def master():
    return scrip_master()


def _hist(symbol: str, interval: str, start: datetime, end: datetime) -> dict:
    df = HIST.get(interval, {}).get(symbol)
    if df is None or df.empty:
        return {"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    a, b = start.replace(tzinfo=IST).timestamp(), end.replace(tzinfo=IST).timestamp()
    d = df[(df["ts"] >= a) & (df["ts"] < b)]
    return {"timestamp": d["ts"].astype(int).tolist(), "open": d["open"].tolist(), "high": d["high"].tolist(),
            "low": d["low"].tolist(), "close": d["close"].tolist(), "volume": d["volume"].astype(int).tolist(),
            "open_interest": [0] * len(d)}


@app.post("/v2/charts/historical")
async def historical(req: Request):
    b = await req.json()
    inst = BY_SEG_SID.get((b["exchangeSegment"], int(b["securityId"])))
    if not inst:
        return {"status": "failure", "errorCode": "813", "errorMessage": "Invalid SecurityId"}
    sym = inst.underlying if inst.kind == "FUT" else inst.symbol
    out = _hist(sym, "D", datetime.fromisoformat(b["fromDate"]), datetime.fromisoformat(b["toDate"]))
    if inst.kind == "FUT" and out["timestamp"]:
        i = SIM.syms.index(inst.underlying) if inst.underlying in SIM.syms else None
        n = len(out["timestamp"])
        oi = int(SIM.oi_prev[i]) if i is not None else 2_000_000
        out["open_interest"] = [int(oi * (0.9 + 0.1 * k / max(1, n - 1))) for k in range(n)]
        if i is not None:                                    # futures trade at a premium to spot
            out["close"] = [round(c * (1 + float(SIM.basis[i])), 2) for c in out["close"]]
    return out


@app.post("/v2/charts/intraday")
async def intraday(req: Request):
    b = await req.json()
    inst = BY_SEG_SID.get((b["exchangeSegment"], int(b["securityId"])))
    if not inst:
        return {"status": "failure", "errorCode": "813", "errorMessage": "Invalid SecurityId"}
    start, end = datetime.fromisoformat(b["fromDate"]), datetime.fromisoformat(b["toDate"])
    out = _hist(inst.symbol, "1m", start, end)
    a, z = start.replace(tzinfo=IST).timestamp(), end.replace(tzinfo=IST).timestamp()
    for row in TODAY_BARS.bars.get(inst.symbol, [])[:-1]:      # completed minutes only
        if a <= row[0] < z:
            for k, v in zip(("timestamp", "open", "high", "low", "close", "volume"), row):
                out[k].append(v)
    out["open_interest"] = [0] * len(out["timestamp"])
    return out


def _index_name(sid: int) -> str | None:
    return next((n for n, (s, _, _) in INDICES.items() if s == sid), None)


@app.post("/v2/optionchain/expirylist")
async def expirylist(req: Request):
    b = await req.json()
    name = _index_name(int(b["UnderlyingScrip"]))
    return {"data": SIM.expiries(name) if name else [], "status": "success"}


@app.post("/v2/optionchain")
async def optionchain(req: Request):
    b = await req.json()
    name = _index_name(int(b["UnderlyingScrip"]))
    data = SIM.option_chain(name, b["Expiry"]) if name else {}
    spot = float(data.get("last_price") or 0)
    for k, row in (data.get("oc") or {}).items():           # Dhan quirk: deep in-the-money legs come back with IV/greeks = 0
        strike = float(k)
        for side, deep in (("ce", strike < spot * 0.97), ("pe", strike > spot * 1.03)):
            if deep and side in row:
                row[side]["implied_volatility"] = 0
                row[side]["greeks"] = {g: 0 for g in ("delta", "theta", "gamma", "vega")}
    return {"data": data, "status": "success"}


@app.post("/v2/data/newsheadline")
async def news(req: Request):
    return {"data": [], "status": "success"}


@app.websocket("/feed")
async def feed(ws: WebSocket):
    await ws.accept()
    subs: set = set()
    subscribers.append((ws, subs))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            for it in msg.get("InstrumentList", []):
                key = (it["ExchangeSegment"], int(it["SecurityId"]))
                subs.add(key)
                inst = BY_SEG_SID.get(key)
                if inst and inst.kind == "IDX" and inst.symbol in SIM.idx_prev:   # Dhan: prev close for indices only
                    await ws.send_bytes(P.encode_prev(0, inst.sid, float(SIM.idx_prev[inst.symbol]), 0))
    except WebSocketDisconnect:
        pass
    finally:
        subscribers.remove((ws, subs))


@app.websocket("/twentydepth")
async def depth(ws: WebSocket):
    await ws.accept()
    want: set = set()

    async def reader():
        while True:
            msg = json.loads(await ws.receive_text())
            for it in msg.get("InstrumentList", []):
                want.add(int(it["SecurityId"]))
    task = asyncio.create_task(reader())
    try:
        while True:
            for sid in list(want):
                for b in SIM.depth20(sid):
                    await ws.send_bytes(b)
            await asyncio.sleep(1)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        task.cancel()


async def pump():
    sent_prev = False
    while True:
        t0 = time.monotonic()
        msgs = SIM.advance(1.0)
        TODAY_BARS.sample()
        if subscribers:
            for _, b in msgs:
                for p in P.parse_market_feed(b):
                    seg = SEG_NAME.get(p.get("seg"))
                    if p.get("t") == "prev" and p.get("seg") != 0:
                        continue                               # Dhan quirk: no prev close for equities/futures
                    for ws, subs in list(subscribers):
                        if (seg, p["sid"]) in subs:
                            try:
                                await ws.send_bytes(b)
                            except Exception:
                                pass
            if not sent_prev:
                sent_prev = True
        await asyncio.sleep(max(0.0, 1.0 - (time.monotonic() - t0)))


@app.on_event("startup")
async def _start():
    asyncio.create_task(pump())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--days", type=int, default=80, help="days of daily history to serve (read at import as --days=N)")
    a = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
