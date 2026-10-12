"""GATEWAY — FastAPI REST + WebSocket fan-out to browsers.

Stateless: every instance subscribes to Redis (td:pub:*) once and fans messages out to its
connected clients, so you can run several gateways behind a load balancer. Messages are
forwarded as raw bytes (no re-encoding) through bounded per-client queues — a slow
client never slows down the others.
Run:  uvicorn td.services.gateway:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import orjson
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__, bus
from ..backfill import JobSpec
from ..analytics import alerts as alerts_mod
from ..analytics import smc
from ..config import ROOT, settings
from ..lake import Lake
from ..timeutil import IST, MARKET_OPEN, at, from_epoch

log = logging.getLogger("td.gateway")
FRONTEND = ROOT / "frontend" / "dist"


class Hub:
    """Topic -> clients registry with raw-bytes fan-out."""

    def __init__(self):
        self.clients: dict[int, "Client"] = {}
        self.redis = None
        self.task: asyncio.Task | None = None
        self.msgs_out = 0

    async def start(self):
        self.redis = bus.connect()
        self.task = asyncio.create_task(self._pump())

    async def stop(self):
        if self.task:
            self.task.cancel()
        if self.redis:
            await self.redis.aclose()

    async def _pump(self):
        while True:
            try:
                ps = self.redis.pubsub()
                await ps.psubscribe(bus.PUB_PREFIX + "*")
                async for msg in ps.listen():
                    if msg.get("type") != "pmessage":
                        continue
                    topic = msg["channel"].decode()[len(bus.PUB_PREFIX):]
                    data = msg["data"]
                    for c in list(self.clients.values()):
                        if c.wants(topic):
                            c.push(topic, data)
                            self.msgs_out += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("pubsub pump error, reconnecting")
                await asyncio.sleep(1)


class Client:
    """Per-browser outbox. Snapshot topics are coalesced (only the newest frame is kept while the
    socket is busy); events (alerts, lock notices) are queued individually."""

    MAX_EVENTS = 200

    def __init__(self, ws: WebSocket, user: str):
        self.ws = ws
        self.user = user
        self.topics: set[str] = set()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.latest: dict[str, bytes] = {}
        self.events: list[bytes] = []
        self.connected = time.time()

    def wants(self, topic: str) -> bool:
        return topic in self.topics or topic in ("alert", "event")

    def push(self, topic: str, data: bytes):
        if topic in ("alert", "event"):
            if len(self.events) < self.MAX_EVENTS:
                self.events.append(data)
                self.queue.put_nowait("__ev")
            return
        pending = topic in self.latest
        self.latest[topic] = data
        if not pending:
            self.queue.put_nowait(topic)

    async def sender(self):
        while True:
            topic = await self.queue.get()
            if topic == "__ev":
                if self.events:
                    await self.ws.send_text(self.events.pop(0).decode())
                continue
            data = self.latest.pop(topic, None)
            if data is not None:
                await self.ws.send_text(data.decode())


hub = Hub()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await hub.start()
    app.state.redis = bus.connect()
    app.state.lake = Lake()
    asyncio.create_task(_status_loop(app))
    yield
    await hub.stop()
    await app.state.redis.aclose()


async def _status_loop(app):
    demand_key = f"td:demand:{os.getpid()}"
    while True:
        try:
            users = sorted({c.user for c in hub.clients.values()})
            await app.state.redis.set(bus.k_status(f"gateway:{os.getpid()}"), bus.dumps({
                "pid": os.getpid(), "clients": len(hub.clients), "users": users, "msgs_out": hub.msgs_out,
                "updated": time.time()}), ex=15)
            # which geo views have viewers -> the GEO worker pauses feeds nobody is watching
            counts: dict[str, int] = {}
            for c in hub.clients.values():
                for t in c.topics:
                    if t.startswith("geo:"):
                        counts[t] = counts.get(t, 0) + 1
            pipe = app.state.redis.pipeline(transaction=False)
            pipe.delete(demand_key)
            if counts:
                pipe.hset(demand_key, mapping=counts)
                pipe.expire(demand_key, 15)
            await pipe.execute()
        except Exception:
            pass
        await asyncio.sleep(3)


class FastJSON(Response):
    """orjson rendering (numpy-aware, NaN -> null) without FastAPI's deprecated ORJSONResponse."""
    media_type = "application/json"

    def render(self, content) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS)


app = FastAPI(title="TradeDesk Gateway", version=__version__, default_response_class=FastJSON,
              lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=2048)
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ auth
def user_for_key(key: str | None) -> str:
    keys = settings.access_key_map()
    if not keys:
        return "local"
    if key and key in keys:
        return keys[key]
    raise HTTPException(401, "invalid or missing access key")


def auth(request: Request) -> str:
    key = request.headers.get("x-access-key") or request.query_params.get("key")
    return user_for_key(key)


LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _local_client(request: Request) -> bool:
    return bool(request.client and request.client.host in LOOPBACK)


def _admin_mode(request: Request) -> str:
    """Effective admin mode for this request: 'open' only for requests from this machine."""
    mode = settings.admin_mode()
    return "off" if mode == "open" and not _local_client(request) else mode


def _same_site(request: Request) -> bool:
    """CORS is open (*) by default, so a page on another site could otherwise make the browser POST here.
    The Host header is not trusted (DNS rebinding); local origins and explicit CORS_ORIGINS are."""
    origin = request.headers.get("origin")
    if not origin:
        return True                                   # scripts / curl / same-origin navigation
    if urlsplit(origin).hostname in LOOPBACK:
        return True                                   # the terminal on localhost or the Vite dev server
    explicit = {o.strip().rstrip("/") for o in settings.cors_origins.split(",") if o.strip() not in ("", "*")}
    return origin.rstrip("/") in explicit


def admin(request: Request) -> str:
    """ADMIN desk: history pulls spend the Dhan quota. Without keys it is open to this machine only; with
    ACCESS_KEYS it needs ADMIN_KEY (and is off without one)."""
    user = auth(request)
    mode = _admin_mode(request)
    if mode == "off":
        raise HTTPException(403, "The admin desk is off here — it is open on the host itself (localhost); "
                                 "set ADMIN_KEY in .env to use it from other machines")
    if mode == "key":
        given = (request.headers.get("x-admin-key") or "").encode()
        if not hmac.compare_digest(given, settings.admin_key.encode()):
            raise HTTPException(403, "admin key required")
    elif request.method not in ("GET", "HEAD") and not _same_site(request):
        raise HTTPException(403, "cross-site admin request blocked")      # (a valid admin key proves intent)
    return user


# -------------------------------------------------------------- websocket
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, key: str | None = Query(None)):
    try:
        user = user_for_key(key)
    except HTTPException:
        # accept then close, so browsers see code 4401 (a handshake rejection only shows 1006)
        await ws.accept()
        await ws.close(code=4401, reason="invalid access key")
        return
    await ws.accept()
    client = Client(ws, user)
    cid = id(client)
    hub.clients[cid] = client
    sender = asyncio.create_task(client.sender())
    r = app.state.redis
    try:
        await ws.send_text(bus.dumps({"topic": "hello", "data": {"user": user, "version": __version__,
                                                                  "source": settings.data_source}}).decode())
        while True:
            msg = await ws.receive_json()
            op = msg.get("op")
            if op == "sub":
                new = [t for t in msg.get("topics", []) if isinstance(t, str)][:40]
                client.topics.update(new)
                if new:
                    snaps = await r.mget([bus.SNAP_PREFIX + t for t in new])
                    for t, s in zip(new, snaps):
                        if s:
                            client.push(t, s)
            elif op == "unsub":
                for t in msg.get("topics", []):
                    client.topics.discard(t)
            elif op == "depth":
                sid = int(msg.get("sid", 0))
                if sid:
                    await r.sadd("td:depth:want", sid)
                    await r.expire("td:depth:want", 120)
            elif op == "ping":
                await ws.send_text(bus.dumps({"topic": "pong", "data": {"t": time.time()}}).decode())
    except WebSocketDisconnect:
        pass
    except Exception:
        log.debug("ws closed", exc_info=True)
    finally:
        sender.cancel()
        hub.clients.pop(cid, None)


# ------------------------------------------------------------------- REST
@app.get("/api/meta")
async def meta(request: Request, user: str = Depends(auth)):
    return {"version": __version__, "user": user, "source": settings.data_source, "plan": settings.dhan_plan,
            "lock_time": settings.lock_time, "top_n": settings.top_n, "chains": settings.chain_list(),
            "or_minutes": settings.or_minutes, "auth": bool(settings.access_key_map()),
            "admin": _admin_mode(request),
            "geo": {"enabled": settings.geo_enabled, "mode": settings.geo_mode,
                    "disabled": sorted(settings.geo_disabled())}}


@app.get("/api/universe")
async def universe(user: str = Depends(auth)):
    raw = await app.state.redis.get(bus.KEY_UNIVERSE)
    return bus.loads(raw) if raw else []


@app.get("/api/snapshot/{topic:path}")
async def snapshot(topic: str, user: str = Depends(auth)):
    raw = await app.state.redis.get(bus.SNAP_PREFIX + topic)
    if not raw:
        raise HTTPException(404, f"no snapshot for {topic}")
    return bus.loads(raw)


async def _inst(symbol: str) -> dict:
    raw = await app.state.redis.get(bus.KEY_UNIVERSE)
    for i in bus.loads(raw) or []:
        if i["symbol"] == symbol.upper():
            return i
    raise HTTPException(404, f"unknown symbol {symbol}")


def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if df.empty or minutes <= 1:
        return df
    t = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(IST)
    g = df.assign(dt=t).set_index("dt")
    o = g.resample(f"{minutes}min", origin="start_day", offset="15min", label="left", closed="left").agg(
        {"ts": "first", "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    o["ts"] = [int(x.timestamp()) for x in o.index]
    return o.reset_index(drop=True)


@app.get("/api/chart/{symbol}")
async def chart(symbol: str, tf: str = "5", days: int = 3, user: str = Depends(auth)):
    """Bars (lake history + today's live minute bars), VWAP, key levels and SMC overlays."""
    inst = await _inst(symbol)
    lake: Lake = app.state.lake
    r = app.state.redis
    raw = await r.get(bus.k_bars(inst.get("key", inst["sid"])))
    today = bus.loads(raw) if raw else None
    live_rows = []
    if today:
        day = pd.Timestamp(today["day"]).date()
        t0 = int(at(day, MARKET_OPEN).timestamp())
        for b in today["bars"]:
            m, o, h, l, c, v = b
            live_rows.append({"ts": t0 + int(m) * 60, "open": o, "high": h, "low": l, "close": c, "volume": v})
    live = pd.DataFrame(live_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    name = inst["underlying"] if inst["kind"] == "FUT" else inst["symbol"]
    if tf == "D":
        hist = await asyncio.to_thread(lake.read, "D", name, None, None, 400)
        df = hist[["ts", "open", "high", "low", "close", "volume"]]
        if not live.empty:
            day_ts = int(pd.Timestamp(today["day"]).tz_localize(IST).timestamp())
            agg = {"ts": day_ts, "open": live["open"].iat[0], "high": live["high"].max(), "low": live["low"].min(),
                   "close": live["close"].iat[-1], "volume": int(live["volume"].sum())}
            df = pd.concat([df[df["ts"] < day_ts], pd.DataFrame([agg])], ignore_index=True)
        k = 3
    else:
        minutes = int(tf)
        hist = await asyncio.to_thread(lake.read, "1m", name, None, None, 375 * max(1, days))
        hist = hist[["ts", "open", "high", "low", "close", "volume"]]
        if not live.empty:
            hist = hist[hist["ts"] < live["ts"].min()]
        df = pd.concat([hist, live], ignore_index=True)
        df = _resample(df, minutes)
        k = 2
    df = df.dropna().reset_index(drop=True)
    # session VWAP for today's bars
    vwap = []
    if not live.empty and tf != "D":
        lv = live.copy()
        tp = (lv["high"] + lv["low"] + lv["close"]) / 3
        cv = lv["volume"].cumsum().replace(0, pd.NA)
        lv["vwap"] = (tp * lv["volume"]).cumsum() / cv
        # one point per chart bar: extra 1-minute points would add empty slots to the time axis on 5m/15m
        step = int(tf) * 60                       # t0 = today's 09:15, same alignment as _resample
        by_bar: dict[int, float] = {}
        for t, v in zip(lv["ts"], lv["vwap"].ffill().fillna(lv["close"])):
            by_bar[t0 + (int(t) - t0) // step * step] = round(float(v), 2)
        vwap = [[t, v] for t, v in sorted(by_bar.items())]
    d = await asyncio.to_thread(lake.read, "D", name, None, None, 3)
    levels = {}
    if not d.empty:
        today_ts = int(pd.Timestamp(today["day"]).tz_localize(IST).timestamp()) if today else None
        prev = d[d["ts"] < today_ts] if today_ts else d
        if not prev.empty:
            levels = {"pdh": float(prev["high"].iat[-1]), "pdl": float(prev["low"].iat[-1]),
                      "pdc": float(prev["close"].iat[-1])}
    if today:
        levels.update({"orh": today.get("orh"), "orl": today.get("orl")})
    sm = await asyncio.to_thread(smc.analyze, df.tail(400).reset_index(drop=True), k, 30)
    bars = [[int(r.ts), round(r.open, 2), round(r.high, 2), round(r.low, 2), round(r.close, 2), int(r.volume)]
            for r in df.tail(1500).itertuples()]
    return {"symbol": inst["symbol"], "sid": inst.get("key", inst["sid"]), "tf": tf, "bars": bars, "vwap": vwap,
            "levels": levels, "smc": sm}


@app.get("/api/depth/{sid}")
async def depth(sid: int, user: str = Depends(auth)):
    r = app.state.redis
    await r.sadd("td:depth:want", sid)
    await r.expire("td:depth:want", 120)
    raw = await r.get(bus.k_depth(sid))
    return bus.loads(raw) if raw else {"sid": sid, "bid": [], "ask": [], "pending": True}


@app.get("/api/research/intraday")
async def research_intraday(user: str = Depends(auth)):
    """Latest walk-forward backtest written by research/evaluate_intraday.py (None if never run)."""
    p = Path(settings.data_dir) / "research" / "intraday_eval.json"
    if not p.exists():
        return {"available": False}
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return {"available": False}
    d["days"] = d.get("days", [])[-60:]
    d["available"] = True
    return d


@app.get("/api/scorecard")
async def scorecard(user: str = Depends(auth)):
    raw = await app.state.redis.lrange(bus.KEY_SCORECARD, 0, 249)
    cards = [bus.loads(x) for x in raw]
    if not cards:
        return {"cards": [], "summary": None}
    n = len(cards)
    ics = [c["ic"] for c in cards if c.get("ic") is not None]
    summary = {
        "days": n,
        "mean_ic": round(sum(ics) / len(ics), 4) if ics else None,
        "ic_positive_days": sum(1 for x in ics if x > 0),
        "avg_long": round(sum(c.get("long_avg", 0) for c in cards) / n, 3),
        "avg_universe": round(sum(c.get("universe_avg", 0) for c in cards) / n, 3),
        "avg_excess": round(sum(c.get("excess_long", 0) for c in cards) / n, 3),
        "top5_hits_per_day": round(sum(c.get("long_hits_top5", 0) for c in cards) / n, 3),
        "top20_hit_rate": round(sum(c.get("long_hits_top20", 0) for c in cards) /
                                max(1, sum(c.get("n_long", 0) for c in cards)) * 100, 1),
        "pct_green": round(sum(c.get("long_green", 0) for c in cards) /
                           max(1, sum(c.get("n_long", 0) for c in cards)) * 100, 1),
    }
    return {"cards": cards, "summary": summary}


class AlertIn(BaseModel):
    symbol: str
    type: str
    value: float | str | None = None
    note: str = ""
    repeat: bool = False


@app.get("/api/alerts")
async def list_alerts(user: str = Depends(auth)):
    r = app.state.redis
    defs = [bus.loads(v) for v in (await r.hgetall(bus.KEY_ALERTS)).values()]
    log_ = [bus.loads(v) for v in await r.lrange(bus.KEY_ALERT_LOG, 0, 99)]
    return {"alerts": sorted([a for a in defs if a.get("user") in (user, "local") or user == "local"],
                             key=lambda a: -a["created"]),
            "log": log_, "types": alerts_mod.TYPES}


@app.post("/api/alerts")
async def create_alert(a: AlertIn, user: str = Depends(auth)):
    try:
        al = alerts_mod.new_alert(user, a.symbol, a.type, a.value, a.note, a.repeat)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    await app.state.redis.hset(bus.KEY_ALERTS, al["id"], bus.dumps(al))
    return al


@app.delete("/api/alerts/{aid}")
async def delete_alert(aid: str, user: str = Depends(auth)):
    await app.state.redis.hdel(bus.KEY_ALERTS, aid)
    return {"ok": True}


@app.get("/api/system")
async def system(user: str = Depends(auth)):
    r = app.state.redis
    keys = [k async for k in r.scan_iter(match="td:status:*")]
    vals = await r.mget(keys) if keys else []
    status = {k.decode().split(":", 2)[2]: bus.loads(v) for k, v in zip(keys, vals) if v}
    lake: Lake = app.state.lake
    cached = getattr(app.state, "lake_summary", None)
    if cached and time.time() - cached[0] < 30:
        summary = cached[1]
    else:
        summary = await asyncio.to_thread(lake.summary)
        app.state.lake_summary = (time.time(), summary)
    info = await r.info("memory")
    streams = {}
    for s in (bus.STREAM_TICKS, bus.STREAM_NEWS, bus.STREAM_EVENTS):
        try:
            streams[s] = await r.xlen(s)
        except Exception:
            streams[s] = 0
    return {"status": status, "lake": summary, "lake_root": str(lake.root),
            "redis": {"used_memory_human": info.get("used_memory_human"), "streams": streams},
            "gateway": {"clients": len(hub.clients), "users": sorted({c.user for c in hub.clients.values()}),
                        "msgs_out": hub.msgs_out},
            "settings": {"source": settings.data_source, "plan": settings.dhan_plan, "universe": settings.universe,
                         "lock_time": settings.lock_time, "top_n": settings.top_n,
                         "limits": {"data_per_sec": settings.limit_data_per_sec,
                                    "data_per_day": settings.limit_data_per_day,
                                    "quote_per_sec": settings.limit_quote_per_sec}}}


@app.get("/api/lake/coverage")
async def lake_coverage(user: str = Depends(auth)):
    return await asyncio.to_thread(app.state.lake.coverage)


# ------------------------------------------------------------------ admin
class PullIn(BaseModel):
    intervals: list[str] = ["D", "1"]
    years: float = 5
    rate: float = 3
    reserve: int = 2500
    market_hours: str = "slow"
    symbols: list[str] = []


def _spec(body: PullIn) -> dict:
    try:
        return JobSpec.clean(body.model_dump()).as_dict()
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


async def _admin_cmd(op: str, user: str, **extra) -> None:
    await app.state.redis.lpush(bus.KEY_ADMIN_CMD, json.dumps({"op": op, "by": user, "ts": time.time(), **extra}))
    await app.state.redis.expire(bus.KEY_ADMIN_CMD, 3600)


@app.get("/api/admin/backfill")
async def admin_backfill(user: str = Depends(admin)):
    """Backfill worker status (None when the worker is not running)."""
    raw = await app.state.redis.get(bus.KEY_ADMIN_STATUS)
    return {"status": json.loads(raw) if raw else None, "source": settings.data_source, "user": user}


@app.post("/api/admin/backfill/plan")
async def admin_backfill_plan(body: PullIn, user: str = Depends(admin)):
    """Ask the worker how many requests the pull still needs (it owns the resume state)."""
    spec = _spec(body)
    rid = uuid.uuid4().hex
    await _admin_cmd("plan", user, id=rid, job=spec)
    r = app.state.redis
    for _ in range(40):
        raw = await r.get(bus.k_admin_plan(rid))
        if raw:
            await r.delete(bus.k_admin_plan(rid))
            return json.loads(raw)
        await asyncio.sleep(0.25)
    raise HTTPException(504, "the backfill worker did not answer — is it running?")


@app.post("/api/admin/backfill/start")
async def admin_backfill_start(body: PullIn, user: str = Depends(admin)):
    spec = _spec(body)
    await _admin_cmd("start", user, job=spec)
    return {"ok": True, "job": spec}


@app.post("/api/admin/backfill/stop")
async def admin_backfill_stop(user: str = Depends(admin)):
    await _admin_cmd("stop", user)
    return {"ok": True}


@app.post("/api/admin/backfill/coverage")
async def admin_backfill_coverage(user: str = Depends(admin)):
    await _admin_cmd("coverage", user)
    return {"ok": True}


@app.get("/api/health")
async def health():
    try:
        await app.state.redis.ping()
        ok = True
    except Exception:
        ok = False
    return {"ok": ok, "time": time.time()}


# ------------------------------------------------------- frontend (built)
if FRONTEND.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")

    @app.get("/{path:path}")
    async def spa(path: str):
        f = FRONTEND / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(FRONTEND / "index.html")
