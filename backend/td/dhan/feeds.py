"""Dhan websocket clients: live market feed, 20-level depth, CAS (closing auction).

One `MarketFeedManager` shards the subscription list across up to 5 connections
(5,000 instruments each, 100 per subscribe message) and reconnects with backoff.
Parsed packets are handed to an async callback in batches. The URL (it carries the access token)
is built on every connect, and an open socket is closed when the token keeper swaps the token, so the
feeds move to a renewed token within seconds.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from ..config import settings
from . import constants as C
from .parser import parse_cas, parse_depth, parse_market_feed

log = logging.getLogger("td.dhan.feed")

PacketHandler = Callable[[list[dict]], Awaitable[None]]
TOKEN_POLL = 5.0          # s, how often an open socket checks whether the access token changed


def _auth_url(base: str, version: bool = False) -> str:
    v = "version=2&" if version else ""
    return f"{base}?{v}token={settings.dhan_access_token}&clientId={settings.dhan_client_id}&authType=2"


@dataclass
class Sub:
    segment: str          # e.g. "NSE_EQ"
    security_id: int
    mode: int = C.REQ_SUB_QUOTE   # 15 ticker / 17 quote / 21 full


@dataclass
class FeedStats:
    connected: int = 0
    messages: int = 0
    packets: int = 0
    reconnects: int = 0
    last_error: str = ""
    per_conn: dict = field(default_factory=dict)


def _scrub(text: str, token: str) -> str:
    return text.replace(token, "***") if token else text


def _sub_messages(subs: list[Sub]) -> list[str]:
    msgs = []
    by_mode: dict[int, list[Sub]] = {}
    for s in subs:
        by_mode.setdefault(s.mode, []).append(s)
    for mode, items in by_mode.items():
        for i in range(0, len(items), C.MAX_INSTRUMENTS_PER_MESSAGE):
            batch = items[i:i + C.MAX_INSTRUMENTS_PER_MESSAGE]
            msgs.append(json.dumps({
                "RequestCode": mode,
                "InstrumentCount": len(batch),
                "InstrumentList": [{"ExchangeSegment": s.segment, "SecurityId": str(s.security_id)} for s in batch],
            }))
    return msgs


class _Conn:
    def __init__(self, idx: int, url: str | Callable[[], str], subs: list[Sub], handler: PacketHandler,
                 stats: FeedStats, parser=parse_market_feed):
        self.idx = idx
        self.url = url              # a factory: called on every (re)connect, so a renewed token is used
        self.subs = subs
        self.handler = handler
        self.stats = stats
        self.parser = parser
        self._stop = asyncio.Event()
        self._rotated = False

    async def _watch_token(self, ws, used: str) -> None:
        while True:
            await asyncio.sleep(TOKEN_POLL)
            if settings.dhan_access_token != used:
                log.info("feed[%d] access token renewed — reconnecting", self.idx)
                self._rotated = True
                with contextlib.suppress(Exception):
                    await ws.close(code=1000, reason="token renewed")
                return

    async def run(self):
        backoff = 1.0
        while not self._stop.is_set():
            used = settings.dhan_access_token
            url = self.url() if callable(self.url) else self.url
            watch: asyncio.Task | None = None
            try:
                async with connect(url, max_size=2 ** 22, ping_interval=None,
                                   open_timeout=15, close_timeout=5) as ws:
                    if callable(self.url):
                        watch = asyncio.create_task(self._watch_token(ws, used))
                    self.stats.connected += 1
                    self.stats.per_conn[self.idx] = "up"
                    for msg in _sub_messages(self.subs):
                        await ws.send(msg)
                    backoff = 1.0
                    log.info("feed[%d] connected, %d instruments", self.idx, len(self.subs))
                    async for raw in ws:
                        if isinstance(raw, str):
                            continue
                        self.stats.messages += 1
                        pkts = self.parser(raw)
                        if pkts:
                            self.stats.packets += len(pkts)
                            for p in pkts:
                                if p.get("t") == "disconnect":
                                    self.stats.last_error = f"server disconnect: {p.get('msg', p)}"
                                    log.warning("feed[%d] %s", self.idx, self.stats.last_error)
                            await self.handler(pkts)
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                self.stats.last_error = f"conn {self.idx}: {type(e).__name__}: {_scrub(str(e), used)}"
            except Exception as e:  # pragma: no cover - defensive
                self.stats.last_error = f"conn {self.idx}: {type(e).__name__}: {_scrub(str(e), used)}"
                if used and used in str(e):              # e.g. an invalid-URI error quoting the URL
                    log.error("feed[%d] error: %s", self.idx, self.stats.last_error)
                else:
                    log.exception("feed[%d] error", self.idx)
            finally:
                if watch:
                    watch.cancel()
                if self.stats.per_conn.get(self.idx) == "up":
                    self.stats.connected = max(0, self.stats.connected - 1)
                self.stats.per_conn[self.idx] = "down"
            if self._stop.is_set():
                break
            self.stats.reconnects += 1
            if self._rotated:                            # not a failure: reconnect at once with the new token
                self._rotated = False
                backoff = 1.0
                continue
            await asyncio.sleep(backoff + random.random())
            backoff = min(backoff * 2, 30)

    def stop(self):
        self._stop.set()


class MarketFeedManager:
    """Shards subscriptions over ≤5 websocket connections (Dhan limit per user)."""

    def __init__(self, subs: list[Sub], handler: PacketHandler, max_connections: int = 4):
        # keep one of the 5 allowed sockets free for depth/CAS feeds
        self.subs = subs
        self.handler = handler
        self.max_connections = max(1, min(max_connections, C.MAX_CONNECTIONS))
        self.stats = FeedStats()
        self._conns: list[_Conn] = []

    def _url(self) -> str:
        return _auth_url(settings.dhan_feed_url, version=True)

    async def run(self):
        per = C.MAX_INSTRUMENTS_PER_CONNECTION
        shards = [self.subs[i:i + per] for i in range(0, len(self.subs), per)][: self.max_connections]
        dropped = len(self.subs) - sum(len(s) for s in shards)
        if dropped > 0:
            log.warning("feed: %d instruments exceed websocket capacity and were dropped", dropped)
        self._conns = [_Conn(i, self._url, shard, self.handler, self.stats) for i, shard in enumerate(shards)]
        await asyncio.gather(*(c.run() for c in self._conns))

    def stop(self):
        for c in self._conns:
            c.stop()


class Depth20Feed:
    """20-level order book for up to 50 instruments on one connection (NSE_EQ / NSE_FNO only)."""

    def __init__(self, subs: list[Sub], handler: PacketHandler):
        self.subs = subs[: C.DEPTH20_MAX_INSTRUMENTS]
        for s in self.subs:
            s.mode = C.REQ_SUB_DEPTH
        self.stats = FeedStats()
        self._conn = _Conn(90, lambda: _auth_url(settings.dhan_depth20_url), self.subs, handler, self.stats,
                           parser=lambda b: parse_depth(b, 20))

    async def run(self):
        await self._conn.run()

    def stop(self):
        self._conn.stop()


class CasFeed:
    """Closing Auction Session feed (Pro plan). Request code 23 on the market-feed endpoint."""

    def __init__(self, subs: list[Sub], handler: PacketHandler):
        for s in subs:
            s.mode = C.REQ_SUB_DEPTH
        self.stats = FeedStats()
        self._conn = _Conn(91, lambda: _auth_url(settings.dhan_feed_url), subs, handler, self.stats, parser=parse_cas)

    async def run(self):
        await self._conn.run()

    def stop(self):
        self._conn.stop()
