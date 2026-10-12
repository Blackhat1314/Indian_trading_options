"""Redis message bus: one place for every key, stream and channel name.

Topology
  ingestor  --XADD td:ticks / td:news-->  analytics  --SET td:snap:* + PUBLISH td:pub:*-->  gateway(s)  --ws-->  browsers

Any number of gateway processes can subscribe, so many users share one Dhan connection.
"""
from __future__ import annotations

from typing import Any

import orjson
import redis.asyncio as aioredis

from .config import settings

STREAM_TICKS = "td:ticks"
STREAM_NEWS = "td:news"
STREAM_EVENTS = "td:events"          # CAS, depth snapshots, market-status
KEY_UNIVERSE = "td:universe"
KEY_CLOCK = "td:clock"
KEY_PREV = "td:prev:k"               # hash instrument key -> [prev close, prev OI] (prev-close packets / history)
KEY_ALERTS = "td:alerts"             # hash id -> alert json
KEY_ALERT_LOG = "td:alerts:log"      # list of fired alerts
KEY_SCORECARD = "td:scorecard"       # list of daily scorecards (json)
PUB_PREFIX = "td:pub:"
SNAP_PREFIX = "td:snap:"
# ADMIN desk <-> backfill worker. Outside td:snap:* / td:status:* on purpose: those are readable by every
# viewer (snapshot API, SYSTEM desk); these are served only through the admin endpoints.
KEY_ADMIN_CMD = "td:admin:backfill:cmd"          # list of commands (gateway LPUSH, worker BRPOP)
KEY_ADMIN_STATUS = "td:admin:backfill:status"    # worker status json (expires when the worker stops)


def k_admin_plan(req_id: str) -> str:
    return f"td:admin:backfill:plan:{req_id}"     # answer to one plan request


def k_status(service: str) -> str:
    return f"td:status:{service}"


def k_chain_raw(name: str) -> str:
    return f"td:chain:raw:{name}"


def k_depth(sid: int) -> str:
    return f"td:depth:{sid}"


def k_bars(key: int) -> str:
    return f"td:bars:{key}"          # today's 1-minute bars by unique instrument key


def k_top(day: str) -> str:
    return f"td:top:{day}"


def k_quote_last() -> str:
    return "td:quote:last"           # hash sid -> last normalized tick json


def dumps(obj: Any) -> bytes:
    return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS)


def loads(b: bytes | str | None) -> Any:
    if b is None:
        return None
    return orjson.loads(b)


def connect(url: str | None = None) -> aioredis.Redis:
    # redis-py >= 6 defaults to a 5 s socket timeout, which kills a blocking XREAD whenever the event loop is
    # busy for a few seconds (e.g. reloading a year of history) — use a generous explicit timeout instead.
    return aioredis.from_url(url or settings.redis_url, decode_responses=False, socket_timeout=60,
                             socket_connect_timeout=10, health_check_interval=30, socket_keepalive=True)


async def publish_snapshot(r: aioredis.Redis, topic: str, payload: Any, ttl: int | None = None) -> None:
    """Store the latest snapshot (for late joiners) and publish it to live subscribers."""
    data = dumps({"topic": topic, "data": payload})
    pipe = r.pipeline(transaction=False)
    if ttl:
        pipe.set(SNAP_PREFIX + topic, data, ex=ttl)
    else:
        pipe.set(SNAP_PREFIX + topic, data)
    pipe.publish(PUB_PREFIX + topic, data)
    await pipe.execute()


async def publish_event(r: aioredis.Redis, topic: str, payload: Any) -> None:
    """Fire-and-forget event (no snapshot), e.g. alert fired, top-5 locked."""
    await r.publish(PUB_PREFIX + topic, dumps({"topic": topic, "data": payload, "event": True}))
