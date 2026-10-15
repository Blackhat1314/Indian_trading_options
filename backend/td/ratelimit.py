"""Rate limiting for Dhan APIs.

- TokenBucket: per-second limits (data 5/s, quote 1/s ...) inside one process.
- SharedRate: the same per-second limits across processes. Dhan counts per ACCOUNT, so the ingestor, the
  gateway, the backfill worker and the backfill CLI take turns through Redis; TokenBucket is its fallback.
- DailyBudget: hard daily caps (data 7,000/day). Persisted in Redis when available so
  the ingestor, gateway and backfill tool share ONE budget; falls back to a JSON file.
- SlidingWindowUnique: option-chain rule "max 10 unique underlyings per 8 seconds".
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import socket
import time
from collections import deque
from pathlib import Path

from .timeutil import now_ist

log = logging.getLogger("td.ratelimit")


class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float | None = None):
        self.rate = rate_per_sec
        self.capacity = capacity if capacity is not None else max(1.0, rate_per_sec)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                await asyncio.sleep((tokens - self.tokens) / self.rate)


def _num(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


async def _sleep_until(t: float) -> None:
    d = t - time.time()
    if d > 0:
        await asyncio.sleep(d)
        d = t - time.time()
        if 0 < d < 0.05:                     # the loop's timer fired a hair before the wall clock got there
            await asyncio.sleep(d)


class SharedRate:
    """`rate` requests per second across every process that shares one Redis.

    Reservation-based: time is cut into slots of (1 + margin) / rate seconds. A caller claims the first free
    slot from now on (SET NX PX on td:rl:<name>:slot:<index>), then sleeps until it starts: no busy waiting,
    and callers go in the order they claimed, so the live app queues behind at most the slots already taken
    and is never starved by the backfill (which keeps its own lower pace on top: backfill.Pacer). Any `rate`
    consecutive slots span 1 + margin seconds, so up to `margin` of uneven delay (network, a busy event loop)
    still keeps every 1-second window at Dhan within `rate`; a caller that wakes later than margin / 2 after its
    slot claims a new one instead of bunching up with the next. Slot keys expire seconds after their slot.

    Waits are bounded: nobody waits more than max_wait (30 s). If more than that is already queued (an
    overload: in TradeDesk at most ~20 calls are in flight, ~4 s of queue) the overflow goes when its max_wait
    is up, in claim order; Dhan may then answer 429 and the shared cooldown below paces everyone.

    penalize() (Dhan answered 429 / DH-904 / 805 / RL001) stores a cooldown in Redis that pauses every
    process — 1 s, doubling on repeats up to 8 s, back to 1 s after the next success (relax()) — instead of
    each process retrying on its own. Waiters already holding a slot re-queue after the pause.

    No Lua (fakeredis has none): SET NX PX, MGET, INCR and DEL only. The td:rl:<name>:tail hint (the last slot
    claimed) only saves probes; correctness rests on SET NX alone.

    Clock: slots are wall-clock (time.time()) instants, so all processes must share a clock — true on one
    host, which is how TradeDesk runs (ingestor, gateway, backfill worker and CLI next to one Redis). Across
    hosts the clocks need NTP, and any skew eats into `margin`. All processes must use the same rate (they
    read LIMIT_*_PER_SEC from the same .env), since a slot index means a time only for one slot length.

    No Redis, or Redis erroring: the in-process `fallback` bucket paces this process alone (the old
    behaviour); Redis is tried again `retry_down` seconds later.
    """

    max_wait = 30.0          # longest wait for a slot; past it the caller goes anyway, in claim order
    cooldown_base = 1.0      # first pause after a rate-limit answer (s), doubled per repeat ...
    cooldown_max = 8.0       # ... up to this
    strikes_ttl = 60.0       # repeats count while they are this close together
    retry_down = 5.0         # after a Redis error, pace locally this long before trying Redis again
    redis_timeout = 2.0      # a hung Redis must not hold every request for the client's socket timeout

    def __init__(self, redis, name: str, rate: float, *, margin: float = 0.1, prefix: str = "td:rl:",
                 fallback: TokenBucket | None = None):
        self.redis = redis
        self.name = name
        self.rate = max(0.05, float(rate))
        self.margin = margin
        self.interval = (1.0 + margin) / self.rate
        self.key = f"{prefix}{name}"
        self.fallback = fallback or TokenBucket(self.rate)
        self.owner = f"{socket.gethostname()}:{os.getpid()}"
        self.stats = {"shared": 0, "local": 0, "cooldowns": 0, "reclaims": 0, "overflow": 0}
        self._lock = asyncio.Lock()
        self._tail = 0                       # last slot this instance claimed
        self._overflow_logged = 0.0
        self._down_until = 0.0
        self._shared_ok = True
        self._dirty = False                  # saw a cooldown: the next success resets the shared strikes
        self._local_cool = 0.0               # pause when pacing locally
        self._local_strikes = 0

    # ------------------------------------------------------------------ acquire
    async def acquire(self) -> float | None:
        """Wait for this caller's turn. Returns the reserved slot time, or None when paced locally."""
        if self.redis is not None and time.time() >= self._down_until:
            try:
                t = await self._acquire_shared()
                self.stats["shared"] += 1
                if not self._shared_ok:
                    self._shared_ok = True
                    log.info("rate limiter %s: Redis is back — pacing shared across processes again", self.name)
                return t
            except Exception as e:           # never let pacing take requests down
                self._redis_failed(e)
        await self._acquire_local()
        return None

    async def _acquire_shared(self) -> float:
        deadline = time.time() + self.max_wait
        t = 0.0
        for attempt in range(4):             # the claim, then re-claims after a cooldown or a late wake-up
            t = await self._claim()
            await _sleep_until(min(t, deadline))
            now = time.time()
            if now >= deadline or attempt == 3:
                return t
            try:
                async with asyncio.timeout(self.redis_timeout):
                    cool = _num(await self.redis.get(self.key + ":cool"))
            except Exception:                # the slot was ours: go
                return t
            if cool > now:                   # a 429 since the claim: everyone pauses, then queues again
                self._dirty = True
            elif now - t > self.margin / 2:  # woke late (busy loop): going now could bunch up with later slots
                self.stats["reclaims"] += 1
            else:
                return t
        return t

    async def _claim(self) -> float:
        """Reserve the first free slot at or after max(now, cooldown end); returns its start time. When every
        slot up to max_wait ahead is taken, returns a later time without a slot: the caller goes when its
        max_wait is up (bounded wait), still in claim order."""
        async with self._lock:               # one claimer per process at a time: fewer collisions, FIFO here
            async with asyncio.timeout(self.redis_timeout):
                now = time.time()
                tail, cool = await self.redis.mget([self.key + ":tail", self.key + ":cool"])
                first = math.ceil(now / self.interval)
                cool = min(_num(cool), now + self.cooldown_max)     # a value from a skewed clock waits no longer
                if cool > now:
                    self._dirty = True
                    first = max(first, math.ceil(cool / self.interval))
                last = first + math.ceil(self.max_wait / self.interval)
                hint = max(self._tail, int(_num(tail)))
                start = max(first, hint + 1)
                if start > last:
                    if await self.redis.exists(f"{self.key}:slot:{hint}"):
                        return self._overflow(start)                 # a real queue longer than max_wait
                    start = first                                    # a stale hint (skewed clock): probe from now
                for i in range(start, last + 1):
                    t = i * self.interval
                    ttl = int((t - now + 5.0) * 1000)        # gone a few seconds after its slot: no growth
                    if await self.redis.set(f"{self.key}:slot:{i}", self.owner, nx=True, px=ttl):
                        self._tail = i
                        await self.redis.set(self.key + ":tail", i, px=ttl)
                        return t
                return self._overflow(last + 1)

    def _overflow(self, slot: int) -> float:
        self.stats["overflow"] += 1
        now = time.time()
        if now - self._overflow_logged > 10:
            self._overflow_logged = now
            log.warning("rate limiter %s: more than %.0f s of Dhan requests queued — the overflow goes after "
                        "%.0f s each (Dhan may answer 429; the shared cooldown then paces everyone)", self.name,
                        self.max_wait, self.max_wait)
        return slot * self.interval

    async def _acquire_local(self) -> None:
        d = self._local_cool - time.time()
        if d > 0:
            await asyncio.sleep(min(d, self.max_wait))
        await self.fallback.acquire()
        self.stats["local"] += 1

    def _redis_failed(self, e: Exception) -> None:
        if self._shared_ok:
            log.warning("rate limiter %s: Redis unavailable (%s: %s) — pacing this process alone", self.name,
                        type(e).__name__, e)
        self._shared_ok = False
        self._down_until = time.time() + self.retry_down

    # ------------------------------------------------------------------ cooldown
    def _pause_for(self, strikes: int) -> float:
        return min(self.cooldown_max, self.cooldown_base * 2 ** max(0, strikes - 1))

    async def penalize(self) -> float:
        """Dhan said slow down: pause every process sharing this limiter. Returns the pause in seconds."""
        self._dirty = True
        if self.redis is not None and time.time() >= self._down_until:
            try:
                async with asyncio.timeout(self.redis_timeout):
                    return await self._penalize_shared()
            except Exception as e:
                self._redis_failed(e)
        now = time.time()
        if self._local_cool > now:
            return self._local_cool - now
        self._local_strikes += 1
        d = self._pause_for(self._local_strikes)
        self._local_cool = now + d
        self.stats["cooldowns"] += 1
        return d

    async def _penalize_shared(self) -> float:
        now = time.time()
        cool = _num(await self.redis.get(self.key + ":cool"))
        if cool > now:                       # already pausing: answers to requests sent before it are no new strike
            return cool - now
        strikes = int(await self.redis.incr(self.key + ":strikes"))
        await self.redis.pexpire(self.key + ":strikes", int(self.strikes_ttl * 1000))
        d = self._pause_for(strikes)
        await self.redis.set(self.key + ":cool", repr(now + d), px=int(d * 1000) + 1000)
        self.stats["cooldowns"] += 1
        log.warning("Dhan rate limit hit (%s) — all processes pause %.1f s", self.name, d)
        return d

    async def relax(self) -> None:
        """A request went through: the next rate-limit answer starts again from the shortest pause."""
        self._local_strikes = 0
        if not self._dirty or self.redis is None:
            return
        self._dirty = False
        try:
            async with asyncio.timeout(self.redis_timeout):
                await self.redis.delete(self.key + ":strikes")
        except Exception:
            pass


class BudgetExhausted(RuntimeError):
    pass


class DailyBudget:
    """Daily request counter keyed by IST date."""

    def __init__(self, name: str, limit: int, redis=None, file: Path | None = None):
        self.name = name
        self.limit = limit
        self.redis = redis
        self.file = file
        self._local: dict[str, int] = {}
        if file and file.exists():
            try:
                self._local = json.loads(file.read_text())
            except Exception:
                self._local = {}

    def _key(self) -> str:
        return f"budget:{self.name}:{now_ist().date().isoformat()}"

    async def used(self) -> int:
        if self.redis is not None:
            try:
                v = await self.redis.get(self._key())
                return int(v or 0)
            except Exception:
                pass
        return int(self._local.get(self._key(), 0))

    async def remaining(self) -> int:
        return max(0, self.limit - await self.used())

    async def consume(self, n: int = 1) -> int:
        key = self._key()
        if self.redis is not None:
            try:
                v = await self.redis.incrby(key, n)
                await self.redis.expire(key, 3 * 86400)
                if int(v) > self.limit:
                    raise BudgetExhausted(f"{self.name}: daily limit {self.limit} reached")
                return int(v)
            except BudgetExhausted:
                raise
            except Exception:
                pass
        v = self._local.get(key, 0) + n
        self._local[key] = v
        if self.file:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            self.file.write_text(json.dumps(self._local))
        if v > self.limit:
            raise BudgetExhausted(f"{self.name}: daily limit {self.limit} reached")
        return v


class SlidingWindowUnique:
    """At most `max_unique` distinct keys inside any `window` seconds, and the same key
    no more than once per `per_key_gap` seconds (Dhan option-chain rule)."""

    def __init__(self, max_unique: int = 10, window: float = 8.0, per_key_gap: float = 3.0):
        self.max_unique = max_unique
        self.window = window
        self.per_key_gap = per_key_gap
        self.events: deque[tuple[float, str]] = deque()
        self.last_by_key: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                while self.events and now - self.events[0][0] > self.window:
                    self.events.popleft()
                uniq = {k for _, k in self.events}
                gap_ok = now - self.last_by_key.get(key, -1e9) >= self.per_key_gap
                uniq_ok = key in uniq or len(uniq) < self.max_unique
                if gap_ok and uniq_ok:
                    self.events.append((now, key))
                    self.last_by_key[key] = now
                    return
                waits = []
                if not gap_ok:
                    waits.append(self.per_key_gap - (now - self.last_by_key[key]))
                if not uniq_ok and self.events:
                    waits.append(self.window - (now - self.events[0][0]))
                await asyncio.sleep(max(0.05, min(waits) if waits else 0.1))
