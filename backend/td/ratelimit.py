"""Rate limiting for Dhan APIs.

- TokenBucket: per-second limits (data 5/s, quote 1/s ...)
- DailyBudget: hard daily caps (data 7,000/day). Persisted in Redis when available so
  the ingestor, gateway and backfill tool share ONE budget; falls back to a JSON file.
- SlidingWindowUnique: option-chain rule "max 10 unique underlyings per 8 seconds".
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path

from .timeutil import now_ist


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
