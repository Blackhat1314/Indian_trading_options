"""Per-second Dhan limits shared across processes (td.ratelimit.SharedRate) and their use in DhanRest.

Processes are simulated by limiter / client instances with their own connections to one fakeredis server
(no Lua there, and none needed). Dhan is an httpx.MockTransport. One test runs against the real Redis at
localhost:6379 when it is reachable (own key prefix, cleaned up)."""
import asyncio
import time
import uuid
from datetime import date

import httpx
import pytest

from td.config import settings
from td.dhan.rest import DhanError, DhanRest
from td.ratelimit import SharedRate, TokenBucket

fakeredis = pytest.importorskip("fakeredis")


def run(coro):
    return asyncio.run(coro)


def max_in_window(ts: list[float], window: float = 1.0) -> int:
    ts = sorted(ts)
    return max((sum(1 for u in ts[i:] if u < t + window) for i, t in enumerate(ts)), default=0)


def processes(n: int, rate: float, **kw) -> list[SharedRate]:
    server = fakeredis.FakeServer()
    return [SharedRate(fakeredis.aioredis.FakeRedis(server=server), "data", rate, **kw) for _ in range(n)]


class Broken:
    """A Redis client whose every command fails (Redis down)."""

    def __init__(self):
        self.calls = 0

    def __getattr__(self, name):
        async def fail(*a, **k):
            self.calls += 1
            raise ConnectionError("Redis is down")
        return fail


# ------------------------------------------------------------------ the limiter
def test_slots_are_exclusive_across_processes():
    async def go():
        lims = processes(4, 200)
        got: list[float] = []

        async def caller(lim):
            for _ in range(8):
                got.append(await lim.acquire())
        await asyncio.gather(*(caller(lim) for lim in lims for _ in range(3)))
        assert None not in got and len(got) == 96
        assert len(set(got)) == 96                                   # no slot handed out twice
        s = sorted(got)
        gap = min(b - a for a, b in zip(s, s[1:]))
        assert gap >= lims[0].interval * 0.999                      # one slot length apart, across all four
        assert all(lim.stats["shared"] == 24 and lim.stats["local"] == 0 for lim in lims)
    run(go())


def test_sustained_rate_never_exceeds_rate_in_any_second():
    async def go():
        rate = 25
        lims = processes(3, rate)
        sent: list[float] = []

        async def caller(lim):
            for _ in range(10):
                await lim.acquire()
                sent.append(time.time())
        t0 = time.time()
        await asyncio.gather(*(caller(lim) for lim in lims for _ in range(2)))
        took = time.time() - t0
        assert max_in_window(sent) <= rate                          # 3 processes together, not 3 x 25
        assert took >= 59 * lims[0].interval * 0.99                 # 60 calls, one slot each
        assert took <= 60 * lims[0].interval + 1.0                  # ... and not much slower than the cap
    run(go())


def test_cooldown_is_shared_grows_on_repeats_and_resets_on_success():
    async def go():
        a, b = processes(2, 50)
        a.cooldown_base = b.cooldown_base = 0.2
        assert await a.penalize() == pytest.approx(0.2)
        assert await b.penalize() <= 0.2                             # answers during a pause are no new strike
        t0 = time.time()
        await b.acquire()                                            # another process waits the pause out
        assert time.time() - t0 >= 0.18
        assert await a.penalize() == pytest.approx(0.4)              # a repeat after it doubles the pause
        t0 = time.time()
        await b.acquire()
        assert time.time() - t0 >= 0.38
        assert b._dirty                                              # b saw the pause ...
        await b.relax()                                              # ... so its next success resets the repeats
        assert await b.redis.get("td:rl:data:strikes") is None
        assert await a.penalize() == pytest.approx(0.2)
    run(go())


def test_waits_are_bounded_and_slot_keys_expire():
    async def go():
        lim, other = processes(2, 5)
        lim.max_wait = 0.3
        first = int(time.time() / lim.interval) + 1
        for i in range(first, first + 40):                          # an 8-second queue of other callers
            await other.redis.set(f"td:rl:data:slot:{i}", "other", px=20_000)
        t0 = time.time()
        await lim.acquire()                                          # goes after max_wait, no deadlock
        assert 0.25 < time.time() - t0 < 0.5 and lim.stats["overflow"] == 1
        await other.redis.set("td:rl:data:tail", first + 39, px=20_000)     # the queue's end, as callers leave it
        t0 = time.time()
        await lim.acquire()                                          # same, found without probing the queue
        assert 0.25 < time.time() - t0 < 0.5 and lim.stats["overflow"] == 2

        lim2, = processes(1, 5)
        await lim2.redis.set("td:rl:data:tail", first + 10**6)      # a hint from a skewed clock is ignored
        t0 = time.time()
        t = await lim2.acquire()
        assert time.time() - t0 < lim2.interval + 0.1 and t < time.time() + 0.01
        keys = [k.decode() for k in await lim2.redis.keys("td:rl:*")]
        ttls = [await lim2.redis.pttl(k) for k in keys]
        assert keys and all(0 < t <= 5_000 + 1_000 * lim2.interval for t in ttls)
    run(go())


def test_falls_back_to_the_local_bucket_when_redis_fails():
    async def go():
        broken = Broken()
        bucket = TokenBucket(5)
        lim = SharedRate(broken, "data", 5, fallback=bucket)
        assert await lim.acquire() is None                          # Redis down: paced locally, no exception
        assert lim.stats == {"shared": 0, "local": 1, "cooldowns": 0, "reclaims": 0, "overflow": 0}
        assert bucket.tokens < 5
        n = broken.calls
        for _ in range(3):
            await lim.acquire()
        assert broken.calls == n                                    # not retried on every call while down
        lim.cooldown_base = 0.2
        assert await lim.penalize() == pytest.approx(0.2)           # local pause
        t0 = time.time()
        await lim.acquire()
        assert time.time() - t0 >= 0.18
        lim.redis, lim._down_until = fakeredis.aioredis.FakeRedis(), 0.0       # Redis is back
        assert await lim.acquire() is not None and lim.stats["shared"] == 1
        none = SharedRate(None, "data", 5)                          # no Redis at all
        assert await none.acquire() is None and none.stats["local"] == 1
    run(go())


# ------------------------------------------------------------------ DhanRest
@pytest.fixture
def creds(monkeypatch):
    monkeypatch.setattr(settings, "dhan_client_id", "1")
    monkeypatch.setattr(settings, "dhan_access_token", "t")
    monkeypatch.setattr(settings, "dhan_base_url", "https://dhan.test/v2")


def _client(redis, handler) -> DhanRest:
    rest = DhanRest(redis=redis)
    rest._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return rest


def _bars():
    return httpx.Response(200, json={"timestamp": [1], "open": [1], "high": [1], "low": [1], "close": [1],
                                     "volume": [1]})


D0, D1 = date(2026, 10, 1), date(2026, 10, 10)


def test_dhan_rest_paces_through_the_shared_limiter(creds, monkeypatch):
    monkeypatch.setattr(settings, "limit_data_per_sec", 20)

    async def go():
        server = fakeredis.FakeServer()
        seen: list[float] = []

        def handler(req):
            seen.append(time.time())
            return _bars()
        a, b = (_client(fakeredis.aioredis.FakeRedis(server=server), handler) for _ in range(2))
        assert isinstance(a.limits["data"], SharedRate) and a.limits["data"].redis is not None
        assert a.limits["data"].fallback is a.data_bucket and a.limits["quote"].fallback is a.quote_bucket
        await asyncio.gather(*(c.daily(13, "IDX_I", "INDEX", D0, D1) for c in (a, b) for _ in range(10)))
        assert a.limits["data"].stats["shared"] == 10 and b.limits["data"].stats["shared"] == 10
        assert max_in_window(seen) <= 20
        assert max(seen) - min(seen) >= 19 * a.limits["data"].interval * 0.99     # one queue for both clients
        assert await a.limits["data"].redis.get("td:rl:data:tail") is not None
        local = _client(None, handler)                              # no Redis: the in-process bucket
        await local.daily(13, "IDX_I", "INDEX", D0, D1)
        assert local.limits["data"].stats == {"shared": 0, "local": 1, "cooldowns": 0, "reclaims": 0, "overflow": 0}
        for c in (a, b, local):
            await c.close()
    run(go())


def test_rate_limit_answers_pause_every_process(creds):
    async def go():
        server = fakeredis.FakeServer()
        seen: list[tuple[str, float, int]] = []
        answers = {"a": [429]}

        def handler_for(name):
            def handler(req):
                q = answers.get(name)
                code = q.pop(0) if q else 200
                seen.append((name, time.time(), code))
                if code == 429:
                    return httpx.Response(429, json={"errorCode": "DH-904"})
                if code != 200:
                    return httpx.Response(200, json={"status": "failure", "errorCode": code,
                                                     "errorMessage": "Too many requests"})
                return _bars()
            return handler
        a = _client(fakeredis.aioredis.FakeRedis(server=server), handler_for("a"))
        b = _client(fakeredis.aioredis.FakeRedis(server=server), handler_for("b"))
        a.limits["data"].cooldown_base = 0.4
        penalized = asyncio.Event()
        real = a.limits["data"].penalize

        async def penalize():
            d = await real()
            penalized.set()
            return d
        a.limits["data"].penalize = penalize

        async def other_process():
            await penalized.wait()
            return await b.daily(13, "IDX_I", "INDEX", D0, D1)
        res = await asyncio.gather(a.daily(13, "IDX_I", "INDEX", D0, D1), other_process())
        assert all(r.get("timestamp") == [1] for r in res)          # both went through after the pause
        t429 = next(t for n, t, c in seen if c == 429)
        after = [t for n, t, c in seen if c == 200]
        assert len(after) == 2 and min(after) >= t429 + 0.38        # A's retry AND B waited the shared pause
        assert a.calls["errors"] == 1
        assert await b.limits["data"].redis.get("td:rl:data:strikes") is None     # reset by the success

        a.limits["data"].cooldown_base = 0.2
        for code in ("DH-904", 805, "RL001"):                       # Dhan's rate-limit codes in a 200 body
            answers["a"] = [code]
            before = a.limits["data"].stats["cooldowns"]
            assert (await a.daily(13, "IDX_I", "INDEX", D0, D1)).get("timestamp") == [1]
            assert a.limits["data"].stats["cooldowns"] == before + 1

        answers["a"] = [429]                                        # backfill style (retries=1): fail fast ...
        with pytest.raises(DhanError, match="retries exhausted") as e:
            await a.daily(13, "IDX_I", "INDEX", D0, D1, retries=1)
        assert e.value.status == 429
        assert await b.limits["data"].redis.get("td:rl:data:cool") is not None   # ... but everyone still pauses
        for c in (a, b):
            await c.close()
    run(go())


# ------------------------------------------------------------------ real Redis (skipped when unreachable)
def test_real_redis_processes_share_one_queue():
    import redis.asyncio as aioredis

    async def go():
        url = "redis://localhost:6379/0"
        probe = aioredis.from_url(url, socket_connect_timeout=1, socket_timeout=2)
        try:
            await probe.ping()
        except Exception:
            await probe.aclose()
            pytest.skip("no Redis at localhost:6379")
        prefix = f"td:rl:test:{uuid.uuid4().hex}:"
        clients = [aioredis.from_url(url, socket_timeout=5) for _ in range(3)]
        try:
            lims = [SharedRate(c, "data", 50, prefix=prefix) for c in clients]
            got: list[float] = []

            async def caller(lim):
                for _ in range(8):
                    got.append(await lim.acquire())
            await asyncio.gather(*(caller(lim) for lim in lims for _ in range(2)))
            assert None not in got and len(set(got)) == 48
            s = sorted(got)
            assert min(b - a for a, b in zip(s, s[1:])) >= lims[0].interval * 0.999
            lims[0].cooldown_base = 0.2
            await lims[0].penalize()
            t0 = time.time()
            await lims[2].acquire()
            assert time.time() - t0 >= 0.18
            keys = [k async for k in probe.scan_iter(match=prefix + "*")]
            ttls = [await probe.pttl(k) for k in keys]
            assert keys and all(t > 0 for t in ttls)                  # everything expires by itself
        finally:
            keys = [k async for k in probe.scan_iter(match=prefix + "*")]
            if keys:
                await probe.delete(*keys)
            for c in clients + [probe]:
                await c.aclose()
    run(go())
