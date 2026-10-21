"""Dhan token keeper: JWT claims, TOTP, the token store, renewal / TOTP rules, followers, feeds and admin API.

No network: Dhan is an httpx.MockTransport, Redis is fakeredis, time is a fake clock.
"""
import asyncio
import base64
import errno
import importlib.util
import json
import logging
import os
import stat
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from td.config import settings
from td.dhan import feeds
from td.dhan import token as tk
from td.dhan.rest import DhanRest
from td.dhan.token import TokenKeeper, TokenStore, best_token, claims, expires_at, record, totp
from td.timeutil import IST

fakeredis = pytest.importorskip("fakeredis")
CID = "1000000001"
SECRET = base64.b32encode(b"12345678901234567890").decode()          # RFC 6238 test key
H = {"X-TD-Request": "1"}


def T(h: int, m: int = 0, d: int = 0) -> float:
    """Epoch of 2026-10-15 h:m IST (+ d days)."""
    return datetime(2026, 10, 15, h, m, tzinfo=IST).timestamp() + d * 86400


_n = [0]


def jwt(exp: float, iat: float | None = None, cid: str = CID) -> str:
    def b64(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    _n[0] += 1
    body = {"iss": "dhan", "exp": int(exp), "iat": int(iat if iat is not None else exp - 86400), "dhanClientId": cid,
            "n": _n[0]}
    return f"{b64({'typ': 'JWT', 'alg': 'HS512'})}.{b64(body)}.sig{_n[0]}"


class FakeDhan:
    """RenewToken / generateAccessToken / profile with Dhan's shapes; renewing retires the old token."""

    def __init__(self, clock: list):
        self.clock = clock
        self.valid: set[str] = set()
        self.calls: list[str] = []
        self.issued: list[str] = []
        self.renew_status = 200
        self.renew_body: dict | None = None    # what a refused RenewToken answers (default: DH-901)
        self.gen_fail = 0              # how many generate calls fail before one succeeds
        self.gen_status = 400
        self.reject_new = False        # newly issued tokens fail /profile
        self.delay = 0.0
        self.gen_params: list[dict] = []
        self.glitches: list[httpx.Response] = []   # answers for the next /profile calls, whatever the token

    def issue(self) -> str:
        t = jwt(self.clock[0] + 86400, self.clock[0])
        self.issued.append(t)
        if not self.reject_new:
            self.valid.add(t)
        return t

    async def __call__(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        self.calls.append(path.rsplit("/", 1)[-1])
        tok = req.headers.get("access-token")
        bad = {"errorType": "Invalid_Authentication", "errorCode": "DH-901",
               "errorMessage": "Client ID or user generated access token is invalid or expired."}
        if path.endswith("/profile"):
            if self.glitches:
                return self.glitches.pop(0)
            if tok in self.valid:
                return httpx.Response(200, json={"dhanClientId": CID, "tokenValidity": "16/10/2026 08:30",
                                                 "dataPlan": "Active", "dataValidity": "2026-12-01 00:00:00.0"})
            return httpx.Response(401, json=bad)
        if path.endswith("/RenewToken"):
            await asyncio.sleep(self.delay)
            assert req.headers.get("dhanClientId") == CID
            if tok not in self.valid or self.renew_status != 200:
                return httpx.Response(self.renew_status if self.renew_status != 200 else 401,
                                      json=self.renew_body or bad)
            self.valid.discard(tok)
            new = self.issue()
            return httpx.Response(200, json={"dhanClientId": CID, "dhanClientName": "X", "accessToken": new,
                                             "expiryTime": "2026-10-16T08:30:00"})
        if path.endswith("/app/generateAccessToken"):
            q = dict(req.url.params)
            self.gen_params.append(q)
            if self.gen_fail > 0:
                self.gen_fail -= 1
                return httpx.Response(self.gen_status, json={"status": "error", "message": "Invalid TOTP"})
            assert q["dhanClientId"] == CID and q["pin"] == "4321" and q["totp"] == totp(SECRET, self.clock[0])
            return httpx.Response(200, json={"dhanClientId": CID, "accessToken": self.issue(),
                                             "expiryTime": "2026-10-16T08:30:00"})
        return httpx.Response(404)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "dhan_client_id", CID)
    monkeypatch.setattr(settings, "dhan_access_token", "")
    monkeypatch.setattr(settings, "dhan_pin", "")
    monkeypatch.setattr(settings, "dhan_totp_secret", "")
    monkeypatch.setattr(settings, "dhan_token_auto_renew", True)
    monkeypatch.setattr(settings, "dhan_token_renew_at", "08:30")
    monkeypatch.setattr(settings, "dhan_base_url", "https://api.dhan.test/v2")
    monkeypatch.setattr(settings, "dhan_auth_url", "https://auth.dhan.test")
    monkeypatch.setattr(tk, "dotenv_token", lambda: "")
    tk._warned.clear()


def setup(clock: list, env: str = "", role: str = "owner"):
    r = fakeredis.aioredis.FakeRedis()
    dhan = FakeDhan(clock)
    settings.dhan_access_token = env
    k = keeper(r, dhan, clock, role)
    return r, dhan, k


def keeper(r, dhan, clock, role="owner") -> TokenKeeper:
    k = TokenKeeper(r, role=role, http=httpx.AsyncClient(transport=httpx.MockTransport(dhan)))
    k.clock = lambda: clock[0]

    async def sleep(s):
        clock[0] += s
    k.sleep = sleep
    return k


# ------------------------------------------------------------------ helpers
def test_claims_and_expiry():
    t = jwt(T(8, 30, 1), T(8, 30))
    c = claims(t)
    assert c["dhanClientId"] == CID and c["exp"] == int(T(8, 30, 1)) and expires_at(t) == int(T(8, 30, 1))
    for bad in (None, "", "abc", "a.b", "a.!!!.c", "a." + base64.urlsafe_b64encode(b"[1]").decode() + ".c"):
        assert claims(bad) is None and expires_at(bad) is None
    rec = record(t, "renewed", exp=T(9, 0, 1), now=5.0)
    assert rec["exp"] == T(9, 0, 1) and rec["iat"] == int(T(8, 30)) and rec["saved_at"] == 5.0   # later hint wins
    assert tk._when("16/10/2026 08:30") == T(8, 30, 1) and tk._when("2026-10-16T08:30:00") == T(8, 30, 1)
    assert tk._when("2026-10-16T03:00:00Z") == T(8, 30, 1) and tk._when(T(1) * 1000) == T(1)


def test_totp_rfc6238_vectors():
    assert totp(SECRET, 59, digits=8) == "94287082" and totp(SECRET, 59) == "287082"
    for t, code in ((1111111109, "07081804"), (1234567890, "89005924"), (2000000000, "69279037")):
        assert totp(SECRET, t, digits=8) == code
    messy = " ".join(SECRET.lower().rstrip("=")[i:i + 4] for i in range(0, 32, 4))     # spaced, lowercase, unpadded
    assert totp(messy, 59) == "287082"
    with pytest.raises(ValueError):
        totp("not base32 !!", 59)


def test_store_roundtrip_file_mode_and_restore(tmp_path):
    async def go():
        r = fakeredis.aioredis.FakeRedis()
        st = TokenStore(r)
        assert await st.load() is None
        t = jwt(T(8, 30, 1))
        await st.save(record(t, "renewed", now=T(8, 30)))
        p = tmp_path / "auth" / "dhan_token.json"
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600 and stat.S_IMODE(os.stat(p.parent).st_mode) == 0o700
        assert (await st.load())["token"] == t and int(await r.get(tk.KEY_VER)) == 1
        await r.flushall()                                    # Docker's Redis restarted: the file brings it back
        rec = await TokenStore(r).load()
        assert rec["token"] == t and rec["source"] == "renewed" and await r.get(tk.KEY_TOKEN)
        assert await r.ttl(tk.KEY_TOKEN) == -1                # no expiry on the key
        newer = jwt(T(9, 0, 1))                               # a later saved file copy (Redis write lost) wins
        st._write(record(newer, "totp", now=T(9)))
        assert (await st.load())["token"] == newer
    asyncio.run(go())


def test_failed_file_write_never_brings_back_the_older_token(monkeypatch):
    async def go():
        clock = [T(10)]
        r, dhan, k = setup(clock, role="follower")                  # the gateway, where the admin pastes
        await k.store.save(record(jwt(T(8, 30, 1)), "renewed", now=T(8, 30)))   # since revoked by Dhan
        await k._pick()
        pasted = jwt(T(8, 0, 1), T(8, 0))                          # made on Dhan Web earlier: expires earlier

        def full(rec):
            raise OSError(errno.ENOSPC, "No space left on device")
        monkeypatch.setattr(k.store, "_write", full)
        dhan.valid.add(pasted)
        v = await k.adopt(pasted)
        assert settings.dhan_access_token == pasted and v["source"] == "admin"
        assert "not on disk" in v["last_error"] and "No space left" in v["last_error"]
        assert (await TokenStore(r).load())["token"] == pasted     # the stale file copy does not win
        clock[0] = T(10, 10)
        await k.sync()
        assert settings.dhan_access_token == pasted
    asyncio.run(go())


def test_concurrent_file_writes_never_corrupt_the_file(monkeypatch):
    """Every container's process is PID 1 on the shared data mount: two writers must not share a temp file."""
    st = TokenStore(None)
    recs = [record(jwt(T(9, i, 1)), "renewed", now=T(9, i)) for i in range(2)]
    errors: list[Exception] = []

    def write(gate, rec):
        gate.wait()
        try:
            st._write(rec)
        except OSError as e:
            errors.append(e)
    for _ in range(40):
        gate = threading.Barrier(2)
        threads = [threading.Thread(target=write, args=(gate, rec)) for rec in recs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert json.loads(st.path.read_text()) in recs
    assert not errors and os.listdir(st.path.parent) == [st.path.name]
    assert stat.S_IMODE(os.stat(st.path).st_mode) == 0o600

    def boom(*a):
        raise OSError(errno.EIO, "I/O error")
    monkeypatch.setattr(tk.os, "replace", boom)
    with pytest.raises(OSError):
        st._write(recs[0])
    assert os.listdir(st.path.parent) == [st.path.name]          # no temp file left behind


def test_best_token_choice():
    now = T(10)
    stale_env = record(jwt(T(9)), "env", now=0)                      # exported variable from yesterday
    stored = record(jwt(T(8, 30, 1)), "renewed", now=T(8, 30))
    assert best_token([stored, stale_env], now=now) is stored
    assert best_token([None, stale_env], now=now) is stale_env       # better an expired token than none
    fresh_env = record(jwt(T(9, 59, 1)), "env", now=0)                # pasted into .env after the stored one
    assert best_token([stored, fresh_env], now=now) is fresh_env
    foreign = record(jwt(T(12, 0, 1), cid="999"), "env", now=0)
    assert best_token([foreign, stored], now=now) is stored and best_token([foreign], now=now) is None
    opaque = record("not-a-jwt", "env", now=0)                        # unknown expiry: below valid, above expired
    assert best_token([opaque, stored], now=now) is stored and best_token([stale_env, opaque], now=now) is opaque
    same = record(stored["token"], "env", now=0)
    assert best_token([stored, same], now=now) is stored             # ties keep the stored record (its source)


# -------------------------------------------------------------------- start
def test_start_prefers_fresher_stored_token_and_owner_stores_env():
    async def go():
        clock = [T(10)]
        r, dhan, k = setup(clock, env=jwt(T(9)))
        good = jwt(T(8, 30, 1))
        await TokenStore(r).save(record(good, "renewed", now=T(8, 30)))
        await k._pick()
        assert settings.dhan_access_token == good and k.current["source"] == "renewed"
        # a fresh token in .env (exported variable stale) beats the stored one, and the owner stores it
        dot = jwt(T(9, 59, 1))
        tk.dotenv_token = lambda: dot
        settings.dhan_access_token = jwt(T(9))
        k2 = keeper(r, dhan, clock)
        await k2._pick()
        assert settings.dhan_access_token == dot
        assert (await TokenStore(r).load())["token"] == dot
    asyncio.run(go())


# ---------------------------------------------------------- owner renewals
def test_owner_renews_near_expiry():
    async def go():
        clock = [T(14)]
        old = jwt(T(15), T(15, 0, -1))                            # 1 h left
        r, dhan, k = setup(clock, env=old)
        dhan.valid.add(old)
        await k._pick()
        await k.step(verify=True)
        new = dhan.issued[-1]
        assert dhan.calls.count("RenewToken") == 1 and settings.dhan_access_token == new
        assert (await TokenStore(r).load())["source"] == "renewed"
        st = json.loads(await r.get(tk.KEY_STATUS))
        assert st["source"] == "renewed" and st["last_renew"] == T(14) and st["valid"] and st["profile"]["dataPlan"]
        assert old.encode() not in await r.get(tk.KEY_STATUS) and new.encode() not in await r.get(tk.KEY_STATUS)
        await k.step(verify=False)
        assert dhan.calls.count("RenewToken") == 1                # 24 h left now: nothing to do
    asyncio.run(go())


def test_owner_daily_window_once_a_day_outside_market_hours():
    async def go():
        clock = [T(8)]
        tok = jwt(T(19), T(19, 0, -1))                            # made yesterday 19:00: would lapse this evening
        r, dhan, k = setup(clock, env=tok)
        dhan.valid.add(tok)
        await k._pick()
        await k.step()
        assert "RenewToken" not in dhan.calls
        assert k.view["next_action"] == "renew" and k.view["next_action_at"] == T(8, 30)
        clock[0] = T(8, 31)
        await k.step()
        assert dhan.calls.count("RenewToken") == 1
        clock[0] = T(8, 40)                                       # still inside the window: not twice
        await k.step()
        assert dhan.calls.count("RenewToken") == 1
        assert k.view["next_action_at"] == T(6, 31, 1)            # 2 h before tomorrow's 08:31 expiry
        clock[0] = T(6, 31, 1)
        await k.step()
        assert dhan.calls.count("RenewToken") == 2
        clock[0] = T(8, 35, 1)                                    # morning window, 22 h left: no second renewal
        await k.step()
        assert dhan.calls.count("RenewToken") == 2
    asyncio.run(go())


def test_window_pulls_an_evening_expiry_back_to_the_morning():
    k = TokenKeeper(None)
    rec = record(jwt(T(20, 30)), "renewed")
    assert k._renew_due(rec, T(3)) == T(8, 30)                    # not 18:30, in market hours' wake
    rec = record(jwt(T(22, 30)), "renewed")
    assert k._renew_due(rec, T(3)) == T(20, 30)                   # > 12 h left at the window: renew at night
    settings.dhan_token_renew_at = "07:15"
    assert k._renew_due(record(jwt(T(12)), "renewed"), T(3)) == T(7, 15)


def test_expired_token_generates_with_totp():
    async def go():
        clock = [T(11)]
        settings.dhan_pin, settings.dhan_totp_secret = "4321", SECRET.lower()
        r, dhan, k = setup(clock, env=jwt(T(8)))                   # lapsed while the server was off
        await k._pick()
        await k.step()
        assert "RenewToken" not in dhan.calls and dhan.calls.count("generateAccessToken") == 1
        assert settings.dhan_access_token == dhan.issued[-1] and k.current["source"] == "totp"
        assert k.view["state"] == "valid" and k.view["last_renew"] == T(11)
    asyncio.run(go())


def test_token_dhan_will_not_renew_is_replaced_by_a_totp_login():
    async def go():
        clock = [T(14)]
        old = jwt(T(15), T(15, 0, -1))
        r, dhan, k = setup(clock, env=old)
        dhan.valid.add(old)
        dhan.renew_status = 400                                   # e.g. a token from an earlier TOTP login
        await k._pick()
        await k.step()                                            # no PIN / TOTP: kept, retried later
        assert dhan.calls.count("RenewToken") == 1 and "generateAccessToken" not in dhan.calls
        assert settings.dhan_access_token == old and "Renewal failed" in k.view["last_error"]
        settings.dhan_pin, settings.dhan_totp_secret = "4321", SECRET
        clock[0] += tk.RETRY_AFTER
        await k.step()
        assert dhan.calls.count("RenewToken") == 2 and dhan.calls.count("generateAccessToken") == 1
        assert settings.dhan_access_token == dhan.issued[-1] and k.current["source"] == "totp"
        assert k.view["state"] == "valid" and not k.view["last_error"]
    asyncio.run(go())


TOTP_TOKEN_REFUSED = {"errorCode": "INVALID_REQUEST", "errorMessage": "Renewal of token not allowed for this token type"}


@pytest.mark.parametrize("body", [{"errorType": "Input_Exception", "errorCode": "DH-905",     # as Dhan sends it
                                   "errorMessage": json.dumps(TOTP_TOKEN_REFUSED, separators=(",", ":"))},
                                  TOTP_TOKEN_REFUSED])
def test_dh905_from_a_totp_token_logs_in_at_once(body):
    """Dhan answers RenewToken for a token from a TOTP login with HTTP 500 + DH-905. That is a final no, not a
    hiccup: the keeper logs in with the PIN + TOTP in the same step instead of retrying until the token lapses."""
    async def go():
        clock = [T(8, 31)]
        old = jwt(T(17), T(17, 0, -1))                           # made by a TOTP login yesterday 17:00
        r, dhan, k = setup(clock, env=old)
        dhan.valid.add(old)
        dhan.renew_status, dhan.renew_body = 500, body
        settings.dhan_pin, settings.dhan_totp_secret = "4321", SECRET
        await k._pick()
        await k.step()                                            # 08:31: inside the morning window
        assert dhan.calls.count("RenewToken") == 1 and dhan.calls.count("generateAccessToken") == 1
        assert settings.dhan_access_token == dhan.issued[-1] and k.current["source"] == "totp"
        assert not k.view["last_error"] and k.view["state"] == "valid"
    asyncio.run(go())


def test_dh905_without_totp_says_what_to_do():
    async def go():
        clock = [T(8, 31)]
        old = jwt(T(17), T(17, 0, -1))
        r, dhan, k = setup(clock, env=old)
        dhan.valid.add(old)
        dhan.renew_status, dhan.renew_body = 500, TOTP_TOKEN_REFUSED
        await k._pick()
        await k.step()
        assert "generateAccessToken" not in dhan.calls and settings.dhan_access_token == old
        assert "DHAN_TOTP_SECRET" in k.view["last_error"]
    asyncio.run(go())


def test_totp_retries_once_with_the_next_code_then_pauses(tmp_path):
    async def go():
        clock = [T(11, 0) + 10]
        settings.dhan_pin, settings.dhan_totp_secret = "4321", SECRET
        r, dhan, k = setup(clock)                                  # no token at all
        dhan.gen_fail = 1
        await k.step()
        assert len(dhan.gen_params) == 2 and dhan.gen_params[0]["totp"] != dhan.gen_params[1]["totp"]
        assert clock[0] >= T(11, 0) + 30 and settings.dhan_access_token == dhan.issued[-1]
        # wrong PIN: both tries refused -> automatic login stops (repeated wrong PINs can lock the account)
        clock[0] = T(12)
        settings.data_dir = tmp_path / "other"                   # a second deployment: nothing stored either
        r2, dhan2, k2 = setup(clock)
        dhan2.gen_fail, dhan2.gen_status = 5, 401
        await k2.step()
        assert len(dhan2.gen_params) == 2
        await k2.step()
        assert len(dhan2.gen_params) == 2
        v = await k2.snapshot()
        assert v["totp_blocked"] and v["next_action"] == "needs_token" and "DHAN_PIN" in v["last_error"]
        with pytest.raises(tk.TokenError):                       # the admin's Renew now tries once more
            await k2.renew_now()
        assert len(dhan2.gen_params) == 4
    asyncio.run(go())


def test_expired_without_totp_asks_for_a_token():
    async def go():
        clock = [T(11)]
        r, dhan, k = setup(clock, env=jwt(T(8)))
        await k._pick()
        await k.step()
        assert "RenewToken" not in dhan.calls and "generateAccessToken" not in dhan.calls
        v = await k.snapshot()
        assert v["state"] == "expired" and not v["valid"] and v["next_action"] == "needs_token"
        assert "paste" in v["message"] and v["configured"] and not v["totp_configured"]
        # a token Dhan refuses (revoked) is treated the same way, even though its JWT has not run out
        clock[0] = T(12)
        revoked = jwt(T(8, 30, 1))
        r, dhan, k = setup(clock, env=revoked)
        await k._pick()
        await k.step()
        v = await k.snapshot()
        assert v["state"] == "rejected" and v["next_action"] == "needs_token" and "DH-901" in v["last_error"]
        assert "RenewToken" not in dhan.calls
    asyncio.run(go())


def test_profile_rejecting_the_new_token_keeps_the_old_one():
    async def go():
        clock = [T(14)]
        old = jwt(T(15), T(15, 0, -1))
        r, dhan, k = setup(clock, env=old)
        dhan.valid.add(old)
        dhan.reject_new = True
        await k._pick()
        await k.step()
        assert dhan.calls.count("RenewToken") == 1 and settings.dhan_access_token == old
        assert (await TokenStore(r).load())["token"] == old
        v = await k.snapshot()
        assert "did not accept" in v["last_error"] and v["next_action_at"] == T(14) + tk.RETRY_AFTER
    asyncio.run(go())


RATE_LIMITED = {"errorType": "Rate_Limit", "errorCode": "DH-904", "errorMessage": "Too many requests"}


def test_a_transient_profile_answer_is_not_a_refusal():
    async def go():
        clock = [T(10)]
        settings.dhan_pin, settings.dhan_totp_secret = "4321", SECRET
        tok = jwt(T(8, 30, 1), T(8, 30))
        r, dhan, k = setup(clock, env=tok)
        dhan.valid.add(tok)
        await k._pick()
        for glitch in (httpx.Response(400, json=RATE_LIMITED), httpx.Response(404, json={"detail": "Not Found"}),
                       httpx.Response(429), httpx.Response(503)):
            dhan.glitches.append(glitch)
            clock[0] += tk.VERIFY_EVERY
            await k.step()
            assert "generateAccessToken" not in dhan.calls and settings.dhan_access_token == tok
            assert k.view["state"] == "valid" and "Could not check" in k.view["last_error"]
        clock[0] += tk.VERIFY_EVERY
        await k.step()
        assert k.view["state"] == "valid" and not k.view["last_error"]
        dhan.valid.discard(tok)                                    # a real refusal still starts the TOTP login
        clock[0] += tk.VERIFY_EVERY
        await k.step()
        assert dhan.calls.count("generateAccessToken") == 1 and settings.dhan_access_token == dhan.issued[-1]
    asyncio.run(go())


def test_a_renewed_token_survives_a_hiccup_on_its_check():
    """RenewToken may retire the old token, so a hiccup while checking the new one must not throw it away."""
    async def go():
        clock = [T(14)]
        old = jwt(T(15), T(15, 0, -1))
        r, dhan, k = setup(clock, env=old)
        dhan.valid.add(old)
        await k._pick()
        dhan.glitches.append(httpx.Response(400, json=RATE_LIMITED))   # once: the retry 2 s later gets through
        await k.step(verify=False)
        new = dhan.issued[-1]
        assert old not in dhan.valid and settings.dhan_access_token == new and not k.view["last_error"]
        assert (await TokenStore(r).load())["token"] == new and k.view["profile"]["dataPlan"]
        clock[0] = T(12, 30, 1)                                    # next renewal: Dhan cannot be asked at all
        dhan.glitches += [httpx.Response(404), httpx.Response(503)]
        await k.step(verify=False)
        assert dhan.calls.count("RenewToken") == 2 and settings.dhan_access_token == dhan.issued[-1] != new
        assert (await TokenStore(r).load())["token"] == dhan.issued[-1]
        assert "without a Dhan check" in k.view["last_error"] and k.view["state"] == "valid"
    asyncio.run(go())


def test_a_token_no_header_can_carry_is_reported_not_raised():
    async def go():
        clock = [T(10)]
        r, dhan, k = setup(clock, env=jwt(T(8, 30, 1)) + "\u200b")   # copied into .env from a document
        await k._pick()
        await k.step()
        st = json.loads(await r.get(tk.KEY_STATUS))
        assert st["owner_updated"] == T(10) and st["state"] == "rejected" and "ASCII" in st["last_error"]
        assert dhan.calls == []

        def refuse(req):
            raise httpx.LocalProtocolError("Illegal header value b'eyJhbGci.eyJpc3Mi\\nSECRETPART'")
        k2 = TokenKeeper(r, http=httpx.AsyncClient(transport=httpx.MockTransport(refuse)))
        for value in ("tok\u00e9n", "fine"):                       # httpx refuses it / h11 refuses it
            got, body, err = await k2._call("GET", "https://api.dhan.test/v2/profile",
                                            headers={"access-token": value})
            assert got is None and body is None and err.startswith("invalid request")
            assert "SECRET" not in err and "eyJ" not in err and "tok" not in err
        await k2.http.aclose()
    asyncio.run(go())


def test_renew_errors_are_kept_and_retried_later():
    async def go():
        clock = [T(14)]
        old = jwt(T(15), T(15, 0, -1))
        r, dhan, k = setup(clock, env=old)
        dhan.valid.add(old)
        dhan.renew_status = 503
        await k._pick()
        await k.step()
        await k.step()
        assert dhan.calls.count("RenewToken") == 1 and "503" in k.view["last_error"]
        clock[0] += tk.RETRY_AFTER
        dhan.renew_status = 200
        await k.step()
        assert dhan.calls.count("RenewToken") == 2 and settings.dhan_access_token != old and not k.view["last_error"]
    asyncio.run(go())


def test_lock_prevents_double_renew():
    async def go():
        clock = [T(14)]
        old = jwt(T(15), T(15, 0, -1))
        r, dhan, owner = setup(clock, env=old)
        dhan.valid.add(old)
        dhan.delay = 0.05
        other = keeper(r, dhan, clock, role="owner")              # e.g. a second ingestor / the admin button
        await owner._pick()
        await other._pick()
        await asyncio.gather(owner.step(verify=False), other.step(verify=False))
        assert dhan.calls.count("RenewToken") == 1
        await other.step(verify=False)                            # after the lock: sees the new token, does nothing
        assert dhan.calls.count("RenewToken") == 1 and await r.get(tk.KEY_LOCK) is None
    asyncio.run(go())


# ---------------------------------------------------------------- followers
def test_follower_applies_a_newer_stored_token():
    async def go():
        clock = [T(10)]
        env = jwt(T(8, 30, 1))
        r, dhan, f = setup(clock, env=env, role="follower")
        await f._pick()
        assert settings.dhan_access_token == env
        assert not await f.sync()
        clock[0] = T(10, 5)
        new = jwt(T(10, 5, 1))
        await TokenStore(r).save(record(new, "renewed", now=T(10, 5)))   # the ingestor renewed
        assert await f.sync() and settings.dhan_access_token == new
        expired = jwt(T(9))
        await TokenStore(r).save(record(expired, "admin", now=T(10, 6)))
        clock[0] = T(10, 7)
        assert not await f.sync() and settings.dhan_access_token == new   # never trade a valid token for a dead one
        foreign = jwt(T(11, 0, 1), cid="999")
        await TokenStore(r).save(record(foreign, "admin", now=T(10, 8)))
        assert not await f.sync() and settings.dhan_access_token == new
        assert dhan.calls == []                                           # followers never call Dhan
    asyncio.run(go())


def test_follower_rereads_the_store_only_when_the_version_moves(monkeypatch):
    async def go():
        clock = [T(10)]
        r, dhan, f = setup(clock, env=jwt(T(8, 30, 1)), role="follower")
        await TokenStore(r).save(record(settings.dhan_access_token, "env", now=T(9)))
        await f._pick()
        loads = []
        real = f.store.load

        async def load():
            loads.append(1)
            return await real()
        monkeypatch.setattr(f.store, "load", load)
        await f.sync()
        await f.sync()
        assert len(loads) == 1
        new = jwt(T(10, 1, 1))
        await TokenStore(r).save(record(new, "renewed", now=T(10, 1)))          # INCR td:dhan:token:ver
        assert await f.sync() and settings.dhan_access_token == new and len(loads) == 2
        clock[0] += tk.RELOAD                                                  # a file-only save is seen too
        await f.sync()
        assert len(loads) == 3
    asyncio.run(go())


def test_httpx_request_log_hides_the_pin(caplog):
    url = "https://auth.dhan.test/app/generateAccessToken?dhanClientId=1000000001&pin=4321&totp=287082"
    with caplog.at_level("INFO", logger="httpx"):
        logging.getLogger("httpx").info('HTTP Request: %s %s "%s %d %s"', "POST", httpx.URL(url), "HTTP/1.1", 200, "OK")
    assert "4321" not in caplog.text and "287082" not in caplog.text and "pin=***" in caplog.text


def test_follower_loop_picks_up_tokens(monkeypatch):
    async def go():
        clock = [T(10)]
        r, dhan, f = setup(clock, env=jwt(T(8, 30, 1)), role="follower")
        monkeypatch.setattr(tk, "TICK", 0.01)
        await f.start()
        new = jwt(T(10, 1, 1))
        clock[0] = T(10, 1)
        await TokenStore(r).save(record(new, "renewed", now=T(10, 1)))
        for _ in range(200):
            if settings.dhan_access_token == new:
                break
            await asyncio.sleep(0.01)
        await f.stop()
        assert settings.dhan_access_token == new
    asyncio.run(go())


def _backfill_cli():
    """backfill/backfill.py is a script, not a package: load it from its file."""
    path = Path(__file__).resolve().parents[2] / "backfill" / "backfill.py"
    spec = importlib.util.spec_from_file_location("backfill_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_backfill_cli_follows_the_kept_token(monkeypatch):
    """`backfill.py update` from cron: the .env token lapsed on day 2, the ingestor's renewed one is stored."""
    cli = _backfill_cli()
    from td import bus
    monkeypatch.setattr(tk, "TICK", 0.01)

    async def go():
        now = time.time()
        r = fakeredis.aioredis.FakeRedis()
        kept = jwt(now + 20 * 3600, now - 4 * 3600)
        await TokenStore(r).save(record(kept, "renewed", now=now - 4 * 3600))
        lapsed = jwt(now - 3600, now - 90000)
        settings.dhan_access_token = lapsed
        monkeypatch.setattr(bus, "connect", lambda *a, **k: r)
        async with cli._Session() as ses:
            assert ses.rest.access_token == kept and ses.rest._headers()["access-token"] == kept
            renewed = jwt(now + 24 * 3600, now)                     # renewed by the ingestor during a long run
            await TokenStore(r).save(record(renewed, "renewed", now=now + 1))
            for _ in range(200):
                if ses.rest.access_token == renewed:
                    break
                await asyncio.sleep(0.01)
            assert ses.rest.access_token == renewed

        class Down:
            async def ping(self):
                raise ConnectionError("Redis is down")
        settings.dhan_access_token = lapsed
        monkeypatch.setattr(bus, "connect", lambda *a, **k: Down())
        async with cli._Session() as ses:                           # no Redis: the file copy
            assert ses.redis is None and ses.rest.access_token == renewed
        settings.dhan_access_token = ""                             # .env without a token: main() reads the file
        await cli._stored_token()
        assert settings.dhan_access_token == renewed
    asyncio.run(go())


# ------------------------------------------------------------ REST + feeds
def test_rest_header_uses_the_applied_token():
    async def go():
        seen = []

        def handler(req):
            seen.append(req.headers["access-token"])
            return httpx.Response(200, json={"data": {}})
        settings.dhan_access_token = "first"
        rest = DhanRest()
        await rest._http.aclose()
        rest._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        assert rest._headers()["access-token"] == "first"
        settings.dhan_access_token = "renewed"
        await rest.quote({"NSE_EQ": [1]}, "ltp")
        assert seen == ["renewed"] and rest.access_token == "renewed"
        settings.dhan_access_token = ""
        with pytest.raises(Exception, match="not configured"):
            await rest._post("/x", {})
        assert DhanRest(access_token="explicit").access_token == "explicit"   # an explicit token sticks
        await rest.close()
    asyncio.run(go())


class _FakeWS:
    def __init__(self):
        self.closed = asyncio.Event()

    async def send(self, msg):
        pass

    async def close(self, code=1000, reason=""):
        self.closed.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self.closed.wait()
        raise StopAsyncIteration


class _FakeConnect:
    def __init__(self):
        self.urls: list[str] = []
        self.sockets: list[_FakeWS] = []

    def __call__(self, url, **kw):
        ws = _FakeWS()
        self.urls.append(url)
        self.sockets.append(ws)

        class CM:
            async def __aenter__(self):
                return ws

            async def __aexit__(self, *a):
                ws.closed.set()
                return False
        return CM()


def test_feed_urls_follow_the_token_and_sockets_reconnect(monkeypatch):
    async def handler(pkts):
        pass

    settings.dhan_access_token = "tokA"
    m = feeds.MarketFeedManager([feeds.Sub("NSE_EQ", 1)], handler)
    d = feeds.Depth20Feed([feeds.Sub("NSE_EQ", 1)], handler)
    c = feeds.CasFeed([feeds.Sub("NSE_EQ", 1)], handler)
    settings.dhan_access_token = "tokB"
    assert "token=tokB&" in m._url() and "version=2" in m._url()
    assert d._conn.url().startswith(settings.dhan_depth20_url + "?token=tokB&clientId=" + CID)
    assert c._conn.url() == f"{settings.dhan_feed_url}?token=tokB&clientId={CID}&authType=2"

    async def go():
        fc = _FakeConnect()
        monkeypatch.setattr(feeds, "connect", fc)
        monkeypatch.setattr(feeds, "TOKEN_POLL", 0.01)

        async def until(cond):
            for _ in range(300):
                if cond():
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("timed out")
        task = asyncio.create_task(m.run())
        await until(lambda: len(fc.urls) == 1)
        assert "token=tokB&" in fc.urls[0]
        settings.dhan_access_token = "tokC"                      # the keeper renewed
        await until(lambda: len(fc.urls) == 2)
        assert fc.sockets[0].closed.is_set() and "token=tokC&" in fc.urls[1]
        assert m.stats.reconnects == 1 and "tokB" not in m.stats.last_error
        m.stop()
        fc.sockets[1].closed.set()
        await asyncio.wait_for(task, 2)
    asyncio.run(go())


# ------------------------------------------------------------------ gateway
def _gateway(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from td import auth
    from td.auth import Throttle, UserStore
    from td.services import gateway as gw
    monkeypatch.setattr(settings, "session_secret", "")
    monkeypatch.setattr(settings, "access_keys", "")
    monkeypatch.setattr(settings, "admin_user", "")
    monkeypatch.setattr(settings, "admin_password", "")
    monkeypatch.setattr(auth, "_SECRET", None)
    r = fakeredis.aioredis.FakeRedis()
    monkeypatch.setattr(gw.bus, "connect", lambda *a, **k: r)
    monkeypatch.setattr(gw, "Lake", lambda *a, **k: None)
    monkeypatch.setattr(gw, "users", UserStore(tmp_path / "users.json"))
    monkeypatch.setattr(gw, "throttle", Throttle())
    return gw, r, TestClient(gw.app, client=("127.0.0.1", 5000), base_url="http://127.0.0.1:8000")


def test_admin_endpoints(monkeypatch, tmp_path):
    import time as _time
    now = _time.time()
    clock = [now]
    dhan = FakeDhan(clock)
    env = jwt(now + 20 * 3600, now - 4 * 3600)
    dhan.valid.add(env)
    settings.dhan_access_token = env
    gw, r, c = _gateway(monkeypatch, tmp_path)
    gw.users.create("boss", "bosspassword", "admin")
    gw.users.create("ravi", "ravipassword", "viewer")
    with c:
        gw.app.state.dhan_token._http = httpx.AsyncClient(transport=httpx.MockTransport(dhan))
        assert c.get("/api/admin/dhan-token").status_code == 401
        c.post("/api/auth/login", json={"username": "ravi", "password": "ravipassword"}, headers=H)
        assert c.get("/api/admin/dhan-token").status_code == 403
        assert c.post("/api/admin/dhan-token/renew", headers=H).status_code == 403
        c.cookies.clear()
        c.post("/api/auth/login", json={"username": "boss", "password": "bosspassword"}, headers=H)
        res = c.get("/api/admin/dhan-token")
        st = res.json()
        assert res.status_code == 200 and env not in res.text
        assert st["configured"] and st["valid"] and st["state"] == "valid" and st["source"] == "env"
        assert st["client_id"] == CID and st["next_action"] == "renew" and not st["totp_configured"]
        assert set(st) >= {"exp", "iat", "expires_in_s", "auto_renew", "renew_at", "next_action_at", "last_check",
                           "last_renew", "last_error", "profile", "message", "owner_alive"}
        assert not set(st) & set(tk.INTERNAL)
        # paste: CSRF header, shape, client id, expiry and Dhan's own check
        new = jwt(now + 86000, now - 400)
        assert c.post("/api/admin/dhan-token", json={"token": new}).status_code == 403
        for body, msg in (({"token": "hello"}, "not a Dhan access token"), ({}, "send JSON"),
                          ({"token": jwt(now + 9000, cid="999")}, "belongs to Dhan client 999"),
                          ({"token": jwt(now - 60)}, "expired"),
                          ({"token": "\u201c" + new + "\u201d"}, "not plain ASCII"),       # curly quotes
                          ({"token": new + "\u200b"}, "not plain ASCII"),                  # zero-width space
                          ({"token": new + "!"}, "not a Dhan access token"),
                          ({"token": new}, "did not accept")):
            res = c.post("/api/admin/dhan-token", json=body, headers=H)
            assert res.status_code == 400 and msg in res.json()["detail"], (body, res.text)
            assert new not in res.text
        assert settings.dhan_access_token == env
        dhan.valid.add(new)
        wrapped = "  " + new[:50] + "\n" + new[50:100] + "\r\n" + new[100:] + "\n"   # copied from a wrapped terminal
        res = c.post("/api/admin/dhan-token", json={"token": wrapped}, headers=H)
        assert res.status_code == 200 and res.json()["source"] == "admin"
        assert new[:50] not in res.text and new[50:100] not in res.text and new.split(".")[1] not in res.text
        assert settings.dhan_access_token == new and (tmp_path / "auth" / "dhan_token.json").exists()
        # renew now
        res = c.post("/api/admin/dhan-token/renew", headers=H)
        renewed = dhan.issued[-1]
        assert res.status_code == 200 and res.json()["source"] == "renewed" and res.json()["last_renew"]
        assert renewed not in res.text and settings.dhan_access_token == renewed
        assert dhan.calls.count("RenewToken") == 1
        assert renewed.encode() not in (asyncio.run(r.get(tk.KEY_STATUS)) or b"")
        # once the token is refused, Renew now explains what to do (no PIN / TOTP configured)
        dhan.valid.clear()
        res = c.post("/api/admin/dhan-token/renew", headers=H)
        assert res.status_code == 400 and "paste" in res.json()["detail"]
        assert c.get("/api/admin/dhan-token").json()["state"] == "rejected"
