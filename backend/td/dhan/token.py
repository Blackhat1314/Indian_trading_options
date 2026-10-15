"""Dhan access-token keeper: the 24-hour token renews itself, so a deployed app needs no daily paste.

Tokens generated on Dhan Web expire 24 h after they are made. Dhan's RenewToken call swaps a still-valid token for
a fresh 24-hour one; once a token has lapsed only a new login replaces it: the account PIN + a TOTP code
(DHAN_PIN + DHAN_TOTP_SECRET), or a token pasted on /admin → SYSTEM.

The ingestor owns renewals; every process follows. The current token lives in Redis (td:dhan:token) and in
data/auth/dhan_token.json (Docker's Redis keeps nothing across restarts), and each process applies it to
`settings.dhan_access_token`, which the REST client and the websocket URLs read on every call / connect.
The token itself is never logged, put in the status key or returned by the API.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import logging
import os
import re
import socket
import struct
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any

import httpx

from .. import bus
from ..config import settings
from ..timeutil import IST

log = logging.getLogger("td.dhan.token")


class _HideLogin(logging.Filter):
    """httpx logs request URLs at INFO, and generateAccessToken takes the PIN + TOTP in its query string."""

    def filter(self, rec: logging.LogRecord) -> bool:
        with contextlib.suppress(Exception):
            msg = rec.getMessage()
            if "pin=" in msg or "totp=" in msg:
                rec.msg, rec.args = re.sub(r"\b(pin|totp)=[^&\s\"']*", r"\1=***", msg), ()
        return True


logging.getLogger("httpx").addFilter(_HideLogin())

KEY_TOKEN = "td:dhan:token"            # the record (with the token): served by no endpoint
KEY_LOCK = "td:dhan:token:lock"
KEY_VER = "td:dhan:token:ver"          # bumped on every save
KEY_STATUS = "td:dhan:token:status"    # what the admin page shows (no token)
SOURCES = ("env", "renewed", "totp", "admin")

RENEW_BEFORE = 2 * 3600     # always renew once less than this is left
ANCHOR_LEFT = 12 * 3600     # ...and in the morning window once at most this is left
WINDOW = 45 * 60            # morning window length from DHAN_TOKEN_RENEW_AT
EXPIRING = 3 * 3600         # "expiring soon" on the admin page
MARGIN = 60                 # a token this close to its expiry counts as expired
VERIFY_EVERY = 300          # owner: GET /profile
TICK = 30                   # every process: pick up a token saved by another one
RETRY_AFTER = 300           # a renewal that failed on the network / a Dhan hiccup is retried after this
LOCK_TTL = 60
RELOAD = 300                # a follower re-reads the store at least this often, version bump or not
INTERNAL = ("rejected_fp", "token_fp", "retry_at", "gen_blocked", "error_op", "owner_updated")


class TokenError(Exception):
    """A token could not be set or renewed; `status` is the HTTP status for the admin API."""

    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


# ------------------------------------------------------------------ helpers
def claims(token: str | None) -> dict | None:
    """The JWT payload (exp, iat, dhanClientId), read without checking the signature: only Dhan can check that,
    this is just to know whose token it is and when it runs out."""
    if not token or not isinstance(token, str) or token.count(".") != 2:
        return None
    part = token.split(".")[1]
    try:
        data = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _num(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x > 0 and x == x else None


def expires_at(token: str | None) -> float | None:
    return _num((claims(token) or {}).get("exp"))


def fp(token: str | None) -> str:
    """Short fingerprint, so the status can say which token it is about without holding it."""
    return hashlib.sha256((token or "").encode()).hexdigest()[:16]


def totp(secret_b32: str, t: float | None = None, step: int = 30, digits: int = 6) -> str:
    """RFC 6238 code (HMAC-SHA1) for the base32 secret Dhan shows under the TOTP QR code."""
    s = "".join(secret_b32.split()).replace("-", "").upper()
    key = base64.b32decode(s + "=" * (-len(s) % 8))
    counter = int((time.time() if t is None else t) // step)
    mac = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    off = mac[-1] & 0x0F
    code = (struct.unpack(">I", mac[off:off + 4])[0] & 0x7FFFFFFF) % 10 ** digits
    return str(code).zfill(digits)


def totp_configured() -> bool:
    return bool(settings.dhan_pin.strip() and settings.dhan_totp_secret.strip())


def record(token: str, source: str, exp: float | None = None, now: float | None = None) -> dict:
    """What the store keeps. `exp` is a hint (Dhan's expiryTime) used when the JWT has none or ends earlier."""
    c = claims(token) or {}
    exps = [x for x in (_num(c.get("exp")), _num(exp)) if x]
    return {"token": token, "exp": max(exps) if exps else None, "iat": _num(c.get("iat")), "source": source,
            "saved_at": time.time() if now is None else now}


def _clean(d: Any) -> dict | None:
    if not isinstance(d, dict) or not isinstance(d.get("token"), str) or not d["token"].strip():
        return None
    return {"token": d["token"].strip(), "exp": _num(d.get("exp")), "iat": _num(d.get("iat")),
            "source": d.get("source") if d.get("source") in SOURCES else "env", "saved_at": _num(d.get("saved_at"))}


_warned: set[str] = set()


def _foreign(rec: dict, client_id: str) -> bool:
    owner = str((claims(rec["token"]) or {}).get("dhanClientId") or "")
    if owner and client_id and owner != client_id:
        if fp(rec["token"]) not in _warned:
            _warned.add(fp(rec["token"]))
            log.warning("ignoring a Dhan token (%s) issued to client %s — DHAN_CLIENT_ID is %s",
                        rec.get("source"), owner, client_id)
        return True
    return False


def best_token(candidates: list[dict | None], client_id: str | None = None, now: float | None = None) -> dict | None:
    """The record to use: a still-valid token beats one of unknown expiry, which beats an expired one; then the
    later expiry wins (ties: the earlier candidate). A token issued to another client id is ignored."""
    now = time.time() if now is None else now
    cid = str(settings.dhan_client_id if client_id is None else client_id)
    best, best_key = None, None
    for rec in candidates:
        if not rec or not rec.get("token") or _foreign(rec, cid):
            continue
        exp = rec.get("exp")
        key = (1 if exp is None else 2 if exp > now + MARGIN else 0, exp or 0)
        if best_key is None or key > best_key:
            best, best_key = rec, key
    return best


def dotenv_token() -> str:
    """DHAN_ACCESS_TOKEN as written in .env: an exported environment variable hides it from `settings`."""
    path = settings.model_config.get("env_file")
    try:
        if path and Path(path).is_file():
            from dotenv import dotenv_values
            return (dotenv_values(path).get("DHAN_ACCESS_TOKEN") or "").strip()
    except Exception:
        log.debug("cannot read %s", path, exc_info=True)
    return ""


def _when(v: Any) -> float | None:
    """Dhan's expiryTime: epoch (s or ms), ISO 8601, or 'dd/mm/yyyy HH:MM' — IST when it names no zone."""
    if v in (None, ""):
        return None
    x = _num(v)
    if x:
        return x / 1000 if x > 1e11 else x
    s = str(v).strip()
    for f in ("%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(s, f).replace(tzinfo=IST).timestamp()
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=IST)).timestamp()


def _field(body: Any, name: str) -> Any:
    if isinstance(body, dict):
        if name in body:
            return body[name]
        if isinstance(body.get("data"), dict):
            return body["data"].get(name)
    return None


def _token_in(body: Any) -> str | None:
    for k in ("accessToken", "access_token", "token"):
        v = _field(body, k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _error_text(status: int, body: Any, text: str) -> str:
    if isinstance(body, dict):
        code = body.get("errorCode") or body.get("errorType") or ""
        msg = body.get("errorMessage") or body.get("message") or body.get("remarks") or ""
        if code or msg:
            return f"HTTP {status} {code} {msg}".strip()[:300]
    return f"HTTP {status} {text.strip()[:200]}".strip()


def _renew_time() -> dtime:
    try:
        h, m = settings.dhan_token_renew_at.strip().split(":")[:2]
        return dtime(int(h), int(m))
    except (ValueError, AttributeError):
        return dtime(8, 30)


def _ist(ts: float | None) -> str:
    return datetime.fromtimestamp(ts, tz=IST).strftime("%d %b %H:%M") if ts else "?"


def _base() -> str:
    return settings.dhan_base_url.rstrip("/")


# -------------------------------------------------------------------- store
class TokenStore:
    """The current token: Redis td:dhan:token (no TTL) for every process, plus data/auth/dhan_token.json
    (owner-only file mode) because Docker's Redis keeps nothing across a restart."""

    def __init__(self, redis, path: Path | str | None = None):
        self.r = redis
        self.path = Path(path) if path else Path(settings.data_dir) / "auth" / "dhan_token.json"
        self._bad_file = False

    async def load(self) -> dict | None:
        """The fresher of the Redis and file copies; a lost Redis key is restored from the file."""
        cached, redis_ok = None, True
        try:
            raw = await self.r.get(KEY_TOKEN)
            cached = _clean(bus.loads(raw)) if raw else None
        except ValueError:
            pass
        except Exception:
            redis_ok = False
            log.debug("token store: Redis read failed", exc_info=True)
        disk = await asyncio.to_thread(self._read)
        if disk and not cached and redis_ok:
            try:
                if await self.r.set(KEY_TOKEN, bus.dumps(disk), nx=True):
                    await self.r.incr(KEY_VER)
            except Exception:
                log.debug("token store: restore to Redis failed", exc_info=True)
        if cached and disk and disk["token"] != cached["token"] and (disk["exp"] or 0) > (cached["exp"] or 0):
            return disk
        return cached or disk

    async def save(self, rec: dict) -> None:
        """File first (survives a Redis restart), then Redis; a file error is logged, a Redis error raises."""
        rec = _clean(rec) or {}
        if not rec:
            raise ValueError("empty token record")
        try:
            await asyncio.to_thread(self._write, rec)
        except OSError as e:
            log.error("cannot write %s (%s) — the token lives in Redis only until this is fixed", self.path, e)
        await self.r.set(KEY_TOKEN, bus.dumps(rec))
        await self.r.incr(KEY_VER)

    def _read(self) -> dict | None:
        try:
            rec = _clean(json.loads(self.path.read_text()))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            if not self._bad_file:
                log.warning("cannot read %s (%s) — ignored", self.path, e)
            self._bad_file = True
            return None
        self._bad_file = False
        return rec

    def _write(self, rec: dict) -> None:
        d = self.path.parent
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.suppress(OSError):
            os.chmod(d, 0o700)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(rec, f)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)


# ------------------------------------------------------------------- keeper
class TokenKeeper:
    """Applies the freshest stored token in this process and, as the owner (the ingestor), keeps it alive.

    Owner rules, every 30 s (GET /profile every 5 min, or soon after kick()):
      1. valid with less than 2 h left → RenewToken now;
      2. valid, inside the morning window [DHAN_TOKEN_RENEW_AT, +45 min] IST and at most 12 h left → RenewToken.
         A renewal lasts until the same time tomorrow, so rule 1 then fires 2 h earlier each night (06:30, 04:30…)
         until the token would lapse in the evening, when the window pulls it back to the morning: about one
         renewal a day, never in market hours;
      3. expired, missing or rejected by Dhan → a new token from the PIN + TOTP when configured, else the admin
         page asks for one. With the PIN + TOTP set, a valid token that Dhan refuses to renew (RenewToken is
         documented for Dhan Web tokens) is replaced the same way, at the time rules 1 and 2 pick.
    Every new token passes GET /profile (same client id) before it is saved and applied. Renewals hold the Redis
    lock td:dhan:token:lock, so the owner and an admin "Renew now" never both renew.
    """

    def __init__(self, redis, role: str = "follower", http: httpx.AsyncClient | None = None,
                 store: TokenStore | None = None):
        if role not in ("owner", "follower"):
            raise ValueError(f"role must be owner or follower, not {role!r}")
        self.r = redis
        self.role = role
        self.store = store or TokenStore(redis)
        self._http = http
        self._own_http = http is None
        self.clock = time.time                 # tests move time
        self.sleep = asyncio.sleep
        self.current: dict | None = None
        self.view: dict = {}
        self._applied_at = 0.0
        self._env: list[dict] = []
        self._kick = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._next_verify = 0.0
        self._last_verify = 0.0
        self._lock_id: str | None = None
        self._ver: bytes | None = None
        self._loaded_at = 0.0

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=10.0))
        return self._http

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Pick the best token (stored vs DHAN_ACCESS_TOKEN), apply it and start the background loop."""
        try:
            await self._pick()
            if self.role == "owner":
                await self._note(gen_blocked=False, retry_at=0)
        except Exception as e:
            log.warning("token keeper start: %s", self._scrub(f"{type(e).__name__}: {e}"))
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        if self._own_http and self._http is not None:
            with contextlib.suppress(Exception):
                await self._http.aclose()
            self._http = None

    def kick(self) -> None:
        """Check the token with Dhan now (e.g. a REST call was refused)."""
        self._kick.set()

    async def _run(self) -> None:
        while True:
            kicked = self._kick.is_set()
            self._kick.clear()
            try:
                if self.role == "owner":
                    now = self.clock()
                    verify = now >= self._next_verify or (kicked and now - self._last_verify >= 60)
                    if verify:
                        self._next_verify = now + VERIFY_EVERY
                    await self.step(verify)
                else:
                    await self.sync()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("token keeper: %s", self._scrub(f"{type(e).__name__}: {e}"))
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._kick.wait(), TICK)

    # ------------------------------------------------------------- choosing
    def _rec(self) -> dict | None:
        tok = settings.dhan_access_token
        if not tok:
            return None
        if self.current and self.current["token"] == tok:
            return self.current
        return record(tok, "env", now=0)

    def _env_tokens(self) -> list[dict]:
        out: list[dict] = []
        for t in (settings.dhan_access_token, dotenv_token()):
            t = (t or "").strip()
            if t and all(r["token"] != t for r in out):
                out.append(record(t, "env", now=0))
        return out

    def _apply(self, rec: dict) -> bool:
        changed = rec["token"] != settings.dhan_access_token
        settings.dhan_access_token = rec["token"]
        self.current = rec
        self._applied_at = self.clock()
        if changed:
            log.info("Dhan token applied (%s, valid until %s IST)", rec.get("source"), _ist(rec.get("exp")))
        return changed

    async def _pick(self) -> None:
        stored = await self.store.load()
        self._env = self._env_tokens()
        now = self.clock()
        best = best_token([stored, *self._env], now=now)
        if best is None:
            return
        self._apply(best)
        if self.role == "owner" and best is not stored:
            rec = dict(best, saved_at=now)
            await self.store.save(rec)               # the followers then use the same token
            self.current = rec

    async def sync(self) -> bool:
        """Apply a token another process saved (renewal, admin paste) when it should replace ours. The store is
        read again when td:dhan:token:ver moves, and every few minutes for a save that only reached the file."""
        now = self.clock()
        try:
            ver = await self.r.get(KEY_VER)
        except Exception:
            ver = None
        if (ver is not None and ver == self._ver and now - self._loaded_at < RELOAD and self.current
                and self.current["token"] == settings.dhan_access_token):
            return False
        stored = await self.store.load()
        self._ver, self._loaded_at = ver, now
        if not stored:
            return False
        if stored["token"] == settings.dhan_access_token:
            self.current = stored
            return False
        if best_token([stored], now=now) is None:
            return False
        fresh = stored["exp"] is None or stored["exp"] > now + MARGIN
        newer = fresh and (stored["saved_at"] or 0) > self._applied_at      # e.g. pasted after ours was applied
        if newer or best_token([stored, self._rec()], now=now) is stored:
            self._apply(stored)
            return True
        return False

    # --------------------------------------------------------------- status
    async def _saved(self) -> dict:
        try:
            d = bus.loads(await self.r.get(KEY_STATUS))
        except Exception:
            return {}
        return d if isinstance(d, dict) else {}

    async def _note(self, **fields) -> dict:
        """Merge observations into td:dhan:token:status and refresh the computed fields (never the token)."""
        saved = await self._saved()
        saved.update(fields)
        if self.role == "owner":
            saved["owner_updated"] = self.clock()
        saved.update(self._view(saved))
        saved["token_fp"] = fp(settings.dhan_access_token)
        with contextlib.suppress(Exception):
            await self.r.set(KEY_STATUS, bus.dumps(saved))
        self.view = saved
        return saved

    def _state(self, saved: dict, now: float) -> str:
        rec = self._rec()
        if not rec:
            return "missing"
        exp = rec.get("exp")
        if exp and exp <= now + MARGIN:             # Dhan refuses it too, but "expired" says why
            return "expired"
        if saved.get("rejected_fp") and saved["rejected_fp"] == fp(rec["token"]):
            return "rejected"
        if exp and exp - now < EXPIRING:
            return "expiring"
        return "valid"

    def _renew_due(self, rec: dict, now: float) -> float:
        exp = rec.get("exp") or ((rec.get("iat") or rec.get("saved_at") or now) + 86400)
        due = exp - RENEW_BEFORE
        day = datetime.fromtimestamp(now, tz=IST).date()
        for d in range(3):
            ws = datetime.combine(day + timedelta(days=d), _renew_time(), tzinfo=IST).timestamp()
            start = max(ws, exp - ANCHOR_LEFT, now)
            if start >= due:
                break
            if start <= ws + WINDOW:
                due = start
                break
        return max(now, due)

    def plan(self, saved: dict, now: float | None = None) -> tuple[str, float | None, str]:
        """(next_action, when, message): renew | generate | needs_token | none."""
        now = self.clock() if now is None else now
        if not settings.dhan_client_id:
            return "needs_token", None, "DHAN_CLIENT_ID is not set in .env"
        state = self._state(saved, now)
        if state in ("missing", "expired", "rejected"):
            why = {"missing": "No Dhan access token", "expired": "The Dhan token has expired",
                   "rejected": "Dhan rejected the token"}[state]
            if saved.get("gen_blocked"):
                return ("needs_token", None, f"{why}, and Dhan refused DHAN_PIN / DHAN_TOTP_SECRET — fix them and "
                                             "restart the ingestor, press Renew now, or paste a new token")
            if totp_configured() and settings.dhan_token_auto_renew:
                return ("generate", max(now, saved.get("retry_at") or 0),
                        f"{why} — a new one is generated with DHAN_PIN + DHAN_TOTP_SECRET")
            return ("needs_token", None, f"{why} — paste a new one (or set DHAN_PIN + DHAN_TOTP_SECRET in .env for "
                                         "automatic recovery)")
        if not settings.dhan_token_auto_renew:
            return "none", None, "Automatic renewal is off (DHAN_TOKEN_AUTO_RENEW=false) — paste a new token in time"
        at = max(self._renew_due(self._rec(), now), saved.get("retry_at") or 0)
        return "renew", at, f"Renews by itself at {_ist(at)} IST"

    def _view(self, saved: dict) -> dict:
        now = self.clock()
        rec = self._rec()
        exp = rec["exp"] if rec else None
        state = self._state(saved, now)
        action, at, msg = self.plan(saved, now)
        return {
            "configured": bool(settings.dhan_client_id and rec),
            "client_id": settings.dhan_client_id,
            "exp": exp, "iat": rec["iat"] if rec else None, "source": rec["source"] if rec else None,
            "state": state, "valid": state in ("valid", "expiring"),
            "expires_in_s": int(exp - now) if exp else None,
            "auto_renew": settings.dhan_token_auto_renew, "renew_at": _renew_time().strftime("%H:%M"),
            "next_action": action, "next_action_at": at, "message": msg,
            "totp_configured": totp_configured(), "totp_blocked": bool(saved.get("gen_blocked")),
            "last_check": saved.get("last_check"), "last_renew": saved.get("last_renew"),
            "last_error": saved.get("last_error") or "", "profile": saved.get("profile"),
            "owner_alive": now - (saved.get("owner_updated") or 0) < 180,
            "updated": now,
        }

    async def snapshot(self) -> dict:
        """The admin page's view: the owner's last observations + values computed now. Never the token."""
        await self.sync()
        view = self._view(await self._saved())
        return {k: v for k, v in view.items() if k not in INTERNAL}

    def brief(self) -> dict:
        """One line for the ingestor's status (/admin → SYSTEM)."""
        return {k: self.view.get(k) for k in ("state", "exp", "source", "next_action", "next_action_at")}

    # ------------------------------------------------------------ the owner
    async def step(self, verify: bool = True) -> None:
        """One owner pass: follow saved changes, check the token with Dhan, renew / generate when due."""
        await self.sync()
        if verify:
            await self._verify()
        now = self.clock()
        action, at, _ = self.plan(await self._saved(), now)
        if at is not None and at <= now:
            if action == "renew":
                await self._renew()
            elif action == "generate":
                await self._generate()
        await self._note()

    async def _verify(self) -> bool | None:
        tok = settings.dhan_access_token
        if not tok or not settings.dhan_client_id:
            return None
        ok, prof, err = await self._profile(tok)
        now = self.clock()
        self._last_verify = now
        saved = await self._saved()
        if ok:
            fields: dict = {"last_check": now, "profile": prof, "rejected_fp": None}
            if saved.get("error_op") == "check":
                fields.update(last_error="", error_op=None)
            await self._note(**fields)
        elif ok is False:
            # .env may hold a newer token than the one Dhan just refused (made on Dhan Web after it)
            for rec in self._env:
                if rec["token"] != tok and (rec["exp"] is None or rec["exp"] > now + MARGIN):
                    ok2, prof2, _ = await self._profile(rec["token"])
                    if ok2:
                        rec = dict(rec, saved_at=now)
                        await self.store.save(rec)
                        self._apply(rec)
                        await self._note(last_check=now, profile=prof2, rejected_fp=None, last_error="", error_op=None)
                        return True
            fields = {"last_check": now, "rejected_fp": fp(tok)}
            if saved.get("rejected_fp") != fp(tok) or not saved.get("last_error"):
                # first refusal of this token; later checks keep what came after it (e.g. a refused TOTP login)
                log.warning("Dhan rejected the access token: %s", err)
                fields.update(last_error=f"Dhan rejected the token: {err}", error_op="check")
            await self._note(**fields)
        else:
            await self._note(last_check=now, last_error=f"Could not check the token with Dhan: {err}", error_op="check")
        return ok

    @asynccontextmanager
    async def _locked(self, wait: float = 0):
        """td:dhan:token:lock (SET NX EX): one renewal at a time across every process."""
        me = f"{self.role}@{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        end = time.monotonic() + wait
        while not await self.r.set(KEY_LOCK, me, nx=True, ex=LOCK_TTL):
            if time.monotonic() >= end:
                yield False
                return
            await asyncio.sleep(0.25)
        self._lock_id = me
        try:
            yield True
        finally:
            self._lock_id = None
            with contextlib.suppress(Exception):
                if (await self.r.get(KEY_LOCK)) == me.encode():
                    await self.r.delete(KEY_LOCK)

    async def _hold_lock(self) -> None:
        if self._lock_id:
            with contextlib.suppress(Exception):
                if (await self.r.get(KEY_LOCK)) == self._lock_id.encode():
                    await self.r.expire(KEY_LOCK, LOCK_TTL)

    async def _due(self, action: str) -> bool:
        a, at, _ = self.plan(await self._saved(), self.clock())
        return a == action and at is not None and at <= self.clock()

    async def _renew(self, force: bool = False) -> tuple[bool, str, int]:
        """RenewToken under the lock → (ok, error message, HTTP status for the admin API)."""
        async with self._locked(wait=10 if force else 0) as got:
            if not got:
                return False, ("another renewal is in progress — try again in a minute" if force else ""), 409
            await self.sync()                      # another process may have just renewed
            if not force and not await self._due("renew"):
                return False, "", 0
            tok, cid = settings.dhan_access_token, str(settings.dhan_client_id or "")
            if not tok or not cid:
                return False, "no token to renew", 400
            st, body, err = await self._call("GET", f"{_base()}/RenewToken", tok,
                                             headers={"access-token": tok, "dhanClientId": cid,
                                                      "Accept": "application/json"})
            new = _token_in(body) if st is not None and st < 400 else None
            now = self.clock()
            if not new:
                err = err or "Dhan's answer had no token"
                if st is not None and 400 <= st < 500 and st != 429:
                    ok, _, perr = await self._profile(tok)
                    if ok is False:
                        msg = f"Dhan rejected the token: {perr}"
                        await self._note(last_check=now, rejected_fp=fp(tok), last_error=msg, error_op="renew")
                        return False, msg, 400
                    if ok and totp_configured() and (force or not (await self._saved()).get("gen_blocked")):
                        # Dhan documents RenewToken for tokens made on Dhan Web: replace any other by a new login
                        log.info("Dhan did not renew the token (%s) — logging in with the PIN + TOTP instead", err)
                        return await self._login()
                msg = f"Renewal failed: {err}"
                log.warning(msg)
                await self._note(retry_at=now + RETRY_AFTER, last_error=msg, error_op="renew")
                return False, msg, 502 if st is None or st >= 500 or st == 429 else 400
            hint = _when(_field(body, "expiryTime")) if (new == tok or expires_at(new) is None) else None
            await self._hold_lock()
            try:
                rec = await self._adopt(new, "renewed", hint)
            except TokenError as e:
                await self._note(retry_at=now + RETRY_AFTER, last_error=str(e), error_op="renew")
                return False, str(e), e.status
            log.info("Dhan token renewed — valid until %s IST", _ist(rec["exp"]))
            return True, "", 200

    async def _generate(self, force: bool = False) -> tuple[bool, str, int]:
        """A new token from DHAN_PIN + a TOTP code (retried once with the next code) → (ok, error, HTTP status)."""
        async with self._locked(wait=10 if force else 0) as got:
            if not got:
                return False, ("another renewal is in progress — try again in a minute" if force else ""), 409
            await self.sync()
            if not force and not await self._due("generate"):
                return False, "", 0
            return await self._login()

    async def _login(self) -> tuple[bool, str, int]:
        """generateAccessToken with DHAN_PIN + a TOTP code, retried once with the next code. The lock is held."""
        cid = str(settings.dhan_client_id or "")
        if not totp_configured() or not cid:
            return False, "DHAN_CLIENT_ID, DHAN_PIN and DHAN_TOTP_SECRET must all be set in .env", 400
        last_err, last_st = "", None
        for attempt in range(2):
            if attempt:                             # a TOTP code works once: wait for the next one
                await self.sleep(30 - self.clock() % 30 + 1)
                await self._hold_lock()
            try:
                code = totp(settings.dhan_totp_secret, self.clock())
            except ValueError:
                msg = "DHAN_TOTP_SECRET is not a valid base32 secret (copy the text under the QR code)"
                await self._note(gen_blocked=True, last_error=msg, error_op="generate")
                return False, msg, 400
            st, body, err = await self._call(
                "POST", f"{settings.dhan_auth_url.rstrip('/')}/app/generateAccessToken",
                params={"dhanClientId": cid, "pin": settings.dhan_pin.strip(), "totp": code})
            new = _token_in(body) if st is not None and st < 400 else None
            if new:
                await self._hold_lock()
                hint = _when(_field(body, "expiryTime")) if expires_at(new) is None else None
                try:
                    rec = await self._adopt(new, "totp", hint)
                except TokenError as e:
                    await self._note(retry_at=self.clock() + RETRY_AFTER, last_error=str(e), error_op="generate")
                    return False, str(e), e.status
                log.info("new Dhan token generated with the PIN + TOTP — valid until %s IST", _ist(rec["exp"]))
                return True, "", 200
            last_err, last_st = err or "Dhan's answer had no token", st
        if last_st is not None and last_st < 500 and last_st != 429:
            # stop here: repeated wrong PINs can lock the Dhan account
            msg = (f"Dhan refused DHAN_PIN / DHAN_TOTP_SECRET ({last_err}) — automatic login is paused: fix them "
                   "and restart the ingestor, or paste a token")
            log.error(msg)
            await self._note(gen_blocked=True, last_error=msg, error_op="generate")
            return False, msg, 400
        msg = f"Could not generate a token: {last_err}"
        log.warning(msg)
        await self._note(retry_at=self.clock() + RETRY_AFTER, last_error=msg, error_op="generate")
        return False, msg, 502

    async def _adopt(self, token: str, source: str, exp_hint: float | None = None, strict: bool = False) -> dict:
        """Check a new token with GET /profile, then save and apply it. Raises TokenError when Dhan refuses it
        (or, `strict`, when Dhan cannot be asked); the current token stays in place."""
        cid = str(settings.dhan_client_id or "")
        if not cid:
            raise TokenError("DHAN_CLIENT_ID is not set in .env")
        owner = str((claims(token) or {}).get("dhanClientId") or "")
        if owner and owner != cid:
            raise TokenError(f"this token belongs to Dhan client {owner}, not DHAN_CLIENT_ID {cid}")
        ok, prof, err = await self._profile(token)
        if ok is None:
            await self.sleep(2)
            ok, prof, err = await self._profile(token)
        if ok is False:
            raise TokenError(f"Dhan did not accept the new token: {err}")
        if ok is None and strict:
            raise TokenError(f"could not check the token with Dhan ({err}) — try again", 502)
        now = self.clock()
        rec = record(token, source, exp_hint, now)
        try:
            await self.store.save(rec)
        except Exception as e:
            log.error("could not store the new Dhan token in Redis (%s) — the other processes read it from %s",
                      type(e).__name__, self.store.path)
        self._apply(rec)
        fields: dict = {"rejected_fp": None, "retry_at": 0, "gen_blocked": False, "last_error": "", "error_op": None}
        if ok:
            fields.update(last_check=now, profile=prof)
        else:
            fields.update(last_error=f"New token saved without a Dhan check ({err})", error_op="check")
        if source in ("renewed", "totp"):
            fields["last_renew"] = now
        await self._note(**fields)
        return rec

    # ---------------------------------------------------- admin (gateway)
    async def adopt(self, token: Any) -> dict:
        """Admin paste: a JWT for DHAN_CLIENT_ID that is not expired and that Dhan accepts → saved and applied."""
        token = token.strip() if isinstance(token, str) else ""
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if not token:
            raise TokenError("paste the access token")
        if len(token) > 4096 or claims(token) is None:
            raise TokenError("that is not a Dhan access token (expected a JWT: three base64 parts separated by dots)")
        exp = expires_at(token)
        if exp is not None and exp <= self.clock() + MARGIN:
            raise TokenError(f"that token expired at {_ist(exp)} IST — generate a new one on web.dhan.co")
        async with self._locked(wait=10) as got:
            if not got:
                raise TokenError("a renewal is in progress — try again in a minute", 409)
            await self._adopt(token, "admin", strict=True)
        log.info("Dhan token replaced from the admin page (valid until %s IST)", _ist(exp))
        return await self.snapshot()

    async def renew_now(self) -> dict:
        """Admin "Renew now": RenewToken while the token is valid; once it has lapsed, a TOTP login if configured.
        Also lifts a paused TOTP login (the admin may have fixed the PIN)."""
        await self.sync()
        if not settings.dhan_client_id:
            raise TokenError("DHAN_CLIENT_ID is not set in .env")
        await self._note(gen_blocked=False, retry_at=0)
        ok, msg, code = False, "", 400
        if self._state(await self._saved(), self.clock()) in ("valid", "expiring"):
            ok, msg, code = await self._renew(force=True)
        if not ok and code != 409 and self._state(await self._saved(), self.clock()) in ("missing", "expired",
                                                                                           "rejected"):
            if totp_configured():
                ok, msg, code = await self._generate(force=True)
            else:
                msg = (f"{msg or 'the token has lapsed and Dhan only renews a valid one'} — paste a new token (or "
                       "set DHAN_PIN + DHAN_TOTP_SECRET in .env for automatic recovery)")
                code = 400
        if not ok:
            raise TokenError(msg or "renewal failed", code or 400)
        return await self.snapshot()

    # ----------------------------------------------------------------- HTTP
    def _scrub(self, text: str, *extra: str) -> str:
        for s in (settings.dhan_access_token, *(r["token"] for r in self._env), *extra,
                  settings.dhan_totp_secret.strip(), settings.dhan_pin.strip()):
            if s and len(s) >= 4:
                text = text.replace(s, "***")
        return text[:300]

    async def _call(self, method: str, url: str, *secrets: str, **kw) -> tuple[int | None, Any, str]:
        """(HTTP status, JSON body, error text). Network problems come back as status None and never raise."""
        try:
            r = await self.http.request(method, url, **kw)
        except httpx.HTTPError as e:
            return None, None, self._scrub(f"network error ({type(e).__name__}: {e})", *secrets)
        try:
            body = r.json()
        except ValueError:
            body = None
        failed = isinstance(body, dict) and str(body.get("status", "")).lower() in ("failure", "failed", "error")
        status = 400 if failed and r.status_code < 400 else r.status_code
        if status < 400:
            return status, body, ""
        return status, body, self._scrub(_error_text(r.status_code, body, r.text), *secrets)

    async def _profile(self, token: str) -> tuple[bool | None, dict | None, str]:
        """GET /profile with `token`: (True, plan info) · (False, why) when Dhan refuses it · (None, why) unsure."""
        cid = str(settings.dhan_client_id or "")
        st, body, err = await self._call("GET", f"{_base()}/profile", token,
                                         headers={"access-token": token, "client-id": cid,
                                                  "Accept": "application/json"})
        if st is None or st == 429 or st >= 500:
            return None, None, err or f"HTTP {st}"
        if st >= 400:
            return False, None, err
        if not isinstance(body, dict):
            return None, None, "unexpected answer from /profile"
        got = str(body.get("dhanClientId") or "")
        if got and cid and got != cid:
            return False, None, f"the token belongs to client {got}, not DHAN_CLIENT_ID {cid}"
        return True, {k: body.get(k) for k in ("dataPlan", "dataValidity", "tokenValidity")}, ""


__all__ = ["TokenKeeper", "TokenStore", "TokenError", "best_token", "claims", "expires_at", "record", "totp",
           "totp_configured"]
