"""User accounts and login sessions for the terminal and the admin page.

Users live in data/auth/users.json: scrypt-hashed passwords and a role (admin = the /admin page and history
pulls, viewer = the terminal). Sessions are signed tokens in an HttpOnly cookie (HMAC-SHA256 with
data/auth/secret.key), so every gateway instance can check them without shared state and they survive
restarts. Changing a password or disabling a user bumps their token version, which ends their sessions.

The first admin comes from ADMIN_USER / ADMIN_PASSWORD in .env, or from the setup screen of /login, which
only the host itself (localhost) can use, and only while there are no users at all.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .config import settings

try:
    import fcntl
except ImportError:                    # Windows: in-process locking only
    fcntl = None

log = logging.getLogger("td.auth")
COOKIE = "td_session"
ROLES = ("admin", "viewer")
USERNAME = re.compile(r"[a-z0-9][a-z0-9._-]{1,31}")
RESERVED = {"local"}                 # the identity of local mode (no accounts): never an account name
MIN_PASSWORD, MAX_PASSWORD = 8, 128


class AuthError(ValueError):
    """A user-facing problem with the request (bad input, last admin, …)."""


def auth_dir() -> Path:
    return Path(settings.data_dir) / "auth"


# ------------------------------------------------------------------ passwords
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    n, r, p = 2 ** 14, 8, 1
    dk = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, dk = stored.split("$")
        if algo != "scrypt":
            return False
        got = hashlib.scrypt(password.encode(), salt=_unb64(salt), n=int(n), r=int(r), p=int(p),
                             dklen=len(_unb64(dk)))
        return hmac.compare_digest(got, _unb64(dk))
    except (ValueError, TypeError):
        return False


_DUMMY = None


def _dummy_hash() -> str:
    """Checked when the user does not exist, so a wrong user name costs as long as a wrong password."""
    global _DUMMY
    if _DUMMY is None:
        _DUMMY = hash_password(secrets.token_hex(8))
    return _DUMMY


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def check_username(name: str) -> str:
    name = (name or "").strip().lower()
    if not USERNAME.fullmatch(name):
        raise AuthError("user names are 2-32 characters: letters, digits, dot, dash, underscore")
    if name in RESERVED:
        raise AuthError(f"'{name}' is reserved")
    return name


def check_password(password: str) -> str:
    if not isinstance(password, str) or not MIN_PASSWORD <= len(password) <= MAX_PASSWORD:
        raise AuthError(f"passwords are {MIN_PASSWORD}-{MAX_PASSWORD} characters")
    return password


# ------------------------------------------------------------------ users
@dataclass
class User:
    name: str
    role: str
    via: str = "session"         # session | key (legacy ACCESS_KEYS) | local (no accounts configured)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin" and self.via == "session"


_guard = threading.Lock()


@contextmanager
def _locked(path: Path):
    """One writer at a time, across threads and processes (several gateways share the file)."""
    with _guard:
        f = None
        if fcntl is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            f = open(path.with_suffix(".lock"), "a+")
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if f:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()


class UserStore:
    """users.json: {name: {role, pw, ver, disabled, created, by, last_login}}. Cached by mtime."""

    def __init__(self, path: Path | None = None):
        self.path = path or auth_dir() / "users.json"
        self._cache: tuple[float, dict] | None = None
        self.broken = False           # users.json exists but cannot be read: fail closed, never overwrite

    def _load(self) -> dict:
        try:
            m = self.path.stat().st_mtime
        except FileNotFoundError:
            self.broken = False
            return {}
        if self._cache and self._cache[0] == m:
            return self._cache[1]
        try:
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict):
                raise ValueError("not an object")
            self.broken = False
        except (OSError, ValueError) as e:
            if not self.broken:
                log.error("cannot read %s (%s) — logins are refused until it is fixed or removed", self.path, e)
            self.broken = True
            return {}
        self._cache = (m, data)
        return data

    def _writable(self) -> dict:
        data = self._load()
        if self.broken:
            raise AuthError(f"{self.path} cannot be read — fix or remove it first (accounts are not changed)")
        return data

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)
        self._cache = None

    def get(self, name: str) -> dict | None:
        return self._load().get((name or "").lower())

    def count(self) -> int:
        n = len(self._load())
        return max(n, 1) if self.broken else n          # a broken file still means "accounts exist"

    def list(self) -> list[dict]:
        return [{"name": n, "role": u["role"], "disabled": bool(u.get("disabled")), "created": u.get("created"),
                 "by": u.get("by"), "last_login": u.get("last_login")} for n, u in sorted(self._load().items())]

    def _admins(self, data: dict, exclude: str | None = None) -> int:
        return sum(1 for n, u in data.items() if n != exclude and u["role"] == "admin" and not u.get("disabled"))

    def create(self, name: str, password: str, role: str, by: str = "") -> dict:
        name, password = check_username(name), check_password(password)
        if role not in ROLES:
            raise AuthError("role must be admin or viewer")
        with _locked(self.path):
            data = dict(self._writable())
            if name in data:
                raise AuthError(f"user {name} already exists")
            # sid: a deleted and re-created name must not revive the old account's sessions
            data[name] = {"role": role, "pw": hash_password(password), "ver": 1, "sid": secrets.token_hex(8),
                          "disabled": False, "created": time.time(), "by": by, "last_login": None}
            self._save(data)
        log.info("user %s created (%s) by %s", name, role, by or "setup")
        return {"name": name, "role": role}

    def update(self, name: str, *, role: str | None = None, password: str | None = None,
               disabled: bool | None = None, by: str = "") -> dict:
        name = (name or "").lower()
        with _locked(self.path):
            data = {k: dict(v) for k, v in self._writable().items()}
            u = data.get(name)
            if not u:
                raise AuthError(f"no user {name}")
            if role is not None:
                if role not in ROLES:
                    raise AuthError("role must be admin or viewer")
                if role != "admin" and u["role"] == "admin" and not self._admins(data, exclude=name):
                    raise AuthError("this is the last admin — make someone else admin first")
                u["role"] = role
            if disabled is not None:
                if disabled and u["role"] == "admin" and not self._admins(data, exclude=name):
                    raise AuthError("this is the last admin — it cannot be disabled")
                if bool(disabled) != bool(u.get("disabled")):
                    u["disabled"] = bool(disabled)
                    u["ver"] = int(u.get("ver", 1)) + 1          # ends their sessions
            if password is not None:
                u["pw"] = hash_password(check_password(password))
                u["ver"] = int(u.get("ver", 1)) + 1              # ends their other sessions
            self._save(data)
        log.info("user %s updated by %s%s%s%s", name, by, f" role={role}" if role else "",
                 " password" if password else "", f" disabled={disabled}" if disabled is not None else "")
        return {"name": name, "role": u["role"], "disabled": bool(u.get("disabled"))}

    def delete(self, name: str, by: str = "") -> None:
        name = (name or "").lower()
        with _locked(self.path):
            data = dict(self._writable())
            u = data.get(name)
            if not u:
                raise AuthError(f"no user {name}")
            if u["role"] == "admin" and not self._admins(data, exclude=name):
                raise AuthError("this is the last admin — it cannot be deleted")
            del data[name]
            self._save(data)
        log.info("user %s deleted by %s", name, by)

    def authenticate(self, name: str, password: str) -> dict | None:
        u = self.get(name)
        if not u or u.get("disabled"):
            verify_password(password or "", _dummy_hash())
            return None
        if not verify_password(password or "", u["pw"]):
            return None
        return {"name": name.lower(), **u}

    def touch_login(self, name: str) -> None:
        try:
            with _locked(self.path):
                data = {k: dict(v) for k, v in self._writable().items()}
                if name in data:
                    data[name]["last_login"] = time.time()
                    self._save(data)
        except (OSError, AuthError) as e:
            log.debug("last_login: %s", e)


# ------------------------------------------------------------------ sessions
_SECRET: bytes | None = None


def secret() -> bytes:
    """Signing key: SESSION_SECRET from .env, else data/auth/secret.key (created once, readable by owner)."""
    global _SECRET
    if _SECRET is None:
        if settings.session_secret:
            _SECRET = settings.session_secret.encode()
        else:
            p = auth_dir() / "secret.key"
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                with _locked(p):
                    if not p.exists():
                        tmp = p.with_name(f"secret.{os.getpid()}.tmp")
                        tmp.write_bytes(secrets.token_bytes(32))
                        os.chmod(tmp, 0o600)
                        tmp.replace(p)
            _SECRET = p.read_bytes()
    return _SECRET


def issue(name: str, ver: int, days: float | None = None, sid: str = "") -> str:
    exp = int(time.time() + 86400 * (days if days is not None else settings.session_days))
    body = _b64(json.dumps({"u": name, "v": ver, "s": sid, "exp": exp}, separators=(",", ":")).encode())
    sig = _b64(hmac.new(secret(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def session_token(name: str, store: "UserStore") -> str:
    u = store.get(name) or {}
    return issue(name, int(u.get("ver", 1)), sid=u.get("sid", ""))


def verify(token: str | None, store: UserStore) -> User | None:
    if not token or not token.isascii() or token.count(".") != 1:
        return None
    body, sig = token.split(".")
    want = _b64(hmac.new(secret(), body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig.encode(), want.encode()):
        return None
    try:
        claims = json.loads(_unb64(body))
        if claims.get("exp", 0) < time.time():
            return None
        u = store.get(str(claims.get("u", "")))
    except (ValueError, TypeError, AttributeError):
        return None
    if not u or u.get("disabled") or int(u.get("ver", 1)) != claims.get("v") or u.get("sid", "") != claims.get("s", ""):
        return None
    return User(claims["u"], u["role"])


# ------------------------------------------------------------------ login throttling
class Throttle:
    """At most `limit` failed logins per key in `window` seconds. Keys are the client address and the
    (user name, address) pair — never the name alone, or anyone could lock any account out."""

    def __init__(self, limit: int = 10, window: float = 900):
        self.limit, self.window = limit, window
        self.fails: dict[str, list[float]] = {}

    def blocked(self, *keys: str) -> bool:
        now = time.time()
        for k in keys:
            ts = [t for t in self.fails.get(k, []) if now - t < self.window]
            self.fails[k] = ts
            if len(ts) >= self.limit:
                return True
        return False

    def fail(self, *keys: str) -> None:
        now = time.time()
        for k in keys:
            self.fails.setdefault(k, []).append(now)
        if len(self.fails) > 10_000:                     # bounded memory under a spray of names
            self.fails = {k: v for k, v in self.fails.items() if v and now - v[-1] < self.window}

    def ok(self, *keys: str) -> None:
        for k in keys:
            self.fails.pop(k, None)


def seed_admin(store: UserStore) -> None:
    """Create ADMIN_USER with ADMIN_PASSWORD from .env if that user does not exist yet (never overwritten)."""
    if not (settings.admin_user and settings.admin_password):
        return
    try:
        name = check_username(settings.admin_user)
        if store.get(name) is None:
            store.create(name, settings.admin_password, "admin", by=".env")
    except AuthError as e:
        log.error("ADMIN_USER / ADMIN_PASSWORD in .env: %s", e)
