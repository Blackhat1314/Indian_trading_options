"""Accounts, login sessions and the admin/user endpoints."""
import asyncio
import json

import pytest

from td import auth
from td.auth import AuthError, Throttle, UserStore, hash_password, issue, verify, verify_password

fakeredis = pytest.importorskip("fakeredis")
H = {"X-TD-Request": "1"}


@pytest.fixture(autouse=True)
def _secret(monkeypatch, tmp_path):
    from td.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "session_secret", "")
    monkeypatch.setattr(settings, "access_keys", "")
    monkeypatch.setattr(settings, "admin_user", "")
    monkeypatch.setattr(settings, "admin_password", "")
    monkeypatch.setattr(auth, "_SECRET", None)


def test_password_hashing():
    h = hash_password("correct horse")
    assert h.startswith("scrypt$") and verify_password("correct horse", h)
    assert not verify_password("wrong horse", h) and not verify_password("x", "garbage")
    assert hash_password("same") != hash_password("same")                  # salted


def test_store_rules(tmp_path):
    st = UserStore(tmp_path / "users.json")
    st.create("Alice", "password1", "admin")
    assert st.get("alice")["role"] == "admin" and st.count() == 1
    for bad in (("a", "password1", "viewer"), ("bob", "short", "viewer"), ("bob", "password1", "root"),
                ("alice", "password1", "viewer"), ("bad name", "password1", "viewer")):
        with pytest.raises(AuthError):
            st.create(*bad)
    with pytest.raises(AuthError):
        st.update("alice", role="viewer")                                    # last admin
    with pytest.raises(AuthError):
        st.update("alice", disabled=True)
    with pytest.raises(AuthError):
        st.delete("alice")
    st.create("bob", "password2", "viewer")
    st.update("bob", role="admin")
    st.update("alice", role="viewer")                                        # fine now: bob is admin
    assert st.authenticate("bob", "password2")["role"] == "admin"
    assert st.authenticate("bob", "nope") is None and st.authenticate("nobody", "password2") is None
    v = st.get("alice")["ver"]
    st.update("alice", password="newpassword")
    assert st.get("alice")["ver"] == v + 1 and st.authenticate("alice", "newpassword")
    st.update("alice", disabled=True)
    assert st.authenticate("alice", "newpassword") is None
    st.delete("alice")
    assert [u["name"] for u in st.list()] == ["bob"]
    assert (tmp_path / "users.json").stat().st_mode & 0o077 == 0              # owner-only file


def test_tokens(tmp_path):
    st = UserStore(tmp_path / "users.json")
    st.create("alice", "password1", "admin")
    tok = auth.session_token("alice", st)
    u = verify(tok, st)
    assert u and u.name == "alice" and u.is_admin
    body, sig = tok.split(".")
    assert verify(body + "." + sig[:-2] + "AA", st) is None                    # tampered signature
    sid = st.get("alice")["sid"]
    assert verify(issue("alice", 1, days=-1, sid=sid), st) is None           # expired
    assert verify(issue("alice", 1), st) is None                             # not bound to this account
    st.update("alice", password="password2")
    assert verify(tok, st) is None                                           # password change revokes
    assert verify(issue("ghost", 1), st) is None and verify("junk", st) is None and verify(None, st) is None


def test_throttle():
    t = Throttle(limit=3, window=60)
    for _ in range(3):
        assert not t.blocked("ip:1", "user:a")
        t.fail("ip:1", "user:a")
    assert t.blocked("ip:1") and t.blocked("user:a") and not t.blocked("ip:2")
    t.ok("user:a")
    assert not t.blocked("user:a") and t.blocked("ip:1")


def test_seed_admin(monkeypatch, tmp_path):
    from td.config import settings
    st = UserStore(tmp_path / "users.json")
    monkeypatch.setattr(settings, "admin_user", "Boss")
    monkeypatch.setattr(settings, "admin_password", "bosspassword")
    auth.seed_admin(st)
    assert st.get("boss")["role"] == "admin"
    monkeypatch.setattr(settings, "admin_password", "changed-in-env")
    auth.seed_admin(st)                                                      # never overwrites an existing user
    assert st.authenticate("boss", "bosspassword")


def _client(monkeypatch, tmp_path, host="127.0.0.1"):
    from fastapi.testclient import TestClient
    from td.services import gateway as gw
    r = fakeredis.aioredis.FakeRedis()
    monkeypatch.setattr(gw.bus, "connect", lambda *a, **k: r)
    monkeypatch.setattr(gw, "Lake", lambda *a, **k: None)
    monkeypatch.setattr(gw, "users", UserStore(tmp_path / "users.json"))
    monkeypatch.setattr(gw, "throttle", Throttle())
    asyncio.run(r.set(gw.bus.KEY_UNIVERSE, json.dumps([])))
    base = "http://127.0.0.1:8000" if host == "127.0.0.1" else "http://testserver"
    return gw, TestClient(gw.app, client=(host, 5000), base_url=base)


def test_local_mode_then_setup_then_login(monkeypatch, tmp_path):
    gw, c = _client(monkeypatch, tmp_path)
    with c:
        me = c.get("/api/auth/me").json()
        assert me["user"] == "local" and not me["login_required"] and me["setup"] and me["can_setup"]
        assert c.get("/api/universe").status_code == 200                     # no accounts yet: open as before
        assert c.get("/api/admin/users").status_code == 403                  # but never the admin page
        assert c.post("/api/auth/setup", json={"username": "boss", "password": "bosspassword"}).status_code == 403
        r = c.post("/api/auth/setup", json={"username": "boss", "password": "bosspassword"}, headers=H)
        assert r.status_code == 200 and r.json()["role"] == "admin" and "td_session" in r.cookies
        assert c.post("/api/auth/setup", json={"username": "x2", "password": "bosspassword"}, headers=H).status_code == 403
        assert c.get("/api/auth/me").json()["admin"] is True
        assert c.get("/api/admin/users").json()["users"][0]["name"] == "boss"
        c.post("/api/auth/logout", headers=H)
        c.cookies.clear()
        assert c.get("/api/universe").status_code == 401                     # accounts exist: login required
        assert c.get("/api/auth/me").json()["login_required"] is True
        assert c.post("/api/auth/login", json={"username": "boss", "password": "nope"}, headers=H).status_code == 401
        r = c.post("/api/auth/login", json={"username": "BOSS", "password": "bosspassword"}, headers=H)
        assert r.status_code == 200 and c.get("/api/universe").status_code == 200


def test_setup_is_local_only(monkeypatch, tmp_path):
    gw, c = _client(monkeypatch, tmp_path, host="10.0.0.9")
    with c:
        assert c.get("/api/auth/me").json()["can_setup"] is False
        assert c.post("/api/auth/setup", json={"username": "boss", "password": "bosspassword"}, headers=H).status_code == 403
    gw, c = _client(monkeypatch, tmp_path, host="127.0.0.1")       # a reverse proxy / tunnel on the same host
    with c:
        fwd = {**H, "X-Forwarded-For": "203.0.113.9"}
        assert c.get("/api/auth/me", headers=fwd).json()["can_setup"] is False
        assert c.post("/api/auth/setup", json={"username": "boss", "password": "bosspassword"}, headers=fwd).status_code == 403


def test_roles_users_admin_and_revocation(monkeypatch, tmp_path):
    gw, c = _client(monkeypatch, tmp_path)
    gw.users.create("boss", "bosspassword", "admin")
    with c:
        c.post("/api/auth/login", json={"username": "boss", "password": "bosspassword"}, headers=H)
        assert c.post("/api/admin/users", json={"username": "ravi", "password": "ravipassword"}).status_code == 403   # CSRF header
        assert c.post("/api/admin/users", json={"username": "ravi", "password": "ravipassword"}, headers=H).json()["role"] == "viewer"
        assert c.post("/api/admin/users", json={"username": "ravi", "password": "x"}, headers=H).status_code == 400
        assert c.delete("/api/admin/users/boss", headers=H).status_code == 400            # yourself
        assert c.patch("/api/admin/users/boss", json={"role": "viewer"}, headers=H).status_code == 400
        assert c.patch("/api/admin/users/ravi", json={"disabled": True}, headers=H).json()["disabled"] is True
        assert c.patch("/api/admin/users/ravi", json={"disabled": False, "password": "ravipass2"}, headers=H).status_code == 200
        boss_cookie = c.cookies.get("td_session")
        # a viewer: terminal yes, admin no
        c.cookies.clear()
        assert c.post("/api/auth/login", json={"username": "ravi", "password": "ravipass2"}, headers=H).status_code == 200
        assert c.get("/api/universe").status_code == 200 and c.get("/api/meta").json()["role"] == "viewer"
        for path in ("/api/admin/users", "/api/admin/backfill", "/api/system", "/api/lake/coverage"):
            assert c.get(path).status_code == 403, path
        # own password change revokes the viewer's old session, renews this one
        old = c.cookies.get("td_session")
        assert c.post("/api/auth/password", json={"old": "wrong", "new": "ravipass3"}, headers=H).status_code == 401
        assert c.post("/api/auth/password", json={"old": "ravipass2", "new": "ravipass3"}, headers=H).status_code == 200
        assert c.get("/api/universe").status_code == 200
        c.cookies.clear()
        c.cookies.set("td_session", old)
        assert c.get("/api/universe").status_code == 401
        # admin deletes the viewer
        c.cookies.clear()
        c.cookies.set("td_session", boss_cookie)
        assert c.delete("/api/admin/users/ravi", headers=H).json()["ok"]
        assert [u["name"] for u in c.get("/api/admin/users").json()["users"]] == ["boss"]


def test_access_keys_still_work_but_not_for_admin(monkeypatch, tmp_path):
    from td.config import settings
    gw, c = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "access_keys", "friend:k1")
    with c:
        assert c.get("/api/universe").status_code == 401
        assert c.get("/api/universe", headers={"X-Access-Key": "k1"}).status_code == 200
        assert c.get("/api/universe?key=k1").status_code == 200
        assert c.get("/api/admin/backfill", headers={"X-Access-Key": "k1"}).status_code == 403
        me = c.get("/api/auth/me?key=k1").json()
        assert me["user"] == "friend" and me["via"] == "key" and not me["admin"]


def test_login_throttled(monkeypatch, tmp_path):
    gw, c = _client(monkeypatch, tmp_path)
    gw.users.create("boss", "bosspassword", "admin")
    with c:
        for _ in range(10):
            c.post("/api/auth/login", json={"username": "boss", "password": "wrongpass"}, headers=H)
        r = c.post("/api/auth/login", json={"username": "boss", "password": "bosspassword"}, headers=H)
        assert r.status_code == 429


def test_websocket_needs_a_session(monkeypatch, tmp_path):
    from starlette.websockets import WebSocketDisconnect
    gw, c = _client(monkeypatch, tmp_path, host="10.0.0.5")     # cookie jar and ws share testserver
    gw.users.create("boss", "bosspassword", "admin")
    with c:
        with c.websocket_connect("/ws") as ws:
            with pytest.raises(WebSocketDisconnect) as e:
                ws.receive_text()
            assert e.value.code == 4401
        c.post("/api/auth/login", json={"username": "boss", "password": "bosspassword"}, headers=H)
        with c.websocket_connect("/ws") as ws:
            hello = json.loads(ws.receive_text())
            assert hello["topic"] == "hello" and hello["data"]["user"] == "boss"
        with c.websocket_connect("/ws", headers={"Origin": "https://evil.example"}) as ws:      # cross-site socket
            with pytest.raises(WebSocketDisconnect) as e:
                ws.receive_text()
            assert e.value.code == 4403
        proxied = {"Origin": "https://desk.example.com", "X-Forwarded-Host": "desk.example.com"}
        with c.websocket_connect("/ws", headers=proxied) as ws:                               # behind a proxy
            assert json.loads(ws.receive_text())["topic"] == "hello"


def test_admin_backfill_endpoints(monkeypatch, tmp_path):
    gw, c = _client(monkeypatch, tmp_path)
    gw.users.create("boss", "bosspassword", "admin")
    with c:
        c.post("/api/auth/login", json={"username": "boss", "password": "bosspassword"}, headers=H)
        assert c.get("/api/admin/backfill").json()["status"] is None
        assert c.post("/api/admin/backfill/start", json={"intervals": ["X"]}, headers=H).status_code == 400
        assert c.post("/api/admin/backfill/start", json={"intervals": ["D"], "years": 2}, headers=H).json()["ok"]
        cmd = json.loads(asyncio.run(gw.app.state.redis.rpop(gw.bus.KEY_ADMIN_CMD)))
        assert cmd["op"] == "start" and cmd["job"]["years"] == 2 and cmd["by"] == "boss"
        assert c.post("/api/admin/backfill/stop").status_code == 403                    # CSRF header required
        assert c.post("/api/admin/backfill/stop", headers=H).json()["ok"]


def test_static_files_never_leave_dist(monkeypatch, tmp_path):
    gw, c = _client(monkeypatch, tmp_path)
    if not gw.FRONTEND.exists():
        pytest.skip("frontend not built")
    with c:
        for path in ("/..%2f..%2f.env", "/..%2f..%2fdata%2fauth%2fsecret.key", "/%2e%2e/%2e%2e/README.md"):
            r = c.get(path)
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/html"), path   # the app shell
            assert b"DHAN_ACCESS_TOKEN" not in r.content and b"TradeDesk" in r.content


def test_csrf_header_is_not_allowed_cross_origin(monkeypatch, tmp_path):
    gw, c = _client(monkeypatch, tmp_path)
    with c:
        pre = c.options("/api/auth/setup", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST",
                                                     "Access-Control-Request-Headers": "content-type,x-td-request"})
        assert "x-td-request" not in pre.headers.get("access-control-allow-headers", "").lower()
        # DNS rebinding: same-origin for the attacker's page, but its Host is not localhost
        rebind = {**H, "Host": "evil.example:8000"}
        assert c.post("/api/auth/setup", json={"username": "evil", "password": "evilpassword"}, headers=rebind).status_code == 403
        assert gw.users.count() == 0


def test_unreadable_users_file_fails_closed(monkeypatch, tmp_path):
    gw, c = _client(monkeypatch, tmp_path)
    gw.users.create("boss", "bosspassword", "admin")
    raw = gw.users.path.read_text()
    gw.users.path.write_text(raw + ",")                                     # a bad hand edit
    with c:
        me = c.get("/api/auth/me").json()
        assert me["login_required"] and not me["setup"] and c.get("/api/universe").status_code == 401
        assert c.post("/api/auth/login", json={"username": "boss", "password": "bosspassword"}, headers=H).status_code == 401
    with pytest.raises(AuthError):
        gw.users.create("x2", "password1", "admin")
    assert gw.users.path.read_text() == raw + ","                           # never overwritten


def test_recreated_name_does_not_revive_old_sessions(tmp_path):
    st = UserStore(tmp_path / "users.json")
    st.create("boss", "bosspassword", "admin")
    st.create("sam", "sampassword", "viewer")
    old = auth.session_token("sam", st)
    assert verify(old, st)
    st.delete("sam")
    st.create("sam", "otherpassword", "admin")
    assert verify(old, st) is None and verify(auth.session_token("sam", st), st).is_admin


def test_reserved_names_and_garbage_cookies(monkeypatch, tmp_path):
    with pytest.raises(AuthError):
        UserStore(tmp_path / "u.json").create("local", "password1", "admin")
    st = UserStore(tmp_path / "users.json")
    assert verify("abc.été", st) is None and verify("a.b.c", st) is None and verify("x." + "=" * 9, st) is None
    gw, c = _client(monkeypatch, tmp_path)
    with c:
        c.cookies.set("td_session", "e30.%%%")
        assert c.get("/api/auth/me").status_code == 200


def test_throttle_never_locks_a_name_for_everyone(monkeypatch, tmp_path):
    gw, c = _client(monkeypatch, tmp_path)
    gw.users.create("boss", "bosspassword", "admin")
    with c:
        attacker = {**H, "X-Forwarded-For": "198.51.100.7"}                 # behind the local proxy
        for _ in range(10):
            c.post("/api/auth/login", json={"username": "boss", "password": "wrongpass"}, headers=attacker)
        assert c.post("/api/auth/login", json={"username": "boss", "password": "bosspassword"}, headers=attacker).status_code == 429
        owner = {**H, "X-Forwarded-For": "203.0.113.5"}
        assert c.post("/api/auth/login", json={"username": "boss", "password": "bosspassword"}, headers=owner).status_code == 200


def test_disabled_user_loses_the_live_feed(monkeypatch, tmp_path):
    from starlette.websockets import WebSocketDisconnect
    gw, c = _client(monkeypatch, tmp_path, host="10.0.0.5")     # cookie jar and ws share testserver
    gw.users.create("boss", "bosspassword", "admin")
    gw.users.create("ravi", "ravipassword", "viewer")
    with c:
        c.post("/api/auth/login", json={"username": "ravi", "password": "ravipassword"}, headers=H)
        with c.websocket_connect("/ws") as ws:
            assert json.loads(ws.receive_text())["topic"] == "hello"
            gw.users.update("ravi", disabled=True, by="boss")
            with pytest.raises(WebSocketDisconnect) as e:                    # closed by the 3 s status loop
                while True:
                    ws.receive_text()
            assert e.value.code == 4401
