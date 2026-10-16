"""REST API of the PAPER TRADE desk: /api/paper/...

Each logged-in user has their own paper account. The paper service (td/services/paper.py) is the only writer of
account files while it runs: settings and resets are queued to it on Redis and the gateway waits briefly for
the updated state. When the service is not running the gateway applies them to the account file itself
(nothing else is writing it then), so the desk stays usable for setting up an account.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Callable, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from .. import bus
from ..paper.backtest import report_dir
from ..paper.broker import Account, validate_settings
from .paper import K_CMD, state_key

SAFE_USER = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SHIPPED_EVIDENCE = Path(__file__).resolve().parents[1] / "paper" / "pretrained"


class IntradayIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool | None = None
    capital: float | None = None
    leverage: float | None = None


class SwingIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool | None = None
    capital: float | None = None


class SettingsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intraday: IntradayIn | None = None
    swing: SwingIn | None = None


class ResetIn(BaseModel):
    strategy: Literal["intraday", "swing", "all"]


def _user(who) -> str:
    name = getattr(who, "name", str(who))
    if not SAFE_USER.match(name):
        raise HTTPException(400, "this login name cannot hold a paper account")
    return name


async def _service_up(r) -> bool:
    raw = await r.get(bus.k_status("paper"))
    st = bus.loads(raw) if raw else None
    return bool(st) and time.time() - float(st.get("updated", 0)) < 20


def _offline_view(user: str) -> dict:
    acct = Account(user)
    svc = {"running": False, "clock": None, "phase": None, "mode": "live", "models": {"source": "none", "trained_to": None},
           "note": "The paper service is not running: settings are saved, trades start once it runs "
                   "(run.py / docker compose start it)."}
    return acct.view({}, svc, None)


def make_router(current: Callable, csrf: Callable) -> APIRouter:
    router = APIRouter(prefix="/api/paper")

    async def state_of(request: Request, user: str) -> dict:
        r = request.app.state.redis
        raw = await r.get(state_key(user))
        if raw and await _service_up(r):
            return json.loads(raw)
        return await asyncio.to_thread(_offline_view, user)

    async def send(request: Request, user: str, cmd: dict, check) -> dict:
        r = request.app.state.redis
        if not await _service_up(r):
            def apply():
                acct = Account(user)
                if cmd["kind"] == "settings":
                    acct.update_settings(cmd["patch"])
                else:
                    acct.reset(cmd["strategy"])
                acct.save()
            await asyncio.to_thread(apply)
            return await state_of(request, user)
        before = await r.get(state_key(user))
        await r.rpush(K_CMD, json.dumps({"user": user, **cmd}))
        for _ in range(30):                                     # the service polls every second
            await asyncio.sleep(0.2)
            raw = await r.get(state_key(user))
            if raw and raw != before:
                st = json.loads(raw)
                if check(st):
                    return st
        return await state_of(request, user)

    @router.get("/state")
    async def get_state(request: Request, who=Depends(current)):
        return await state_of(request, _user(who))

    @router.post("/settings", dependencies=[Depends(csrf)])
    async def post_settings(body: SettingsIn, request: Request, who=Depends(current)):
        user = _user(who)
        patch = body.model_dump(exclude_none=True)
        if not patch:
            raise HTTPException(422, "nothing to change")
        cur = (await state_of(request, user))["settings"]
        try:
            want = validate_settings(cur, patch)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return await send(request, user, {"kind": "settings", "patch": patch}, lambda st: st.get("settings") == want)

    @router.post("/reset", dependencies=[Depends(csrf)])
    async def post_reset(body: ResetIn, request: Request, who=Depends(current)):
        user = _user(who)

        def done(st: dict) -> bool:
            strats = ("intraday", "swing") if body.strategy == "all" else (body.strategy,)
            return all(not st[s]["positions"] and st[s]["stats"]["trades"] == 0 for s in strats)
        return await send(request, user, {"kind": "reset", "strategy": body.strategy}, done)

    @router.get("/book")
    async def get_book(request: Request, strategy: Literal["intraday", "swing", "all"] = "all",
                       limit: int = Query(200, ge=1, le=2000), who=Depends(current)):
        user = _user(who)
        return {"trades": await asyncio.to_thread(lambda: Account(user).book(strategy, limit))}

    @router.get("/signals")
    async def get_signals(request: Request, who=Depends(current)):
        raw = await request.app.state.redis.get(bus.SNAP_PREFIX + "paper")
        data = (bus.loads(raw) or {}).get("data") if raw else None
        return data or {"date": None, "intraday": {"decisions": []}, "swing": {"asof": None, "regime": None, "candidates": []}}

    @router.get("/evidence")
    async def get_evidence(who=Depends(current)):
        def read(kind: str):
            for d in (report_dir(), SHIPPED_EVIDENCE):
                p = d / f"{kind}.json" if d == report_dir() else d / f"evidence_{kind}.json"
                if p.exists():
                    try:
                        rep = json.loads(p.read_text())
                        rep.pop("trades", None)
                        rep["source"] = "local" if d == report_dir() else "shipped"
                        return rep
                    except ValueError:
                        continue
            return None
        return await asyncio.to_thread(lambda: {"intraday": read("intraday"), "swing": read("swing")})

    return router
