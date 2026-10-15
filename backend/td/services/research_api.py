"""REST API of the research desks (AI analyst, Screener, Monte Carlo, Quant): /api/research/...

One `Research` container per app holds the services: ResearchData (history, symbols, live context), the
Screener.in client (shared with the AI analyst so both read the same fundamentals and cache), the quant model,
the AI analyst and a semaphore that lets at most two CPU-heavy simulations run at once (they run in worker
threads; the event loop keeps serving the live desks). The model modules are imported on first use, so a broken
or missing research module costs only its own endpoints (503), never the gateway.

Errors are for people: an unknown symbol is a 404 that says so, too little history a 422 with the reason, an
unreachable Screener.in a 502 — and validation errors come back as one readable sentence, not a list.
"""
from __future__ import annotations

import asyncio
import logging
import math
import zlib
from contextlib import asynccontextmanager
from typing import Annotated, Any, Callable, Literal

import orjson
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from ..config import settings
from ..research.data import ResearchData, bar_date

log = logging.getLogger("td.research.api")

SYMBOL_PATTERN = r"^[A-Za-z0-9&._-]{1,40}$"
JOB_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
Symbol = Annotated[str, Path(pattern=SYMBOL_PATTERN, description="NSE trading symbol, e.g. RELIANCE or M&M")]
McModel = Literal["ensemble", "garch", "bootstrap", "jump", "regime", "gbm"]
McDrift = Literal["capm", "historical", "zero", "quant"]
Target = Annotated[float, Field(gt=0, lt=1e8, allow_inf_nan=False)]
QUANT_HORIZONS = (1, 5, 21)
NO_EDGE_AUC = 0.52


# ------------------------------------------------------------------ request bodies
class McRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon_days: int = Field(252, ge=5, le=756)
    paths: int = Field(10_000, ge=500, le=50_000)
    model: McModel = "ensemble"
    drift: McDrift = "capm"
    lookback_years: float = Field(5.0, ge=1.0, le=10.0, allow_inf_nan=False)
    targets: list[Target] = Field(default_factory=list, max_length=6)
    seed: int | None = Field(None, ge=0, le=2**32 - 1)


class McBacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon_days: int = Field(63, ge=5, le=756)
    model: McModel = "ensemble"
    drift: McDrift = "capm"
    lookback_years: float = Field(3.0, ge=1.0, le=10.0, allow_inf_nan=False)
    origins: int = Field(40, ge=10, le=80)
    paths: int = Field(2_000, ge=500, le=5_000)


class AiStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    refresh: bool = False


# ------------------------------------------------------------------ plumbing
class _JSON(Response):
    """orjson rendering (numpy-aware, NaN -> null), like the gateway's FastJSON: the model modules return numpy
    scalars that FastAPI's own encoder rejects."""
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS)


def _readable(errors: list[dict]) -> str:
    parts = []
    for e in errors:
        loc = [str(x) for x in e.get("loc", ()) if x not in ("body", "path", "query")]
        parts.append(f"{'.'.join(loc) or 'request'}: {e.get('msg', 'invalid')}")
    return "; ".join(parts) or "invalid request"


class _Route(APIRoute):
    """Validation errors as one sentence in `detail` (the terminal shows `detail` as the error message)."""

    def get_route_handler(self) -> Callable:
        handler = super().get_route_handler()

        async def run(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as e:
                return _JSON({"detail": _readable(e.errors())}, status_code=422)
        return run


class Unavailable(RuntimeError):
    """A research module failed to import or start."""


def _msg(e: BaseException) -> str:
    if isinstance(e, KeyError) and e.args:
        return str(e.args[0])
    return str(e) or type(e).__name__


def _fail(what: str, e: Exception, missing: bool = True) -> HTTPException:
    """Expected failures -> 404 / 422 / 502 / 503 with the reason; anything else is logged and becomes a 500
    that still says what failed. `missing`: a KeyError means "no such symbol" (services that look one up);
    elsewhere it is a bug."""
    if isinstance(e, HTTPException):
        return e
    if isinstance(e, KeyError) and missing:
        return HTTPException(404, _msg(e))
    if isinstance(e, Unavailable):
        return HTTPException(503, _msg(e))
    if type(e).__name__ == "ScreenerError":
        return HTTPException(502, f"Screener.in: {_msg(e)}")
    if isinstance(e, ValueError):                    # e.g. too little history for the requested lookback
        log.info("%s rejected: %s", what, e)
        return HTTPException(422, _msg(e))
    if isinstance(e, asyncio.TimeoutError):
        return HTTPException(504, f"{what} timed out")
    log.exception("%s failed", what)
    return HTTPException(500, f"{what} failed: {type(e).__name__}: {_msg(e)}")


def _dedupe(items: list) -> list[str]:
    return list(dict.fromkeys(str(x) for x in items if x))


# ------------------------------------------------------------------ the container
class Research:
    """Services of the research desks for one app (one Redis connection, one lake). Each service can be
    injected (tests); the rest are built on first use."""

    def __init__(self, redis, lake=None, *, data: ResearchData | None = None, screener=None, quant=None,
                 ai=None, mc=None, cpu_slots: int = 2):
        self.redis = redis
        self.data = data or ResearchData(redis, lake)
        self.cpu = asyncio.Semaphore(cpu_slots)
        self._screener, self._quant, self._ai, self._mc = screener, quant, ai, mc
        self.tasks: set[asyncio.Task] = set()

    def _lazy(self, attr: str, what: str, build: Callable[[], Any]) -> Any:
        obj = getattr(self, attr)
        if obj is None:
            try:
                obj = build()
            except Exception as e:
                log.exception("research: %s unavailable", what)
                raise Unavailable(f"the {what} is unavailable on this server ({type(e).__name__}: {e})") from e
            setattr(self, attr, obj)
        return obj

    @property
    def mc(self):
        def build():
            from ..research import montecarlo
            return montecarlo
        return self._lazy("_mc", "Monte Carlo engine", build)

    @property
    def screener(self):
        def build():
            from ..research.screener import ScreenerClient
            return ScreenerClient()
        return self._lazy("_screener", "Screener.in client", build)

    @property
    def quant(self):
        def build():
            from ..research.quant import QuantService
            return QuantService(self.data)
        return self._lazy("_quant", "quant model", build)

    @property
    def ai(self):
        def build():
            from ..research.ai import AiService
            screener = quant = None
            try:
                screener = self.screener
            except Unavailable:
                pass
            try:
                quant = self.quant
            except Unavailable:
                pass
            return AiService(self.data, screener, quant, redis=self.redis, mc=self._mc)
        return self._lazy("_ai", "AI analyst", build)

    async def aclose(self) -> None:
        """App shutdown: close the history layer's shared Dhan client (built again on the next top-up)."""
        close = getattr(self.data, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception as e:
                log.warning("research: closing the Dhan client failed: %s", e)

    def background(self, coro, what: str) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.tasks.add(task)

        def done(t: asyncio.Task) -> None:
            self.tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                log.error("%s failed: %s", what, t.exception(), exc_info=t.exception())
        task.add_done_callback(done)
        return task


def research(request: Request) -> Research:
    """The app's container, rebuilt when the app's Redis connection changes (a new lifespan)."""
    app = request.app
    rs = getattr(app.state, "research", None)
    redis = app.state.redis
    if rs is None or rs.redis is not redis:
        rs = Research(redis, getattr(app.state, "lake", None))
        app.state.research = rs
    return rs


State = Annotated[Research, Depends(research)]      # module level: FastAPI resolves string annotations here


def default_seed(symbol: str, last_bar_ts: int) -> int:
    """Same symbol and same data -> same paths (results are reproducible until a new bar arrives)."""
    return zlib.crc32(f"{symbol}:{bar_date(last_bar_ts).isoformat()}".encode())


def quant_drift(result: dict | None) -> tuple[float | None, str | None]:
    """Annual arithmetic drift implied by the quant model's 1-month view, or (None, why not).

    The model gives the expected 21-day log return m and its sd s. Twelve such months compound to an expected
    arithmetic annual return exp(12·(m + s²/2)) − 1 — the s²/2 term turns a log-return view into an arithmetic
    one, so p_up = 0.5 (m = 0, a flat median) maps to the small positive mean a flat median implies."""
    if not isinstance(result, dict):
        return None, "quant model result unavailable — CAPM drift used instead"
    if result.get("status") != "ready":
        return None, f"quant model is {result.get('status') or 'not ready'} — CAPM drift used instead"
    h21 = next((h for h in result.get("horizons") or [] if isinstance(h, dict) and h.get("h") == 21), None)
    try:
        m, s = float(h21["exp_ret"]), float(h21.get("ret_sd") or 0.0)
    except (TypeError, KeyError, ValueError):
        return None, "quant model has no 1-month forecast for this stock — CAPM drift used instead"
    if not (math.isfinite(m) and math.isfinite(s)):
        return None, "quant model's 1-month forecast is not a number — CAPM drift used instead"
    drift = math.expm1(min(5.0, max(-5.0, 12.0 * (m + 0.5 * s * s))))
    if h21.get("edge") is False:
        # without a measured edge the 1-month "view" is the training base rate, the same tilt for every stock
        why = h21.get("edge_note") or "walk-forward AUC too close to 0.5"
        return None, f"the quant model has no reliable 1-month edge ({why}) — CAPM drift used instead"
    auc = h21.get("oos_auc")
    if not isinstance(auc, (int, float)) or not math.isfinite(auc):
        return drift, "the quant model's 1-month edge has not been measured — treat its drift as unproven"
    if auc < NO_EDGE_AUC:
        return drift, (f"the quant model shows no reliable 1-month edge (walk-forward AUC {auc:.2f}) — its drift "
                       "tilt is close to noise")
    return drift, None


# ------------------------------------------------------------------ router
@asynccontextmanager
async def _lifespan(app):
    """Nested inside the app's own lifespan (FastAPI merges a router's), so Redis is still open here."""
    yield
    rs = getattr(app.state, "research", None)
    if rs is not None:
        await rs.aclose()


def make_router(current: Callable, admin: Callable, csrf: Callable) -> APIRouter:
    router = APIRouter(prefix="/api/research", route_class=_Route, lifespan=_lifespan)
    write = [Depends(csrf)]

    async def resolve(rs: Research, symbol: str):
        try:
            return await rs.data.resolve(symbol.upper())
        except KeyError as e:
            raise HTTPException(404, _msg(e)) from e

    async def history_for_model(rs: Research, symbol: str, what: str):
        info = await resolve(rs, symbol)
        df, meta = await rs.data.daily(info.symbol, fetch=True)
        if df.empty:
            raise HTTPException(422, f"{info.symbol}: no daily price history to run the {what} on"
                                + (f" ({meta['warning']})" if meta.get("warning") else ""))
        try:
            index_df = await rs.data.index_daily("NIFTY")
        except Exception as e:                       # beta falls back to 1 with a warning inside the model
            log.info("NIFTY history unavailable: %s", e)
            index_df = None
        if index_df is not None and index_df.empty:
            index_df = None
        return info, df, meta, index_df

    # ---- symbols & history
    @router.get("/search")
    async def search(rs: State, q: Annotated[str, Query(max_length=64)] = "",
                     limit: Annotated[int, Query(ge=1, le=50)] = 12, who=Depends(current)):
        try:
            return _JSON(await rs.data.search(q, limit))
        except Exception as e:
            raise _fail("symbol search", e) from e

    @router.get("/history/{symbol}")
    async def history(symbol: Symbol, rs: State, fetch: bool = True, who=Depends(current)):
        try:
            return _JSON(await rs.data.history_info(symbol.upper(), fetch=fetch))
        except Exception as e:
            raise _fail("history", e) from e

    # ---- Monte Carlo
    @router.post("/montecarlo/{symbol}", dependencies=write)
    async def montecarlo(symbol: Symbol, body: McRequest, rs: State, who=Depends(current)):
        try:
            mc = rs.mc
            info, df, meta, index_df = await history_for_model(rs, symbol, "simulation")
            if info.symbol == "INDIAVIX":
                raise HTTPException(422, "India VIX is a volatility index, not a traded price — simulate NIFTY "
                                         "or a stock instead")
            spot, source = await rs.data.spot(info.symbol)
            if spot is None or source not in ("live", "close"):
                spot, source = float(df["close"].iat[-1]), "close"
            seed = body.seed if body.seed is not None else default_seed(info.symbol, int(df["ts"].iat[-1]))
            notes: list[str] = []
            qd = None
            if body.drift == "quant":
                try:
                    qd, note = quant_drift(await asyncio.wait_for(rs.quant.result(info.symbol), 90))
                except Exception as e:
                    qd, note = None, f"quant model unavailable ({_msg(e)}) — CAPM drift used instead"
                notes.append(note)
            async with rs.cpu:
                res = await asyncio.to_thread(
                    mc.simulate, df, horizon_days=body.horizon_days, paths=body.paths, model=body.model,
                    drift=body.drift, lookback_years=body.lookback_years, targets=list(body.targets), seed=seed,
                    spot=float(spot), spot_source=source, index_df=index_df, rf=settings.risk_free_rate,
                    erp=settings.equity_risk_premium, quant_drift=qd)
            out = {"symbol": info.symbol, **res}
            out["warnings"] = _dedupe(meta.get("warnings", []) + notes + list(res.get("warnings") or []))
            return _JSON(out)
        except Exception as e:
            raise _fail("Monte Carlo simulation", e, missing=False) from e

    @router.post("/montecarlo/{symbol}/backtest", dependencies=write)
    async def montecarlo_backtest(symbol: Symbol, body: McBacktestRequest, rs: State, who=Depends(current)):
        try:
            mc = rs.mc
            info, df, meta, index_df = await history_for_model(rs, symbol, "calibration backtest")
            notes: list[str] = []
            drift = body.drift
            if drift == "quant":
                # the quant model is trained on the whole history, so replaying its drift at past dates would
                # leak the future into the test — calibrate the same model with CAPM drift instead
                drift = "capm"
                notes.append("quant drift cannot be backtested without look-ahead (the model is trained on the "
                             "whole history) — calibration uses CAPM drift")
            seed = default_seed(info.symbol, int(df["ts"].iat[-1]))
            async with rs.cpu:
                res = await asyncio.to_thread(
                    mc.backtest, df, horizon_days=body.horizon_days, model=body.model, drift=drift,
                    lookback_years=body.lookback_years, origins=body.origins, paths=body.paths, index_df=index_df,
                    rf=settings.risk_free_rate, erp=settings.equity_risk_premium, seed=seed)
            out = {"symbol": info.symbol, **res}
            out["warnings"] = _dedupe(meta.get("warnings", []) + notes + list(res.get("warnings") or []))
            return _JSON(out)
        except Exception as e:
            raise _fail("Monte Carlo backtest", e, missing=False) from e

    # ---- quant model
    @router.get("/quant/{symbol}")
    async def quant(symbol: Symbol, rs: State, who=Depends(current)):
        try:
            q = rs.quant
            info = await resolve(rs, symbol)
            return _JSON(await q.result(info.symbol))
        except Exception as e:
            raise _fail("quant model", e) from e

    @router.get("/quant-model")
    async def quant_model(rs: State, who=Depends(current)):
        try:
            return _JSON(await rs.quant.info())
        except Exception as e:
            raise _fail("quant model info", e) from e

    @router.post("/quant-model/train", dependencies=write)
    async def quant_train(rs: State, who=Depends(admin)):
        try:
            q = rs.quant
            if not any(not t.done() for t in rs.tasks if t.get_name() == "quant-train"):
                rs.background(q.train(force=True), "quant training").set_name("quant-train")
                await asyncio.sleep(0)                   # let it register as "training" before we answer
            log.info("quant model retrain requested by %s", getattr(who, "name", who))
            return _JSON(await q.info())
        except Exception as e:
            raise _fail("quant training", e) from e

    @router.get("/quant-leaderboard")
    async def quant_leaderboard(rs: State, h: int = 21, n: Annotated[int, Query(ge=1, le=50)] = 15,
                                who=Depends(current)):
        if h not in QUANT_HORIZONS:
            raise HTTPException(422, f"h must be one of {', '.join(map(str, QUANT_HORIZONS))} (trading days)")
        try:
            return _JSON(await rs.quant.leaderboard(h, n))
        except Exception as e:
            raise _fail("quant leaderboard", e) from e

    # ---- Screener.in fundamentals
    @router.get("/screener/{symbol}")
    async def screener(symbol: Symbol, rs: State, refresh: bool = False,
                       view: Literal["auto", "consolidated", "standalone"] = "auto", who=Depends(current)):
        try:
            info = await rs.data.resolve(symbol.upper())
        except KeyError:
            info = None                                  # not in our lists — Screener.in may still know it
        if info is not None and info.kind == "IDX":
            raise HTTPException(422, f"{info.symbol} is an index — fundamentals exist only for companies")
        try:
            return _JSON(await rs.screener.fundamentals(symbol.upper(), view=view, refresh=refresh))
        except Exception as e:
            raise _fail("Screener.in fundamentals", e) from e

    # ---- AI analyst
    @router.get("/ai-config")
    async def ai_config(rs: State, who=Depends(current)):
        try:
            return _JSON(await rs.ai.config(who.name, who.is_admin))
        except Exception as e:
            raise _fail("AI configuration", e) from e

    @router.post("/ai/{symbol}", dependencies=write)
    async def ai_start(symbol: Symbol, rs: State, body: AiStartRequest | None = None, who=Depends(current)):
        try:
            ai = rs.ai
            info = await resolve(rs, symbol)
            return _JSON(await ai.start(info.symbol, who.name, who.is_admin, refresh=bool(body and body.refresh)))
        except Exception as e:
            raise _fail("AI analysis", e) from e

    @router.get("/ai-job/{job_id}")
    async def ai_job(job_id: Annotated[str, Path(pattern=JOB_PATTERN)], rs: State, who=Depends(current)):
        try:
            job = rs.ai.job(job_id)
        except Exception as e:
            raise _fail("AI job", e) from e
        if job is None:
            raise HTTPException(404, "unknown or expired AI job (jobs are kept for an hour) — start a new analysis")
        return _JSON(job)

    @router.get("/ai/{symbol}/latest")
    async def ai_latest(symbol: Symbol, rs: State, who=Depends(current)):
        sym = symbol.upper()
        try:
            rep = await rs.ai.latest(sym)
        except Exception as e:
            raise _fail("AI report", e) from e
        if rep is None:
            raise HTTPException(404, f"no AI analysis of {sym} yet")
        return _JSON(rep)

    return router


__all__ = ["make_router", "Research", "McRequest", "McBacktestRequest", "quant_drift", "default_seed",
           "SYMBOL_PATTERN"]
