"""Async Dhan v2 REST client (data endpoints only — this project never places orders).

Every call goes through the shared rate limiters so the whole system respects
Dhan's limits: data 5/s + 7,000/day, quotes 1/s (1,000 instruments per call),
option chain 1 per 3 s per underlying and max 10 unique underlyings per 8 s.
With `redis`, the per-second limits and the daily budget are shared by every process
(ingestor, gateway, backfill worker, backfill CLI): Dhan counts them per account.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Any

import httpx

from ..config import settings
from ..ratelimit import BudgetExhausted, DailyBudget, SharedRate, SlidingWindowUnique, TokenBucket
from .constants import DATA_API_ERRORS

log = logging.getLogger("td.dhan.rest")

RATE_CODES = {"RL001", "DH-904", "805"}          # Dhan's "too many requests" answers (besides HTTP 429)


class DhanError(RuntimeError):
    def __init__(self, msg: str, code: Any = None, status: int | None = None):
        super().__init__(msg)
        self.code = code
        self.status = status


class DhanRest:
    def __init__(self, client_id: str | None = None, access_token: str | None = None, redis=None,
                 budget_file=None):
        self.client_id = client_id or settings.dhan_client_id
        self._token = access_token          # None: follow settings, where the token keeper swaps in renewed tokens
        self.base = settings.dhan_base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0),
                                       limits=httpx.Limits(max_connections=10))
        self.data_bucket = TokenBucket(settings.limit_data_per_sec)
        self.quote_bucket = TokenBucket(settings.limit_quote_per_sec)
        # per-second pacing shared through Redis with every other process on this account (the buckets above
        # pace this process alone when there is no Redis or it errors)
        self.limits = {"data": SharedRate(redis, "data", settings.limit_data_per_sec, fallback=self.data_bucket),
                       "quote": SharedRate(redis, "quote", settings.limit_quote_per_sec, fallback=self.quote_bucket)}
        self.chain_window = SlidingWindowUnique(settings.limit_chain_unique_per_8s, 8.0, 3.0)
        self.data_budget = DailyBudget("dhan_data", settings.limit_data_per_day, redis=redis, file=budget_file)
        self.calls = {"data": 0, "quote": 0, "chain": 0, "errors": 0}

    @property
    def access_token(self) -> str:
        return self._token or settings.dhan_access_token

    async def close(self):
        await self._http.aclose()

    def _headers(self) -> dict:
        return {
            "access-token": self.access_token,
            "client-id": self.client_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _post(self, path: str, payload: dict, kind: str = "data", retries: int = 3) -> Any:
        if not self.access_token or not self.client_id:
            raise DhanError("DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not configured")
        if kind == "data":
            await self.data_budget.consume(1)
        limit = self.limits.get(kind)        # option chains are paced by chain_window in the caller
        self.calls[kind] = self.calls.get(kind, 0) + 1
        delay = 1.0
        last_status = None
        for attempt in range(retries):
            if limit is not None:
                await limit.acquire()        # every attempt is a request at Dhan, retries included
            try:
                r = await self._http.post(self.base + path, json=payload, headers=self._headers())
            except httpx.HTTPError as e:
                self.calls["errors"] += 1
                if attempt == retries - 1:
                    raise DhanError(f"network error on {path}: {e}") from e
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 429:
                self.calls["errors"] += 1
                last_status = r.status_code
                delay = await self._rate_limited(limit, delay)
                continue
            if r.status_code >= 500 and attempt < retries - 1:
                self.calls["errors"] += 1
                last_status = r.status_code
                await asyncio.sleep(delay)
                delay *= 2
                continue
            try:
                body = r.json()
            except ValueError:
                body = {"raw": r.text}
            if r.status_code >= 400 or (isinstance(body, dict) and body.get("status") == "failure"):
                self.calls["errors"] += 1
                code = body.get("errorCode") if isinstance(body, dict) else None
                msg = (body.get("errorMessage") if isinstance(body, dict) else None) or str(body)[:300]
                try:
                    msg = DATA_API_ERRORS.get(int(code), msg) if code is not None else msg
                except (TypeError, ValueError):
                    pass
                if str(code) in RATE_CODES:
                    delay = await self._rate_limited(limit, delay, attempt < retries - 1)
                    if attempt < retries - 1:
                        continue
                raise DhanError(f"{path}: {msg}", code=code, status=r.status_code)
            if limit is not None:
                await limit.relax()
            return body
        raise DhanError(f"{path}: retries exhausted", status=last_status)

    @staticmethod
    async def _rate_limited(limit: SharedRate | None, delay: float, retrying: bool = True) -> float:
        """Dhan answered "too many requests": a pause shared by every process (the next acquire waits it out)
        rather than each retrying on its own. Returns the next backoff delay for the unshared case."""
        if limit is not None:
            await limit.penalize()
            return delay
        if retrying:
            await asyncio.sleep(delay)
        return delay * 2

    # ------------------------------------------------------------------ quotes
    async def quote(self, instruments: dict[str, list[int]], mode: str = "quote") -> dict:
        """mode: ltp | ohlc | quote. Max 1000 instruments per call (auto-chunked)."""
        flat = [(seg, sid) for seg, ids in instruments.items() for sid in ids]
        out: dict = {}
        for i in range(0, len(flat), 1000):
            chunk: dict[str, list[int]] = {}
            for seg, sid in flat[i:i + 1000]:
                chunk.setdefault(seg, []).append(int(sid))
            body = await self._post(f"/marketfeed/{mode}", chunk, kind="quote")
            for seg, rows in (body.get("data") or {}).items():
                out.setdefault(seg, {}).update(rows)
        return out

    # -------------------------------------------------------------- historical
    async def daily(self, security_id: str | int, segment: str, instrument: str,
                    start: date, end: date, oi: bool = False, expiry_code: int | None = None,
                    retries: int = 3) -> dict:
        payload = {"securityId": str(security_id), "exchangeSegment": segment, "instrument": instrument,
                   "oi": oi, "fromDate": start.isoformat(), "toDate": end.isoformat()}
        if expiry_code is not None:
            payload["expiryCode"] = expiry_code
        body = await self._post("/charts/historical", payload, retries=retries)
        return body if isinstance(body, dict) else {}

    async def intraday(self, security_id: str | int, segment: str, instrument: str, interval: int,
                       start: datetime, end: datetime, oi: bool = False, retries: int = 3) -> dict:
        """interval ∈ {1,5,15,25,60}; max 90 days per request."""
        payload = {"securityId": str(security_id), "exchangeSegment": segment, "instrument": instrument,
                   "interval": str(interval), "oi": oi,
                   "fromDate": start.strftime("%Y-%m-%d %H:%M:%S"), "toDate": end.strftime("%Y-%m-%d %H:%M:%S")}
        body = await self._post("/charts/intraday", payload, retries=retries)
        return body if isinstance(body, dict) else {}

    async def rolling_option(self, security_id: int, segment: str, instrument: str, expiry_flag: str,
                             expiry_code: int, strike: str, option_type: str, start: date, end: date,
                             interval: int = 1, fields: list[str] | None = None) -> dict:
        """Expired options data (max 30 days per call)."""
        payload = {"securityId": security_id, "exchangeSegment": segment, "instrument": instrument,
                   "expiryFlag": expiry_flag, "expiryCode": expiry_code, "strike": strike,
                   "drvOptionType": option_type,
                   "requiredData": fields or ["open", "high", "low", "close", "iv", "volume", "oi", "spot"],
                   "fromDate": start.isoformat(), "toDate": end.isoformat(), "interval": interval}
        body = await self._post("/charts/rollingoption", payload)
        return body if isinstance(body, dict) else {}

    # ------------------------------------------------------------ option chain
    async def expiry_list(self, underlying_id: int, underlying_seg: str) -> list[str]:
        await self.chain_window.acquire(f"exp:{underlying_id}")
        body = await self._post("/optionchain/expirylist",
                                {"UnderlyingScrip": int(underlying_id), "UnderlyingSeg": underlying_seg},
                                kind="chain")
        return list(body.get("data") or [])

    async def option_chain(self, underlying_id: int, underlying_seg: str, expiry: str) -> dict:
        await self.chain_window.acquire(f"{underlying_id}:{expiry}")
        self.calls["chain"] += 1
        body = await self._post("/optionchain",
                                {"UnderlyingScrip": int(underlying_id), "UnderlyingSeg": underlying_seg,
                                 "Expiry": expiry}, kind="chain")
        return body.get("data") or {}

    # --------------------------------------------- market analytics (Pro plan)
    async def news(self, limit: int = 50, categories: list[str] | None = None,
                   stock_ids: list[int] | None = None) -> list[dict]:
        payload: dict = {"dhanClientId": self.client_id, "categories": categories or ["ALL"],
                         "limit": max(1, min(50, limit))}
        if stock_ids:
            payload["stockList"] = [str(s) for s in stock_ids]
        body = await self._post("/data/newsheadline", payload)
        data = body.get("data") or {}
        return list(data.get("latestNews") or []) + list(data.get("nextNews") or [])

    async def market_movers(self, segment: str, instruments: list[str], category: str,
                            universe: str | None = None, expiry: str | None = None, limit: int = 100) -> list[dict]:
        payload: dict = {"exchangeSegment": segment, "instrument": instruments, "category": category,
                         "limit": limit}
        if universe:
            payload["universe"] = universe
        if expiry:
            payload["expiry"] = expiry
        body = await self._post("/data/marketmovers", payload)
        return list(body.get("data") or [])

    async def technical(self, security_id: int, segment: str, instrument: str, timeframe: str,
                        indicators: list[str]) -> dict:
        body = await self._post("/data/technical", {"securityId": str(security_id), "exchangeSegment": segment,
                                                    "instrument": instrument, "timeframe": timeframe,
                                                    "indicators": indicators})
        return body.get("data") or {}

    async def company_info(self, security_id: int, segment: str = "NSE_EQ",
                           metrics: list[str] | None = None) -> dict:
        body = await self._post("/data/companyinfo", {"securityId": str(security_id), "exchangeSegment": segment,
                                                      "instrument": "EQUITY",
                                                      "metrics": metrics or ["CO", "RATIOS", "SHP"]})
        return body.get("data") or {}


__all__ = ["DhanRest", "DhanError", "BudgetExhausted"]
