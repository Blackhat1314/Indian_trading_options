"""Price history, symbols and market context for the research desks.

Every research model works from daily bars in the lake. The F&O universe is kept fresh by the ingestor's
auto-history, but a user can research any NSE equity: in live mode (DATA_SOURCE=dhan) a stock that is missing,
short (< 2 years) or stale is topped up with ONE call to Dhan's historical API (it returns 10+ years at once),
and only while the shared daily data budget keeps what the live app needs for its own day (the larger of
`research_fetch_reserve` and the backfill's reserve) and research's own allowance of FETCH_PER_DAY top-ups a day
is not used up. All top-ups share one Dhan client, so its 5/s rate limit covers them together. A failed fetch
never fails the caller: the lake is used, the reason comes back as a warning and the symbol is not retried for
10 minutes. Sim mode never fetches and only knows the simulator's universe.

Live prices, news and market breadth come from the snapshots the analytics worker already publishes to Redis
(in live mode without the simulator's headlines: sim and live sessions share one Redis).
"""
from __future__ import annotations

import asyncio
import csv
import io
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import bus
from ..config import settings
from ..dhan.instruments import download_scrip_master, ikey, sim_universe
from ..dhan.rest import DhanError, DhanRest
from ..lake import Lake, bars_from_dhan
from ..ratelimit import BudgetExhausted, DailyBudget
from ..services.bootstrap import HIST
from ..timeutil import IST, from_epoch, now_ist, previous_weekday
from ..universe_data import INDICES, SECTOR_OF

log = logging.getLogger("td.research.data")

BAR_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]
YEAR_S = 365.25 * 86400
IST_OFFSET_S = 19800                    # 05:30
MIN_YEARS = 2.0                          # below this a stock's history is topped up from Dhan (live mode)
FETCH_YEARS_DAYS = 3653                  # one daily request reaches back 10 years
NEG_CACHE_S = 600.0                      # a failed fetch is not retried for 10 minutes
CACHE_S = 30.0                           # lake listing cache (search runs on every keystroke)
FETCH_PER_DAY = 300                      # research's own cap on Dhan history top-ups per IST day (all users)
EQUITY_SERIES = {"EQ", "BE", "BZ", "SM", "ST"}   # NSE equity shares: main board, trade-for-trade, SME


@dataclass
class SymbolInfo:
    symbol: str
    name: str
    sector: str
    sid: int | None
    seg: str | None
    kind: str                            # "EQ" | "IDX"
    in_universe: bool

    def as_dict(self) -> dict:
        return asdict(self)


def clean_daily(df: pd.DataFrame | None) -> pd.DataFrame:
    """Lake/Dhan bars -> ts, open, high, low, close, volume: sorted, one bar per ts, finite closes > 0.
    Missing open/high/low fall back to the close (they only widen nothing); missing volume is 0."""
    if df is None or df.empty or "close" not in df.columns:
        return pd.DataFrame({c: pd.Series(dtype="int64" if c in ("ts", "volume") else "float64")
                             for c in BAR_COLUMNS})
    d = pd.DataFrame({"ts": pd.to_numeric(df["ts"], errors="coerce"),
                      "close": pd.to_numeric(df["close"], errors="coerce")})
    for c in ("open", "high", "low"):
        d[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    d["volume"] = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else 0
    d = d[np.isfinite(d["ts"]) & np.isfinite(d["close"]) & (d["close"] > 0)].copy()
    for c in ("open", "high", "low"):
        bad = ~np.isfinite(d[c]) | (d[c] <= 0)
        d.loc[bad, c] = d.loc[bad, "close"]
    d["volume"] = d["volume"].fillna(0).clip(lower=0)
    d["ts"] = d["ts"].astype("int64")
    d["volume"] = d["volume"].astype("int64")
    d = d.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    return d[BAR_COLUMNS]


def day_start(ts: np.ndarray | pd.Series) -> np.ndarray:
    """Epoch seconds of the IST midnight of each timestamp's date (the lake's daily-bar convention)."""
    t = np.asarray(ts, dtype="int64")
    return (t + IST_OFFSET_S) // 86400 * 86400 - IST_OFFSET_S


def bar_date(ts: int | float) -> date:
    return from_epoch(int(ts)).date()


def span_years(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["ts"].iat[-1] - df["ts"].iat[0]) / YEAR_S


INDEX_NAMES = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank", "FINNIFTY": "Nifty Financial Services",
               "MIDCPNIFTY": "Nifty Midcap Select", "INDIAVIX": "India VIX", "SENSEX": "BSE Sensex"}
_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def _alnum(s: str) -> str:
    return _NON_ALNUM.sub("", s.upper())


@lru_cache(maxsize=8192)
def _search_keys(symbol: str, name: str) -> tuple[str, str, str, tuple[str, ...]]:
    """(symbol alnum, NAME, NAME alnum, NAME words) — computed once per symbol, search runs per keystroke."""
    up = name.upper()
    return _alnum(symbol), up, _alnum(up), tuple(up.split())


def _env_data(raw: bytes | None) -> dict | None:
    """Payload of a td:snap:* envelope ({"topic", "data"})."""
    if not raw:
        return None
    try:
        env = bus.loads(raw)
    except ValueError:
        return None
    if isinstance(env, dict) and isinstance(env.get("data"), dict):
        return env["data"]
    return None


def _parse_scrip_master(text: str) -> dict[str, SymbolInfo]:
    """NSE equity shares from Dhan's detailed scrip master, by trading symbol (main-board series preferred)."""
    out: dict[str, tuple[int, SymbolInfo]] = {}
    for r in csv.DictReader(io.StringIO(text)):
        if (r.get("EXCH_ID") or "").strip() != "NSE" or (r.get("INSTRUMENT") or "").strip() != "EQUITY":
            continue
        series = (r.get("SERIES") or "").strip()
        if (r.get("INSTRUMENT_TYPE") or "").strip() != "ES" or series not in EQUITY_SERIES:
            continue
        sym = (r.get("UNDERLYING_SYMBOL") or r.get("SYMBOL_NAME") or "").strip().upper()
        try:
            sid = int((r.get("SECURITY_ID") or "").strip())
        except ValueError:
            continue
        if not sym:
            continue
        rank = 0 if series == "EQ" else 1
        if sym in out and out[sym][0] <= rank:
            continue
        name = (r.get("DISPLAY_NAME") or r.get("SYMBOL_NAME") or sym).strip()
        out[sym] = (rank, SymbolInfo(sym, name, SECTOR_OF.get(sym, "OTHER"), sid, "NSE_EQ", "EQ", False))
    return {k: v for k, (_, v) in out.items()}


class ResearchData:
    """History / symbols / live context shared by the Monte Carlo, quant, screener and AI desks."""

    def __init__(self, redis, lake: Lake | None = None, rest_factory: Callable[[], DhanRest] | None = None):
        self.redis = redis
        self.lake = lake or Lake()
        self.rest_factory = rest_factory or (lambda: DhanRest(redis=redis))
        self._rest: DhanRest | None = None                   # one shared client, built on the first top-up
        self.fetch_budget = DailyBudget("research_fetch", int(getattr(settings, "research_fetch_per_day",
                                                                      FETCH_PER_DAY)), redis)
        self._locks: dict[str, asyncio.Lock] = {}
        self._neg: dict[str, tuple[float, str]] = {}          # symbol -> (retry after, warning)
        self._topped: dict[str, date] = {}                    # symbol -> day a fetch succeeded (once per day)
        self._lake_syms: tuple[float, set[str]] = (0.0, set())
        self._sim_uni: list[dict] | None = None
        self._master: dict[str, SymbolInfo] = {}
        self._master_day: date | None = None
        self._master_task: asyncio.Task | None = None
        self._master_error: tuple[float, str] | None = None

    # ------------------------------------------------------------ symbols
    @property
    def live(self) -> bool:
        return settings.data_source == "dhan"

    async def _universe(self) -> list[dict]:
        raw = await self.redis.get(bus.KEY_UNIVERSE)
        rows = bus.loads(raw) if raw else None
        if not rows and not self.live:
            if self._sim_uni is None:                           # gateway running without the ingestor
                self._sim_uni = sim_universe().to_json()
            rows = self._sim_uni
        return [r for r in rows or [] if isinstance(r, dict) and r.get("kind") in ("EQ", "IDX")]

    @staticmethod
    def _from_inst(r: dict) -> SymbolInfo:
        sym = str(r["symbol"]).upper()
        kind = "IDX" if r.get("kind") == "IDX" else "EQ"
        sector = r.get("sector") or SECTOR_OF.get(sym, "INDEX" if kind == "IDX" else "OTHER")
        sid = r.get("sid")
        name = (r.get("name") or "").strip() or INDEX_NAMES.get(sym, sym)
        return SymbolInfo(sym, name, sector, int(sid) if sid is not None else None, r.get("seg"), kind, True)

    @staticmethod
    def _lake_only(sym: str) -> SymbolInfo:
        """A symbol the lake has bars for but no instrument list knows (no Dhan id: it can't be topped up)."""
        kind = "IDX" if sym in INDICES else "EQ"
        return SymbolInfo(sym, INDEX_NAMES.get(sym, sym), SECTOR_OF.get(sym, "INDEX" if kind == "IDX" else "OTHER"),
                          None, None, kind, False)

    async def lake_symbols(self) -> set[str]:
        ts, syms = self._lake_syms
        if time.monotonic() - ts > CACHE_S:
            syms = set(await asyncio.to_thread(self.lake.symbols, "D"))
            self._lake_syms = (time.monotonic(), syms)
        return syms

    async def _load_master(self) -> dict[str, SymbolInfo]:
        text = await asyncio.to_thread(download_scrip_master)
        idx = await asyncio.to_thread(_parse_scrip_master, text)
        self._master, self._master_day, self._master_error = idx, now_ist().date(), None
        log.info("scrip master: %d NSE equities", len(idx))
        return idx

    def _master_job(self) -> asyncio.Task | None:
        """Start (or join) the daily scrip-master load; None when it is not needed or recently failed."""
        if not self.live:
            return None
        if self._master and self._master_day == now_ist().date():
            return None
        if self._master_error and time.monotonic() < self._master_error[0]:
            return None
        if self._master_task is None or self._master_task.done():
            async def run():
                try:
                    await self._load_master()
                except Exception as e:                     # network / disk: retry in 10 minutes
                    self._master_error = (time.monotonic() + NEG_CACHE_S, f"{type(e).__name__}: {e}"[:160])
                    log.warning("scrip master unavailable: %s", e)
            self._master_task = asyncio.create_task(run())
        return self._master_task

    async def _master_index(self, wait: bool) -> dict[str, SymbolInfo]:
        task = self._master_job()
        if task is not None and wait:
            await asyncio.shield(task)
        return self._master

    async def resolve(self, symbol: str) -> SymbolInfo:
        sym = (symbol or "").strip().upper()
        if not sym:
            raise KeyError("empty symbol")
        for r in await self._universe():
            if str(r.get("symbol", "")).upper() == sym:
                return self._from_inst(r)
        master = await self._master_index(wait=True)
        if sym in master:
            return master[sym]
        if sym in await self.lake_symbols():
            return self._lake_only(sym)
        if not self.live:
            raise KeyError(f"{sym} is not in the simulator's universe (DATA_SOURCE=sim knows only its "
                           f"{len(await self._universe())} instruments)")
        if self._master_error:
            raise KeyError(f"{sym} is not in the F&O universe and Dhan's scrip master could not be loaded "
                           f"({self._master_error[1]})")
        raise KeyError(f"{sym} is not an NSE-listed equity (check the trading symbol, e.g. RELIANCE, M&M)")

    async def search(self, q: str, limit: int = 12) -> list[dict]:
        qa = _alnum(q or "")
        qn = " ".join((q or "").upper().split())
        if not qa:
            return []
        lake_syms = await self.lake_symbols()
        cands: dict[str, SymbolInfo] = {}
        for r in await self._universe():
            info = self._from_inst(r)
            cands[info.symbol] = info
        for sym, info in (await self._master_index(wait=False)).items():
            cands.setdefault(sym, info)
        for sym in lake_syms:
            if sym not in cands:
                cands[sym] = self._lake_only(sym)
        scored = []
        for info in cands.values():
            s = self._score(qa, qn, info)
            if s is not None:
                has = info.symbol in lake_syms
                scored.append(((s, not info.in_universe, not has, len(info.symbol), info.symbol), info, has))
        scored.sort(key=lambda x: x[0])
        return [{"symbol": i.symbol, "name": i.name, "sector": i.sector, "in_universe": i.in_universe,
                 "has_history": has} for _, i, has in scored[:max(1, limit)]]

    @staticmethod
    def _score(qa: str, qn: str, info: SymbolInfo) -> int | None:
        sa, name, name_a, words = _search_keys(info.symbol, info.name)
        if sa == qa:
            return 0
        if sa.startswith(qa):
            return 1
        if name.startswith(qn) or name_a.startswith(qa):
            return 2
        if any(w.startswith(qn) for w in words):
            return 3
        if qa in sa:
            return 4
        if qn in name:
            return 5
        return None

    # ------------------------------------------------------------ history
    async def daily(self, symbol: str, fetch: bool = True, min_years: float = 0.0) -> tuple[pd.DataFrame, dict]:
        info = await self.resolve(symbol)
        sym = info.symbol
        lock = self._locks.setdefault(sym, asyncio.Lock())
        warnings: list[str] = []
        fetched = False
        async with lock:
            df = await self._read(sym)
            today = now_ist().date()
            reason = self._needs_fetch(df, today, max(MIN_YEARS, min_years))
            if fetch and self.live and reason and self._topped.get(sym) != today:
                neg = self._neg.get(sym)
                if neg and time.monotonic() < neg[0]:
                    warnings.append(neg[1])
                else:
                    n, warn = await self._fetch(info, df, today, reason)
                    if warn:
                        self._neg[sym] = (time.monotonic() + NEG_CACHE_S, warn)
                        warnings.append(warn)
                    else:
                        self._neg.pop(sym, None)
                        self._topped[sym] = today
                        if n:
                            fetched = True
                            df = await self._read(sym)
        years = span_years(df)
        if df.empty:
            warnings.append(f"no daily history for {sym}" + ("" if self.live else " in the simulator's lake"))
        else:
            if years < max(MIN_YEARS, min_years):
                warnings.append(f"only {years:.1f} years of daily history")
            last = bar_date(df["ts"].iat[-1])
            if last < now_ist().date() - timedelta(days=7):
                warnings.append(f"the last daily bar is from {last.isoformat()}")
        if not self.live:
            warnings.append("simulated prices (DATA_SOURCE=sim) — not real market history")
        warnings = list(dict.fromkeys(warnings))
        meta = {"fetched": fetched, "warning": "; ".join(warnings) or None, "warnings": warnings,
                "source": "lake+dhan" if fetched else "lake"}
        return df, meta

    async def _read(self, sym: str) -> pd.DataFrame:
        raw = await asyncio.to_thread(self.lake.read, "D", sym)
        df = clean_daily(raw)
        tomorrow = now_ist().date() + timedelta(days=1)
        end = int(datetime.combine(tomorrow, datetime.min.time(), tzinfo=IST).timestamp())
        return df[df["ts"] < end].reset_index(drop=True)                    # never a bar dated in the future

    @staticmethod
    def _needs_fetch(df: pd.DataFrame, today: date, min_years: float) -> str | None:
        if df.empty:
            return "empty"
        if span_years(df) < min_years:
            return "short"
        if bar_date(df["ts"].iat[-1]) < previous_weekday(today):
            return "stale"
        return None

    def _client(self) -> DhanRest:
        """The Dhan client every top-up shares (one rate limiter, one connection pool), built on first use."""
        if self._rest is None:
            self._rest = self.rest_factory()
        return self._rest

    async def aclose(self) -> None:
        """Close the shared Dhan client (app shutdown); the next top-up builds a new one."""
        rest, self._rest = self._rest, None
        close = getattr(rest, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass

    async def _fetch(self, info: SymbolInfo, df: pd.DataFrame, today: date, reason: str) -> tuple[int, str | None]:
        """One Dhan daily request for the missing range. Returns (bars written, warning or None)."""
        from ..backfill import classify, min_reserve

        hist = HIST.get((info.kind, info.seg or ""))
        if info.sid is None or hist is None:
            return 0, f"no Dhan security id for {info.symbol} — using the lake only"
        using = "using the lake" if not df.empty else "no history available"
        reserve = max(settings.research_fetch_reserve, min_reserve())   # never eat into the live app's own day
        budget = DailyBudget("dhan_data", settings.limit_data_per_day, self.redis)
        left = await budget.remaining()
        if left <= reserve:
            return 0, (f"Dhan daily data budget nearly used ({left} calls left, {reserve} kept for the live "
                       f"desks) — {using}")
        if reason in ("empty", "short"):
            start = today - timedelta(days=FETCH_YEARS_DAYS)
        else:
            start = bar_date(df["ts"].iat[-1]) + timedelta(days=1)
        while start.weekday() >= 5:
            start += timedelta(days=1)
        if start >= today:
            return 0, None
        seg, instrument = hist
        try:
            rest = self._client()
        except Exception as e:                          # e.g. no credentials: the lake still serves
            log.warning("history fetch %s: Dhan client unavailable: %s", info.symbol, e)
            return 0, f"Dhan client unavailable ({type(e).__name__}: {e}) — {using}"
        try:
            await self.fetch_budget.consume(1)          # atomic: concurrent top-ups cannot overshoot the cap
        except BudgetExhausted:
            return 0, (f"research's {self.fetch_budget.limit} Dhan history top-ups for today are used up — "
                       f"{using}")
        try:
            payload = await rest.daily(info.sid, seg, instrument, start, today)
        except BudgetExhausted:
            return 0, f"Dhan daily data budget used up — {using}"
        except DhanError as e:
            if reason == "stale" and not df.empty and classify(e) == "nodata":
                return 0, None                          # DH-907 on a gap-only range (a holiday): nothing new yet
            return 0, f"{_dhan_reason(e)} — {using}"
        except Exception as e:                          # never fail the caller over a history top-up
            log.warning("history fetch %s failed: %s", info.symbol, e, exc_info=True)
            return 0, f"Dhan history fetch failed ({type(e).__name__}) — {using}"
        bars = clean_daily(bars_from_dhan(payload or {}))
        if not bars.empty:
            bars["ts"] = day_start(bars["ts"])              # Dhan stamps daily bars at IST midnight; make sure
            today_ts = int(datetime.combine(today, datetime.min.time(), tzinfo=IST).timestamp())
            bars = clean_daily(bars[bars["ts"] < today_ts])                # today's bar is still forming
        if bars.empty:
            return 0, (f"Dhan returned no daily bars for {info.symbol}" if df.empty else None)
        await asyncio.to_thread(self.lake.write, "D", info.symbol, bars)
        self._lake_syms = (0.0, set())
        log.info("research history: %s +%d daily bars from Dhan (%s)", info.symbol, len(bars), reason)
        return len(bars), None

    async def index_daily(self, name: str = "NIFTY") -> pd.DataFrame:
        try:
            df, _ = await self.daily(name, fetch=True)
            return df
        except KeyError:
            return await self._read(name.upper())

    # ------------------------------------------------------------ live / context
    async def spot(self, symbol: str) -> tuple[float | None, str | None]:
        info = await self.resolve(symbol)
        last = await asyncio.to_thread(self.lake.read, "D", info.symbol, None, None, 5)
        last = clean_daily(last)
        last_close = round(float(last["close"].iat[-1]), 2) if not last.empty else None
        last_day = bar_date(last["ts"].iat[-1]) if not last.empty else None
        live = await self._live_price(info, last_day)
        if live is not None:
            return live, "live"
        if last_close is not None:
            return last_close, "close"
        return None, None

    async def _live_price(self, info: SymbolInfo, last_day: date | None) -> float | None:
        """The analytics worker's last price, only while it is running (its status key lives 30 s) and only
        when its session is not older than the lake's last bar."""
        if not info.in_universe or info.sid is None or not info.seg:
            return None
        status, raw = await self.redis.mget([bus.k_status("analytics"), bus.SNAP_PREFIX + "quotes"])
        q = _env_data(raw)
        if not status or not q:
            return None
        row = (q.get("q") or {}).get(str(ikey(info.seg, info.sid)))
        try:
            px = float(row[0])
        except (TypeError, ValueError, IndexError):
            return None
        if not math.isfinite(px) or px <= 0:
            return None
        clock = q.get("ts")
        if last_day is not None and isinstance(clock, (int, float)) and bar_date(clock) < last_day:
            return None
        return round(px, 2)

    async def news(self, symbol: str, days: int = 30, limit: int = 25) -> list[dict]:
        """Headlines that name the stock (or that the news engine maps onto it), newest first. In live mode the
        simulator's headlines are skipped: an earlier sim session leaves them in the shared Redis for weeks."""
        from ..analytics.news import classify

        sym = (symbol or "").strip().upper()
        since = time.time() - days * 86400
        found: dict[str, dict] = {}

        def add(it: dict, impact: int | None) -> None:
            if self.live and (str(it.get("source") or "").upper() == "SIM"
                              or str(it.get("id") or "").startswith("sim-")):
                return                                  # invented by the simulator, not news about a real stock
            title = str(it.get("title") or "").strip()
            try:
                ts = float(it.get("ts") or 0)
            except (TypeError, ValueError):
                return
            if not title or ts < since:
                return
            key = str(it.get("id") or title.lower())
            if key in found or title.lower() in titles:
                if impact is not None and key in found:
                    found[key]["impact"] = impact        # the news engine's sign for this stock
                return
            titles.add(title.lower())
            cls = classify(title, it.get("sentiment"))
            sent = it.get("sentiment") if it.get("sentiment") in ("positive", "negative", "neutral") else \
                {1: "positive", -1: "negative"}.get(cls["sign"], "neutral")
            found[key] = {"id": key, "ts": round(ts, 3), "date": from_epoch(ts).strftime("%Y-%m-%d %H:%M"),
                          "title": title, "text": str(it.get("text") or "")[:400],
                          "source": str(it.get("source") or ""), "category": str(it.get("category") or ""),
                          "type": it.get("type") or cls["type"], "sentiment": sent,
                          "impact": impact if impact is not None else {"positive": 1, "negative": -1}.get(sent, 0)}

        titles: set[str] = set()
        try:
            entries = await self.redis.xrevrange(bus.STREAM_NEWS, count=5000)
        except Exception as e:                          # no stream yet: the snapshot below still counts
            log.debug("news stream: %s", e)
            entries = []
        for _, fields in entries:                       # headlines the feed tagged with this stock
            try:
                it = bus.loads(fields.get(b"j") or fields.get("j"))
            except ValueError:
                continue
            if isinstance(it, dict) and sym in [str(x).upper() for x in it.get("symbols") or []]:
                add(it, None)
        snap = _env_data(await self.redis.get(bus.SNAP_PREFIX + "news")) or {}
        for it in snap.get("items") or []:              # + sector/macro headlines the news engine maps onto it
            impacts = it.get("impacts") if isinstance(it, dict) else None
            if isinstance(impacts, dict) and sym in impacts:
                try:
                    add(it, int(impacts[sym]))
                except (TypeError, ValueError):
                    add(it, None)
        return sorted(found.values(), key=lambda x: -x["ts"])[:max(1, limit)]

    async def market_context(self, symbol: str | None = None) -> dict:
        """What the live desks see right now: clock/phase, indices + breadth + sector moves (`market`), and the
        swing scanner — the row for `symbol` (plus its sector's move in `sector`), or every row
        ({ts, rules, rows}) without a symbol. `live` is false when the analytics worker is not running (the
        snapshots are then from its last session). Pieces that were never published are {}."""
        keys = [bus.k_status("analytics")] + [bus.SNAP_PREFIX + t for t in ("clock", "market", "swing")]
        status, clock_raw, market_raw, swing_raw = await self.redis.mget(keys)
        clock = _env_data(clock_raw) or {}
        market = _env_data(market_raw) or {}
        swing = _env_data(swing_raw) or {}
        rows = [{k: v for k, v in r.items() if k != "stats"} for r in swing.get("rows") or [] if isinstance(r, dict)]
        out: dict = {"live": bool(status),
                     "clock": {k: clock.get(k) for k in ("ts", "phase", "date", "time", "source")} if clock else {},
                     "market": {}, "sector": {}, "swing": {}}
        if market:
            sectors = [{k: v for k, v in s.items() if k != "members"} for s in market.get("sectors") or []
                       if isinstance(s, dict)]
            breadth = market.get("breadth") if isinstance(market.get("breadth"), dict) else {}
            if not breadth.get("n", (breadth.get("adv") or 0) + (breadth.get("dec") or 0)):
                breadth = {}                            # analytics' sentinel before any price (n = 0): no reading
            indices = market.get("indices") or []
            if indices or breadth or sectors:
                out["market"] = {"ts": market.get("ts"), "phase": market.get("phase"), "indices": indices,
                                 "breadth": breadth, "sectors": sectors}
        if not symbol:
            if rows:
                out["swing"] = {"ts": swing.get("ts"), "rules": swing.get("rules") or {}, "rows": rows}
            return out
        sym = symbol.strip().upper()
        sector = SECTOR_OF.get(sym)
        if sector is None:
            try:
                sector = (await self.resolve(sym)).sector
            except KeyError:
                sector = None
        for s in (out["market"] or {}).get("sectors", []):
            if s.get("sector") == sector:
                out["sector"] = s
                break
        for row in rows:
            if str(row.get("symbol", "")).upper() == sym:
                out["swing"] = {**row, "rules": swing.get("rules") or {}, "ts": swing.get("ts")}
                break
        return out

    async def history_info(self, symbol: str, fetch: bool = True) -> dict:
        info = await self.resolve(symbol)
        df, meta = await self.daily(info.symbol, fetch=fetch)
        spot, source = await self.spot(info.symbol)
        recent = df[df["ts"] >= df["ts"].iat[-1] - 5 * YEAR_S] if not df.empty else df
        return {
            "symbol": info.symbol, "name": info.name, "sector": info.sector,
            "bars": int(len(df)),
            "first": bar_date(df["ts"].iat[0]).isoformat() if not df.empty else None,
            "last": bar_date(df["ts"].iat[-1]).isoformat() if not df.empty else None,
            "years": round(span_years(df), 2),
            "spot": spot, "spot_source": source,
            "fetched": bool(meta["fetched"]), "warning": meta["warning"],
            "closes": [[int(t), round(float(c), 2)] for t, c in zip(recent["ts"], recent["close"])],
        }


def _dhan_reason(e: DhanError) -> str:
    """Short, user-facing reason for a failed Dhan history call."""
    from ..backfill import classify

    msg = str(e)
    if "not configured" in msg:
        return "Dhan credentials not configured (DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN)"
    kind = classify(e)
    if kind == "auth":
        return f"Dhan rejected the access token ({msg.split(': ', 1)[-1][:80]}) — renew DHAN_ACCESS_TOKEN"
    if kind == "rate":
        return "Dhan rate limit hit — history top-up skipped for 10 minutes"
    if kind == "network":
        return "Dhan is unreachable"
    if kind == "nodata":
        return "Dhan has no daily history for this symbol"
    return f"Dhan history fetch failed ({msg[:100]})"


__all__ = ["ResearchData", "SymbolInfo", "clean_daily", "day_start", "span_years", "bar_date"]
