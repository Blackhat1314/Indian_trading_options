"""ANALYTICS WORKER — turns the tick stream into desk views.

Consumes td:ticks / td:news / td:events and publishes (SET td:snap:<topic> + PUBLISH td:pub:<topic>):
  clock     market time & phase
  market    indices, breadth, sector heatmap, movers
  rank      live intraday factor ranking (every F&O stock)
  top       09:30 locked top-N long/short picks + live tracking (provisional before lock)
  fno       futures OI build-up board (LB / SB / SC / LU)
  chain:<X> option-chain analytics (PCR, max pain, GEX, walls, IV, skew)
  swing     swing scanner + historical setup statistics
  news      classified headline feed with catalyst scores
  quotes    compact last-price map for every instrument
  scorecard daily accuracy record of the locked picks (the system's own receipts)
Run:  python -m td.services.analytics
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .. import bus
from ..config import settings
from ..dhan.instruments import Universe
from ..lake import Lake
from ..timeutil import IST, MARKET_CLOSE, MARKET_OPEN, at, from_epoch, minute_of_session, parse_hms, phase
from ..universe_data import SECTOR_OF
from ..analytics import alerts as alerts_mod
from ..analytics import intraday, market, optmath, swing
from ..analytics.context import load_context
from ..analytics.news import NewsEngine
from ..analytics.state import InstState, safe

log = logging.getLogger("td.analytics")
LOCK_GRACE_S = 600      # picks may lock up to 10 minutes after LOCK_TIME (e.g. a slow start), never later


class Engine:
    def __init__(self, r):
        self.r = r
        self.lake = Lake()
        self.states: dict[int, InstState] = {}
        self.eqs: list[InstState] = []
        self.futs_by_symbol: dict[str, InstState] = {}
        self.eq_by_symbol: dict[str, InstState] = {}
        self.idx: dict[str, InstState] = {}
        self.universe: Universe | None = None
        self.clock = time.time()
        self.day = None
        self.cfg = intraday.RankConfig(min_turnover_cr=settings.min_turnover_cr, top_n=settings.top_n)
        self._load_weights()
        self.news = NewsEngine(SECTOR_OF)
        self.alerts = alerts_mod.AlertEngine()
        self.rows: list[dict] = []
        self.top: dict = {"locked": False}
        self.scorecard_done = False
        self.swing_rows: list[dict] = []
        self.swing_stats: dict = {}
        self.daily: dict[str, pd.DataFrame] = {}
        self.chain_cache: dict[str, dict] = {}
        self.last_bars_push = 0.0
        self.new_news_symbols: set[str] = set()
        self.ticks_seen = 0
        self.lock_t = parse_hms(settings.lock_time)

    def _load_weights(self):
        p = Path(settings.data_dir) / "config" / "ranker_weights.json"
        if p.exists():
            try:
                w = json.loads(p.read_text())
                self.cfg.weights.update({k: float(v) for k, v in w.items() if k in self.cfg.weights})
                log.info("ranker weights loaded from %s", p)
            except Exception:
                log.exception("bad ranker weights file")

    # ------------------------------------------------------------------ setup
    async def setup(self):
        raw = None
        for _ in range(120):
            raw = await self.r.get(bus.KEY_UNIVERSE)
            if raw:
                break
            log.info("waiting for ingestor to publish the universe ...")
            await asyncio.sleep(1)
        if not raw:
            raise RuntimeError("universe not available — is the ingestor running?")
        self.universe = Universe.from_json(bus.loads(raw))
        for i in self.universe.instruments:
            # keyed by the unique instrument key: Dhan ids repeat across segments (IDX_I 13 = NIFTY, NSE_EQ 13 = ABB)
            s = InstState(i.key, i.seg, i.symbol, i.kind, i.underlying, i.sector, i.lot)
            self.states[i.key] = s
            if i.kind == "EQ":
                self.eqs.append(s)
                self.eq_by_symbol[i.symbol] = s
            elif i.kind == "FUT":
                self.futs_by_symbol[i.underlying] = s
            elif i.kind == "IDX":
                self.idx[i.symbol] = s
        log.info("engine: %d eq, %d fut, %d idx", len(self.eqs), len(self.futs_by_symbol), len(self.idx))

    async def _apply_history(self, day) -> int:
        """Load pre-open context (ATR, beta, RVOL curve, daily bars) from the lake. Returns #series with data."""
        syms = [s.symbol for s in self.eqs] + list(self.idx)
        ctx, daily = await asyncio.to_thread(load_context, self.lake, syms, day)
        for s in self.eqs:
            if s.symbol in ctx:
                s.ctx = ctx[s.symbol]
        for name, s in self.idx.items():
            if name in ctx:
                s.ctx = ctx[name]
        self.daily = daily
        return sum(1 for d in daily.values() if d is not None and len(d))

    async def _wait_for_lake(self, day):
        """Lake empty (first sim run still generating it, or backfill not run yet): retry until it appears."""
        while self.day == day:
            await asyncio.sleep(15)
            n = await self._apply_history(day)
            if n:
                for s in list(self.eqs) + list(self.idx.values()):
                    if not s.pc and s.ctx.pdc:
                        s.pc = s.ctx.pdc
                log.info("history now available: %d daily series — context reloaded", n)
                await self.refresh_swing(initial=True)
                return

    async def load_day_context(self, day):
        self.day = day
        n_hist = await self._apply_history(day)
        if not n_hist:
            log.warning("no history in %s yet — RVOL/ATR/swing need it; the ingestor's auto-history "
                        "(AUTO_HISTORY) or backfill/backfill.py fills it — checking every 15 s", self.lake.root)
            asyncio.create_task(self._wait_for_lake(day))
        # previous close / previous OI: from the feed's prev-close packets, else the lake
        prev = await self.r.hgetall(bus.KEY_PREV)
        for k, v in prev.items():
            st = self.states.get(int(k))
            if st:
                pc, poi = bus.loads(v)
                st.pc = pc or st.pc
                st.poi = poi or st.poi
        for s in list(self.eqs) + list(self.idx.values()):
            if not s.pc and s.ctx.pdc:
                s.pc = s.ctx.pdc
        # restore today's lock if analytics restarted after 09:30
        raw = await self.r.get(bus.k_top(day.isoformat()))
        if raw:
            self.top = bus.loads(raw)
            for sid, px in (self.top.get("lock_px") or {}).items():
                st = self.states.get(int(sid))
                if st:
                    st.lock_px = px
                    st.max_since_lock = st.min_since_lock = px
        self.scorecard_done = bool(await self.r.get(f"td:scorecard:done:{day.isoformat()}"))
        # restore today's minute bars (analytics restart mid-session)
        pipe = self.r.pipeline(transaction=False)
        for s in self.eqs:
            pipe.get(bus.k_bars(s.sid))
        for s, raw in zip(self.eqs, await pipe.execute()):
            if raw:
                d = bus.loads(raw)
                if d.get("day") == day.isoformat():
                    s.bars = {int(b[0]): list(b[1:]) for b in d.get("bars", [])}
                    # rebuild opening range and since-lock extremes from the restored bars
                    orb = [b for m, b in s.bars.items() if m < settings.or_minutes]
                    if orb:
                        s.orh = max(b[1] for b in orb)
                        s.orl = min(b[2] for b in orb)
                    if s.lock_px:
                        lock_m = minute_of_session(at(day, parse_hms(settings.lock_time)))
                        after = [b for m, b in s.bars.items() if m >= lock_m]
                        if after:
                            s.max_since_lock = max(s.lock_px, *(b[1] for b in after))
                            s.min_since_lock = min(s.lock_px, *(b[2] for b in after))
        await self._merge_seed_bars(day)
        asyncio.create_task(self.refresh_swing(initial=True))
        log.info("context loaded for %s (%d daily series with history)", day, n_hist)

    async def _merge_seed_bars(self, day) -> int:
        """Today's 1-minute bars fetched by the ingestor's history bootstrap (late start): fill the minutes
        this process never saw, then rebuild the opening range from them."""
        from .bootstrap import k_seed
        pipe = self.r.pipeline(transaction=False)
        for s in self.eqs:
            pipe.get(k_seed(s.sid))
        n = 0
        for s, raw in zip(self.eqs, await pipe.execute()):
            d = bus.loads(raw) if raw else None
            if not d or d.get("day") != day.isoformat():
                continue
            for m, o, h, l, c, v in d.get("bars", []):
                s.bars.setdefault(int(m), [o, h, l, c, v])
            orb = [b for m, b in s.bars.items() if m < settings.or_minutes]
            if orb:
                s.orh = max(b[1] for b in orb)
                s.orl = min(b[2] for b in orb)
            if not s.open and s.bars:
                s.open = s.bars[min(s.bars)][0]
            n += 1
        return n

    async def reload_history(self, what: str):
        """The ingestor finished a history stage (live mode): reload context, previous closes and seed bars."""
        if self.day is None:
            return
        n_hist = await self._apply_history(self.day)
        for s in list(self.eqs) + list(self.idx.values()):
            if not s.pc and s.ctx.pdc:
                s.pc = s.ctx.pdc
        seeded = await self._merge_seed_bars(self.day)
        if what in ("daily", "minutes"):
            asyncio.create_task(self.refresh_swing(initial=True))
        log.info("history reloaded after '%s': %d daily series, %d stocks with today's bars", what, n_hist, seeded)

    # -------------------------------------------------------------- consumers
    async def consume_ticks(self):
        last_id = "$"
        while True:
            resp = await self.r.xread({bus.STREAM_TICKS: last_id}, count=200, block=1000)
            if not resp:
                continue
            for _, entries in resp:
                for eid, fields in entries:
                    last_id = eid
                    batch = bus.loads(fields[b"b"])
                    clock = batch["ts"]
                    day = from_epoch(clock).date()
                    if self.day != day:
                        await self.new_day(day)
                    self.clock = clock
                    for p in batch["p"]:
                        st = self.states.get(p.get("sid"))
                        if st is not None:
                            st.apply(p, clock, settings.or_minutes)
                    self.ticks_seen += len(batch["p"])

    async def new_day(self, day):
        log.info("new trading day %s", day)
        for s in self.states.values():
            s.bars.clear()
            s.orh = s.orl = 0.0
            s.lock_px = 0.0
            s.open = s.high = s.low = 0.0
        self.top = {"locked": False}
        self.scorecard_done = False
        await self.load_day_context(day)

    async def consume_news(self):
        last_id = "0"
        while True:
            resp = await self.r.xread({bus.STREAM_NEWS: last_id}, count=100, block=2000)
            if not resp:
                continue
            changed = False
            for _, entries in resp:
                for eid, fields in entries:
                    last_id = eid
                    ev = self.news.add(bus.loads(fields[b"j"]))
                    if ev:
                        changed = True
                        self.new_news_symbols.update(ev["symbols"])
            if changed:
                await bus.publish_snapshot(self.r, "news", {"items": self.news.latest(150)})

    async def consume_events(self):
        last_id = "$"
        while True:
            resp = await self.r.xread({bus.STREAM_EVENTS: last_id}, count=50, block=2000)
            if not resp:
                continue
            for _, entries in resp:
                for eid, fields in entries:
                    last_id = eid
                    ev = bus.loads(fields[b"j"])
                    if ev.get("type") == "chain":
                        await self.update_chain(ev["name"])
                    elif ev.get("type") == "history":
                        asyncio.create_task(self.reload_history(ev.get("what", "")))

    async def update_chain(self, name: str):
        raw = await self.r.get(bus.k_chain_raw(name))
        if not raw:
            return
        d = bus.loads(raw)
        lot = self.futs_by_symbol.get(name).lot if name in self.futs_by_symbol else 1
        a = optmath.chain_analytics(d["raw"], lot=lot, name=name, expiry=d.get("expiry"))
        a.update({"expiry": d.get("expiry"), "expiries": d.get("expiries", []), "ts": d.get("ts")})
        prev = self.chain_cache.get(name)
        a["pcr_hist"] = ((prev or {}).get("pcr_hist") or [])[-240:] + [[d.get("ts"), a.get("pcr_oi")]]
        self.chain_cache[name] = a
        await bus.publish_snapshot(self.r, f"chain:{name}", a)

    # ------------------------------------------------------------------ views
    def minute(self) -> float:
        dt = from_epoch(self.clock)
        return (dt - at(dt.date(), MARKET_OPEN)).total_seconds() / 60.0

    def nifty_chg(self) -> float:
        n = self.idx.get("NIFTY")
        return n.chg if n and n.ltp and n.pc else 0.0

    def compute_rank(self) -> list[dict]:
        m = self.minute()
        session_open = at(from_epoch(self.clock).date(), MARKET_OPEN).timestamp()
        scores = {s.symbol: self.news.score(s.symbol, self.clock, session_open) for s in self.eqs}
        rows = intraday.rank(self.eqs, self.futs_by_symbol, self.nifty_chg(), m, scores, self.cfg,
                             settings.or_minutes)
        return rows

    async def maybe_lock(self):
        dt = from_epoch(self.clock)
        if self.top.get("locked") or self.top.get("missed") or phase(dt) != "OPEN" \
                or dt.timetz().replace(tzinfo=None) < self.lock_t:
            return
        # a lock taken long after LOCK_TIME would not be the 09:30 strategy — skip the day instead
        late = (datetime.combine(dt.date(), dt.timetz().replace(tzinfo=None))
                - datetime.combine(dt.date(), self.lock_t)).total_seconds()
        if late > LOCK_GRACE_S:
            self.top = {"locked": False, "missed": True, "date": dt.date().isoformat()}
            log.warning("missed today's %s lock (no ranking at lock time) — live ranking only, not scored",
                        settings.lock_time)
            return
        if not self.rows:
            return
        longs = intraday.pick(self.rows, self.cfg, "long")
        shorts = intraday.pick(self.rows, self.cfg, "short")
        lock_px = {str(s.sid): s.ltp for s in self.eqs if s.ltp}
        for s in self.eqs:
            if s.ltp:
                s.lock_px = s.ltp
                s.max_since_lock = s.min_since_lock = s.ltp
        self.top = {
            "locked": True, "date": dt.date().isoformat(), "lock_ts": self.clock, "lock_time": settings.lock_time,
            "long": [self._pick_view(r, "long") for r in longs],
            "short": [self._pick_view(r, "short") for r in shorts],
            "lock_px": lock_px,
            "lock_scores": {str(r["sid"]): [r["score"], r["score_s"]] for r in self.rows},
            "abstained": {"long": self.cfg.top_n - len(longs), "short": self.cfg.top_n - len(shorts)},
        }
        await self.r.set(bus.k_top(dt.date().isoformat()), bus.dumps(self.top), ex=7 * 86400)
        await bus.publish_event(self.r, "event", {"kind": "LOCK", "msg": f"Top picks locked at {settings.lock_time}",
                                                  "long": [p["symbol"] for p in self.top["long"]],
                                                  "short": [p["symbol"] for p in self.top["short"]]})
        log.info("LOCKED long=%s short=%s", [p["symbol"] for p in self.top["long"]],
                 [p["symbol"] for p in self.top["short"]])

    def _pick_view(self, r: dict, side: str) -> dict:
        s = self.eq_by_symbol[r["symbol"]]
        atr = r["atr"] or s.ltp * 0.02
        entry = r["ltp"]
        if side == "long":
            inval = r["orl"] if r["orl"] and r["orl"] < entry else entry - 0.6 * atr
            tgt = entry + 1.0 * atr
        else:
            inval = r["orh"] if r["orh"] and r["orh"] > entry else entry + 0.6 * atr
            tgt = entry - 1.0 * atr
        return {"sid": r["sid"], "symbol": r["symbol"], "sector": r["sector"], "entry": entry,
                "score": r["score"] if side == "long" else r["score_s"], "chips": intraday.factor_chips(r, side),
                "f": r["f"], "rvol": r["rvol"], "gap": r["gap"], "chg_at_lock": r["chg"], "headline": r["headline"],
                "invalidation": safe(inval), "target": safe(tgt), "oi_tag": r["oi_tag"]}

    def top_view(self) -> dict:
        if self.top.get("locked"):
            out = {k: v for k, v in self.top.items() if k not in ("lock_px", "lock_scores")}
            for side in ("long", "short"):
                live = []
                for p in self.top.get(side, []):
                    s = self.eq_by_symbol.get(p["symbol"])
                    q = dict(p)
                    if s and s.ltp:
                        sgn = 1 if side == "long" else -1
                        q["ltp"] = safe(s.ltp)
                        q["chg"] = safe(s.chg)
                        q["pnl"] = safe(sgn * (s.ltp / p["entry"] - 1) * 100)
                        q["mfe"] = safe(((s.max_since_lock / p["entry"] - 1) * 100) if side == "long"
                                        else ((1 - s.min_since_lock / p["entry"]) * 100))
                        q["mae"] = safe(((s.min_since_lock / p["entry"] - 1) * 100) if side == "long"
                                        else ((1 - s.max_since_lock / p["entry"]) * 100))
                        q["status"] = ("TARGET" if (sgn * (s.ltp - p["target"]) >= 0) else
                                       "INVALID" if (sgn * (s.ltp - p["invalidation"]) <= 0) else "LIVE")
                        q["spark"] = [b[3] for _, b in sorted(s.bars.items())][-90:]
                    live.append(q)
                out[side] = live
            # how the picks rank vs everything since the lock
            out["universe_since_lock"] = self._since_lock_stats()
            return out
        longs = intraday.pick(self.rows, self.cfg, "long") if self.rows else []
        shorts = intraday.pick(self.rows, self.cfg, "short") if self.rows else []
        dt = from_epoch(self.clock)
        lock_dt = datetime.combine(dt.date(), self.lock_t, tzinfo=IST)
        def prov(r: dict, side: str) -> dict:
            s = self.eq_by_symbol.get(r["symbol"])
            spark = [b[3] for _, b in sorted(s.bars.items())][-90:] if s else []
            return self._pick_view(r, side) | {"ltp": r["ltp"], "chg": r["chg"], "spark": spark}

        return {"locked": False, "provisional": True, "lock_time": settings.lock_time,
                "missed": bool(self.top.get("missed")),
                "seconds_to_lock": max(0, int((lock_dt - dt).total_seconds())),
                "long": [prov(r, "long") for r in longs],
                "short": [prov(r, "short") for r in shorts]}

    def _since_lock_stats(self) -> dict:
        rets = [(s.symbol, (s.ltp / s.lock_px - 1) * 100) for s in self.eqs if s.lock_px and s.ltp]
        if not rets:
            return {}
        rets.sort(key=lambda x: -x[1])
        avg = sum(r for _, r in rets) / len(rets)
        top5 = {sym for sym, _ in rets[:5]}
        top20 = {sym for sym, _ in rets[:20]}
        bot5 = {sym for sym, _ in rets[-5:]}
        longs = [p["symbol"] for p in self.top.get("long", [])]
        shorts = [p["symbol"] for p in self.top.get("short", [])]
        retmap = dict(rets)
        lavg = sum(retmap.get(s, 0) for s in longs) / len(longs) if longs else 0
        savg = sum(-retmap.get(s, 0) for s in shorts) / len(shorts) if shorts else 0
        return {"universe_avg": safe(avg), "long_avg": safe(lavg), "short_avg": safe(savg),
                "long_hits_top5": sum(1 for s in longs if s in top5),
                "long_hits_top20": sum(1 for s in longs if s in top20),
                "short_hits_bottom5": sum(1 for s in shorts if s in bot5),
                "leaders": [{"symbol": s, "ret": safe(r)} for s, r in rets[:5]],
                "laggards": [{"symbol": s, "ret": safe(r)} for s, r in rets[-5:][::-1]]}

    async def maybe_scorecard(self):
        dt = from_epoch(self.clock)
        if self.scorecard_done or not self.top.get("locked"):
            return
        if dt.timetz().replace(tzinfo=None) < MARKET_CLOSE:
            return
        st = self._since_lock_stats()
        scores = self.top.get("lock_scores", {})
        pairs = [(scores[str(s.sid)][0], (s.ltp / s.lock_px - 1) * 100) for s in self.eqs
                 if s.lock_px and s.ltp and str(s.sid) in scores]
        ic = intraday.spearman([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else float("nan")
        longs = [p["symbol"] for p in self.top.get("long", [])]
        green = sum(1 for p in self.top.get("long", [])
                    if self.eq_by_symbol[p["symbol"]].ltp > p["entry"])
        card = {"date": dt.date().isoformat(), "ic": safe(ic, 3) if ic == ic else None,
                "n_long": len(longs), "long_green": green, **st,
                "excess_long": safe(st.get("long_avg", 0) - st.get("universe_avg", 0)),
                "picks": longs, "shorts": [p["symbol"] for p in self.top.get("short", [])]}
        await self.r.lpush(bus.KEY_SCORECARD, bus.dumps(card))
        await self.r.ltrim(bus.KEY_SCORECARD, 0, 499)
        await self.r.set(f"td:scorecard:done:{dt.date().isoformat()}", "1", ex=3 * 86400)
        out = Path(settings.data_dir) / "scorecards"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{card['date']}.json").write_text(json.dumps(card, indent=2))
        self.scorecard_done = True
        await bus.publish_event(self.r, "event", {"kind": "SCORECARD", "msg": "Daily scorecard recorded", "card": card})
        log.info("scorecard %s", card)

    async def refresh_swing(self, initial: bool = False):
        try:
            live = {}
            m = self.minute()
            for s in self.eqs:
                if s.ltp and s.open:
                    frac = 1.0
                    if s.ctx.base_cum and 0 <= m < 375:
                        frac = max(0.05, s.ctx.base_cum[int(m)] / s.ctx.base_cum[-1])
                    live[s.symbol] = {"ts": int(at(from_epoch(self.clock).date(), MARKET_OPEN).replace(hour=0, minute=0).timestamp()),
                                      "open": s.open, "high": s.high, "low": s.low, "close": s.ltp,
                                      "volume_proj": s.vol / frac if frac else s.vol}
            nifty = self.daily.get("NIFTY")
            eq_daily = {k: v for k, v in self.daily.items() if k in self.eq_by_symbol}
            rows, feats = await asyncio.to_thread(swing.scan, eq_daily, SECTOR_OF, nifty, live)
            if initial or not self.swing_stats:
                self.swing_stats = await asyncio.to_thread(_swing_stats, eq_daily)
            for r in rows:
                st = self.eq_by_symbol.get(r["symbol"])
                r["sid"] = st.sid if st else None
                r["chg"] = safe(st.chg) if st else 0
                r["stats"] = self.swing_stats.get(r["setup"]) if r["setup"] else None
            self.swing_rows = rows
            await bus.publish_snapshot(self.r, "swing", {"ts": self.clock, "rows": rows, "stats": self.swing_stats,
                                                         "rules": {"target": swing.TARGET * 100, "stop": swing.STOP * 100,
                                                                   "horizon": swing.HORIZON}})
        except Exception:
            log.exception("swing refresh failed")

    async def publish_loop(self):
        tick = 0
        while True:
            t0 = time.monotonic()
            try:
                await self.publish_once(tick)
            except Exception:
                log.exception("publish failed")
            tick += 1
            await asyncio.sleep(max(0.05, 1.0 - (time.monotonic() - t0)))

    async def publish_once(self, tick: int):
        if self.day is None:
            return
        dt = from_epoch(self.clock)
        ph = phase(dt)
        await bus.publish_snapshot(self.r, "clock", {"ts": self.clock, "phase": ph, "date": dt.date().isoformat(),
                                                     "time": dt.strftime("%H:%M:%S"), "source": settings.data_source,
                                                     "minute": safe(self.minute(), 1)})
        self.rows = self.compute_rank()
        await self.maybe_lock()
        await self.maybe_scorecard()
        rows_by_symbol = {r["symbol"]: r for r in self.rows}
        indices = []
        for name, s in self.idx.items():
            if s.ltp:
                indices.append({"name": name, "sid": s.sid, "ltp": safe(s.ltp), "pc": safe(s.pc), "chg": safe(s.chg),
                                "o": safe(s.open), "h": safe(s.high), "l": safe(s.low)})
        await bus.publish_snapshot(self.r, "market", {
            "ts": self.clock, "phase": ph, "indices": indices, "breadth": market.breadth(self.eqs),
            "sectors": market.sectors(self.eqs), "movers": market.movers(self.rows)})
        slim = [{k: r[k] for k in ("sid", "symbol", "sector", "ltp", "chg", "gap", "rvol", "resid", "vwap_atr",
                                   "or_pos", "orb", "oi_tag", "oi_chg", "news", "news_type", "headline", "obi",
                                   "turnover", "score", "score_s", "rank", "eligible", "f", "atr_pct", "vwap",
                                   "orh", "orl", "high", "low", "open", "pc")} for r in self.rows]
        if tick % 2 == 0:   # the full cross-section (~90 KB) every 2 s keeps phones on mobile data happy
            await bus.publish_snapshot(self.r, "rank", {"ts": self.clock, "minute": safe(self.minute(), 1),
                                                        "rows": slim, "weights": self.cfg.weights})
        await bus.publish_snapshot(self.r, "top", self.top_view())
        quotes = {s.sid: [safe(s.ltp), safe(s.chg), int(s.vol), safe(s.vwap), int(s.oi)] for s in self.states.values()
                  if s.ltp}
        await bus.publish_snapshot(self.r, "quotes", {"ts": self.clock, "q": quotes})
        if tick % 2 == 0:
            await bus.publish_snapshot(self.r, "fno", market.fno_board(list(self.futs_by_symbol.values()),
                                                                     self.eq_by_symbol))
        # alerts
        raw_alerts = await self.r.hgetall(bus.KEY_ALERTS)
        defs = [bus.loads(v) for v in raw_alerts.values()]
        top_syms = {p["symbol"] for p in (self.top.get("long", []) if self.top.get("locked")
                                          else intraday.pick(self.rows, self.cfg, "long"))}
        fired = self.alerts.evaluate(defs, rows_by_symbol, top_syms, self.new_news_symbols, self.clock)
        self.new_news_symbols = set()
        for f in fired:
            a = f["alert"]
            await self.r.hset(bus.KEY_ALERTS, a["id"], bus.dumps(a))
            await self.r.lpush(bus.KEY_ALERT_LOG, bus.dumps(f))
            await self.r.ltrim(bus.KEY_ALERT_LOG, 0, 499)
            await bus.publish_event(self.r, "alert", f)
        # persist today's minute bars for charts (every 5 s)
        if time.monotonic() - self.last_bars_push > 5:
            self.last_bars_push = time.monotonic()
            pipe = self.r.pipeline(transaction=False)
            day = dt.date().isoformat()
            for s in list(self.eqs) + list(self.futs_by_symbol.values()) + list(self.idx.values()):
                if s.dirty and s.bars:
                    pipe.set(bus.k_bars(s.sid), bus.dumps({"day": day, "bars": s.bars_list(), "orh": s.orh,
                                                           "orl": s.orl, "vwap": s.vwap}), ex=2 * 86400)
                    s.dirty = False
            await pipe.execute()
        if tick % 60 == 30:
            asyncio.create_task(self.refresh_swing())
        if tick % 2 == 0:
            await self.r.set(bus.k_status("analytics"), bus.dumps({
                "pid": os.getpid(), "clock": self.clock, "phase": ph, "ticks": self.ticks_seen,
                "ranked": len(self.rows), "news": len(self.news.items), "chains": list(self.chain_cache),
                "swing_rows": len(self.swing_rows), "locked": self.top.get("locked", False), "updated": time.time(),
                "weights": self.cfg.weights}), ex=30)


def _swing_stats(eq_daily: dict) -> dict:
    """Historical hit-rate per swing setup (CPU heavy: runs in a worker thread)."""
    return swing.setup_stats({k: swing.features(v) for k, v in eq_daily.items() if len(v) >= 60})


async def supervise(fn):
    """Run a long-lived loop; if it ever raises, log the traceback and restart it (never die silently)."""
    while True:
        try:
            await fn()
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("%s crashed — restarting in 1 s", getattr(fn, "__name__", fn))
            await asyncio.sleep(1)


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    r = bus.connect()
    await r.ping()
    eng = Engine(r)
    await eng.setup()
    # initial clock from the ingestor (sim time may differ from wall time)
    st = bus.loads(await r.get(bus.k_status("ingestor")) or b"{}") or {}
    clock = st.get("clock") or time.time()
    eng.clock = clock
    await eng.load_day_context(from_epoch(clock).date())
    for name in settings.chain_list():
        await eng.update_chain(name)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    tasks = [asyncio.create_task(supervise(f)) for f in (eng.consume_ticks, eng.consume_news, eng.consume_events,
                                                         eng.publish_loop)]
    await stop.wait()
    for t in tasks:
        t.cancel()
    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
