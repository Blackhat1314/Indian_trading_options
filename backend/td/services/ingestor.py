"""INGESTOR — the only process that talks to Dhan.

  Dhan websockets / REST  (or the simulator)
        │  parse binary → normalized packets
        ▼
  Redis: XADD td:ticks (100 ms batches), XADD td:news, SET td:chain:raw:*, SET td:depth:*
        ▼
  analytics worker(s) → gateway(s) → every browser

One Dhan subscription therefore serves any number of users.
Run:  python -m td.services.ingestor
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from datetime import datetime

from .. import bus
from ..config import settings
from ..dhan import constants as C
from ..dhan.instruments import Universe, build_universe
from ..dhan.parser import parse_cas, parse_depth, parse_market_feed
from ..lake import Lake
from ..timeutil import IST, from_epoch, now_ist, phase
from ..dhan.instruments import ikey
from .recorder import Recorder

log = logging.getLogger("td.ingestor")


class Publisher:
    """Buffers normalized packets and flushes them to a Redis stream in small batches."""

    def __init__(self, r, recorder: Recorder):
        self.r = r
        self.recorder = recorder
        self.buf: list[dict] = []
        self.clock = time.time()
        self.last_by_sid: dict[int, dict] = {}
        self.prev_updates: dict[int, list] = {}
        self.stats = {"packets": 0, "batches": 0, "pps": 0.0, "news": 0, "chains": 0}
        self._t0 = time.monotonic()
        self._n0 = 0

    def add(self, packets: list[dict], clock: float | None = None) -> None:
        if clock is not None:
            self.clock = clock
        for p in packets:
            # Dhan ids repeat across segments (NIFTY = IDX_I 13, ABB = NSE_EQ 13): publish the unique key
            if "sid" in p and "seg" in p and "dsid" not in p:
                p["dsid"] = p["sid"]
                p["sid"] = ikey(p["seg"], p["sid"])
            t = p.get("t")
            if t in ("quote", "full", "tick"):
                self.last_by_sid[p["sid"]] = p
            elif t == "prev":
                self.prev_updates[p["sid"]] = [p.get("pc"), p.get("poi")]
        self.buf.extend(packets)

    async def flush(self) -> None:
        if not self.buf:
            return
        batch, self.buf = self.buf, []
        self.stats["packets"] += len(batch)
        self.stats["batches"] += 1
        await self.r.xadd(bus.STREAM_TICKS, {"b": bus.dumps({"ts": self.clock, "p": batch})},
                          maxlen=settings.stream_maxlen, approximate=True)
        if self.prev_updates:
            upd, self.prev_updates = self.prev_updates, {}
            await self.r.hset(bus.KEY_PREV, mapping={str(k): bus.dumps(v) for k, v in upd.items()})
        self.recorder.ticks(self.clock, batch)
        now = time.monotonic()
        if now - self._t0 >= 2:
            self.stats["pps"] = round((self.stats["packets"] - self._n0) / (now - self._t0), 1)
            self._t0, self._n0 = now, self.stats["packets"]

    async def news(self, items: list[dict]) -> None:
        for it in items:
            await self.r.xadd(bus.STREAM_NEWS, {"j": bus.dumps(it)}, maxlen=5000, approximate=True)
            self.recorder.add("news", self.clock, it)
            self.stats["news"] += 1


# =============================================================== simulator mode
async def run_sim(r, universe: Universe, pub: Publisher, stop: asyncio.Event) -> None:
    from ..sim.simulator import MarketSimulator, generate_history, history_end_for, sim_trading_date

    lake = Lake()
    by_key = {i.key: i for i in universe.instruments}
    sim_date = sim_trading_date()
    hist_end = history_end_for(sim_date)
    last = lake.last_ts("D", "NIFTY")
    if last is None or from_epoch(last).date() < hist_end:
        log.info("generating synthetic history up to %s (first run only) ...", hist_end)
        seed = settings.sim_seed or 7
        hist = await asyncio.to_thread(generate_history, universe, hist_end, 320, 25, seed)
        for interval, by_sym in hist.items():
            for sym, df in by_sym.items():
                await asyncio.to_thread(lake.write, interval, sym, df)
        log.info("synthetic history written to %s", lake.root)
    prev_close, adv = {}, {}
    for e in universe.equities():
        d = lake.read("D", e.symbol, last_n=25)
        if not d.empty:
            prev_close[e.symbol] = float(d["close"].iat[-1])
            adv[e.symbol] = float(d["volume"].tail(20).mean())
    for i in universe.indices():
        d = lake.read("D", i.symbol, last_n=2)
        if not d.empty:
            prev_close[i.symbol] = float(d["close"].iat[-1])
    sim = MarketSimulator(universe, sim_date, prev_close, adv, seed=settings.sim_seed, start=settings.sim_start)
    await pub.news(sim.drain_news())
    log.info("simulator: %s, %d stocks, speed %.1fx, start %s", sim_date, sim.n, settings.sim_speed,
             settings.sim_start)

    async def chains():
        while not stop.is_set():
            for name in settings.chain_list():
                exps = sim.expiries(name)
                if not exps:
                    continue
                raw = sim.option_chain(name, exps[0])
                if raw:
                    await r.set(bus.k_chain_raw(name), bus.dumps({"name": name, "expiry": exps[0],
                                                                  "expiries": exps, "ts": pub.clock, "raw": raw}))
                    await r.xadd(bus.STREAM_EVENTS, {"j": bus.dumps({"type": "chain", "name": name})},
                                 maxlen=2000, approximate=True)
                    pub.stats["chains"] += 1
            await asyncio.sleep(max(2.0, 15.0 / max(1.0, settings.sim_speed)))

    async def depth():
        while not stop.is_set():
            want = await r.smembers("td:depth:want")
            for raw in list(want)[:50]:
                inst = by_key.get(int(raw))
                if not inst:
                    continue
                for msg in sim.depth20(inst.sid):
                    sides = parse_depth(msg, 20)
                    book = {"sid": inst.key, "ts": pub.clock}
                    for s in sides:
                        book[s["side"]] = s["levels"]
                    await r.set(bus.k_depth(inst.key), bus.dumps(book), ex=30)
            await asyncio.sleep(1.0)

    async def ticker():
        step = 0.1
        while not stop.is_set():
            t0 = time.monotonic()
            msgs = sim.advance(step * settings.sim_speed)
            pkts: list[dict] = []
            for ch, m in msgs:
                if ch == "cas":
                    for c in parse_cas(m):
                        pkts.append(c)
                else:
                    pkts.extend(parse_market_feed(m))
            pub.add(pkts, sim.sim_t.timestamp())
            await pub.flush()
            news = sim.drain_news()
            if news:
                await pub.news(news)
            pub.recorder.maybe_flush(pub.clock)
            await asyncio.sleep(max(0.0, step - (time.monotonic() - t0)))

    async def status():
        while not stop.is_set():
            await r.set(bus.k_status("ingestor"), bus.dumps({
                "mode": "sim", "pid": os.getpid(), "sim_date": sim_date.isoformat(), "speed": settings.sim_speed,
                "clock": pub.clock, "phase": phase(from_epoch(pub.clock)), "instruments": len(universe.instruments),
                "stats": pub.stats, "feed": {"connected": 1, "reconnects": 0, "last_error": ""},
                "recorder_rows": pub.recorder.rows_written, "updated": time.time(),
            }), ex=30)
            await asyncio.sleep(2)

    await asyncio.gather(ticker(), chains(), depth(), status())


# ==================================================================== dhan mode
async def run_dhan(r, universe: Universe, pub: Publisher, stop: asyncio.Event) -> None:
    from ..dhan.feeds import CasFeed, Depth20Feed, MarketFeedManager, Sub
    from ..dhan.rest import DhanError, DhanRest
    from ..dhan.token import TokenKeeper
    from ..ratelimit import BudgetExhausted

    # owns the 24 h access token: renews it every day and, with DHAN_PIN + DHAN_TOTP_SECRET, logs in again after a
    # lapse; the REST client and the feeds read the current token on every call / connect
    keeper = TokenKeeper(r, role="owner")
    await keeper.start()
    rest = DhanRest(redis=r)

    def refused(e: DhanError) -> None:
        if e.status in (401, 403) or str(e.code) in ("807", "808", "809", "DH-901"):
            keeper.kick()

    subs: list[Sub] = []
    for i in universe.instruments:
        mode = {"IDX": C.REQ_SUB_TICKER, "EQ": C.REQ_SUB_QUOTE, "FUT": C.REQ_SUB_FULL}.get(i.kind, C.REQ_SUB_QUOTE)
        subs.append(Sub(i.seg, i.sid, mode))

    async def on_packets(pkts: list[dict]) -> None:
        pub.add(pkts, time.time())

    feed = MarketFeedManager(subs, on_packets, max_connections=3)
    by_symbol = {i.symbol: i for i in universe.instruments}
    from .bootstrap import HistoryBootstrap
    history = HistoryBootstrap(r, rest, universe, pub)

    async def flusher():
        while not stop.is_set():
            await pub.flush()
            pub.recorder.maybe_flush(pub.clock)
            await asyncio.sleep(0.1)

    async def depth_mgr():
        """Keep one 20-level depth socket on the instruments users are looking at (≤50)."""
        current: tuple = ()
        task: asyncio.Task | None = None
        latest: dict[int, dict] = {}

        async def on_depth(pkts):
            for p in pkts:
                if p.get("t") != "depth":
                    continue
                k = ikey(p["seg"], p["sid"])
                book = latest.setdefault(k, {"sid": k})
                book[p["side"]] = p["levels"]
                book["ts"] = time.time()
                await r.set(bus.k_depth(k), bus.dumps(book), ex=30)
        by_key = {i.key: i for i in universe.instruments}
        while not stop.is_set():
            want = tuple(sorted(int(x) for x in await r.smembers("td:depth:want")))[:50]
            if want != current:
                if task:
                    task.cancel()
                current = want
                dsubs = [Sub(by_key[k].seg, by_key[k].sid) for k in want
                         if k in by_key and by_key[k].seg in ("NSE_EQ", "NSE_FNO")]
                if dsubs:
                    task = asyncio.create_task(Depth20Feed(dsubs, on_depth).run())
            await asyncio.sleep(5)

    async def cas_mgr():
        """Closing Auction Session feed (Pro plan) between 15:29 and 16:00 IST."""
        if not settings.is_pro:
            return
        task = None
        while not stop.is_set():
            t = now_ist().time()
            active = t >= datetime.strptime("15:29", "%H:%M").time() and t < datetime.strptime("16:00", "%H:%M").time()
            if active and task is None:
                csubs = [Sub(i.seg, i.sid) for i in universe.equities()]

                async def on_cas(pkts):
                    pub.add(pkts, time.time())
                    for p in pkts:
                        pub.recorder.add("cas", time.time(), p)
                task = asyncio.create_task(CasFeed(csubs, on_cas).run())
            elif not active and task is not None:
                task.cancel()
                task = None
            await asyncio.sleep(10)

    async def chain_poller():
        expiries: dict[str, list[str]] = {}
        while not stop.is_set():
            for name in settings.chain_list():
                inst = by_symbol.get(name)
                if not inst:
                    continue
                seg = "IDX_I" if inst.kind == "IDX" else "NSE_EQ"
                try:
                    if name not in expiries:
                        expiries[name] = await rest.expiry_list(inst.sid, seg)
                    exps = [e for e in expiries[name] if e >= now_ist().date().isoformat()]
                    if not exps:
                        continue
                    raw = await rest.option_chain(inst.sid, seg, exps[0])
                    await r.set(bus.k_chain_raw(name), bus.dumps({"name": name, "expiry": exps[0], "expiries": exps[:6],
                                                                  "ts": time.time(), "raw": raw}))
                    await r.xadd(bus.STREAM_EVENTS, {"j": bus.dumps({"type": "chain", "name": name})},
                                 maxlen=2000, approximate=True)
                    pub.stats["chains"] += 1
                except DhanError as e:
                    refused(e)
                    log.warning("chain %s: %s", name, e)
            await asyncio.sleep(15)

    async def news_poller():
        if not settings.is_pro:
            return
        sid_to_symbol = {i.sid: i.symbol for i in universe.equities()}
        seen: set[str] = set()
        while not stop.is_set():
            try:
                items = await rest.news(limit=50)
                out = []
                for it in items:
                    obj = it.get("newsObject") or {}
                    title = obj.get("title") or ""
                    key = f"{title}|{it.get('publishDate')}"
                    if not title or key in seen:
                        continue
                    seen.add(key)
                    sym = sid_to_symbol.get(it.get("nseScripCode")) or (it.get("smSymbol") or "").upper()
                    out.append({"id": key, "ts": time.time(), "title": title, "text": obj.get("text", ""),
                                "sentiment": obj.get("overallSentiment"), "category": it.get("category", ""),
                                "subCategory": it.get("subCategory", ""), "symbols": [sym] if sym else [],
                                "isin": it.get("isinCode"), "source": "DHAN", "publishDate": it.get("publishDate")})
                if out:
                    await pub.news(out)
            except DhanError as e:
                refused(e)
                log.warning("news: %s", e)
            except BudgetExhausted as e:
                # out of today's data calls: pause news (a crash here would take the whole ingestor down)
                log.warning("news paused: %s", e)
                await asyncio.sleep(900)
                continue
            ph = phase(now_ist())
            await asyncio.sleep(20 if ph in ("OPEN", "PRE_OPEN", "PRE_OPEN_MATCH") else 180)

    async def status():
        while not stop.is_set():
            await r.set(bus.k_status("ingestor"), bus.dumps({
                "mode": "dhan", "plan": settings.dhan_plan, "pid": os.getpid(), "clock": time.time(),
                "phase": phase(now_ist()), "instruments": len(universe.instruments), "stats": pub.stats,
                "feed": {"connected": feed.stats.connected, "reconnects": feed.stats.reconnects,
                         "messages": feed.stats.messages, "last_error": feed.stats.last_error},
                "rest_calls": rest.calls, "data_budget_used": await rest.data_budget.used(),
                "data_budget_limit": settings.limit_data_per_day, "token": keeper.brief(),
                "recorder_rows": pub.recorder.rows_written, "updated": time.time(),
            }), ex=30)
            await asyncio.sleep(2)

    try:
        await asyncio.gather(feed.run(), flusher(), depth_mgr(), cas_mgr(), chain_poller(), news_poller(), status(),
                             history.run(stop))
    finally:
        await keeper.stop()
        await rest.close()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    r = bus.connect()
    await r.ping()
    universe = await asyncio.to_thread(build_universe)
    await r.set(bus.KEY_UNIVERSE, bus.dumps(universe.to_json()))
    log.info("universe published: %d instruments (%s mode)", len(universe.instruments), settings.data_source)
    recorder = Recorder(settings.recorder_dir, enabled=os.getenv("RECORDER", "1") == "1")
    pub = Publisher(r, recorder)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    runner = run_sim if settings.data_source == "sim" else run_dhan
    task = asyncio.create_task(runner(r, universe, pub, stop))
    await stop.wait()
    task.cancel()
    recorder.maybe_flush(pub.clock, force=True)
    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
