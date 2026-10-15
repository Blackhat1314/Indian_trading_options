"""GEO — world-events worker for the GEO desk.

One process polls the free world feeds (ships, conflict, fires, aircraft, chokepoint transits,
geopolitical risk, macro series), derives chokepoint status, fires at energy sites, an airspace
monitor, a live geo-risk index and a market-impact map, and publishes compact snapshots on the
same Redis bus the gateways already fan out (topics geo:*). Viewers cost nothing extra.

Demand-driven: the gateways report which geo topics have viewers; the ship websocket and the
aircraft polling pause when nobody is looking. GEO_DISABLE switches a source off for good.
Run:  python -m td.services.geo
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import signal
import time
from datetime import date, datetime, timedelta, timezone

from .. import bus
from ..config import settings
from ..geo import engine, sources
from ..geo.sim import GeoSim
from ..geo.sites import AIS_BOXES, FIRE_BOXES, load_sites

log = logging.getLogger("td.geo")
IST = timezone(timedelta(hours=5, minutes=30))
CLS_CODE = {"TANKER": 1, "CARGO": 2, "PASSENGER": 3, "MILITARY": 4, "FISHING": 5, "SERVICE": 6, "OTHER": 7,
            "UNKNOWN": 0}
K_FIREHIST = "td:geo:firehist"
K_AIRBASE = "td:geo:airbase"
K_RISKHIST = "td:geo:riskhist"
K_AISSTATIC = "td:geo:aisstatic"      # mmsi -> ship type / name / destination, kept across restarts
AIS_STATIC_TTL = 14 * 86400
AIS_ALL = ["PositionReport", "StandardClassBPositionReport", "ShipStaticData", "StaticDataReport"]
AIS_TYPES_ONLY = ["ShipStaticData", "StaticDataReport"]


def _calendar(today: date) -> list[dict]:
    """Scheduled macro events. RBI MPC FY27 and FOMC 2026-27 dates from the official calendars;
    CPI (MOSPI, 12th, 16:00 IST) and GDP (last working day of Feb/May/Aug/Nov) follow the usual pattern."""
    ev = []
    for d in ("2026-04-08", "2026-06-05", "2026-08-05", "2026-10-07", "2026-12-04", "2027-02-05"):
        ev.append({"date": d, "time": "10:00 IST", "title": "RBI policy decision", "kind": "RBI"})
    for d in ("2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28",
              "2026-12-09", "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09"):
        ev.append({"date": d, "time": "~23:30 IST", "title": "FOMC decision", "kind": "FED"})
    for k in range(0, 14):
        m = (today.month - 1 + k) % 12 + 1
        y = today.year + (today.month - 1 + k) // 12
        ev.append({"date": f"{y}-{m:02d}-12", "time": "16:00 IST", "title": "India CPI (MOSPI)", "kind": "CPI",
                   "approx": True})
        if m in (2, 5, 8, 11):
            last = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))
            while last.weekday() >= 5:
                last -= timedelta(days=1)
            ev.append({"date": last.isoformat(), "time": "16:00 IST", "title": "India GDP (quarterly)",
                       "kind": "GDP", "approx": True})
    try:
        from pathlib import Path
        extra = json.loads((Path(settings.data_dir) / "config" / "macro_calendar.json").read_text())
        ev += [e for e in extra if isinstance(e, dict) and "date" in e and "title" in e]
    except (OSError, ValueError):
        pass
    ev = [e for e in ev if e["date"] >= (today - timedelta(days=1)).isoformat()]
    ev.sort(key=lambda e: e["date"])
    return ev[:14]


class GeoService:
    def __init__(self):
        self.mode = settings.geo_mode
        self.off = settings.geo_disabled()
        self.sites = load_sites()
        self.r = bus.connect()
        self.sim = GeoSim(settings.sim_seed or None) if self.mode == "sim" else None
        self.stop = asyncio.Event()
        self.ships: dict[int, dict] = {}
        self.static: dict[int, dict] = {}
        self.transits: dict[str, dict[int, float]] = {}
        self.events: dict[str, dict] = {}
        self.hotspots: list[dict] = []
        self.fire_hist: dict[str, list[int]] = {}
        self.aircraft: list[dict] = []
        self.air_counts: dict[str, int] = {}
        self.air_base: dict[str, list[int]] = {}
        self.portwatch: dict[str, dict] = {}
        self.gpr: dict = {}
        self.fred: dict[str, list] = {}
        self.macro_live: dict[str, dict] = {}
        self.india_vix: float | None = None
        self.status: dict[str, dict] = {}
        self.demand: dict[str, int] = {}
        self.risk_hist: list[list] = []
        self.dirty: set[str] = {"sites"}
        self.pushed_news: set[str] = set()
        for layer, src in (("ships", "AISStream"), ("chokepoints", "IMF PortWatch + AIS"), ("conflict", "GDELT 2.0"),
                           ("fires", "NASA FIRMS (VIIRS)"), ("air", settings.adsb_api_url.split("//")[-1]),
                           ("risk", "Caldara–Iacoviello GPR + live proxy"), ("macro", "FRED + Dhan feed"),
                           ("satellite", "NASA GIBS / EOX (loaded by the browser)")):
            self._st(layer, "WAITING", src)
        self._st("satellite", "CLIENT", self.status["satellite"]["source"])

    # ------------------------------------------------------------ helpers
    def _st(self, layer: str, state: str, source: str | None = None, detail: str = "", n: int | None = None):
        cur = self.status.get(layer, {})
        if self.mode == "sim" and state in ("LIVE",):
            state = "SIM"
        if layer in self.off:
            state, detail = "DISABLED", "GEO_DISABLE on the host"
        self.status[layer] = {"state": state, "source": source or cur.get("source", ""), "detail": detail,
                              "updated": time.time() if state in ("LIVE", "SIM") else cur.get("updated"),
                              "n": n if n is not None else cur.get("n")}
        self.dirty.add("status")

    def wants(self, *topics: str) -> bool:
        return any(self.demand.get(f"geo:{t}", 0) > 0 for t in topics)

    async def _demand_loop(self):
        while not self.stop.is_set():
            try:
                tot: dict[str, int] = {}
                async for k in self.r.scan_iter(match="td:demand:*", count=50):
                    for t, n in (await self.r.hgetall(k)).items():
                        tot[t.decode()] = tot.get(t.decode(), 0) + int(n)
                self.demand = tot
            except Exception:
                log.debug("demand read failed", exc_info=True)
            await asyncio.sleep(4)

    async def _sleep(self, s: float):
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=s)
        except asyncio.TimeoutError:
            pass

    async def _nap(self, s: float, topic: str):
        """Sleep up to s seconds, but wake as soon as someone starts viewing `topic` (demand is re-read every 4 s)."""
        was, end = self.wants(topic), time.monotonic() + s
        while not self.stop.is_set() and time.monotonic() < end:
            await self._sleep(min(2.0, end - time.monotonic()))
            if not was and self.wants(topic):
                return

    # ------------------------------------------------------------ sim mode
    async def _sim_loop(self):
        sim = self.sim
        sites = self.sites["energy"]
        self.fire_hist = sim.fire_history(sites)
        pw_rows = sim.portwatch(self.sites["chokepoints"])
        self.portwatch = {k: sources.transit_stats(v) for k, v in pw_rows.items()}
        self.gpr = sim.gpr()
        self.fred = sim.fred()
        for layer in ("ships", "chokepoints", "conflict", "fires", "air", "risk", "macro"):
            self._st(layer, "SIM", detail="simulated feed — set GEO_SOURCE=live for real data")
        k = 0
        while not self.stop.is_set():
            sim.step()
            self.ships = {m: s for m, s in sim.ships.items()}
            engine.update_transits(self.sites["chokepoints"], self.ships, self.transits, time.time())
            self.events = {e["id"]: e for e in sim.events}
            if k % 60 == 0:
                self.hotspots = sim.fires(sites)
                self.dirty.add("fires")
            if k % 6 == 0:
                self.aircraft = sim.aircraft()
                self.air_counts, self.air_base = sim.air_counts()
                self.dirty.add("air")
            self.macro_live = sim.macro_live()
            self.dirty.update({"ships", "conflict", "chokepoints", "macro"})
            self._st("ships", "SIM", n=len(self.ships))
            self._st("conflict", "SIM", n=len(self.events))
            self._st("fires", "SIM", n=len(self.hotspots))
            self._st("air", "SIM", n=len(self.aircraft))
            k += 1
            await self._sleep(5)

    # ------------------------------------------------------------ live: ships
    async def _ais_loop(self):
        if "ships" in self.off:
            return
        if not settings.aisstream_api_key:
            self._st("ships", "NEEDS KEY", detail="free key from aisstream.io → AISSTREAM_API_KEY")
            return
        import websockets
        # a ship's type (tanker, cargo…) only arrives in its static report, every ~6 min: remember what we learnt
        try:
            for k, v in (await self.r.hgetall(K_AISSTATIC)).items():
                self.static[int(k)] = json.loads(v)
            log.info("AIS: %d known ship types loaded", len(self.static))
        except Exception:
            log.debug("AIS static cache unreadable", exc_info=True)
        learnt: dict[int, dict] = {}
        backoff = 2
        while not self.stop.is_set():
            # Someone viewing: positions + types. No one viewing: types only (~15 messages a minute), so tankers are
            # already labelled when the map opens instead of showing up grey for the first half hour.
            full = self.wants("ships", "chokepoints")
            if not full:
                self._st("ships", "PAUSED", detail="no one is viewing ships — learning ship types in the background")
            try:
                async with websockets.connect(sources.AIS_URL, max_size=2 ** 22, ping_interval=20) as ws:
                    await ws.send(json.dumps({"APIKey": settings.aisstream_api_key, "BoundingBoxes": AIS_BOXES,
                                              "FilterMessageTypes": AIS_ALL if full else AIS_TYPES_ONLY}))
                    if full:
                        self._st("ships", "LIVE", detail="connected")
                    backoff = 2
                    idle_since = None
                    last_prune = time.time()
                    while not self.stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), 5)
                        except asyncio.TimeoutError:
                            raw = None
                        p = sources.parse_ais(json.loads(raw)) if raw else None
                        if p:
                            kind, d = p
                            if kind == "pos":
                                st = self.static.get(d["mmsi"], {})
                                d.update(ts=time.time(), type=st.get("type"), cls=sources.ship_class(st.get("type")),
                                         dest=st.get("dest", ""), name=d["name"] or st.get("name", ""))
                                self.ships[d["mmsi"]] = d
                            else:
                                self.static[d["mmsi"]] = d
                                learnt[d["mmsi"]] = d
                                if d["mmsi"] in self.ships:
                                    self.ships[d["mmsi"]].update(type=d["type"], cls=sources.ship_class(d["type"]),
                                                                 dest=d["dest"] or self.ships[d["mmsi"]].get("dest", ""))
                        if not full and self.wants("ships", "chokepoints"):
                            break                                   # a viewer arrived: reconnect with positions
                        now = time.time()
                        if now - last_prune > 30:
                            last_prune = now
                            if learnt:
                                try:
                                    await self.r.hset(K_AISSTATIC, mapping={str(m): json.dumps(v)
                                                                            for m, v in learnt.items()})
                                    await self.r.expire(K_AISSTATIC, AIS_STATIC_TTL)
                                    learnt.clear()
                                except Exception:
                                    log.debug("AIS static cache write failed", exc_info=True)
                            if not full:
                                self._st("ships", "PAUSED", detail="no one is viewing ships — learning ship types in "
                                                                   f"the background ({len(self.static)} known)")
                                continue
                            for m in [m for m, s in self.ships.items() if now - s["ts"] > 1800]:
                                del self.ships[m]
                            engine.update_transits(self.sites["chokepoints"], self.ships, self.transits, now)
                            self.dirty.update({"ships", "chokepoints"})
                            self._st("ships", "LIVE", n=len(self.ships))
                            if not self.wants("ships", "chokepoints"):
                                idle_since = idle_since or now
                                if now - idle_since > 60:
                                    break                           # back to types only
                            else:
                                idle_since = None
            except Exception as e:
                self._st("ships", "ERROR", detail=str(e)[:160])
                log.warning("AIS stream: %s", e)
                await self._sleep(backoff)
                backoff = min(120, backoff * 2)

    # ------------------------------------------------------------ live: conflict (GDELT)
    def _add_events(self, evs: list[dict]):
        for e in evs:
            self.events[e["id"]] = e
        cutoff = time.time() - 26 * 3600
        self.events = {k: v for k, v in self.events.items() if v["ts"] >= cutoff}
        merged = sources.gdelt_dedupe(list(self.events.values()))
        self.events = {e["id"]: e for e in merged}
        self.dirty.update({"conflict", "chokepoints"})

    async def _gdelt_loop(self):
        if "conflict" in self.off:
            return
        seen: set[str] = set()
        async with sources.client(60) as c:
            # backfill recent hours so the map is not empty after a restart
            stamps = sources.gdelt_stamps_back(datetime.now(timezone.utc), settings.geo_gdelt_backfill_hours)
            for i, st in enumerate(stamps):
                if self.stop.is_set():
                    return
                url = sources.GDELT_FILE.format(stamp=st)
                try:
                    text = await sources.gdelt_fetch(c, url)
                    if text:
                        evs = await asyncio.to_thread(sources.parse_gdelt_export, text)
                        self._add_events(evs)
                    seen.add(url)
                    self._st("conflict", "LIVE", detail=f"backfilling {i + 1}/{len(stamps)}", n=len(self.events))
                except Exception as e:
                    self._st("conflict", "ERROR", detail=f"GDELT: {str(e)[:140]}")
                    log.warning("gdelt backfill %s: %s", st, e)
                    break
                await asyncio.sleep(0.5)
            while not self.stop.is_set():
                try:
                    url = await sources.gdelt_latest_url(c)
                    if url and url not in seen:
                        text = await sources.gdelt_fetch(c, url)
                        evs = await asyncio.to_thread(sources.parse_gdelt_export, text)
                        self._add_events(evs)
                        seen.add(url)
                    self._st("conflict", "LIVE", detail="15-minute GDELT updates", n=len(self.events))
                except Exception as e:
                    self._st("conflict", "ERROR", detail=f"GDELT: {str(e)[:140]}")
                    log.warning("gdelt: %s", e)
                await self._sleep(300)

    # ------------------------------------------------------------ live: fires (FIRMS)
    async def _firms_loop(self):
        if "fires" in self.off:
            return
        if not settings.firms_map_key:
            self._st("fires", "NEEDS KEY", detail="free MAP_KEY from firms.modaps.eosdis.nasa.gov → FIRMS_MAP_KEY")
            return
        raw = await self.r.hgetall(K_FIREHIST)
        hist = {k.decode(): json.loads(v) for k, v in raw.items()}        # site -> {date: count}
        first = True
        async with sources.client(60) as c:
            while not self.stop.is_set():
                pts: list[dict] = []
                try:
                    for box in FIRE_BOXES:
                        for src in sources.FIRMS_SOURCES:
                            pts += await sources.firms_fetch(c, settings.firms_map_key, box, src, 5 if first else 2)
                    pts = sources.dedupe_fires(pts)
                    now = time.time()
                    self.hotspots = [p for p in pts if now - p["ts"] < 48 * 3600]
                    self.hotspots.sort(key=lambda p: -p["frp"])
                    self.hotspots = self.hotspots[:6000]
                    # per-site daily counts for completed UTC days -> each site's own baseline
                    utc_today = datetime.now(timezone.utc).date()
                    span = [(utc_today - timedelta(days=k)).isoformat() for k in range(1, 5 if first else 2)]
                    for s in self.sites["energy"]:
                        days: dict[str, int] = hist.get(s["id"], {})
                        counts: dict[str, int] = {}
                        for p in pts:
                            if abs(p["lat"] - s["lat"]) > 1:
                                continue
                            if engine.haversine_km(s["lat"], s["lon"], p["lat"], p["lon"]) <= s.get("r_km", 8):
                                d = datetime.fromtimestamp(p["ts"], timezone.utc).date().isoformat()
                                counts[d] = counts.get(d, 0) + 1
                        for d in span:
                            days[d] = max(days.get(d, 0), counts.get(d, 0))
                        hist[s["id"]] = dict(sorted(days.items())[-21:])
                    await self.r.hset(K_FIREHIST, mapping={k: json.dumps(v) for k, v in hist.items()})
                    self.fire_hist = {k: list(v.values()) for k, v in hist.items()}
                    first = False
                    self.dirty.add("fires")
                    self._st("fires", "LIVE", detail="VIIRS S-NPP / NOAA-20 / NOAA-21, every 30 min",
                             n=len(self.hotspots))
                except Exception as e:
                    self._st("fires", "ERROR", detail=f"FIRMS: {str(e)[:140]}")
                    log.warning("firms: %s", e)
                await self._sleep(1800)

    # ------------------------------------------------------------ live: aircraft (adsb.lol)
    async def _adsb_loop(self):
        if "air" in self.off:
            return
        regions = self.sites["air_regions"]
        base_raw = await self.r.hgetall(K_AIRBASE)
        base = {k.decode(): json.loads(v) for k, v in base_raw.items()}   # region -> {hour: [counts]}
        i, last_mil, strikes = 0, 0.0, 0
        async with sources.client(20) as c:
            while not self.stop.is_set():
                viewing = self.wants("air")
                try:
                    now = time.time()
                    if viewing and now - last_mil > 120:
                        js = await sources.adsb_get(c, settings.adsb_api_url, "/v2/mil")
                        mil = sources.parse_adsb(js, mil=True)
                        civ = [a for a in self.aircraft if not a["mil"]]
                        self.aircraft = mil + civ
                        last_mil = now
                    reg = regions[i % len(regions)]
                    i += 1
                    js = await sources.adsb_get(c, settings.adsb_api_url,
                                                f"/v2/point/{reg['lat']}/{reg['lon']}/{reg['nm']}")
                    acs = sources.parse_adsb(js)
                    airborne = [a for a in acs if (a["alt"] or 0) > 3000]
                    self.air_counts[reg["id"]] = len(airborne)
                    hour = str(datetime.now(IST).hour)
                    hb = base.setdefault(reg["id"], {})
                    lst = hb.get(hour, [])
                    stamp = datetime.now(IST).strftime("%Y%m%d%H")
                    if not lst or lst[-1][0] != stamp:
                        lst = (lst + [[stamp, len(airborne)]])[-14:]
                        hb[hour] = lst
                        await self.r.hset(K_AIRBASE, reg["id"], json.dumps(hb))
                    self.air_base[reg["id"]] = [n for s, n in hb.get(hour, [])[:-1]]
                    if viewing:
                        seen = {a["hex"] for a in self.aircraft}
                        self.aircraft += [a for a in acs if a["hex"] not in seen]
                        self.aircraft = self.aircraft[-3000:]
                    self.dirty.add("air")
                    strikes = 0
                    self._st("air", "LIVE", detail="military + airspace watch" if viewing else
                             "airspace watch only (no one viewing aircraft)", n=len(self.aircraft))
                except Exception as e:
                    wait = sources.rate_limit_wait(e, strikes)
                    if wait is not None:
                        # adsb.lol asking us to slow down is not an outage: keep the aircraft we have, back off
                        strikes += 1
                        self._st("air", "LIVE", detail=f"adsb.lol rate limit — pausing {wait:.0f} s",
                                 n=len(self.aircraft))
                        log.info("adsb: rate limited, pausing %.0f s", wait)
                        await self._sleep(wait)
                        continue
                    self._st("air", "ERROR", detail=f"ADS-B: {str(e)[:140]}")
                    log.warning("adsb: %s", e)
                await self._nap(30 if viewing else 60, "air")

    # ------------------------------------------------------------ live: slow series
    async def _portwatch_loop(self):
        if "chokepoints" in self.off:
            return
        async with sources.client(60) as c:
            while not self.stop.is_set():
                ok, err, total = 0, "", len(self.sites["chokepoints"])
                for n, cp in enumerate(self.sites["chokepoints"], 1):
                    try:
                        rows = await sources.portwatch_fetch(c, cp["portwatch"], 430)
                        self.portwatch[cp["id"]] = sources.transit_stats(rows)
                        ok += 1
                    except Exception as e:
                        err = f"{type(e).__name__}: {str(e)[:100]}"
                        log.warning("portwatch %s: %s", cp["name"], e)
                    if self.status.get("chokepoints", {}).get("state") == "WAITING":
                        self._st("chokepoints", "WAITING", detail=f"loading PortWatch {n}/{total}")
                    await asyncio.sleep(1)
                self.dirty.add("chokepoints")
                self._st("chokepoints", "LIVE" if ok else "ERROR",
                         detail=(f"PortWatch transits for {ok}/{total} chokepoints (published with a few days' lag)"
                                 if ok else f"IMF PortWatch unreachable from this server — {err}"))
                await self._sleep(6 * 3600 if ok else 900)

    async def _gpr_loop(self):
        if "risk" in self.off:
            return
        import pandas as pd
        async with sources.client(90) as c:
            while not self.stop.is_set():
                try:
                    page = (await c.get(sources.GPR_PAGE)).text
                    links = sources.find_gpr_links(page)
                    d = await c.get(links["daily"])
                    d.raise_for_status()
                    df = await asyncio.to_thread(pd.read_excel, io.BytesIO(d.content))
                    daily = sources.parse_gpr_daily(df)
                    m = await c.get(links["monthly"])
                    m.raise_for_status()
                    dfm = await asyncio.to_thread(pd.read_excel, io.BytesIO(m.content))
                    self.gpr = {"daily": daily[-400:], "monthly": sources.parse_gpr_monthly(dfm, "IND")}
                    self.dirty.add("risk")
                    self._st("risk", "LIVE", detail=f"GPR daily to {daily[-1][0] if daily else '?'}")
                    await self._sleep(12 * 3600)
                except ImportError:
                    self._st("risk", "ERROR", detail="pip install xlrd (needed to read the GPR .xls files)")
                    return
                except Exception as e:
                    self._st("risk", "ERROR", detail=f"GPR: {str(e)[:140]}")
                    log.warning("gpr: %s", e)
                    await self._sleep(1800)

    async def _fred_loop(self):
        if "macro" in self.off:
            return
        async with sources.client(60) as c:
            while not self.stop.is_set():
                ok, err = 0, ""
                for sid in sources.FRED_SERIES:
                    try:
                        self.fred[sid] = await sources.fred_fetch(c, sid, date.today() - timedelta(days=400),
                                                                  settings.fred_api_key)
                        ok += 1
                    except Exception as e:
                        err = f"{type(e).__name__}: {str(e)[:100]}"
                        log.warning("fred %s: %s", sid, e)
                self.dirty.add("macro")
                hint = "" if settings.fred_api_key else " — a free FRED_API_KEY (fred.stlouisfed.org) usually fixes this"
                self._st("macro", "LIVE" if ok else "ERROR",
                         detail=(f"FRED {ok}/{len(sources.FRED_SERIES)} series" if ok
                                 else f"FRED unreachable from this server ({err}){hint}"))
                await self._sleep(6 * 3600 if ok else 900)

    async def _quotes_loop(self):
        """Live MCX / currency futures and India VIX from the market snapshot the analytics publishes."""
        wanted = {s.strip().upper() for s in settings.geo_macro_symbols.split(",") if s.strip()}
        while not self.stop.is_set():
            try:
                uni = bus.loads(await self.r.get(bus.KEY_UNIVERSE)) or []
                q = bus.loads(await self.r.get(bus.SNAP_PREFIX + "quotes"))
                quotes = (q or {}).get("data", {}).get("q", {})
                live = {}
                for i in uni:
                    v = quotes.get(str(i.get("key", i["sid"])))
                    if not v:
                        continue
                    if i["kind"] == "IDX" and i["symbol"] == "INDIAVIX":
                        self.india_vix = v[0]
                    if i["kind"] == "MACRO" and i["underlying"] in wanted:
                        live[i["underlying"]] = {"ltp": v[0], "chg": v[1], "src": "DHAN", "expiry": i.get("expiry")}
                if live and self.mode != "sim":
                    self.macro_live = live
                    self.dirty.add("macro")
            except Exception:
                log.debug("quotes read failed", exc_info=True)
            await self._sleep(5)

    # ------------------------------------------------------------ derive + publish
    def _macro_inputs(self) -> dict:
        def fchg(sid):
            return sources.pct_change(self.fred.get(sid, []))
        crude = (self.macro_live.get("CRUDEOIL") or {}).get("chg")
        inr = (self.macro_live.get("USDINR") or {}).get("chg")
        gold = (self.macro_live.get("GOLD") or {}).get("chg")
        return {"crude_chg": crude if crude is not None else fchg("DCOILBRENTEU"),
                "usdinr_chg": inr if inr is not None else fchg("DEXINUS"),
                "gold_chg": gold, "india_vix": self.india_vix}

    async def _publish(self, topic: str, payload: dict):
        await bus.publish_snapshot(self.r, f"geo:{topic}", payload)

    async def _publish_loop(self):
        await self._publish("sites", {"ts": time.time(), **self.sites})
        raw = await self.r.lrange(K_RISKHIST, 0, -1)
        self.risk_hist = [json.loads(x) for x in raw]
        last = {"ships": 0.0, "air": 0.0, "conflict": 0.0, "risk": 0.0, "hist": 0.0, "macro": 0.0, "fires": 0.0}
        while not self.stop.is_set():
            now = time.time()
            try:
                events = sorted(self.events.values(), key=lambda e: -e["sev"])
                chokes = engine.chokepoint_view(self.sites["chokepoints"], self.ships, self.transits,
                                                self.portwatch, events, now)
                fires = engine.fire_sites(self.sites["energy"], self.hotspots, self.fire_hist, now)
                air = engine.airspace(self.sites["air_regions"], self.air_counts, self.air_base)
                regions = engine.conflict_summary(events, now)
                macro_in = self._macro_inputs()
                base_vals = [h[2] for h in self.risk_hist if len(h) > 2 and h[2]]
                cbase = sorted(base_vals)[len(base_vals) // 2] if len(base_vals) >= 24 else None
                risk = engine.risk_index(regions, chokes, fires, air, macro_in, cbase)
                signals = engine.impact(chokes, fires, regions, air, macro_in, now)

                if "ships" in self.dirty and now - last["ships"] >= 8:
                    rows = [[s["mmsi"], s["lat"], s["lon"], s.get("sog"), s.get("cog"), CLS_CODE.get(s.get("cls") or
                             "UNKNOWN", 0), s.get("name", ""), s.get("dest", "")] for s in self.ships.values()]
                    counts: dict[str, int] = {}
                    for s in self.ships.values():
                        counts[s.get("cls") or "UNKNOWN"] = counts.get(s.get("cls") or "UNKNOWN", 0) + 1
                    await self._publish("ships", {"ts": now, "n": len(rows), "rows": rows[:6000], "counts": counts,
                                                  "cls": CLS_CODE})
                    last["ships"] = now
                    self.dirty.discard("ships")
                await self._publish("chokepoints", {"ts": now, "rows": chokes})
                if "conflict" in self.dirty and now - last["conflict"] >= 30:
                    top = events[:1500]
                    await self._publish("conflict", {"ts": now, "n": len(events), "events": top, "regions": regions})
                    last["conflict"] = now
                    self.dirty.discard("conflict")
                if "fires" in self.dirty or now - last["fires"] >= 300:
                    pts = [[p["lat"], p["lon"], p["frp"], p["ts"], p["conf"]] for p in self.hotspots]
                    await self._publish("fires", {"ts": now, "n": len(pts), "points": pts, "sites": fires})
                    last["fires"] = now
                    self.dirty.discard("fires")
                if "air" in self.dirty and now - last["air"] >= 10:
                    rows = [[a["hex"], a["flight"], a["lat"], a["lon"], a["alt"], a["gs"], a["track"], a["type"],
                             1 if a["mil"] else 0] for a in self.aircraft]
                    await self._publish("air", {"ts": now, "n": len(rows), "rows": rows, "regions": air})
                    last["air"] = now
                    self.dirty.discard("air")
                if now - last["macro"] >= 10:
                    daily = {sid: {"label": lab, "last": (self.fred.get(sid) or [[None, None]])[-1],
                                   "chg1d": sources.pct_change(self.fred.get(sid, [])),
                                   "chg5d": sources.pct_change(self.fred.get(sid, []), 5),
                                   "spark": [v for _, v in self.fred.get(sid, [])[-60:]]}
                             for sid, lab in sources.FRED_SERIES.items() if self.fred.get(sid)}
                    await self._publish("macro", {"ts": now, "live": self.macro_live, "daily": daily,
                                                  "india_vix": self.india_vix, "inputs": macro_in,
                                                  "calendar": _calendar(datetime.now(IST).date())})
                    last["macro"] = now
                # risk history every 15 min (5 in sim) — also the baseline for the conflict component
                step = 300 if self.mode == "sim" else 900
                if now - last["hist"] >= step:
                    point = [round(now), risk["value"], risk["conflict_raw"]]
                    self.risk_hist = (self.risk_hist + [point])[-700:]
                    await self.r.rpush(K_RISKHIST, json.dumps(point))
                    await self.r.ltrim(K_RISKHIST, -700, -1)
                    last["hist"] = now
                if now - last["risk"] >= 10:
                    gd = self.gpr.get("daily", [])
                    await self._publish("risk", {
                        "ts": now, **risk, "history": [[h[0], h[1]] for h in self.risk_hist[-672:]],
                        "regions": regions[:10],
                        "gpr": {"latest": gd[-1] if gd else None, "series": gd[-365:],
                                "ma30": round(sum(v for _, v in gd[-30:]) / len(gd[-30:]), 1) if gd else None,
                                "monthly": self.gpr.get("monthly", {})},
                        "mode": self.mode})
                    await self._publish("impact", {"ts": now, "signals": signals, "inputs": macro_in})
                    last["risk"] = now
                    if settings.geo_to_news:
                        await self._push_news(signals)
                if "status" in self.dirty:
                    await self._publish("status", {"ts": now, "mode": self.mode, "layers": self.status,
                                                   "demand": {k: v for k, v in self.demand.items()
                                                              if k.startswith("geo:")}})
                    self.dirty.discard("status")
                await self.r.set(bus.k_status("geo"), bus.dumps({
                    "mode": self.mode, "updated": now, "ships": len(self.ships), "events": len(self.events),
                    "hotspots": len(self.hotspots), "aircraft": len(self.aircraft), "risk": risk["value"],
                    "layers": {k: v["state"] for k, v in self.status.items()}}), ex=30)
            except Exception:
                log.exception("publish loop")
            await self._sleep(5)

    async def _push_news(self, signals: list[dict]):
        day = datetime.now(IST).date().isoformat()
        for s in signals:
            if s["sev"] < 3 or not s["symbols"]:
                continue
            for sign in (1, -1):
                syms = [x["s"] for x in s["symbols"] if x["dir"] == sign]
                key = f"{s['id']}:{sign}:{day}"
                if not syms or key in self.pushed_news:
                    continue
                self.pushed_news.add(key)
                item = {"id": f"geo-{key}", "ts": time.time(), "title": f"GEO: {s['title']}", "text": s["why"],
                        "source": "GEO", "symbols": syms, "sentiment": "positive" if sign > 0 else "negative"}
                await self.r.xadd(bus.STREAM_NEWS, {"j": bus.dumps(item)}, maxlen=5000, approximate=True)

    # ------------------------------------------------------------ run
    async def run(self):
        log.info("GEO worker: mode=%s disabled=%s", self.mode, sorted(self.off) or "-")
        tasks = [self._demand_loop(), self._publish_loop(), self._quotes_loop()]
        if self.mode == "sim":
            tasks.append(self._sim_loop())
        else:
            tasks += [self._ais_loop(), self._gdelt_loop(), self._firms_loop(), self._adsb_loop(),
                      self._portwatch_loop(), self._gpr_loop(), self._fred_loop()]
        await asyncio.gather(*(asyncio.create_task(t) for t in tasks))


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not settings.geo_enabled:
        log.info("GEO_ENABLED=false — nothing to do")
        return
    svc = GeoService()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, svc.stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    runner = asyncio.create_task(svc.run())
    await svc.stop.wait()
    runner.cancel()
    try:
        await runner
    except (asyncio.CancelledError, Exception):
        pass
    await svc.r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
