"""Check every GEO desk feed from THIS machine: one line per source, OK or the reason it fails.

A feed that works on one machine can be refused on another (cloud-server IP ranges, firewalls, rate limits),
so run this on the server itself:
    docker compose exec geo python -m td.geo.diagnose        (Docker)
    cd backend && python -m td.geo.diagnose                   (plain Python)
Keys come from .env as the app reads them; they are never printed.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone

from ..config import settings
from . import sources

PASS, FAIL, SKIP = "OK  ", "FAIL", "SKIP"


def _scrub(text: str) -> str:
    for k in (settings.firms_map_key, settings.aisstream_api_key, settings.fred_api_key):
        if k:
            text = text.replace(k, "***")
    return text


async def _gdelt(c) -> str:
    url = await sources.gdelt_latest_url(c)
    if not url:
        raise RuntimeError("no export file listed in lastupdate.txt")
    stamp = url.rsplit("/", 1)[-1][:14]
    age = (datetime.now(timezone.utc) - datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc))
    return f"latest 15-minute file is {age.total_seconds() / 60:.0f} min old"


async def _firms(c) -> str:
    key = settings.firms_map_key
    if not key:
        return "SKIP:FIRMS_MAP_KEY not set (free key: firms.modaps.eosdis.nasa.gov)"
    r = await c.get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_NRT/world/1")
    r.raise_for_status()
    if not r.text.startswith("latitude"):
        raise RuntimeError(f"unexpected answer: {r.text[:80]!r} (wrong or expired MAP_KEY?)")
    rows = list(csv.DictReader(io.StringIO(r.text)))
    if not rows:
        return "key works, but no detections in the last 24 h"
    newest = max(datetime.strptime(f"{x['acq_date']} {x['acq_time'].zfill(4)}", "%Y-%m-%d %H%M") for x in rows)
    hours = (datetime.now(timezone.utc).replace(tzinfo=None) - newest).total_seconds() / 3600
    return f"{len(rows)} fire detections in 24 h, newest {hours:.1f} h old"


async def _ais() -> str:
    key = settings.aisstream_api_key
    if not key:
        return "SKIP:AISSTREAM_API_KEY not set (free key: aisstream.io)"
    import websockets
    n, typed, newest = 0, 0, None
    async with websockets.connect(sources.AIS_URL, open_timeout=20) as ws:
        await ws.send(json.dumps({"APIKey": key, "BoundingBoxes": [[[0, 30], [32, 105]]],
                                  "FilterMessageTypes": ["PositionReport", "ShipStaticData"]}))
        end = time.time() + 30
        while time.time() < end:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), 5))
            except asyncio.TimeoutError:
                continue
            if "error" in m:
                raise RuntimeError(f"AISStream refused the key: {str(m['error'])[:80]}")
            if m.get("MessageType") == "ShipStaticData":
                typed += 1
            elif m.get("MessageType") == "PositionReport":
                n += 1
                t = (m.get("MetaData") or {}).get("time_utc", "")[:19]
                try:
                    newest = datetime.strptime(t, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
    if not n:
        raise RuntimeError("connected, but no ship positions in 30 s")
    age = (datetime.now(timezone.utc) - newest).total_seconds() if newest else float("nan")
    return f"{n} ship positions + {typed} ship-type reports in 30 s, newest {age:.0f} s old"


async def _adsb(c) -> str:
    mil = sources.parse_adsb(await sources.adsb_get(c, settings.adsb_api_url, "/v2/mil"), mil=True)
    near = sources.parse_adsb(await sources.adsb_get(c, settings.adsb_api_url, "/v2/point/25.0/55.0/250"))
    if not mil and not near:
        raise RuntimeError("answered, but with no aircraft at all (blocked or rate-limited?)")
    return f"{len(mil)} military aircraft worldwide, {len(near)} aircraft over the Gulf"


async def _portwatch(c) -> str:
    rows = await sources.portwatch_fetch(c, "Strait of Hormuz", 14)
    if not rows:
        raise RuntimeError("no rows returned")
    return f"Strait of Hormuz transits to {rows[-1]['date']} (published with a few days' lag)"


async def _gpr(c) -> str:
    page = (await c.get(sources.GPR_PAGE)).text
    links = sources.find_gpr_links(page)
    if not links.get("daily"):
        raise RuntimeError("download links not found on the GPR page")
    r = await c.get(links["daily"])
    r.raise_for_status()
    return f"daily index file reachable ({len(r.content) // 1024} KB)"


async def _fred(c) -> str:
    rows = await sources.fred_fetch(c, "DCOILBRENTEU", date.today() - timedelta(days=30), settings.fred_api_key)
    if not rows:
        raise RuntimeError("no observations returned")
    via = "API key" if settings.fred_api_key else "CSV download (no key)"
    return f"Brent {rows[-1][1]} on {rows[-1][0]} via {via}"


async def main() -> int:
    print(f"GEO feed check — {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC — GEO_SOURCE={settings.geo_source}, "
          f"GEO_DISABLE={settings.geo_disable or '-'}")
    failed = 0
    async with sources.client(30) as c:
        checks = [("Conflict  (GDELT)", _gdelt(c)), ("Fires     (NASA FIRMS)", _firms(c)),
                  ("Ships     (AISStream)", _ais()), ("Aircraft  (adsb.lol)", _adsb(c)),
                  ("Chokepoints (IMF PortWatch)", _portwatch(c)), ("Geo-risk  (GPR index)", _gpr(c)),
                  ("Macro     (FRED)", _fred(c))]
        for name, coro in checks:
            t0 = time.time()
            try:
                msg = await asyncio.wait_for(coro, 60)
                state = SKIP if msg.startswith("SKIP:") else PASS
                msg = msg.removeprefix("SKIP:")
            except Exception as e:
                state, msg = FAIL, f"{type(e).__name__}: {e}"
                failed += 1
            print(f"  [{state}] {name:28} {_scrub(str(msg))[:160]}  ({time.time() - t0:.1f}s)", flush=True)
    if failed:
        print("\nA FAIL here means this server cannot reach that feed (firewall, blocked IP range, rate limit or a "
              "bad key). FRED: set a free FRED_API_KEY. Aircraft: try ADSB_API_URL=https://api.adsb.lol later, "
              "or a different server region.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
