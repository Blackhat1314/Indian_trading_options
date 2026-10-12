"""Free world-event data sources: fetchers + pure parsers (parsers are unit-tested).

  GDELT 2.0 events        conflict / military events, every 15 min, no key
  NASA FIRMS (VIIRS)      fires & thermal anomalies, near-real-time, free MAP_KEY
  AISStream.io            live ship positions (websocket), free API key
  adsb.lol                military aircraft + regional air traffic, no key
  IMF PortWatch           daily chokepoint transits by vessel type, no key
  Caldara–Iacoviello GPR  daily geopolitical risk index (+ monthly country indices), no key
  FRED                    Brent, WTI, USD/INR, US 10Y, broad dollar, Henry Hub (CSV, no key needed)
"""
from __future__ import annotations

import csv
import io
import logging
import math
import re
import zipfile
from datetime import date, datetime, timezone
from typing import Any

import httpx

log = logging.getLogger("td.geo.sources")
UA = {"User-Agent": "TradeDesk/1.0 (personal market research; data only)"}


def client(timeout: float = 30) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, headers=UA, follow_redirects=True)


# =========================================================================== GDELT
GDELT_LAST = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
GDELT_FILE = "http://data.gdeltproject.org/gdeltv2/{stamp}.export.CSV.zip"

# CAMEO codes we keep: 15x force posture, 18x assault, 19x fight, 20x mass violence
CAMEO = {
    "150": "Military/police show of force", "151": "Raise police alert", "152": "Raise military alert",
    "153": "Mobilise police", "154": "Mobilise armed forces", "155": "Mobilise cyber forces",
    "180": "Unconventional violence", "181": "Abduction / hijacking", "182": "Physical assault",
    "1821": "Sexual assault", "1822": "Torture", "1823": "Killing", "183": "Bombing",
    "1831": "Suicide bombing", "1832": "Vehicle bombing", "1833": "Roadside bombing", "1834": "Location bombing",
    "184": "Human shield", "185": "Assassination attempt", "186": "Assassination",
    "190": "Military force", "191": "Blockade / movement restriction", "192": "Occupation of territory",
    "193": "Small-arms fighting", "194": "Artillery / tank fighting", "195": "Air strike",
    "1951": "Precision air strike", "1952": "Drone strike", "196": "Ceasefire violation",
    "200": "Mass violence", "201": "Mass expulsion", "202": "Mass killing", "203": "Ethnic cleansing",
    "204": "WMD use", "2041": "Chemical / biological / radiological", "2042": "Nuclear detonation",
}


def cameo_weight(code: str) -> float:
    if code.startswith("204"):
        return 6.0
    if code.startswith("20"):
        return 5.0
    if code.startswith(("194", "195")):
        return 4.0
    if code.startswith("19"):
        return 3.0
    if code.startswith(("183", "186")):
        return 3.0
    if code.startswith("18"):
        return 2.0
    if code.startswith("15"):
        return 1.0
    return 0.0


def cameo_label(code: str) -> str:
    for k in (code, code[:3], code[:2] + "0"):
        if k in CAMEO:
            return CAMEO[k]
    return "Conflict event"


def _f(x: str) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def parse_gdelt_export(text: str, min_articles: int = 2) -> list[dict]:
    """GDELT 2.0 export (tab separated, 61 columns) -> conflict events with coordinates."""
    out: list[dict] = []
    for line in text.splitlines():
        c = line.split("\t")
        if len(c) < 61:
            continue
        root = c[28]
        if root not in ("15", "18", "19", "20") or c[25] != "1":
            continue
        if root != "15" and c[29] != "4":                 # QuadClass 4 = material conflict
            continue
        lat, lon = _f(c[56]), _f(c[57])
        if lat is None or lon is None or (lat == 0 and lon == 0):
            continue
        articles = int(c[33] or 0)
        if articles < min_articles:
            continue
        code = c[26]
        w = cameo_weight(code)
        if w <= 0:
            continue
        try:
            ts = datetime.strptime(c[59], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
        out.append({
            "id": c[0], "ts": ts, "lat": round(lat, 4), "lon": round(lon, 4), "code": code, "root": root,
            "label": cameo_label(code), "a1": c[6].title(), "a2": c[16].title(), "a1c": c[7], "a2c": c[17],
            "place": c[52], "country": c[53], "precise": c[51] in ("3", "4"),
            "articles": articles, "sources": int(c[32] or 0), "goldstein": _f(c[30]) or 0.0,
            "tone": round(_f(c[34]) or 0.0, 2), "url": c[60],
            "sev": round(w * (1 + math.log2(1 + articles)), 2),
        })
    return out


def gdelt_dedupe(events: list[dict]) -> list[dict]:
    """GDELT logs the same incident many times; merge by code, ~10 km cell, actors and day."""
    best: dict[tuple, dict] = {}
    for e in events:
        k = (e["code"][:3], round(e["lat"], 1), round(e["lon"], 1), e["a1c"], e["a2c"],
             int(e["ts"] // 86400))
        cur = best.get(k)
        if cur is None:
            best[k] = dict(e)
        else:
            cur["articles"] += e["articles"]
            cur["sources"] = max(cur["sources"], e["sources"])
            cur["ts"] = max(cur["ts"], e["ts"])
            w = cameo_weight(cur["code"])
            cur["sev"] = round(w * (1 + math.log2(1 + cur["articles"])), 2)
    return list(best.values())


def gdelt_stamps_back(now_utc: datetime, hours: int) -> list[str]:
    """15-minute GDELT file stamps going back `hours` (newest first)."""
    t = now_utc.replace(second=0, microsecond=0)
    t = t.replace(minute=(t.minute // 15) * 15)
    out = []
    for i in range(1, hours * 4 + 1):
        ts = t.timestamp() - i * 900
        out.append(datetime.fromtimestamp(ts, timezone.utc).strftime("%Y%m%d%H%M%S"))
    return out


async def gdelt_latest_url(c: httpx.AsyncClient) -> str | None:
    r = await c.get(GDELT_LAST)
    r.raise_for_status()
    for line in r.text.splitlines():
        parts = line.split()
        if parts and parts[-1].endswith(".export.CSV.zip"):
            return parts[-1]
    return None


async def gdelt_fetch(c: httpx.AsyncClient, url: str) -> str:
    r = await c.get(url)
    if r.status_code == 404:
        return ""
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        return z.read(name).decode("utf-8", errors="replace")


# =========================================================================== FIRMS
FIRMS_AREA = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{days}"
FIRMS_SOURCES = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")


def parse_firms_csv(text: str, source: str = "") -> list[dict]:
    out = []
    rdr = csv.DictReader(io.StringIO(text))
    for r in rdr:
        lat, lon = _f(r.get("latitude")), _f(r.get("longitude"))
        if lat is None or lon is None:
            continue
        conf = (r.get("confidence") or "").strip().lower()
        if conf in ("l", "low"):
            continue
        if conf.isdigit() and int(conf) < 30:
            continue
        d, t = r.get("acq_date", ""), (r.get("acq_time") or "0").zfill(4)
        try:
            ts = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
        out.append({"lat": round(lat, 4), "lon": round(lon, 4), "frp": round(_f(r.get("frp")) or 0.0, 1),
                    "conf": conf[:1] or "n", "ts": ts, "sat": source.replace("_NRT", "").replace("VIIRS_", "")})
    return out


async def firms_fetch(c: httpx.AsyncClient, key: str, box: tuple, source: str, days: int = 2) -> list[dict]:
    area = ",".join(f"{v:g}" for v in box)
    r = await c.get(FIRMS_AREA.format(key=key, source=source, area=area, days=days))
    r.raise_for_status()
    if r.text.lstrip().lower().startswith(("invalid", "error")):
        raise RuntimeError(r.text[:120])
    return parse_firms_csv(r.text, source)


def dedupe_fires(points: list[dict]) -> list[dict]:
    """Different satellites see the same fire; keep the strongest detection per ~400 m cell per hour."""
    best: dict[tuple, dict] = {}
    for p in points:
        k = (round(p["lat"] * 250), round(p["lon"] * 250), int(p["ts"] // 3600))
        if k not in best or p["frp"] > best[k]["frp"]:
            best[k] = p
    return list(best.values())


# =========================================================================== AIS (AISStream)
AIS_URL = "wss://stream.aisstream.io/v0/stream"


def ship_class(t: int | None) -> str:
    if t is None:
        return "UNKNOWN"
    if 80 <= t <= 89:
        return "TANKER"
    if 70 <= t <= 79:
        return "CARGO"
    if 60 <= t <= 69:
        return "PASSENGER"
    if t == 35:
        return "MILITARY"
    if t == 30:
        return "FISHING"
    if 50 <= t <= 59 or t in (31, 32, 33, 34):
        return "SERVICE"
    return "OTHER"


def parse_ais(msg: dict) -> tuple[str, dict] | None:
    """AISStream message -> ("pos", {...}) or ("static", {...})."""
    mtype = msg.get("MessageType", "")
    meta = msg.get("MetaData") or {}
    body = (msg.get("Message") or {}).get(mtype) or {}
    mmsi = meta.get("MMSI") or body.get("UserID")
    if not mmsi:
        return None
    if mtype in ("PositionReport", "StandardClassBPositionReport", "ExtendedClassBPositionReport"):
        lat = body.get("Latitude", meta.get("latitude", meta.get("Latitude")))
        lon = body.get("Longitude", meta.get("longitude", meta.get("Longitude")))
        if lat is None or lon is None or abs(lat) > 90 or abs(lon) > 180:
            return None
        sog = body.get("Sog")
        cog = body.get("Cog")
        hdg = body.get("TrueHeading")
        return "pos", {"mmsi": int(mmsi), "lat": round(float(lat), 5), "lon": round(float(lon), 5),
                       "sog": None if sog is None or sog >= 102.3 else round(float(sog), 1),
                       "cog": None if cog is None or cog >= 360 else round(float(cog), 1),
                       "hdg": None if hdg is None or hdg >= 360 else int(hdg),
                       "name": (meta.get("ShipName") or "").strip(), "nav": body.get("NavigationalStatus")}
    if mtype == "ShipStaticData":
        return "static", {"mmsi": int(mmsi), "type": body.get("Type"), "name": (body.get("Name") or "").strip(),
                          "dest": (body.get("Destination") or "").strip(), "imo": body.get("ImoNumber"),
                          "draught": body.get("MaximumStaticDraught")}
    return None


# =========================================================================== ADS-B (adsb.lol / airplanes.live)
def parse_adsb(js: dict, mil: bool = False) -> list[dict]:
    out = []
    for a in js.get("ac") or []:
        lat, lon = a.get("lat"), a.get("lon")
        if lat is None or lon is None:
            continue
        alt = a.get("alt_baro")
        if alt == "ground":
            alt = 0
        out.append({
            "hex": a.get("hex", ""), "flight": (a.get("flight") or "").strip(), "reg": a.get("r", ""),
            "type": a.get("t", ""), "lat": round(float(lat), 4), "lon": round(float(lon), 4),
            "alt": int(alt) if isinstance(alt, (int, float)) else None,
            "gs": round(float(a["gs"])) if a.get("gs") is not None else None,
            "track": round(float(a["track"])) if a.get("track") is not None else None,
            "mil": bool(mil or (int(a.get("dbFlags") or 0) & 1)), "sqk": a.get("squawk", ""),
        })
    return out


async def adsb_get(c: httpx.AsyncClient, base: str, path: str) -> dict:
    r = await c.get(base.rstrip("/") + path)
    r.raise_for_status()
    return r.json()


# =========================================================================== IMF PortWatch
PORTWATCH = ("https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/Daily_Chokepoints_Data/"
             "FeatureServer/0/query")


def parse_portwatch(js: dict) -> list[dict]:
    out = []
    for f in js.get("features") or []:
        a = f.get("attributes") or {}
        d = a.get("date")
        if isinstance(d, (int, float)):
            d = datetime.fromtimestamp(d / 1000, timezone.utc).date().isoformat()
        if not d:
            continue
        out.append({"date": str(d)[:10], "name": a.get("portname", ""), "total": int(a.get("n_total") or 0),
                    "tanker": int(a.get("n_tanker") or 0), "container": int(a.get("n_container") or 0),
                    "capacity": int(a.get("capacity") or 0)})
    return out


async def portwatch_fetch(c: httpx.AsyncClient, portname: str, days: int = 430) -> list[dict]:
    """Most recent `days` daily records for one chokepoint (newest-first query, returned oldest-first)."""
    params = {"where": f"portname='{portname}'", "outFields": "date,portname,n_total,n_tanker,n_container,capacity",
              "orderByFields": "date DESC", "resultRecordCount": min(1000, days), "f": "json"}
    r = await c.get(PORTWATCH, params=params)
    r.raise_for_status()
    js = r.json()
    if "error" in js:
        raise RuntimeError(str(js["error"])[:160])
    return sorted(parse_portwatch(js), key=lambda x: x["date"])


def transit_stats(rows: list[dict]) -> dict:
    """Last 7 days vs the same chokepoint's typical day over the prior year (excl. last 30 days)."""
    if not rows:
        return {}
    rows = sorted(rows, key=lambda x: x["date"])
    last = rows[-7:]
    hist = rows[:-30][-365:]
    avg = lambda xs, k: sum(x[k] for x in xs) / len(xs) if xs else None  # noqa: E731
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
    base_total = med([x["total"] for x in hist])
    base_tanker = med([x["tanker"] for x in hist])
    l7t, l7k = avg(last, "total"), avg(last, "tanker")
    return {
        "as_of": rows[-1]["date"], "last7_total": round(l7t, 1) if l7t is not None else None,
        "last7_tanker": round(l7k, 1) if l7k is not None else None,
        "base_total": base_total, "base_tanker": base_tanker,
        "ratio": round(l7t / base_total, 2) if base_total else None,
        "tanker_ratio": round(l7k / base_tanker, 2) if base_tanker else None,
        "series": [[x["date"], x["total"], x["tanker"]] for x in rows[-90:]],
    }


# =========================================================================== GPR index
GPR_PAGE = "https://www.matteoiacoviello.com/gpr.htm"
GPR_BASE = "https://www.matteoiacoviello.com/"


def find_gpr_links(html: str) -> dict:
    """The site names files by vintage (data_gpr_daily_recent_YYYYMMDD.xls); take whatever the page links."""
    daily = re.findall(r'href="([^"]*data_gpr_daily_recent[^"]*\.xls)"', html, re.I)
    monthly = re.findall(r'href="([^"]*data_gpr_export[^"]*\.xls)"', html, re.I)
    fix = lambda u: u if u.startswith("http") else GPR_BASE + u.lstrip("./")  # noqa: E731
    return {"daily": fix(daily[0]) if daily else GPR_BASE + "gpr_files/data_gpr_daily_recent.xls",
            "monthly": fix(monthly[0]) if monthly else GPR_BASE + "gpr_files/data_gpr_export.xls"}


def parse_gpr_daily(df) -> list[list]:
    cols = {c.lower(): c for c in df.columns}
    val = cols.get("gprd")
    if val is None:
        return []
    if "day" in cols:
        dates = df[cols["day"]].astype(str).str.slice(0, 8)
        dates = dates.str.slice(0, 4) + "-" + dates.str.slice(4, 6) + "-" + dates.str.slice(6, 8)
    elif "date" in cols:
        dates = df[cols["date"]].astype(str).str.slice(0, 10)
    else:
        return []
    out = []
    for d, v in zip(dates, df[val]):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            out.append([d, round(fv, 2)])
    return out


def parse_gpr_monthly(df, country: str = "IND") -> dict:
    cols = {c.upper(): c for c in df.columns}
    month_col = cols.get("MONTH") or cols.get("DATE")
    out = {}
    for key in ("GPR", f"GPRC_{country}", f"GPRHC_{country}"):
        if key in cols and month_col:
            series = []
            for m, v in zip(df[month_col], df[cols[key]]):
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv):
                    series.append([str(m)[:7], round(fv, 3)])
            out[key] = series[-60:]
    return out


# =========================================================================== FRED
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES = {
    "DCOILBRENTEU": "Brent crude ($/bbl)", "DCOILWTICO": "WTI crude ($/bbl)", "DEXINUS": "USD/INR",
    "DGS10": "US 10Y yield (%)", "DTWEXBGS": "Broad US dollar index", "DHHNGSP": "Henry Hub gas ($/MMBtu)",
}


def parse_fred_csv(text: str) -> list[list]:
    out = []
    rdr = csv.reader(io.StringIO(text))
    header = next(rdr, None)
    if not header:
        return out
    for row in rdr:
        if len(row) < 2 or row[1] in ("", "."):
            continue
        try:
            out.append([row[0][:10], round(float(row[1]), 4)])
        except ValueError:
            continue
    return out


async def fred_fetch(c: httpx.AsyncClient, sid: str, start: date, api_key: str = "") -> list[list]:
    if api_key:
        r = await c.get(FRED_API, params={"series_id": sid, "api_key": api_key, "file_type": "json",
                                         "observation_start": start.isoformat()})
        r.raise_for_status()
        return [[o["date"], round(float(o["value"]), 4)] for o in r.json().get("observations", [])
                if o.get("value") not in (None, "", ".")]
    r = await c.get(FRED_CSV.format(sid=sid, start=start.isoformat()))
    r.raise_for_status()
    return parse_fred_csv(r.text)


def pct_change(series: list[list], n: int = 1) -> float | None:
    if len(series) <= n:
        return None
    a, b = series[-1 - n][1], series[-1][1]
    return round((b / a - 1) * 100, 2) if a else None


def jsonable(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x
