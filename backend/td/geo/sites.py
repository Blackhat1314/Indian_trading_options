"""Static geography for the GEO desk: chokepoints, ports, energy infrastructure, airspace
watch points, conflict regions and the market-impact map.

Coordinates of chokepoints are IMF PortWatch's own reference points. Ports and energy sites are
approximate (a few km) — good enough for matching satellite fire detections within a radius and
for drawing on the map. Override or extend any list with data/config/geo_sites.json
(same field names); entries with the same id replace the built-ins.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..config import settings
from ..universe_data import FNO_STOCKS

FNO = {s for s, *_ in FNO_STOCKS}

# ------------------------------------------------------------------ chokepoints
# id, name, lat, lon (PortWatch), radius km used for live AIS counting, India relevance 0..1
CHOKEPOINTS: list[dict] = [
    {"id": "hormuz", "name": "Strait of Hormuz", "portwatch": "Strait of Hormuz", "lat": 26.2969, "lon": 56.8598,
     "r_km": 70, "india": 1.0, "why": "Gulf crude, LNG and LPG into India"},
    {"id": "bab", "name": "Bab el-Mandeb", "portwatch": "Bab el-Mandeb Strait", "lat": 12.7886, "lon": 43.3495,
     "r_km": 60, "india": 0.9, "why": "India–Europe trade, Red Sea route"},
    {"id": "suez", "name": "Suez Canal", "portwatch": "Suez Canal", "lat": 30.5933, "lon": 32.4369,
     "r_km": 90, "india": 0.8, "why": "India–Europe container and product tanker route"},
    {"id": "malacca", "name": "Malacca Strait", "portwatch": "Malacca Strait", "lat": 1.5170, "lon": 102.6651,
     "r_km": 140, "india": 0.7, "why": "India–East Asia trade, coal and LNG"},
    {"id": "cape", "name": "Cape of Good Hope", "portwatch": "Cape of Good Hope", "lat": -34.9273, "lon": 20.8827,
     "r_km": 160, "india": 0.5, "why": "Diversion route when the Red Sea is unsafe"},
    {"id": "bosporus", "name": "Bosporus", "portwatch": "Bosporus Strait", "lat": 41.1693, "lon": 29.0915,
     "r_km": 30, "india": 0.4, "why": "Black Sea crude (Novorossiysk / CPC) and grain"},
    {"id": "kerch", "name": "Kerch Strait", "portwatch": "Kerch Strait", "lat": 45.2668, "lon": 36.5439,
     "r_km": 35, "india": 0.2, "why": "Azov / Black Sea shipping"},
    {"id": "gibraltar", "name": "Gibraltar", "portwatch": "Gibraltar Strait", "lat": 35.9423, "lon": -5.7549,
     "r_km": 45, "india": 0.3, "why": "Mediterranean–Atlantic link"},
    {"id": "taiwan", "name": "Taiwan Strait", "portwatch": "Taiwan Strait", "lat": 24.7235, "lon": 119.8314,
     "r_km": 130, "india": 0.5, "why": "Electronics and East Asia supply chains"},
    {"id": "sunda", "name": "Sunda Strait", "portwatch": "Sunda Strait", "lat": -5.9668, "lon": 105.7752,
     "r_km": 50, "india": 0.2, "why": "Malacca bypass"},
    {"id": "lombok", "name": "Lombok Strait", "portwatch": "Lombok Strait", "lat": -8.4191, "lon": 115.8014,
     "r_km": 45, "india": 0.2, "why": "Deep-draft Malacca bypass"},
    {"id": "panama", "name": "Panama Canal", "portwatch": "Panama Canal", "lat": 9.1205, "lon": -79.7672,
     "r_km": 45, "india": 0.1, "why": "US Gulf LPG / LNG to Asia"},
]

# ------------------------------------------------------------------ ports
# kind: port | oil_terminal.  symbols: listed operator (sign = effect of a disruption)
PORTS: list[dict] = [
    {"id": "jnpa", "name": "JNPA (Nhava Sheva)", "lat": 18.95, "lon": 72.95, "country": "IN", "kind": "port"},
    {"id": "mumbai", "name": "Mumbai Port", "lat": 18.93, "lon": 72.85, "country": "IN", "kind": "port"},
    {"id": "mundra", "name": "Mundra", "lat": 22.74, "lon": 69.70, "country": "IN", "kind": "port",
     "symbols": {"ADANIPORTS": -1}},
    {"id": "kandla", "name": "Deendayal (Kandla)", "lat": 23.02, "lon": 70.22, "country": "IN", "kind": "port"},
    {"id": "hazira", "name": "Hazira", "lat": 21.09, "lon": 72.64, "country": "IN", "kind": "port",
     "symbols": {"ADANIPORTS": -1}},
    {"id": "pipavav", "name": "Pipavav", "lat": 20.92, "lon": 71.51, "country": "IN", "kind": "port"},
    {"id": "sikka", "name": "Sikka (Jamnagar)", "lat": 22.44, "lon": 69.83, "country": "IN", "kind": "oil_terminal",
     "symbols": {"RELIANCE": -1}},
    {"id": "vadinar", "name": "Vadinar", "lat": 22.50, "lon": 69.71, "country": "IN", "kind": "oil_terminal"},
    {"id": "mormugao", "name": "Mormugao", "lat": 15.41, "lon": 73.80, "country": "IN", "kind": "port"},
    {"id": "nmpt", "name": "New Mangalore", "lat": 12.92, "lon": 74.81, "country": "IN", "kind": "port"},
    {"id": "kochi", "name": "Kochi", "lat": 9.97, "lon": 76.26, "country": "IN", "kind": "port"},
    {"id": "vizhinjam", "name": "Vizhinjam", "lat": 8.38, "lon": 76.99, "country": "IN", "kind": "port",
     "symbols": {"ADANIPORTS": -1}},
    {"id": "tuticorin", "name": "V.O. Chidambaranar (Tuticorin)", "lat": 8.75, "lon": 78.20, "country": "IN",
     "kind": "port"},
    {"id": "chennai", "name": "Chennai", "lat": 13.10, "lon": 80.30, "country": "IN", "kind": "port"},
    {"id": "kamarajar", "name": "Kamarajar (Ennore)", "lat": 13.26, "lon": 80.33, "country": "IN", "kind": "port"},
    {"id": "krishnapatnam", "name": "Krishnapatnam", "lat": 14.25, "lon": 80.13, "country": "IN", "kind": "port",
     "symbols": {"ADANIPORTS": -1}},
    {"id": "vizag", "name": "Visakhapatnam", "lat": 17.69, "lon": 83.29, "country": "IN", "kind": "port"},
    {"id": "gangavaram", "name": "Gangavaram", "lat": 17.62, "lon": 83.23, "country": "IN", "kind": "port",
     "symbols": {"ADANIPORTS": -1}},
    {"id": "paradip", "name": "Paradip", "lat": 20.26, "lon": 86.67, "country": "IN", "kind": "port"},
    {"id": "dhamra", "name": "Dhamra", "lat": 20.83, "lon": 86.97, "country": "IN", "kind": "port",
     "symbols": {"ADANIPORTS": -1}},
    {"id": "haldia", "name": "Haldia", "lat": 22.03, "lon": 88.08, "country": "IN", "kind": "port"},
    # region
    {"id": "fujairah", "name": "Fujairah", "lat": 25.17, "lon": 56.36, "country": "AE", "kind": "oil_terminal"},
    {"id": "jebelali", "name": "Jebel Ali", "lat": 25.01, "lon": 55.06, "country": "AE", "kind": "port"},
    {"id": "rastanura_t", "name": "Ras Tanura terminal", "lat": 26.64, "lon": 50.16, "country": "SA",
     "kind": "oil_terminal"},
    {"id": "kharg", "name": "Kharg Island", "lat": 29.24, "lon": 50.31, "country": "IR", "kind": "oil_terminal"},
    {"id": "basra_t", "name": "Basra Oil Terminal", "lat": 29.68, "lon": 48.81, "country": "IQ",
     "kind": "oil_terminal"},
    {"id": "yanbu", "name": "Yanbu", "lat": 24.09, "lon": 38.05, "country": "SA", "kind": "oil_terminal"},
    {"id": "jeddah", "name": "Jeddah", "lat": 21.48, "lon": 39.17, "country": "SA", "kind": "port"},
    {"id": "salalah", "name": "Salalah", "lat": 16.94, "lon": 54.00, "country": "OM", "kind": "port"},
    {"id": "djibouti", "name": "Djibouti", "lat": 11.60, "lon": 43.13, "country": "DJ", "kind": "port"},
    {"id": "portsaid", "name": "Port Said", "lat": 31.26, "lon": 32.31, "country": "EG", "kind": "port"},
    {"id": "colombo", "name": "Colombo", "lat": 6.95, "lon": 79.84, "country": "LK", "kind": "port"},
    {"id": "singapore", "name": "Singapore", "lat": 1.26, "lon": 103.84, "country": "SG", "kind": "port"},
    {"id": "chabahar", "name": "Chabahar", "lat": 25.29, "lon": 60.62, "country": "IR", "kind": "port"},
    {"id": "gwadar", "name": "Gwadar", "lat": 25.12, "lon": 62.33, "country": "PK", "kind": "port"},
    {"id": "karachi", "name": "Karachi", "lat": 24.83, "lon": 66.98, "country": "PK", "kind": "port"},
    {"id": "novorossiysk", "name": "Novorossiysk", "lat": 44.71, "lon": 37.79, "country": "RU",
     "kind": "oil_terminal"},
    {"id": "primorsk", "name": "Primorsk", "lat": 60.34, "lon": 28.71, "country": "RU", "kind": "oil_terminal"},
    {"id": "ustluga", "name": "Ust-Luga", "lat": 59.67, "lon": 28.40, "country": "RU", "kind": "oil_terminal"},
]

# ------------------------------------------------------------------ energy infrastructure
# kind: refinery | processing | terminal | field | lng.  flare: persistent gas flaring expected.
# symbols: listed companies exposed if this site is hit (sign of the likely first move).
ENERGY: list[dict] = [
    # India — refineries
    {"id": "jamnagar", "name": "Jamnagar refinery complex", "kind": "refinery", "lat": 22.36, "lon": 69.86,
     "country": "IN", "owner": "Reliance", "symbols": {"RELIANCE": -1}, "weight": 3},
    {"id": "vadinar_r", "name": "Vadinar refinery", "kind": "refinery", "lat": 22.43, "lon": 69.70,
     "country": "IN", "owner": "Nayara Energy", "weight": 2},
    {"id": "koyali", "name": "Gujarat (Koyali) refinery", "kind": "refinery", "lat": 22.38, "lon": 73.11,
     "country": "IN", "owner": "IOC", "symbols": {"IOC": -1}, "weight": 2},
    {"id": "panipat", "name": "Panipat refinery", "kind": "refinery", "lat": 29.47, "lon": 76.88, "country": "IN",
     "owner": "IOC", "symbols": {"IOC": -1}, "weight": 2},
    {"id": "mathura", "name": "Mathura refinery", "kind": "refinery", "lat": 27.43, "lon": 77.66, "country": "IN",
     "owner": "IOC", "symbols": {"IOC": -1}, "weight": 1},
    {"id": "paradip_r", "name": "Paradip refinery", "kind": "refinery", "lat": 20.26, "lon": 86.60,
     "country": "IN", "owner": "IOC", "symbols": {"IOC": -1}, "weight": 2},
    {"id": "haldia_r", "name": "Haldia refinery", "kind": "refinery", "lat": 22.05, "lon": 88.10, "country": "IN",
     "owner": "IOC", "symbols": {"IOC": -1}, "weight": 1},
    {"id": "barauni", "name": "Barauni refinery", "kind": "refinery", "lat": 25.43, "lon": 86.02, "country": "IN",
     "owner": "IOC", "symbols": {"IOC": -1}, "weight": 1},
    {"id": "mumbai_bpcl", "name": "Mumbai refinery (BPCL)", "kind": "refinery", "lat": 19.01, "lon": 72.89,
     "country": "IN", "owner": "BPCL", "symbols": {"BPCL": -1}, "weight": 2},
    {"id": "mumbai_hpcl", "name": "Mumbai refinery (HPCL)", "kind": "refinery", "lat": 19.03, "lon": 72.89,
     "country": "IN", "owner": "HPCL", "symbols": {"HINDPETRO": -1}, "weight": 2},
    {"id": "kochi_r", "name": "Kochi refinery", "kind": "refinery", "lat": 9.97, "lon": 76.37, "country": "IN",
     "owner": "BPCL", "symbols": {"BPCL": -1}, "weight": 2},
    {"id": "bina", "name": "Bina refinery", "kind": "refinery", "lat": 24.20, "lon": 78.13, "country": "IN",
     "owner": "BPCL", "symbols": {"BPCL": -1}, "weight": 1},
    {"id": "visakh", "name": "Visakh refinery", "kind": "refinery", "lat": 17.70, "lon": 83.25, "country": "IN",
     "owner": "HPCL", "symbols": {"HINDPETRO": -1}, "weight": 2},
    {"id": "bathinda", "name": "Guru Gobind Singh refinery (Bathinda)", "kind": "refinery", "lat": 30.11,
     "lon": 74.97, "country": "IN", "owner": "HPCL-Mittal", "symbols": {"HINDPETRO": -1}, "weight": 1},
    {"id": "mrpl", "name": "Mangalore refinery (MRPL)", "kind": "refinery", "lat": 12.98, "lon": 74.83,
     "country": "IN", "owner": "ONGC/MRPL", "symbols": {"ONGC": -1}, "weight": 1},
    {"id": "cpcl", "name": "Manali refinery (CPCL)", "kind": "refinery", "lat": 13.17, "lon": 80.26,
     "country": "IN", "owner": "IOC/CPCL", "symbols": {"IOC": -1}, "weight": 1},
    {"id": "numaligarh", "name": "Numaligarh refinery", "kind": "refinery", "lat": 26.61, "lon": 93.72,
     "country": "IN", "owner": "NRL (OIL)", "symbols": {"OIL": -1}, "weight": 1},
    # India — gas / LNG / fields
    {"id": "dahej", "name": "Dahej LNG", "kind": "lng", "lat": 21.70, "lon": 72.55, "country": "IN",
     "owner": "Petronet", "symbols": {"PETRONET": -1, "GAIL": -1}, "weight": 2},
    {"id": "kochi_lng", "name": "Kochi LNG", "kind": "lng", "lat": 9.98, "lon": 76.25, "country": "IN",
     "owner": "Petronet", "symbols": {"PETRONET": -1}, "weight": 1},
    {"id": "hazira_lng", "name": "Hazira LNG", "kind": "lng", "lat": 21.10, "lon": 72.65, "country": "IN",
     "owner": "Shell", "weight": 1},
    {"id": "mumbaihigh", "name": "Mumbai High (offshore)", "kind": "field", "lat": 19.40, "lon": 71.30,
     "country": "IN", "owner": "ONGC", "symbols": {"ONGC": -1}, "weight": 2, "flare": True, "r_km": 35},
    {"id": "kgd6", "name": "KG basin (offshore)", "kind": "field", "lat": 16.25, "lon": 82.35, "country": "IN",
     "owner": "Reliance / ONGC", "symbols": {"RELIANCE": -1, "ONGC": -1}, "weight": 1, "flare": True, "r_km": 40},
    # Gulf
    {"id": "abqaiq", "name": "Abqaiq processing", "kind": "processing", "lat": 25.94, "lon": 49.68,
     "country": "SA", "owner": "Saudi Aramco", "weight": 3, "flare": True},
    {"id": "rastanura", "name": "Ras Tanura refinery", "kind": "refinery", "lat": 26.66, "lon": 50.16,
     "country": "SA", "owner": "Saudi Aramco", "weight": 3, "flare": True},
    {"id": "ghawar", "name": "Ghawar field", "kind": "field", "lat": 25.40, "lon": 49.35, "country": "SA",
     "owner": "Saudi Aramco", "weight": 3, "flare": True, "r_km": 60},
    {"id": "kharg_t", "name": "Kharg Island terminal", "kind": "terminal", "lat": 29.24, "lon": 50.32,
     "country": "IR", "owner": "NIOC", "weight": 3, "flare": True},
    {"id": "abadan", "name": "Abadan refinery", "kind": "refinery", "lat": 30.34, "lon": 48.28, "country": "IR",
     "owner": "NIORDC", "weight": 2, "flare": True},
    {"id": "bandarabbas", "name": "Bandar Abbas refinery", "kind": "refinery", "lat": 27.20, "lon": 56.10,
     "country": "IR", "owner": "NIORDC", "weight": 2, "flare": True},
    {"id": "southpars", "name": "South Pars / Asaluyeh", "kind": "processing", "lat": 27.48, "lon": 52.61,
     "country": "IR", "owner": "NIOC", "weight": 3, "flare": True, "r_km": 25},
    {"id": "rumaila", "name": "Rumaila field", "kind": "field", "lat": 30.40, "lon": 47.40, "country": "IQ",
     "owner": "BP / BOC", "weight": 2, "flare": True, "r_km": 40},
    {"id": "ruwais", "name": "Ruwais refinery", "kind": "refinery", "lat": 24.11, "lon": 52.73, "country": "AE",
     "owner": "ADNOC", "weight": 2, "flare": True},
    {"id": "fujairah_t", "name": "Fujairah storage hub", "kind": "terminal", "lat": 25.20, "lon": 56.35,
     "country": "AE", "owner": "various", "weight": 2},
    {"id": "raslaffan", "name": "Ras Laffan LNG", "kind": "lng", "lat": 25.92, "lon": 51.55, "country": "QA",
     "owner": "QatarEnergy", "symbols": {"PETRONET": -1, "GAIL": -1}, "weight": 3, "flare": True},
    {"id": "ahmadi", "name": "Mina al-Ahmadi refinery", "kind": "refinery", "lat": 29.07, "lon": 48.15,
     "country": "KW", "owner": "KNPC", "weight": 2, "flare": True},
    {"id": "yanbu_r", "name": "Yanbu refining hub", "kind": "refinery", "lat": 24.00, "lon": 38.10,
     "country": "SA", "owner": "Saudi Aramco", "weight": 2, "flare": True},
    {"id": "jizan", "name": "Jazan refinery", "kind": "refinery", "lat": 16.75, "lon": 42.65, "country": "SA",
     "owner": "Saudi Aramco", "weight": 1},
    # Russia (India buys Russian crude)
    {"id": "sheskharis", "name": "Novorossiysk (Sheskharis)", "kind": "terminal", "lat": 44.68, "lon": 37.80,
     "country": "RU", "owner": "Transneft", "weight": 3},
    {"id": "tuapse", "name": "Tuapse refinery", "kind": "refinery", "lat": 44.10, "lon": 39.08, "country": "RU",
     "owner": "Rosneft", "weight": 1},
    {"id": "primorsk_t", "name": "Primorsk terminal", "kind": "terminal", "lat": 60.35, "lon": 28.67,
     "country": "RU", "owner": "Transneft", "weight": 2},
    {"id": "ustluga_t", "name": "Ust-Luga terminal", "kind": "terminal", "lat": 59.68, "lon": 28.43,
     "country": "RU", "owner": "various", "weight": 2},
    {"id": "kirishi", "name": "Kirishi refinery", "kind": "refinery", "lat": 59.45, "lon": 32.05, "country": "RU",
     "owner": "Surgutneftegas", "weight": 1},
    {"id": "ryazan", "name": "Ryazan refinery", "kind": "refinery", "lat": 54.58, "lon": 39.75, "country": "RU",
     "owner": "Rosneft", "weight": 1},
    {"id": "volgograd", "name": "Volgograd refinery", "kind": "refinery", "lat": 48.62, "lon": 44.47,
     "country": "RU", "owner": "Lukoil", "weight": 1},
]

# ------------------------------------------------------------------ airspace watch points
# adsb.lol point queries (radius up to 250 nm). Traffic far below its usual level for the hour
# usually means closed or avoided airspace.
AIR_REGIONS: list[dict] = [
    {"id": "tehran", "name": "Iran (Tehran FIR)", "lat": 35.69, "lon": 51.39, "nm": 250, "india": 0.9},
    {"id": "baghdad", "name": "Iraq", "lat": 33.31, "lon": 44.36, "nm": 250, "india": 0.7},
    {"id": "gulf", "name": "Gulf (Doha–Dubai)", "lat": 25.30, "lon": 53.50, "nm": 250, "india": 1.0},
    {"id": "telaviv", "name": "Israel / Levant", "lat": 32.01, "lon": 34.88, "nm": 200, "india": 0.5},
    {"id": "karachi", "name": "Pakistan south", "lat": 24.90, "lon": 67.16, "nm": 250, "india": 0.9},
    {"id": "lahore", "name": "Pakistan north", "lat": 31.52, "lon": 74.40, "nm": 200, "india": 1.0},
    {"id": "delhi", "name": "Delhi", "lat": 28.56, "lon": 77.10, "nm": 200, "india": 1.0},
    {"id": "mumbai", "name": "Mumbai", "lat": 19.09, "lon": 72.87, "nm": 200, "india": 1.0},
    {"id": "redsea", "name": "Red Sea", "lat": 20.00, "lon": 39.50, "nm": 250, "india": 0.6},
    {"id": "kyiv", "name": "Ukraine", "lat": 50.45, "lon": 30.52, "nm": 250, "india": 0.3},
]

# ------------------------------------------------------------------ conflict regions (India relevance)
# bbox = west, south, east, north
REGIONS: list[dict] = [
    {"id": "india_pak", "name": "India–Pakistan", "bbox": (66, 23, 81, 37), "india": 1.0},
    {"id": "gulf", "name": "Gulf & Iran", "bbox": (44, 22, 63, 39), "india": 1.0},
    {"id": "redsea", "name": "Red Sea & Yemen", "bbox": (32, 10, 46, 30), "india": 0.9},
    {"id": "levant", "name": "Israel & Levant", "bbox": (33, 29, 38.5, 37.5), "india": 0.6},
    {"id": "russia_ukr", "name": "Russia–Ukraine", "bbox": (22, 44, 50, 57), "india": 0.5},
    {"id": "scs", "name": "South China Sea & Taiwan", "bbox": (105, 5, 125, 27), "india": 0.5},
    {"id": "china_india", "name": "India–China border", "bbox": (74, 26, 98, 36), "india": 0.9},
    {"id": "myanmar_bd", "name": "Myanmar & Bangladesh", "bbox": (88, 9, 102, 29), "india": 0.4},
    {"id": "horn", "name": "Horn of Africa & Sudan", "bbox": (21, -2, 52, 10), "india": 0.3},
]
OTHER_REGION = {"id": "other", "name": "Rest of world", "india": 0.15}

# AIS subscription boxes: [[lat, lon], [lat, lon]] corners (AISStream order is latitude first)
AIS_BOXES: list[list[list[float]]] = [
    [[22.0, 47.0], [31.0, 60.5]],     # Persian Gulf, Hormuz, Gulf of Oman
    [[11.0, 32.0], [30.5, 52.0]],     # Red Sea, Bab el-Mandeb, Gulf of Aden
    [[29.0, 28.0], [35.0, 36.0]],     # Suez, East Med
    [[6.0, 60.5], [24.5, 78.0]],      # Arabian Sea, India west coast
    [[5.0, 78.0], [23.0, 95.0]],      # Bay of Bengal
    [[-2.0, 95.0], [8.0, 106.0]],     # Malacca, Singapore
]

# FIRMS area boxes (west, south, east, north)
FIRE_BOXES: list[tuple[float, float, float, float]] = [
    (32, 8, 98, 38),        # Red Sea -> Gulf -> India -> Bay of Bengal
    (22, 43, 50, 61),       # Ukraine / western Russia
]


# ------------------------------------------------------------------ helpers
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(min(1.0, math.sqrt(a)))


def region_of(lat: float, lon: float) -> dict:
    for r in REGIONS:
        w, s, e, n = r["bbox"]
        if w <= lon <= e and s <= lat <= n:
            return r
    return OTHER_REGION


def _overrides() -> dict:
    p = Path(settings.data_dir) / "config" / "geo_sites.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def _merge(base: list[dict], extra: list[dict]) -> list[dict]:
    by = {x["id"]: dict(x) for x in base}
    for x in extra or []:
        if "id" in x:
            by[x["id"]] = {**by.get(x["id"], {}), **x}
    return list(by.values())


def load_sites() -> dict:
    """All static geography, with user overrides applied and impact symbols limited to the F&O list."""
    o = _overrides()
    out = {
        "chokepoints": _merge(CHOKEPOINTS, o.get("chokepoints", [])),
        "ports": _merge(PORTS, o.get("ports", [])),
        "energy": _merge(ENERGY, o.get("energy", [])),
        "air_regions": _merge(AIR_REGIONS, o.get("air_regions", [])),
    }
    for key in ("ports", "energy"):
        for x in out[key]:
            x["symbols"] = {k: v for k, v in (x.get("symbols") or {}).items() if k in FNO}
            x.setdefault("weight", 1)
            x.setdefault("flare", False)
            x.setdefault("r_km", 8 if key == "energy" else 10)
            x["approx"] = True
    return out
