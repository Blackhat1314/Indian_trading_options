"""Simulated world feeds for the GEO desk (DATA_SOURCE=sim, or GEO_SOURCE=sim).

Produces the same normalized records as the live connectors — ships moving along real
shipping lanes, conflict events in the usual hot spots, gas flares plus Punjab stubble fires
and the occasional refinery incident, civil and military aircraft, PortWatch-style transit
history, a GPR-like index and macro series — so every view and every derived signal works
without keys or internet. Nothing here is real data.
"""
from __future__ import annotations

import math
import random
import time
from datetime import date, datetime, timedelta, timezone

from .sites import haversine_km
from .sources import CAMEO, cameo_label, cameo_weight, ship_class

LANES = {
    "hormuz_mumbai": [(26.4, 56.5), (25.0, 57.6), (22.5, 60.0), (20.0, 66.0), (19.0, 72.6)],
    "hormuz_sikka": [(26.4, 56.5), (24.5, 58.5), (23.0, 64.0), (22.6, 69.2)],
    "rastanura_hormuz": [(26.7, 50.4), (26.9, 52.5), (26.6, 55.0), (26.4, 56.5)],
    "kharg_hormuz": [(29.2, 50.3), (27.6, 51.9), (26.6, 55.0), (26.4, 56.5)],
    "laffan_hormuz": [(25.95, 51.7), (26.3, 53.5), (26.6, 55.0), (26.4, 56.5)],
    "suez_bab": [(29.9, 32.6), (27.5, 34.0), (22.0, 37.6), (16.0, 41.0), (12.6, 43.3)],
    "bab_mumbai": [(12.6, 43.3), (12.0, 46.0), (13.0, 51.0), (15.0, 60.0), (18.0, 68.0), (18.9, 72.8)],
    "westcoast_malacca": [(18.9, 72.8), (12.0, 74.5), (7.0, 77.5), (5.8, 80.5), (6.0, 94.5)],
    "chennai_malacca": [(13.1, 80.4), (10.0, 86.0), (6.0, 94.5)],
    "paradip_malacca": [(20.2, 86.8), (15.0, 90.0), (6.0, 94.5)],
    "malacca": [(6.0, 94.5), (4.5, 98.5), (2.5, 101.3), (1.3, 103.6)],
}
LANE_SHIPS = {"hormuz_mumbai": 70, "hormuz_sikka": 45, "rastanura_hormuz": 50, "kharg_hormuz": 30,
              "laffan_hormuz": 35, "suez_bab": 30, "bab_mumbai": 35, "westcoast_malacca": 60,
              "chennai_malacca": 35, "paradip_malacca": 30, "malacca": 90}
GULF_LANES = {"hormuz_mumbai", "hormuz_sikka", "rastanura_hormuz", "kharg_hormuz", "laffan_hormuz"}
NAMES = ["OCEAN", "STAR", "PACIFIC", "GULF", "PEARL", "EAGLE", "NORDIC", "GLORY", "MARINER", "SPIRIT", "HARMONY",
         "AURORA", "LEGEND", "PIONEER", "ATLAS", "VEGA", "ORION", "LOTUS", "GANGA", "SAGAR"]

CONFLICT_SPOTS = [  # lat, lon, sigma deg, events/hour, typical codes, actors
    (48.0, 37.8, 1.3, 5.0, ["190", "193", "194", "1952", "195"], ("Russia", "Ukraine")),
    (31.45, 34.40, 0.15, 2.5, ["190", "195", "1951", "193"], ("Israel", "Militants")),
    (33.30, 35.40, 0.4, 0.8, ["195", "190"], ("Israel", "Hezbollah")),
    (15.30, 43.00, 1.6, 0.9, ["190", "195", "1952", "191"], ("Houthi", "Shipping")),
    (15.60, 32.50, 2.5, 1.6, ["190", "193", "202"], ("Sudan Army", "Rsf")),
    (21.00, 96.00, 2.0, 1.2, ["190", "193", "195"], ("Military", "Rebels")),
    (34.00, 74.30, 0.5, 0.25, ["193", "190", "196"], ("India", "Pakistan")),
    (28.50, 65.50, 1.5, 0.5, ["183", "193", "1823"], ("Militants", "Security Forces")),
    (33.00, 45.00, 1.5, 0.3, ["183", "190"], ("Militia", "Security Forces")),
    (2.0, 45.3, 1.5, 0.5, ["183", "193"], ("Al Shabaab", "Military")),
]

AIR_BASE = {"tehran": 55, "baghdad": 45, "gulf": 190, "telaviv": 35, "karachi": 45, "lahore": 30, "delhi": 140,
            "mumbai": 115, "redsea": 60, "kyiv": 2}


def _interp(lane: list, t: float) -> tuple[float, float, float]:
    """Point at fraction t of a polyline + bearing (deg)."""
    segs = [haversine_km(*lane[i], *lane[i + 1]) for i in range(len(lane) - 1)]
    total = sum(segs)
    d = t * total
    for i, L in enumerate(segs):
        if d <= L or i == len(segs) - 1:
            f = d / L if L else 0
            (a1, o1), (a2, o2) = lane[i], lane[i + 1]
            lat, lon = a1 + (a2 - a1) * f, o1 + (o2 - o1) * f
            brg = (math.degrees(math.atan2(o2 - o1, a2 - a1)) + 360) % 360
            return lat, lon, brg
        d -= L
    return lane[-1][0], lane[-1][1], 0.0


class GeoSim:
    def __init__(self, seed: int | None = None):
        today = date.today()
        self.seed = seed if seed is not None else int(today.strftime("%Y%m%d"))
        self.rng = random.Random(self.seed)
        self.t0 = time.time()
        self.last_step = self.t0
        self.ships: dict[int, dict] = {}
        self._lane_state: dict[int, dict] = {}
        self._init_ships()
        self.events: list[dict] = []
        self._eid = 1
        self._backfill_events(hours=24)
        # scenario of the day
        r = random.Random(self.seed + 1)
        self.incident_site = r.choice(["ryazan", "volgograd", "tuapse", "sheskharis", "jizan", "mrpl", "bandarabbas"])
        self.incident_at = self.t0 - r.uniform(2, 20) * 3600
        self.closed_region = r.choice(["lahore", "tehran", "telaviv", "baghdad"])
        self.closed_ratio = r.uniform(0.15, 0.45)

    # ---------------------------------------------------------------- ships
    def _init_ships(self):
        mmsi = 419000000
        for lane, n in LANE_SHIPS.items():
            for _ in range(n):
                mmsi += self.rng.randint(1, 999)
                r = self.rng.random()
                if lane in GULF_LANES:
                    t = 80 + self.rng.randint(0, 9) if r < 0.65 else 70 + self.rng.randint(0, 9)
                else:
                    t = 80 + self.rng.randint(0, 9) if r < 0.3 else 70 + self.rng.randint(0, 9) if r < 0.92 else 60
                self._lane_state[mmsi] = {"lane": lane, "pos": self.rng.random(), "dir": self.rng.choice((1, -1)),
                                          "sog": round(self.rng.uniform(9, 16), 1),
                                          "off": self.rng.uniform(-0.18, 0.18)}
                name = f"{self.rng.choice(NAMES)} {self.rng.choice(NAMES)}"
                self.ships[mmsi] = {"mmsi": mmsi, "type": t, "cls": ship_class(t), "name": name,
                                    "dest": self.rng.choice(["MUMBAI", "SIKKA", "KANDLA", "CHENNAI", "SINGAPORE",
                                                             "FUJAIRAH", "RAS TANURA", "SUEZ", "COLOMBO"])}
        # anchorage off Fujairah (waiting ships, ~0 kn)
        for _ in range(40):
            mmsi += self.rng.randint(1, 999)
            self._lane_state[mmsi] = {"lane": None, "lat": 25.15 + self.rng.uniform(-0.2, 0.25),
                                      "lon": 56.45 + self.rng.uniform(-0.1, 0.25)}
            t = 80 + self.rng.randint(0, 9)
            self.ships[mmsi] = {"mmsi": mmsi, "type": t, "cls": ship_class(t), "name": f"{self.rng.choice(NAMES)} {mmsi % 100}",
                                "dest": "FUJAIRAH ANCH"}
        self._place_ships(0)

    def _place_ships(self, dt: float):
        now = time.time()
        for m, st in self._lane_state.items():
            s = self.ships[m]
            if st["lane"] is None:
                s.update(lat=round(st["lat"], 4), lon=round(st["lon"], 4), sog=round(self.rng.uniform(0, 0.4), 1),
                         cog=None, hdg=None, ts=now)
                continue
            lane = LANES[st["lane"]]
            total = sum(haversine_km(*lane[i], *lane[i + 1]) for i in range(len(lane) - 1))
            st["pos"] += st["dir"] * st["sog"] * 1.852 * dt / 3600 / total
            if st["pos"] > 1 or st["pos"] < 0:
                st["dir"] *= -1
                st["pos"] = min(1.0, max(0.0, st["pos"]))
            lat, lon, brg = _interp(lane, st["pos"])
            if st["dir"] < 0:
                brg = (brg + 180) % 360
            # side of the lane depends on direction (traffic separation)
            off = st["off"] * (1 if st["dir"] > 0 else -1)
            s.update(lat=round(lat + off * math.cos(math.radians(brg + 90)), 4),
                     lon=round(lon + off * math.sin(math.radians(brg + 90)), 4),
                     sog=st["sog"], cog=round(brg, 1), hdg=int(brg), ts=now)

    def step(self):
        now = time.time()
        dt = (now - self.last_step) * 20          # ships move 20x faster so the map visibly changes
        self.last_step = now
        self._place_ships(dt)
        self._spawn_events(now)

    # ---------------------------------------------------------------- conflict
    def _make_event(self, ts: float, spot: tuple) -> dict:
        lat0, lon0, sig, _, codes, actors = spot
        code = self.rng.choice(codes)
        lat = lat0 + self.rng.gauss(0, sig)
        lon = lon0 + self.rng.gauss(0, sig)
        arts = max(2, int(self.rng.expovariate(1 / 6)))
        w = cameo_weight(code)
        self._eid += 1
        return {"id": f"sim{self._eid}", "ts": ts, "lat": round(lat, 4), "lon": round(lon, 4), "code": code,
                "root": code[:2], "label": cameo_label(code), "a1": actors[0], "a2": actors[1], "a1c": "", "a2c": "",
                "place": f"{lat:.2f}, {lon:.2f}", "country": "", "precise": self.rng.random() > 0.25,
                "articles": arts, "sources": max(1, arts // 2), "goldstein": -10.0 if code.startswith("19") else -9.0,
                "tone": round(self.rng.uniform(-9, -3), 2), "url": "", "sev": round(w * (1 + math.log2(1 + arts)), 2)}

    def _backfill_events(self, hours: int):
        now = time.time()
        for spot in CONFLICT_SPOTS:
            n = int(spot[3] * hours * self.rng.uniform(0.7, 1.3))
            for _ in range(n):
                self.events.append(self._make_event(now - self.rng.uniform(0, hours * 3600), spot))
        for _ in range(int(hours * 1.5)):
            spot = (self.rng.uniform(-30, 55), self.rng.uniform(-80, 140), 0.5, 0, list(CAMEO)[6:20],
                    ("Police", "Protesters"))
            self.events.append(self._make_event(now - self.rng.uniform(0, hours * 3600), spot))

    def _spawn_events(self, now: float):
        for spot in CONFLICT_SPOTS:
            # 10x accelerated arrivals so the feed moves while you watch
            if self.rng.random() < spot[3] * 10 * 5 / 3600:
                self.events.append(self._make_event(now, spot))
        self.events = [e for e in self.events if now - e["ts"] < 26 * 3600]

    # ---------------------------------------------------------------- fires
    def fires(self, sites: list[dict]) -> list[dict]:
        now = time.time()
        r = random.Random(self.seed + int(now // 1800))
        out = []
        for s in sites:
            if s.get("flare"):
                for _ in range(r.randint(1, 4)):
                    out.append({"lat": s["lat"] + r.gauss(0, 0.02), "lon": s["lon"] + r.gauss(0, 0.02),
                                "frp": round(r.uniform(4, 45), 1), "conf": "n", "ts": now - r.uniform(0, 30) * 3600,
                                "sat": r.choice(["SNPP", "NOAA20", "NOAA21"])})
            if s["id"] == self.incident_site and now > self.incident_at:
                for _ in range(r.randint(7, 12)):
                    out.append({"lat": s["lat"] + r.gauss(0, 0.012), "lon": s["lon"] + r.gauss(0, 0.012),
                                "frp": round(r.uniform(60, 380), 1), "conf": "h",
                                "ts": self.incident_at + r.uniform(0, 4 * 3600), "sat": "NOAA20"})
        # Punjab/Haryana crop-residue burning (Oct–Nov) + scattered fires
        month = datetime.now().month
        n_ag = 700 if month in (10, 11) else 120
        for _ in range(n_ag):
            out.append({"lat": r.uniform(29.4, 31.6), "lon": r.uniform(74.0, 77.2), "frp": round(r.uniform(1.5, 18), 1),
                        "conf": "n", "ts": now - r.uniform(0, 30) * 3600, "sat": "SNPP"})
        for _ in range(250):
            out.append({"lat": r.uniform(10, 30), "lon": r.uniform(70, 92), "frp": round(r.uniform(1, 15), 1),
                        "conf": "n", "ts": now - r.uniform(0, 30) * 3600, "sat": "NOAA21"})
        for _ in range(140):
            out.append({"lat": r.uniform(45, 55), "lon": r.uniform(30, 45), "frp": round(r.uniform(2, 40), 1),
                        "conf": "n", "ts": now - r.uniform(0, 30) * 3600, "sat": "NOAA20"})
        return out

    def fire_history(self, sites: list[dict]) -> dict[str, list[int]]:
        r = random.Random(self.seed + 7)
        return {s["id"]: [r.randint(1, 4) if s.get("flare") else (1 if r.random() < 0.15 else 0) for _ in range(14)]
                for s in sites}

    # ---------------------------------------------------------------- aircraft
    def aircraft(self) -> list[dict]:
        now = time.time()
        r = random.Random(self.seed + int(now // 30))
        out = []
        routes = [((28.56, 77.10), (25.25, 55.36)), ((19.09, 72.87), (25.25, 55.36)), ((28.56, 77.10), (51.47, -0.45)),
                  ((25.25, 55.36), (51.47, -0.45)), ((19.09, 72.87), (1.36, 103.99)), ((12.95, 77.66), (25.27, 51.61)),
                  ((28.56, 77.10), (13.69, 100.75)), ((25.27, 51.61), (48.35, 11.78))]
        for i in range(260):
            (a1, o1), (a2, o2) = routes[i % len(routes)]
            f = (i * 0.137 + now / 5400) % 1
            lat, lon = a1 + (a2 - a1) * f, o1 + (o2 - o1) * f
            brg = (math.degrees(math.atan2(o2 - o1, a2 - a1)) + 360) % 360
            if i % 2:
                lat, lon, brg = a2 + (a1 - a2) * f, o2 + (o1 - o2) * f, (brg + 180) % 360
            out.append({"hex": f"8{i:05x}", "flight": f"{r.choice(['AIC', 'IGO', 'UAE', 'QTR', 'ETD', 'SIA', 'AXB'])}{100 + i}",
                        "reg": "", "type": r.choice(["A20N", "B77W", "A359", "B788", "A21N", "B738"]),
                        "lat": round(lat + r.gauss(0, 0.3), 3), "lon": round(lon + r.gauss(0, 0.3), 3),
                        "alt": r.choice([33000, 35000, 37000, 39000]), "gs": r.randint(430, 510), "track": round(brg),
                        "mil": False, "sqk": ""})
        mil = [("RCH", "C17", 26.0, 52.0), ("QID", "K35R", 27.5, 50.5), ("FORTE", "Q4", 43.5, 33.5),
               ("IN", "P8", 17.5, 69.0), ("IN", "P8", 12.0, 84.0), ("NAVY", "P8", 13.0, 47.0),
               ("LAGR", "K35R", 32.0, 35.5), ("HOMER", "R135", 44.0, 31.0), ("DUKE", "C30J", 25.0, 47.5)]
        for j, (cs, typ, la, lo) in enumerate(mil * 3):
            ang = (now / 600 + j) % (2 * math.pi)
            out.append({"hex": f"ae{j:04x}", "flight": f"{cs}{10 + j}", "reg": "", "type": typ,
                        "lat": round(la + 1.2 * math.sin(ang) + r.gauss(0, 0.4), 3),
                        "lon": round(lo + 1.6 * math.cos(ang) + r.gauss(0, 0.4), 3), "alt": r.choice([24000, 28000, 31000]),
                        "gs": r.randint(280, 420), "track": round(math.degrees(ang + math.pi / 2) % 360), "mil": True, "sqk": ""})
        return out

    def air_counts(self) -> tuple[dict[str, int], dict[str, list[int]]]:
        r = random.Random(self.seed + int(time.time() // 180))
        counts, base = {}, {}
        for k, b in AIR_BASE.items():
            hist = [max(0, int(b * r.uniform(0.85, 1.15))) for _ in range(10)]
            base[k] = hist
            n = b * r.uniform(0.85, 1.15)
            if k == self.closed_region:
                n = b * self.closed_ratio
            counts[k] = max(0, int(n))
        return counts, base

    # ---------------------------------------------------------------- slow series
    def portwatch(self, chokepoints: list[dict]) -> dict[str, list[dict]]:
        r = random.Random(self.seed + 3)
        typical = {"hormuz": 120, "bab": 70, "suez": 65, "malacca": 230, "cape": 60, "bosporus": 120, "kerch": 30,
                   "gibraltar": 290, "taiwan": 260, "sunda": 50, "lombok": 15, "panama": 35}
        level_now = {"bab": 0.45, "suez": 0.55, "cape": 1.5}
        out = {}
        end = date.today() - timedelta(days=5)
        for cp in chokepoints:
            base = typical.get(cp["id"], 50)
            rows = []
            for i in range(420, -1, -1):
                d = end - timedelta(days=i)
                lvl = level_now.get(cp["id"], 1.0) if i < 18 else 1.0
                tot = max(0, int(base * lvl * r.uniform(0.85, 1.15)))
                rows.append({"date": d.isoformat(), "name": cp["name"], "total": tot,
                             "tanker": int(tot * r.uniform(0.25, 0.45)), "container": int(tot * r.uniform(0.2, 0.4)),
                             "capacity": tot * 50000})
            out[cp["id"]] = rows
        return out

    def gpr(self) -> dict:
        r = random.Random(self.seed + 5)
        v, daily = 110.0, []
        start = date.today() - timedelta(days=400)
        for i in range(395):
            v = max(40.0, v + r.gauss(0, 6) - (v - 115) * 0.05 + (60 if r.random() < 0.01 else 0))
            daily.append([(start + timedelta(days=i)).isoformat(), round(v, 1)])
        months, ind = [], []
        m = date.today().replace(day=1)
        for k in range(36, 0, -1):
            mm = (m.month - k - 1) % 12 + 1
            yy = m.year + (m.month - k - 1) // 12
            ind.append([f"{yy}-{mm:02d}", round(max(0.02, 0.12 + r.gauss(0, 0.04)), 3)])
            months.append([f"{yy}-{mm:02d}", round(max(50, 115 + r.gauss(0, 25)), 1)])
        return {"daily": daily, "monthly": {"GPR": months, "GPRC_IND": ind}}

    def fred(self) -> dict[str, list[list]]:
        r = random.Random(self.seed + 9)
        spec = {"DCOILBRENTEU": (72, 1.6), "DCOILWTICO": (68, 1.6), "DEXINUS": (87.8, 0.18), "DGS10": (4.2, 0.04),
                "DTWEXBGS": (121, 0.35), "DHHNGSP": (3.1, 0.12)}
        out = {}
        start = date.today() - timedelta(days=400)
        for sid, (lvl, vol) in spec.items():
            v, rows = lvl, []
            for i in range(400):
                d = start + timedelta(days=i)
                if d.weekday() >= 5:
                    continue
                v = max(0.1, v + r.gauss(0, vol) - (v - lvl) * 0.03)
                rows.append([d.isoformat(), round(v, 3)])
            out[sid] = rows
        return out

    def macro_live(self) -> dict[str, dict]:
        now = time.time()
        r = random.Random(self.seed + 11)
        spec = {"CRUDEOIL": (6050, 0.012), "NATURALGAS": (265, 0.02), "GOLD": (118000, 0.006),
                "SILVER": (142000, 0.01), "USDINR": (88.4, 0.002)}
        out = {}
        for k, (lvl, vol) in spec.items():
            drift = r.gauss(0, vol * 1.5)
            wig = math.sin(now / 900 + r.random() * 6) * vol * 0.4
            chg = (drift + wig) * 100
            out[k] = {"ltp": round(lvl * (1 + chg / 100), 2), "chg": round(chg, 2), "src": "SIM"}
        return out

    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)
