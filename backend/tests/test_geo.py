import time

import pandas as pd

from td.geo import engine, sources
from td.geo.sim import GeoSim
from td.geo.sites import load_sites


def _gdelt_row(**kw):
    c = [""] * 61
    c[0] = kw.get("id", "1")
    c[6], c[7], c[16], c[17] = "ISRAEL", "ISR", "MILITANT", ""
    c[25] = kw.get("root_event", "1")
    c[26] = kw.get("code", "195")
    c[28] = kw.get("root", "19")
    c[29] = kw.get("quad", "4")
    c[30], c[32], c[33], c[34] = "-10", "3", str(kw.get("articles", 6)), "-7.5"
    c[51], c[52], c[53] = "4", "Gaza, Gaza Strip", "GZ"
    c[56], c[57] = str(kw.get("lat", 31.5)), str(kw.get("lon", 34.45))
    c[59] = "20261012041500"
    c[60] = "https://example.org/story"
    return "\t".join(c)


def test_gdelt_parse_filters_and_fields():
    text = "\n".join([
        _gdelt_row(id="1"),
        _gdelt_row(id="2", root="04", code="040", quad="1"),   # cooperation -> dropped
        _gdelt_row(id="3", root_event="0"),                     # not a root event -> dropped
        _gdelt_row(id="4", articles=1),                         # single article -> dropped
        _gdelt_row(id="5", lat="", lon=""),                     # no location -> dropped
    ])
    ev = sources.parse_gdelt_export(text)
    assert [e["id"] for e in ev] == ["1"]
    e = ev[0]
    assert e["label"] == "Air strike" and e["precise"] and e["url"].startswith("https://")
    assert e["sev"] > sources.cameo_weight("195")


def test_gdelt_dedupe_merges_same_incident():
    a = sources.parse_gdelt_export(_gdelt_row(id="1", articles=4))[0]
    b = dict(a, id="2", articles=5, lat=a["lat"] + 0.01)
    merged = sources.gdelt_dedupe([a, b])
    assert len(merged) == 1 and merged[0]["articles"] == 9


def test_gdelt_stamps_are_quarter_hours():
    from datetime import datetime, timezone
    st = sources.gdelt_stamps_back(datetime(2026, 10, 12, 4, 37, tzinfo=timezone.utc), 1)
    assert st[0] == "20261012041500" and len(st) == 4 and all(s[-4:] in ("0000", "1500", "3000", "4500") for s in st)


def test_firms_csv():
    csv = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,"
           "bright_ti5,frp,daynight\n"
           "22.36,69.86,340.1,0.4,0.4,2026-10-11,0805,N,VIIRS,n,2.0NRT,290.2,45.3,D\n"
           "22.37,69.87,300.1,0.4,0.4,2026-10-11,0805,N,VIIRS,l,2.0NRT,290.2,3.1,D\n")
    pts = sources.parse_firms_csv(csv, "VIIRS_NOAA20_NRT")
    assert len(pts) == 1 and pts[0]["frp"] == 45.3 and pts[0]["sat"] == "NOAA20"


def test_ais_parsing():
    pos = {"MessageType": "PositionReport", "MetaData": {"MMSI": 419001234, "ShipName": "GANGA STAR ",
                                                        "latitude": 26.3, "longitude": 56.4},
           "Message": {"PositionReport": {"UserID": 419001234, "Latitude": 26.31, "Longitude": 56.41, "Sog": 12.3,
                                          "Cog": 101.0, "TrueHeading": 511}}}
    kind, d = sources.parse_ais(pos)
    assert kind == "pos" and d["lat"] == 26.31 and d["hdg"] is None and d["name"] == "GANGA STAR"
    st = {"MessageType": "ShipStaticData", "MetaData": {"MMSI": 419001234},
          "Message": {"ShipStaticData": {"Type": 84, "Name": "GANGA STAR", "Destination": "SIKKA"}}}
    kind, d = sources.parse_ais(st)
    assert kind == "static" and sources.ship_class(d["type"]) == "TANKER"
    assert sources.parse_ais({"MessageType": "PositionReport", "MetaData": {}, "Message": {}}) is None


def test_adsb_parsing():
    js = {"ac": [{"hex": "ae1234", "flight": "RCH123 ", "t": "C17", "lat": 25.1, "lon": 55.2, "alt_baro": 31000,
                  "gs": 430.2, "track": 95.4, "dbFlags": 1},
                 {"hex": "800abc", "flight": "AIC101", "lat": 28.5, "lon": 77.1, "alt_baro": "ground"},
                 {"hex": "nolatlon"}]}
    rows = sources.parse_adsb(js)
    assert len(rows) == 2 and rows[0]["mil"] and rows[0]["flight"] == "RCH123" and rows[1]["alt"] == 0


def test_portwatch_and_transit_stats():
    js = {"features": [{"attributes": {"date": f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", "portname": "X",
                                       "n_total": 100 if i < 200 else 40, "n_tanker": 30 if i < 200 else 10}}
                       for i in range(240)]}
    rows = sources.parse_portwatch(js)
    st = sources.transit_stats(rows)
    assert st["ratio"] == 0.4 and st["tanker_ratio"] < 0.5 and st["base_total"] == 100


def test_fred_csv_both_headers():
    for head in ("observation_date", "DATE"):
        rows = sources.parse_fred_csv(f"{head},DCOILBRENTEU\n2026-10-08,70.1\n2026-10-09,.\n2026-10-10,71.5\n")
        assert rows == [["2026-10-08", 70.1], ["2026-10-10", 71.5]]
    assert sources.pct_change([["a", 100.0], ["b", 102.0]]) == 2.0


def test_gpr_links_and_daily():
    html = '<a href="gpr_files/data_gpr_daily_recent_20261005.xls">daily</a> <a href="gpr_files/data_gpr_export_202610.xls">m</a>'
    links = sources.find_gpr_links(html)
    assert links["daily"].endswith("data_gpr_daily_recent_20261005.xls") and links["monthly"].startswith("https://")
    df = pd.DataFrame({"DAY": [20261004, 20261005], "GPRD": [110.5, 131.2]})
    assert sources.parse_gpr_daily(df) == [["2026-10-04", 110.5], ["2026-10-05", 131.2]]


def test_engine_rules():
    sites = load_sites()
    now = time.time()
    cps = [c for c in sites["chokepoints"] if c["id"] in ("hormuz", "bab")]
    pw = {"bab": {"ratio": 0.45, "series": []}, "hormuz": {"ratio": 0.98, "series": []}}
    ships = {1: {"lat": 26.3, "lon": 56.8, "sog": 11, "cls": "TANKER"}}
    rows = {r["id"]: r for r in engine.chokepoint_view(cps, ships, {}, pw, [], now)}
    assert rows["bab"]["status"] == "DISRUPTED" and rows["hormuz"]["status"] == "NORMAL"
    assert rows["hormuz"]["tankers_now"] == 1

    jam = [s for s in sites["energy"] if s["id"] == "jamnagar"]
    hot = [{"lat": 22.36, "lon": 69.86, "frp": 60.0, "ts": now - 3600} for _ in range(9)]
    f = engine.fire_sites(jam, hot, {"jamnagar": [1, 0, 2, 1, 1, 0, 1]}, now)[0]
    assert f["status"] == "ANOMALY" and f["symbols"] == {"RELIANCE": -1}
    f2 = engine.fire_sites(jam, hot[:2], {"jamnagar": [1, 2, 2, 1, 2, 1, 2]}, now)[0]
    assert f2["status"] != "ANOMALY"

    air = engine.airspace([{"id": "x", "name": "X", "lat": 0, "lon": 0, "nm": 100, "india": 1}], {"x": 5},
                          {"x": [60, 55, 62, 58]})
    assert air[0]["status"] == "CLOSED?"

    risk = engine.risk_index([], list(rows.values()), [f], air, {"crude_chg": 3.0, "usdinr_chg": 0.5, "india_vix": 18})
    assert 0 <= risk["value"] <= 100 and risk["level"] in ("LOW", "ELEVATED", "HIGH", "SEVERE")
    sig = engine.impact(list(rows.values()), [f], [], air, {"crude_chg": 3.0, "usdinr_chg": 0.5})
    ids = {s["id"] for s in sig}
    assert {"crude", "inr", "chokepoints", "fire:jamnagar", "airspace"} <= ids
    crude = next(s for s in sig if s["id"] == "crude")
    assert {"s": "BPCL", "dir": -1} in crude["symbols"] and {"s": "ONGC", "dir": 1} in crude["symbols"]


def test_geo_sim_shapes():
    sim = GeoSim(seed=7)
    sim.step()
    assert len(sim.ships) > 300 and all(-90 <= s["lat"] <= 90 for s in sim.ships.values())
    assert len(sim.events) > 50
    sites = load_sites()["energy"]
    fires = sim.fires(sites)
    inc = next(s for s in sites if s["id"] == sim.incident_site)
    near = [p for p in fires if abs(p["lat"] - inc["lat"]) < 0.1 and abs(p["lon"] - inc["lon"]) < 0.1 and p["frp"] >= 60]
    assert len(near) >= 7
    counts, base = sim.air_counts()
    assert counts[sim.closed_region] < min(base[sim.closed_region])
