import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import type { GeoJSONSource, Map as MLMap, MapGeoJSONFeature } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { feature, mesh } from "topojson-client";
import type { Topology } from "topojson-specification";
import type { Feature, FeatureCollection, Point, Polygon } from "geojson";
import type { GeoAir, GeoChokepoint, GeoConflict, GeoFires, GeoShips, GeoSites } from "./types";
import { PRESETS, ROOT_COLOR, SATELLITE, SHIP_COLORS, SHIP_LABEL, STATUS_COLOR, type Views } from "./views";

export interface Focus { lon: number; lat: number; zoom: number; key: number }

interface Props {
  views: Views; satId: string; satDate: string; focus: Focus | null;
  sites?: GeoSites; ships?: GeoShips; chokepoints?: GeoChokepoint[]; conflict?: GeoConflict; fires?: GeoFires; air?: GeoAir;
}

const EMPTY: FeatureCollection = { type: "FeatureCollection", features: [] };
const fc = (features: Feature[]): FeatureCollection => ({ type: "FeatureCollection", features });
const pt = (lon: number, lat: number, properties: Record<string, unknown>): Feature<Point> =>
  ({ type: "Feature", geometry: { type: "Point", coordinates: [lon, lat] }, properties });

/** Geodesic circle (radius km) as a polygon. */
function circle(lon: number, lat: number, km: number, n = 64): Feature<Polygon> {
  const R = 6371, d = km / R, la = (lat * Math.PI) / 180, lo = (lon * Math.PI) / 180;
  const ring: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const b = (2 * Math.PI * i) / n;
    const la2 = Math.asin(Math.sin(la) * Math.cos(d) + Math.cos(la) * Math.sin(d) * Math.cos(b));
    const lo2 = lo + Math.atan2(Math.sin(b) * Math.sin(d) * Math.cos(la), Math.cos(d) - Math.sin(la) * Math.sin(la2));
    ring.push([(lo2 * 180) / Math.PI, (la2 * 180) / Math.PI]);
  }
  return { type: "Feature", geometry: { type: "Polygon", coordinates: [ring] }, properties: {} };
}

/** White shapes drawn on a canvas, used as SDF icons so MapLibre can tint them per feature. */
function icon(shape: "arrow" | "plane" | "square" | "diamond" | "triangle" | "dot"): ImageData {
  const s = 32, c = document.createElement("canvas");
  c.width = c.height = s;
  const g = c.getContext("2d")!;
  g.fillStyle = "#fff";
  g.beginPath();
  if (shape === "arrow") { g.moveTo(16, 2); g.lineTo(26, 28); g.lineTo(16, 22); g.lineTo(6, 28); }
  else if (shape === "plane") {
    g.moveTo(16, 1); g.lineTo(19, 11); g.lineTo(30, 18); g.lineTo(30, 21); g.lineTo(19, 17); g.lineTo(18, 26);
    g.lineTo(22, 29); g.lineTo(22, 31); g.lineTo(16, 29); g.lineTo(10, 31); g.lineTo(10, 29); g.lineTo(14, 26);
    g.lineTo(13, 17); g.lineTo(2, 21); g.lineTo(2, 18); g.lineTo(13, 11);
  } else if (shape === "square") g.rect(8, 8, 16, 16);
  else if (shape === "diamond") { g.moveTo(16, 3); g.lineTo(29, 16); g.lineTo(16, 29); g.lineTo(3, 16); }
  else if (shape === "triangle") { g.moveTo(16, 4); g.lineTo(29, 27); g.lineTo(3, 27); }
  else g.arc(16, 16, 9, 0, Math.PI * 2);
  g.closePath();
  g.fill();
  return g.getImageData(0, 0, s, s);
}

const KIND_ICON: Record<string, string> = { refinery: "i-diamond", processing: "i-diamond", terminal: "i-square",
  field: "i-triangle", lng: "i-dot" };

function el(tag: string, cls?: string, text?: string): HTMLElement {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function row(parent: HTMLElement, k: string, v: string) {
  const r = el("div", "gp-row");
  r.append(el("span", "gp-k", k), el("span", "gp-v", v));
  parent.append(r);
}
function link(parent: HTMLElement, href: string, text: string) {
  if (!/^https?:\/\//.test(href)) return;
  const a = el("a", "gp-link", text) as HTMLAnchorElement;
  a.href = href;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  parent.append(a);
}
const ago = (ts: number) => {
  const s = Math.max(0, Date.now() / 1000 - ts);
  return s < 3600 ? `${Math.round(s / 60)} min ago` : s < 86400 ? `${(s / 3600).toFixed(1)} h ago` : `${(s / 86400).toFixed(1)} d ago`;
};
const ymd = (ts: number) => new Date(ts * 1000).toISOString().slice(0, 10);
export const worldview = (lon: number, lat: number, date: string) =>
  `https://worldview.earthdata.nasa.gov/?v=${lon - 1.5},${lat - 1},${lon + 1.5},${lat + 1}&t=${date}`;
export const copernicus = (lon: number, lat: number) =>
  `https://browser.dataspace.copernicus.eu/?zoom=13&lat=${lat.toFixed(4)}&lng=${lon.toFixed(4)}`;

function popupFor(f: MapGeoJSONFeature, layer: string): HTMLElement {
  const p = f.properties as Record<string, string | number | boolean>;
  const box = el("div", "gp");
  const [lon, lat] = (f.geometry as Point).coordinates;
  if (layer === "ships") {
    box.append(el("div", "gp-title", String(p.name || `MMSI ${p.mmsi}`)));
    row(box, "Type", SHIP_LABEL[Number(p.cls)] ?? "—");
    row(box, "Speed / course", `${p.sog ?? "—"} kn · ${p.cog ?? "—"}°`);
    if (p.dest) row(box, "Destination", String(p.dest));
    row(box, "MMSI", String(p.mmsi));
    link(box, `https://www.marinetraffic.com/en/ais/details/ships/mmsi:${p.mmsi}`, "MarineTraffic ↗");
  } else if (layer === "conflict") {
    box.append(el("div", "gp-title", String(p.label)));
    row(box, "Where", String(p.place || `${lat.toFixed(2)}, ${lon.toFixed(2)}`));
    row(box, "Actors", [p.a1, p.a2].filter(Boolean).join(" → ") || "—");
    row(box, "When", ago(Number(p.ts)));
    row(box, "Severity", `${p.sev} · ${p.articles} articles${p.precise ? "" : " · location approximate"}`);
    link(box, String(p.url), "Source article ↗");
  } else if (layer === "fires") {
    box.append(el("div", "gp-title", "Satellite hotspot"));
    row(box, "Fire power", `${p.frp} MW`);
    row(box, "Seen", ago(Number(p.ts)));
    link(box, worldview(lon, lat, ymd(Number(p.ts))), "NASA Worldview ↗");
  } else if (layer === "energy" || layer === "ports") {
    box.append(el("div", "gp-title", String(p.name)));
    row(box, "Type", `${p.kind}${p.owner ? ` · ${p.owner}` : ""}`);
    if (p.status) row(box, "Fire activity", `${p.status} · ${p.hotspots ?? 0} hotspots (usual ${p.baseline ?? "?"}/day)`);
    if (p.symbols) row(box, "NSE exposure", String(p.symbols));
    link(box, copernicus(lon, lat), "Copernicus Browser ↗");
    link(box, worldview(lon, lat, ymd(Date.now() / 1000 - 86400)), "NASA Worldview ↗");
  } else if (layer === "aircraft") {
    box.append(el("div", "gp-title", String(p.flight || p.hex)));
    row(box, "Type", `${p.type || "—"}${Number(p.mil) ? " · military" : ""}`);
    row(box, "Alt / speed", `${p.alt ?? "—"} ft · ${p.gs ?? "—"} kn`);
    link(box, `https://globe.adsb.lol/?icao=${p.hex}`, "adsb.lol ↗");
  } else if (layer === "choke-pt") {
    box.append(el("div", "gp-title", String(p.name)));
    row(box, "Status", String(p.status));
    row(box, "Vessels now", `${p.ships} (${p.tankers} tankers)`);
    if (p.ratio !== "") row(box, "Transits 7d vs normal", `${p.ratio}%`);
    if (p.reasons) row(box, "Why", String(p.reasons));
  }
  return box;
}

export function GeoMap(props: Props) {
  const box = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const labels = useRef<maplibregl.Marker[]>([]);
  const [ready, setReady] = useState(false);
  const [cursor, setCursor] = useState<string>("");
  // the map draws with WebGL (the graphics card): if the browser cannot give it a context, or the graphics driver
  // resets mid-session, only the map stops — the GEO panels keep working and the map can be reloaded in place
  const [glError, setGlError] = useState<"create" | "lost" | null>(null);
  const [attempt, setAttempt] = useState(0);
  const { views } = props;

  // ---------------------------------------------------------------- create
  useEffect(() => {
    if (!box.current) return;
    setReady(false);
    let map: MLMap;
    try {
      map = new maplibregl.Map({
        container: box.current, center: [62, 21], zoom: 2.9, minZoom: 1, maxZoom: 14, attributionControl: false,
        renderWorldCopies: false, dragRotate: false, pitchWithRotate: false,
        style: { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#03070d" } }] },
      });
    } catch (err) {
      console.warn("GEO map: WebGL unavailable", err);
      setGlError("create");
      return;
    }
    setGlError(null);
    mapRef.current = map;
    map.on("webglcontextlost", () => setGlError("lost"));
    map.on("webglcontextrestored", () => setGlError(null));
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true,
      customAttribution: "Boundaries: Natural Earth · Data: GDELT, NASA FIRMS/GIBS, IMF PortWatch, AISStream, adsb.lol" }), "bottom-right");
    map.on("mousemove", (e) => setCursor(`${e.lngLat.lat.toFixed(2)}°, ${e.lngLat.lng.toFixed(2)}°`));
    map.on("load", async () => {
      for (const [n, s] of [["i-arrow", "arrow"], ["i-plane", "plane"], ["i-square", "square"], ["i-diamond", "diamond"],
        ["i-triangle", "triangle"], ["i-dot", "dot"]] as const) map.addImage(n, icon(s), { sdf: true });
      for (const id of ["countries", "coast", "borders", "air-regions", "choke-area", "choke-pt", "ports", "energy", "fires",
        "conflict", "ships", "aircraft"]) map.addSource(id, { type: "geojson", data: EMPTY });
      map.addLayer({ id: "land", type: "fill", source: "countries", paint: { "fill-color": "#172232" } });
      map.addLayer({ id: "coast", type: "line", source: "coast", paint: { "line-color": "#41587a", "line-width": 0.9 } });
      map.addLayer({ id: "borders", type: "line", source: "borders", paint: { "line-color": "#2c3a50", "line-width": 0.6 } });
      map.addLayer({ id: "air-regions", type: "line", source: "air-regions",
        paint: { "line-color": ["get", "color"], "line-width": 1, "line-dasharray": [2, 2], "line-opacity": 0.7 } });
      map.addLayer({ id: "choke-area", type: "fill", source: "choke-area",
        paint: { "fill-color": ["get", "color"], "fill-opacity": 0.12, "fill-outline-color": ["get", "color"] } });
      map.addLayer({ id: "fires", type: "circle", source: "fires", paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 2, ["interpolate", ["linear"], ["get", "frp"], 0, 1.2, 100, 3, 400, 5],
          8, ["interpolate", ["linear"], ["get", "frp"], 0, 3, 100, 7, 400, 11]],
        "circle-color": ["step", ["get", "frp"], "#ffd166", 10, "#ff9f1a", 50, "#ff5c39", 150, "#ff2d55"],
        "circle-opacity": 0.85 } });
      map.addLayer({ id: "conflict", type: "circle", source: "conflict", paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 2, ["interpolate", ["linear"], ["get", "sev"], 1, 2, 30, 6],
          8, ["interpolate", ["linear"], ["get", "sev"], 1, 5, 30, 14]],
        "circle-color": ["case", ["get", "precise"], ["get", "color"], "rgba(0,0,0,0)"],
        "circle-stroke-color": ["get", "color"], "circle-stroke-width": 1.2,
        "circle-opacity": ["interpolate", ["linear"], ["get", "age"], 0, 0.9, 24, 0.25] } });
      map.addLayer({ id: "ports", type: "symbol", source: "ports", layout: { "icon-image": "i-square", "icon-size": 0.42,
        "icon-allow-overlap": true }, paint: { "icon-color": "#45d3f2", "icon-opacity": 0.85 } });
      map.addLayer({ id: "energy-halo", type: "circle", source: "energy", filter: ["==", ["get", "status"], "ANOMALY"],
        paint: { "circle-radius": 14, "circle-color": "rgba(255,45,85,0.18)", "circle-stroke-color": "#ff2d55",
          "circle-stroke-width": 1.5 } });
      map.addLayer({ id: "energy", type: "symbol", source: "energy", layout: { "icon-image": ["get", "icon"],
        "icon-size": ["interpolate", ["linear"], ["zoom"], 2, 0.35, 8, 0.6], "icon-allow-overlap": true },
        paint: { "icon-color": ["get", "color"] } });
      map.addLayer({ id: "ships", type: "symbol", source: "ships", layout: { "icon-image": "i-arrow",
        "icon-size": ["interpolate", ["linear"], ["zoom"], 2, 0.22, 8, 0.45], "icon-rotate": ["get", "cog"],
        "icon-rotation-alignment": "map", "icon-allow-overlap": true }, paint: { "icon-color": ["get", "color"] } });
      map.addLayer({ id: "aircraft", type: "symbol", source: "aircraft", layout: { "icon-image": "i-plane",
        "icon-size": ["interpolate", ["linear"], ["zoom"], 2, ["case", ["==", ["get", "mil"], 1], 0.42, 0.28], 8, 0.7],
        "icon-rotate": ["get", "track"], "icon-rotation-alignment": "map", "icon-allow-overlap": true },
        paint: { "icon-color": ["case", ["==", ["get", "mil"], 1], "#ffb000", "#6e8bb3"],
          "icon-opacity": ["case", ["==", ["get", "mil"], 1], 1, 0.7] } });
      map.addLayer({ id: "choke-pt", type: "circle", source: "choke-pt", paint: { "circle-radius": 5,
        "circle-color": ["get", "color"], "circle-stroke-color": "#04070c", "circle-stroke-width": 1.5 } });
      for (const layer of ["ships", "conflict", "fires", "energy", "ports", "aircraft", "choke-pt"]) {
        map.on("click", layer, (e) => {
          const f = e.features?.[0];
          if (!f) return;
          new maplibregl.Popup({ closeButton: true, maxWidth: "300px", className: "geo-popup" })
            .setLngLat((f.geometry as Point).coordinates as [number, number]).setDOMContent(popupFor(f, layer)).addTo(map);
        });
        map.on("mouseenter", layer, () => { map.getCanvas().style.cursor = "pointer"; });
        map.on("mouseleave", layer, () => { map.getCanvas().style.cursor = ""; });
      }
      try {
        const topo = (await (await fetch("/geo/countries-50m.json")).json()) as Topology;
        const obj = topo.objects.countries;
        (map.getSource("countries") as GeoJSONSource).setData(feature(topo, obj) as FeatureCollection);
        (map.getSource("coast") as GeoJSONSource).setData(mesh(topo, obj as never, (a, b) => a === b));
        (map.getSource("borders") as GeoJSONSource).setData(mesh(topo, obj as never, (a, b) => a !== b));
      } catch (err) {
        console.warn("world outline failed to load", err); // map still works without outlines
      }
      setReady(true);
    });
    return () => {
      try {
        map.remove();
      } catch {
        /* a map whose context is already gone can fail to tear down: nothing left to free */
      }
      mapRef.current = null;
    };
  }, [attempt]);

  // ---------------------------------------------------------------- visibility
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const vis = (id: string, on: boolean) => map.getLayer(id) && map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
    vis("ships", views.ships);
    vis("choke-area", views.chokepoints);
    vis("choke-pt", views.chokepoints);
    vis("ports", views.chokepoints);
    vis("conflict", views.conflict);
    vis("fires", views.fires);
    vis("energy", views.oil || views.fires);
    vis("energy-halo", views.fires || views.oil);
    vis("aircraft", views.air);
    vis("air-regions", views.air);
    labels.current.forEach((m) => (m.getElement().style.display = views.chokepoints ? "" : "none"));
  }, [ready, views]);

  // ---------------------------------------------------------------- satellite
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    if (map.getLayer("sat")) map.removeLayer("sat");
    if (map.getSource("sat")) map.removeSource("sat");
    const s = SATELLITE.find((x) => x.id === props.satId) ?? SATELLITE[0];
    if (views.satellite) {
      map.addSource("sat", { type: "raster", tiles: [s.tiles(props.satDate)], tileSize: 256, maxzoom: s.maxzoom,
        attribution: s.attribution });
      map.addLayer({ id: "sat", type: "raster", source: "sat", paint: { "raster-opacity": 0.95 } }, "coast");
    }
  }, [ready, views.satellite, props.satId, props.satDate]);

  // ---------------------------------------------------------------- data
  const set = (id: string, data: FeatureCollection) => {
    const src = mapRef.current?.getSource(id) as GeoJSONSource | undefined;
    src?.setData(data);
  };

  useEffect(() => {
    if (!ready || !props.sites) return;
    set("ports", fc(props.sites.ports.map((p) => pt(p.lon, p.lat, { name: p.name, kind: p.kind.replace("_", " "),
      symbols: Object.keys(p.symbols ?? {}).join(", ") }))));
    set("air-regions", fc(props.sites.air_regions.map((r) => ({ ...circle(r.lon, r.lat, r.nm * 1.852),
      properties: { color: "#4a5a72" } }))));
  }, [ready, props.sites]);

  useEffect(() => {
    if (!ready || !props.sites) return;
    const status = new Map((props.fires?.sites ?? []).map((s) => [s.id, s]));
    set("energy", fc(props.sites.energy.map((s) => {
      const st = status.get(s.id);
      const stat = st?.status ?? "QUIET";
      return pt(s.lon, s.lat, { name: s.name, kind: s.kind, owner: s.owner ?? "", icon: KIND_ICON[s.kind] ?? "i-dot",
        status: stat, color: stat === "QUIET" ? "#8d7bd6" : STATUS_COLOR[stat], hotspots: st?.hotspots ?? 0,
        baseline: st?.baseline ?? "?", symbols: Object.entries(s.symbols ?? {}).map(([k, v]) => `${k} ${v > 0 ? "▲" : "▼"}`).join(", ") });
    })));
  }, [ready, props.sites, props.fires]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    const rows = props.chokepoints ?? [];
    set("choke-area", fc(rows.map((c) => ({ ...circle(c.lon, c.lat, c.r_km), properties: { color: STATUS_COLOR[c.status] } }))));
    set("choke-pt", fc(rows.map((c) => pt(c.lon, c.lat, { name: c.name, status: c.status, color: STATUS_COLOR[c.status],
      ships: c.ships_now, tankers: c.tankers_now, ratio: c.pw?.ratio != null ? Math.round(c.pw.ratio * 100) : "",
      reasons: c.reasons.join("; ") }))));
    labels.current.forEach((m) => m.remove());
    labels.current = rows.map((c) => {
      const e = el("div", `geo-label ${c.status}`);
      e.append(el("b", "", c.name.toUpperCase()), el("span", "", c.status));
      e.style.display = views.chokepoints ? "" : "none";
      return new maplibregl.Marker({ element: e, anchor: "left", offset: [9, 0] }).setLngLat([c.lon, c.lat]).addTo(map);
    });
  }, [ready, props.chokepoints]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!ready) return;
    const rows = props.ships?.rows ?? [];
    set("ships", fc(rows.map(([mmsi, lat, lon, sog, cog, c, name, dest]) =>
      pt(lon, lat, { mmsi, sog, cog: cog ?? 0, cls: c, color: SHIP_COLORS[c] ?? "#9aa8ba", name, dest }))));
  }, [ready, props.ships]);

  useEffect(() => {
    if (!ready) return;
    const now = Date.now() / 1000;
    set("conflict", fc((props.conflict?.events ?? []).map((e) => pt(e.lon, e.lat, { label: e.label, place: e.place, a1: e.a1,
      a2: e.a2, ts: e.ts, sev: e.sev, articles: e.articles, precise: e.precise, url: e.url,
      color: ROOT_COLOR[e.root] ?? "#ff5263", age: Math.min(24, (now - e.ts) / 3600) }))));
  }, [ready, props.conflict]);

  useEffect(() => {
    if (!ready) return;
    set("fires", fc((props.fires?.points ?? []).map(([lat, lon, frp, ts]) => pt(lon, lat, { frp, ts }))));
  }, [ready, props.fires]);

  useEffect(() => {
    if (!ready) return;
    set("aircraft", fc((props.air?.rows ?? []).map(([hex, flight, lat, lon, alt, gs, track, type, mil]) =>
      pt(lon, lat, { hex, flight, alt, gs, track: track ?? 0, type, mil }))));
    if (props.air && props.sites) {
      const st = new Map(props.air.regions.map((r) => [r.id, r.status]));
      set("air-regions", fc(props.sites.air_regions.map((r) => ({ ...circle(r.lon, r.lat, r.nm * 1.852),
        properties: { color: STATUS_COLOR[st.get(r.id) ?? "NO DATA"] ?? "#4a5a72" } }))));
    }
  }, [ready, props.air, props.sites]);

  // ---------------------------------------------------------------- focus
  useEffect(() => {
    if (props.focus && mapRef.current) {
      mapRef.current.flyTo({ center: [props.focus.lon, props.focus.lat], zoom: props.focus.zoom,
        essential: false, duration: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 900 });
    }
  }, [props.focus]);

  const go = (p: (typeof PRESETS)[number]) => mapRef.current?.flyTo({ center: p.center, zoom: p.zoom, duration: 700 });

  return (
    <div className="geo-map-wrap">
      <div ref={box} className="geo-map" role="region" aria-label="World events map" />
      {glError && (
        <div className="geo-gl-error" role="alert">
          <b>{glError === "lost" ? "The map lost its graphics context" : "The map could not start: WebGL is unavailable"}</b>
          <p>
            The map draws with your graphics card. {glError === "lost"
              ? "The browser's graphics process reset (often a driver hiccup or memory pressure)."
              : "The browser has WebGL switched off right now — usually after its graphics process crashed, or with hardware acceleration disabled."}{" "}
            Everything else on the GEO desk keeps working.
          </p>
          <ul>
            <li>Click <b>Reload map</b> below.</li>
            <li>Still failing: close <i>all</i> browser windows and reopen (Chrome re-enables WebGL on a fresh start).</li>
            <li>Chrome / Edge: Settings → System → turn on <i>Use graphics acceleration when available</i>; check <code>chrome://gpu</code>.</li>
            <li>Keep the graphics driver up to date.</li>
          </ul>
          <button type="button" className="btn" onClick={() => { setGlError(null); setAttempt((a) => a + 1); }}>Reload map</button>
        </div>
      )}
      <div className="geo-presets" role="group" aria-label="Map regions">
        {PRESETS.map((p) => <button type="button" key={p.id} onClick={() => go(p)}>{p.label}</button>)}
      </div>
      <div className="geo-legend" aria-label="Map legend">
        {views.ships && props.ships && (
          <span>{Object.entries(props.ships.counts).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k, n]) => (
            <span key={k} className="lg"><i style={{ background: SHIP_COLORS[props.ships!.cls[k]] }} />{k.toLowerCase()} {n}</span>))}</span>
        )}
        {views.conflict && <span className="lg"><i style={{ background: "#ff5263", borderRadius: "50%" }} />conflict {props.conflict?.n ?? 0}</span>}
        {views.fires && <span className="lg"><i style={{ background: "#ff9f1a", borderRadius: "50%" }} />hotspots {props.fires?.n ?? 0}</span>}
        {views.air && <span className="lg"><i style={{ background: "#ffb000" }} />aircraft {props.air?.n ?? 0}</span>}
        <span className="muted">{cursor}</span>
      </div>
    </div>
  );
}
