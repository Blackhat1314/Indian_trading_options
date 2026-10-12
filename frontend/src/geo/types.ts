export interface GeoChokepointDef { id: string; name: string; lat: number; lon: number; r_km: number; india: number; why: string }
export interface GeoSite {
  id: string; name: string; lat: number; lon: number; kind: string; country: string; owner?: string;
  symbols: Record<string, number>; weight: number; flare: boolean; r_km: number;
}
export interface GeoAirRegionDef { id: string; name: string; lat: number; lon: number; nm: number; india: number }
export interface GeoSites { ts: number; chokepoints: GeoChokepointDef[]; ports: GeoSite[]; energy: GeoSite[]; air_regions: GeoAirRegionDef[] }

/** [mmsi, lat, lon, sog, cog, classCode, name, destination] */
export type ShipRow = [number, number, number, number | null, number | null, number, string, string];
export interface GeoShips { ts: number; n: number; rows: ShipRow[]; counts: Record<string, number>; cls: Record<string, number> }

export interface GeoChokepoint {
  id: string; name: string; lat: number; lon: number; r_km: number; india: number; why: string;
  status: "NORMAL" | "WATCH" | "STRAINED" | "DISRUPTED"; reasons: string[];
  ships_now: number; tankers_now: number; avg_speed: number | null; entered_24h: number;
  conflict: number; conflict_n: number;
  pw: null | { as_of: string; last7_total: number; last7_tanker: number; base_total: number; base_tanker: number;
    ratio: number | null; tanker_ratio: number | null };
  series: [string, number, number][];
}

export interface GeoEvent {
  id: string; ts: number; lat: number; lon: number; code: string; root: string; label: string; a1: string; a2: string;
  place: string; country: string; precise: boolean; articles: number; sources: number; tone: number; url: string; sev: number;
}
export interface GeoRegion { id: string; name: string; india: number; n: number; sev: number; weighted: number;
  top: { label: string; place: string; sev: number; url: string } | null }
export interface GeoConflict { ts: number; n: number; events: GeoEvent[]; regions: GeoRegion[] }

/** [lat, lon, frp MW, ts, confidence] */
export type FireRow = [number, number, number, number, string];
export interface GeoFireSite {
  id: string; name: string; kind: string; lat: number; lon: number; country: string; owner: string;
  symbols: Record<string, number>; weight: number; hotspots: number; frp: number; baseline: number | null;
  status: "ANOMALY" | "ACTIVE" | "LEARNING" | "FLARING" | "QUIET"; last: number | null;
}
export interface GeoFires { ts: number; n: number; points: FireRow[]; sites: GeoFireSite[] }

/** [hex, flight, lat, lon, alt, gs, track, type, mil] */
export type AirRow = [string, string, number, number, number | null, number | null, number | null, string, number];
export interface GeoAirRegion { id: string; name: string; lat: number; lon: number; nm: number; india: number;
  count: number | null; baseline: number | null; ratio: number | null; status: string }
export interface GeoAir { ts: number; n: number; rows: AirRow[]; regions: GeoAirRegion[] }

export interface GeoRisk {
  ts: number; value: number; level: string; components: Record<string, number>; weights: Record<string, number>;
  drivers: string[]; conflict_raw: number; history: [number, number][]; regions: GeoRegion[]; mode: string;
  gpr: { latest: [string, number] | null; series: [string, number][]; ma30: number | null;
    monthly: Record<string, [string, number][]> };
}

export interface GeoMacro {
  ts: number;
  live: Record<string, { ltp: number; chg: number; src: string; expiry?: string }>;
  daily: Record<string, { label: string; last: [string, number]; chg1d: number | null; chg5d: number | null; spark: number[] }>;
  india_vix: number | null;
  calendar: { date: string; time: string; title: string; kind: string; approx?: boolean }[];
}

export interface GeoSignal { id: string; title: string; sev: number; symbols: { s: string; dir: number }[]; why: string; ts: number }
export interface GeoImpact { ts: number; signals: GeoSignal[] }

export interface GeoLayerStatus { state: string; source: string; detail: string; updated: number | null; n: number | null }
export interface GeoStatus { ts: number; mode: string; layers: Record<string, GeoLayerStatus>; demand: Record<string, number> }
