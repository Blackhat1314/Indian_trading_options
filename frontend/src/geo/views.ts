/** The GEO desk's switchable views. Each switch controls its map layer and its panel together,
 *  and decides which geo:* topics this browser subscribes to (switched off = no data sent). */
export type ViewId =
  | "ships" | "chokepoints" | "conflict" | "risk" | "fires" | "satellite" | "air" | "oil" | "macro" | "impact";

export interface ViewDef { id: ViewId; label: string; hint: string; topics: string[]; status?: string; on: boolean }

export const VIEWS: ViewDef[] = [
  { id: "ships", label: "Ship movement (live)", hint: "AIS positions in the Gulf, Red Sea, Arabian Sea, Bay of Bengal and Malacca",
    topics: ["geo:ships"], status: "ships", on: true },
  { id: "chokepoints", label: "Chokepoints & ports", hint: "Transit counts vs normal (IMF PortWatch) + live vessels + nearby conflict",
    topics: ["geo:chokepoints"], status: "chokepoints", on: true },
  { id: "conflict", label: "War & conflict events", hint: "GDELT event feed: strikes, fighting, bombings, mobilisation (15-min updates)",
    topics: ["geo:conflict"], status: "conflict", on: true },
  { id: "risk", label: "Geopolitical risk index", hint: "Live India-weighted geo-risk 0–100 + the Caldara–Iacoviello GPR index",
    topics: ["geo:risk"], status: "risk", on: true },
  { id: "fires", label: "Fires / explosions / refinery hits", hint: "NASA FIRMS satellite hotspots matched to refineries, terminals and fields",
    topics: ["geo:fires"], status: "fires", on: true },
  { id: "satellite", label: "Satellite imagery", hint: "NASA GIBS daily imagery, night lights and Sentinel-2 — loaded straight from NASA/EOX",
    topics: [], status: "satellite", on: false },
  { id: "air", label: "Military / air traffic", hint: "Military aircraft (ADS-B) + airspace watch: traffic vs its usual level",
    topics: ["geo:air"], status: "air", on: true },
  { id: "oil", label: "Oil rigs & supply", hint: "Refineries, terminals, fields and LNG on the map; crude, gas and tanker flows",
    topics: ["geo:fires", "geo:macro", "geo:chokepoints"], status: "macro", on: true },
  { id: "macro", label: "India macro", hint: "USD/INR, MCX crude & gold, India VIX, US yields, dollar, RBI / CPI / FOMC calendar",
    topics: ["geo:macro"], status: "macro", on: true },
  { id: "impact", label: "Market impact", hint: "Which NSE names each signal tends to move, with the reason",
    topics: ["geo:impact"], on: true },
];

export const BASE_GEO_TOPICS = ["geo:sites", "geo:status"];
export const GLOBAL_TOPICS = ["geo:risk"];       // also used by the top-bar chip — never unsubscribed here

export type Views = Record<ViewId, boolean>;

export const SATELLITE: { id: string; label: string; tiles: (date: string) => string; maxzoom: number; dated: boolean; attribution: string }[] = [
  { id: "viirs", label: "VIIRS true colour · daily", dated: true, maxzoom: 9,
    tiles: (d) => `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/${d}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`,
    attribution: "Imagery: NASA GIBS / Worldview" },
  { id: "modis", label: "MODIS Terra true colour · daily", dated: true, maxzoom: 9,
    tiles: (d) => `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/${d}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`,
    attribution: "Imagery: NASA GIBS / Worldview" },
  { id: "night", label: "Night lights (VIIRS day/night band) · daily", dated: true, maxzoom: 8,
    tiles: (d) => `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_DayNightBand_ENCC/default/${d}/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png`,
    attribution: "Imagery: NASA GIBS (VIIRS DNB)" },
  { id: "s2", label: "Sentinel-2 cloudless · 10 m mosaic", dated: false, maxzoom: 15,
    tiles: () => "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2021_3857/default/g/{z}/{y}/{x}.jpg",
    attribution: "Sentinel-2 cloudless by EOX IT Services (CC BY-NC-SA 4.0), contains modified Copernicus data" },
];

export const PRESETS: { id: string; label: string; center: [number, number]; zoom: number }[] = [
  { id: "india", label: "INDIA", center: [76, 20], zoom: 3.6 },
  { id: "gulf", label: "GULF", center: [53, 26.5], zoom: 4.8 },
  { id: "redsea", label: "RED SEA", center: [40, 20], zoom: 4.3 },
  { id: "malacca", label: "MALACCA", center: [99, 4], zoom: 4.8 },
  { id: "blacksea", label: "BLACK SEA", center: [35, 46], zoom: 4.3 },
  { id: "world", label: "WORLD", center: [55, 22], zoom: 1.6 },
];

export const SHIP_COLORS: Record<number, string> = {
  1: "#ff9f1a", 2: "#45d3f2", 3: "#a78bfa", 4: "#ff5263", 5: "#7a889d", 6: "#6b7a90", 7: "#9aa8ba", 0: "#9aa8ba",
};
export const SHIP_LABEL: Record<number, string> = {
  1: "Tanker", 2: "Cargo", 3: "Passenger", 4: "Military", 5: "Fishing", 6: "Service", 7: "Other", 0: "Unknown",
};
export const STATUS_COLOR: Record<string, string> = {
  NORMAL: "#27d17f", WATCH: "#f5a524", STRAINED: "#ff8a3d", DISRUPTED: "#ff5263",
  ANOMALY: "#ff2d55", ACTIVE: "#ff8a3d", LEARNING: "#f5a524", FLARING: "#c9a227", QUIET: "#4a5a72",
  "CLOSED?": "#ff5263", REDUCED: "#f5a524", "NO DATA": "#4a5a72",
};
export const ROOT_COLOR: Record<string, string> = { "15": "#f5a524", "18": "#ff8a3d", "19": "#ff5263", "20": "#e040fb" };
