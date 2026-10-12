const IST_OFFSET = 19800; // seconds

/** Values that round to zero at `d` decimals print as 0, never "-0.00". */
const z = (v: number, d: number) => (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v);

export function fx(v: number | null | undefined, d = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return z(v, d).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });
}

export function px(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return fx(v, v >= 10000 ? 1 : 2);
}

/** Signed percentage with an arrow so direction never relies on colour alone. */
export function pct(v: number | null | undefined, d = 2, arrow = true): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  v = z(v, d);
  const s = `${v > 0 ? "+" : ""}${v.toFixed(d)}%`;
  if (!arrow) return s;
  return `${v > 0.004 ? "▲" : v < -0.004 ? "▼" : "•"} ${s}`;
}

export function signed(v: number | null | undefined, d = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  v = z(v, d);
  return `${v > 0 ? "+" : ""}${v.toFixed(d)}`;
}

export function cls(v: number | null | undefined, eps = 0.004): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "flat";
  return v > eps ? "up" : v < -eps ? "down" : "flat";
}

/** Indian compact notation: 1.2 Cr, 3.4 L, 12.5 K */
export function compact(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e7) return `${(v / 1e7).toFixed(a >= 1e9 ? 0 : 2)} Cr`;
  if (a >= 1e5) return `${(v / 1e5).toFixed(2)} L`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(1)} K`;
  return v.toFixed(0);
}

export function crore(v: number | null | undefined, d = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `₹${fx(v, d)} Cr`;
}

export function timeIST(ts: number, withSec = false): string {
  const d = new Date((ts + IST_OFFSET) * 1000);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  const ss = String(d.getUTCSeconds()).padStart(2, "0");
  return withSec ? `${hh}:${mm}:${ss}` : `${hh}:${mm}`;
}

export function dateIST(ts: number): string {
  const d = new Date((ts + IST_OFFSET) * 1000);
  return d.toISOString().slice(0, 10);
}

export function ago(ts: number, now: number): string {
  const s = Math.max(0, now - ts);
  if (s < 60) return `${Math.floor(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
}

/** Lightweight-charts renders UTC; shift epoch seconds so axes read in IST. */
export const toChartTime = (ts: number) => (ts + IST_OFFSET) as unknown as import("lightweight-charts").UTCTimestamp;

export const BUILDUP: Record<string, { label: string; long: string; cls: string }> = {
  LB: { label: "LB", long: "Long build-up", cls: "up" },
  SB: { label: "SB", long: "Short build-up", cls: "down" },
  SC: { label: "SC", long: "Short covering", cls: "cyan" },
  LU: { label: "LU", long: "Long unwinding", cls: "amber" },
};

/** Colour for a % change on a dark tile (heatmap). */
export function heatColor(chg: number): string {
  const c = Math.max(-3, Math.min(3, chg)) / 3;
  if (c >= 0) {
    const a = 0.18 + 0.72 * c;
    return `rgba(22, 163, 92, ${a.toFixed(3)})`;
  }
  const a = 0.18 + 0.72 * -c;
  return `rgba(220, 48, 66, ${a.toFixed(3)})`;
}

const EVENT_SHORT: Record<string, string> = {
  ORDER_WIN: "ORDER", RESULTS: "RESULT", BLOCK_DEAL: "BLOCK", REGULATORY: "REG", MANAGEMENT: "MGMT", GEOPOLITICAL: "GEO",
};
/** Compact news-event label for dense cells. */
export const eventLabel = (t: string | null | undefined) => (t ? EVENT_SHORT[t] ?? t.replace(/_/g, " ") : "NEWS");
