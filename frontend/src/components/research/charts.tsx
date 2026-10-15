/* Research-desk charts — hand-written SVG so the four research desks share one look without a charting dependency.
 * Every chart fills its container's width (ResizeObserver), has a hover/keyboard readout (crosshair on time charts,
 * per-mark tooltip on bars), clean ticks (₹ with Indian grouping), a legend for ≥ 2 series, an aria-label summary and
 * a TABLE toggle that shows the same numbers as a table (the accessible twin).
 * Colours: --viz-1 cyan, --viz-2 amber, --viz-3 violet (research.css) in that fixed order; gains/losses use --up/--down.
 *
 * Usage (props are documented on each *Props interface below):
 *   import { FanChart, Histogram, MC_BANDS, inr } from "../components/research/charts";
 *   <FanChart label={`${r.symbol}: simulated price range`} data={r.fan} median="p50" mean="mean" bands={MC_BANDS}
 *     history={r.history} paths={r.sample_paths} spot={r.spot} targets={r.targets.map((t) => t.price)}
 *     checkpoints={r.horizons.map((h) => ({ ts: h.ts, label: h.label }))} height="fill" />
 *   <Histogram label="Price at horizon" edges={r.terminal.edges} counts={r.terminal.counts} splitAt={r.spot}
 *     markers={[{ value: r.spot, label: "Spot", kind: "spot" }, { value: r.terminal.p5, label: "p5" }]} />
 *   <BarChart label="Quarterly sales & profit (₹ Cr)" categories={q.columns} format="cr"
 *     series={[{ name: "Sales", values: sales }, { name: "Net profit", values: profit }]} />
 *   <LineChart label="Price & Kalman trend" x={c.ts} series={[{ name: "Close", values: c.close, tone: "text" },
 *     { name: "Trend", values: c.trend }]} bands={[{ name: "Trend ± 2σ", lo: c.lower, hi: c.upper }]}
 *     shade={{ name: "Bull-regime probability", values: c.bull_prob }} />
 *   <ProbBar p={h.p_up} />  <CoverageChart data={bt.coverage} n={bt.effective_n} />  <PitHistogram bins={bt.pit_hist} />
 * Notes: height="fill" needs a parent with a definite height (a Panel body in a desk grid); otherwise pass a number.
 * Array props built inline (series, bands, checkpoints…) are fine, but wrap them in useMemo in desks that re-render on
 * every live tick, so the chart's static layer is not redrawn each time. Pass busy while refetching.
 */
import { memo, useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import type { CSSProperties, FocusEvent as ReactFocusEvent, KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent, ReactNode } from "react";
import { dateIST, fx } from "../../lib/format";
import type { McBand, QuantResult } from "../../lib/research";
import "./research.css";

// =================================================================== formatting
/** Value format shared by axes, tooltips and tables:
 *  "inr" rupees (₹2,812.40) · "cr" ₹ crore figures (ticks plain, tooltips "₹1,234 Cr") · "pct" a number already in % ·
 *  "frac" a fraction shown as % (0.12 → 12%) · "num" plain Indian-grouped number · or your own function. */
export type Fmt = "inr" | "cr" | "pct" | "frac" | "num" | ((v: number) => string);

/** Series colour: 1/2/3 = --viz-1/2/3 (default by series order), or a text/status token. */
export type Tone = 1 | 2 | 3 | "text" | "muted" | "up" | "down";

const ok = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const CH = 6.4;                       // px per character of the 10.5 px Fira Code axis text (monospace)
const textW = (s: string) => s.length * CH;

/** ₹ with Indian grouping: inr(281240.5) → "₹2,81,240.50"; default 2 decimals below ₹10,000, else 0. */
export function inr(v: number | null | undefined, d?: number): string {
  if (!ok(v)) return "—";
  const a = Math.abs(v);
  return `${v < 0 ? "-" : ""}₹${fx(a, d ?? (a >= 10000 ? 0 : 2))}`;
}

/** Fraction → percent text: fracPct(0.1234) → "12.3%"; signed adds "+" for gains. */
export function fracPct(v: number | null | undefined, d = 1, signed = false): string {
  if (!ok(v)) return "—";
  const s = fx(v * 100, d);
  return `${signed && v * 100 >= 0.5 * 10 ** -d ? "+" : ""}${s}%`;
}

/** Full-precision text for tooltips, labels and tables. */
export function fmtValue(fmt: Fmt, v: number | null | undefined): string {
  if (!ok(v)) return "—";
  if (typeof fmt === "function") return fmt(v);
  const a = Math.abs(v);
  switch (fmt) {
    case "inr": return inr(v);
    case "cr": return `${v < 0 ? "-" : ""}₹${fx(a, a >= 100 ? 0 : 1)} Cr`;
    case "pct": return `${fx(v, a >= 100 ? 0 : 1)}%`;
    case "frac": return fracPct(v, 1);
    default: return fx(v, a >= 100 || Number.isInteger(v) ? 0 : 2);
  }
}

/** Short text for direct labels on the plot (end values, target lines): ₹ without paise above ₹1,000. */
export function fmtLabel(fmt: Fmt, v: number | null | undefined): string {
  if (fmt !== "inr" || !ok(v)) return fmtValue(fmt, v);
  const a = Math.abs(v);
  return inr(v, a >= 1000 ? 0 : a >= 100 ? 1 : 2);
}

/** Fewest decimals that print every multiple of `step` exactly. */
function decimals(step: number): number {
  for (let d = 0; d < 6; d++) {
    const s = step * 10 ** d;
    if (Math.abs(s - Math.round(s)) < 1e-6 * Math.max(1, s)) return d;
  }
  return 6;
}

/** Axis tick text: as short as the tick spacing allows. */
export function fmtTick(fmt: Fmt, v: number, step: number): string {
  if (typeof fmt === "function") return fmt(v);
  const d = decimals(step);
  switch (fmt) {
    case "inr": return `${v < 0 ? "-" : ""}₹${fx(Math.abs(v), d)}`;
    case "pct": return `${fx(v, d)}%`;
    case "frac": return `${fx(v * 100, decimals(step * 100))}%`;
    default: return fx(v, d);     // "cr" and "num": the unit lives in the chart title
  }
}

export function toneColor(t: Tone | undefined, index = 0): string {
  const tone = t ?? ((index < 3 ? index + 1 : "muted") as Tone);
  switch (tone) {
    case 1: return "var(--viz-1)";
    case 2: return "var(--viz-2)";
    case 3: return "var(--viz-3)";
    case "text": return "var(--text-2)";
    case "up": return "var(--up)";
    case "down": return "var(--down)";
    default: return "var(--muted)";
  }
}

// =================================================================== scales & ticks
interface YScale { (v: number): number; ticks: number[]; step: number; lo: number; hi: number }

function extent(vals: Iterable<number>): [number, number] | null {
  let lo = Infinity, hi = -Infinity;
  for (const v of vals) if (ok(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return lo <= hi ? [lo, hi] : null;
}

function niceStep(span: number, count: number): number {
  const raw = span / Math.max(1, count);
  const p = 10 ** Math.floor(Math.log10(raw));
  const e = raw / p;
  return (e >= 7.07 ? 10 : e >= 3.16 ? 5 : e >= 1.41 ? 2 : 1) * p;
}
const clean = (v: number) => +v.toPrecision(12);

function linearTicks(a: number, b: number, count: number): { ticks: number[]; step: number } {
  const step = niceStep(b - a, count);
  const ticks: number[] = [];
  for (let k = Math.ceil(a / step - 1e-9); k * step <= b + step * 1e-9; k++) ticks.push(clean(k * step));
  return { ticks, step };
}

/** 1-2-5 style ticks on a log axis; falls back to linear ticks when the range is under a decade step. */
function logTicks(a: number, b: number, count: number): { ticks: number[]; step: number } {
  const sets = [[1, 1.5, 2, 3, 4, 5, 6, 8], [1, 2, 3, 5, 7], [1, 2, 5], [1, 3], [1]];
  for (const ms of sets) {
    const out: number[] = [];
    for (let k = Math.floor(Math.log10(a)); k <= Math.ceil(Math.log10(b)); k++)
      for (const m of ms) {
        const v = clean(m * 10 ** k);
        if (v >= a && v <= b) out.push(v);
      }
    if (out.length <= count) {
      if (out.length >= 2) {
        let step = Infinity;
        for (let i = 1; i < out.length; i++) step = Math.min(step, out[i] - out[i - 1]);
        return { ticks: out, step: niceStep(step, 1) };
      }
      break;
    }
  }
  return linearTicks(a, b, count);
}

/** y scale mapping data → pixel (bottom → top), domain padded and extended to clean tick values. */
function makeY(ext: [number, number] | null, bottom: number, top: number, o: { log?: boolean; zero?: boolean; count: number }): YScale {
  let [lo, hi] = ext ?? [0, 1];
  if (o.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  let a: number, b: number, t: { ticks: number[]; step: number }, f: (v: number) => number;
  if (o.log && lo > 0) {
    a = lo / 1.04; b = hi * 1.04;
    t = logTicks(a, b, o.count);
    const la = Math.log(a), lb = Math.log(b);
    f = (v) => bottom + ((Math.log(Math.max(v, 1e-12)) - la) / (lb - la)) * (top - bottom);
  } else {
    if (hi - lo < 1e-12) { const d = Math.abs(hi) * 0.02 || 1; lo -= d; hi += d; }
    const pad = (hi - lo) * 0.04;
    a = o.zero && lo === 0 ? 0 : lo - pad;
    b = o.zero && hi === 0 ? 0 : hi + pad;
    const step = niceStep(b - a, o.count);
    const na = Math.floor(a / step + 1e-9) * step, nb = Math.ceil(b / step - 1e-9) * step;
    if (a - na <= step / 2) a = na;      // snap to a clean bound unless that wastes over half a tick step
    if (nb - b <= step / 2) b = nb;
    t = linearTicks(a, b, o.count);
    const a0 = a, span = b - a || 1;
    f = (v) => bottom + ((v - a0) / span) * (top - bottom);
  }
  return Object.assign(f, { ticks: t.ticks, step: t.step, lo: a, hi: b });
}

const IST = 19800;
const DAY = 86400;
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const istDate = (ts: number) => new Date((ts + IST) * 1000);
const istMidnight = (y: number, m: number, d: number) => Date.UTC(y, m, d) / 1000 - IST;

/** Calendar-aligned ticks (weeks, months, quarters, years) in IST for an epoch-second range. */
function timeTicks(t0: number, t1: number, maxTicks: number): { ts: number; label: string }[] {
  const days = (t1 - t0) / DAY;
  const units: ["d" | "m" | "y", number, number][] = [
    ["d", 1, 1], ["d", 7, 7], ["d", 14, 14], ["m", 1, 30.44], ["m", 2, 60.9], ["m", 3, 91.3], ["m", 6, 182.6], ["y", 1, 365.25], ["y", 2, 730.5], ["y", 5, 1826],
  ];
  const [kind, n] = units.find(([, , len]) => days / len <= maxTicks) ?? ["y", 10, 3652];
  const out: { ts: number; label: string }[] = [];
  const s = istDate(t0);
  if (kind === "d") {
    let d = new Date(Date.UTC(s.getUTCFullYear(), s.getUTCMonth(), s.getUTCDate()));
    if (n >= 7) while (d.getUTCDay() !== 1) d = new Date(d.getTime() + DAY * 1000);   // weeks start on Monday
    for (let ts = d.getTime() / 1000 - IST; ts <= t1 && out.length < 400; ts += n * DAY) {
      if (ts < t0) continue;
      const x = istDate(ts);
      out.push({ ts, label: `${x.getUTCDate()} ${MON[x.getUTCMonth()]}` });
    }
  } else if (kind === "m") {
    let y = s.getUTCFullYear(), m = s.getUTCMonth();
    for (let i = 0; i < 400; i++) {
      const ts = istMidnight(y, m, 1);
      if (ts > t1) break;
      if (ts >= t0 && m % n === 0) out.push({ ts, label: m === 0 || out.length === 0 ? `${MON[m]} '${String(y).slice(2)}` : MON[m] });
      m += 1;
      if (m === 12) { m = 0; y += 1; }
    }
  } else {
    for (let y = s.getUTCFullYear(); y <= istDate(t1).getUTCFullYear() + 1; y++) {
      const ts = istMidnight(y, 0, 1);
      if (ts >= t0 && ts <= t1 && y % n === 0) out.push({ ts, label: String(y) });
    }
  }
  return out;
}

/** Nearest index in a sorted array. */
function nearest(xs: number[], v: number): number {
  let lo = 0, hi = xs.length - 1;
  if (hi <= 0) return 0;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= v) lo = mid; else hi = mid;
  }
  return v - xs[lo] <= xs[hi] - v ? lo : hi;
}

function barPath(x: number, w: number, base: number, tip: number, r: number): string {
  const h = Math.abs(tip - base);
  r = Math.max(0, Math.min(r, w / 2, h));
  const s = tip <= base ? 1 : -1;     // +1: bar grows upward
  return `M${x},${base}V${tip + s * r}Q${x},${tip} ${x + r},${tip}H${x + w - r}Q${x + w},${tip} ${x + w},${tip + s * r}V${base}Z`;
}

type Anchor = "start" | "middle" | "end";
const f1 = (v: number) => v.toFixed(1);
const crisp = (v: number) => Math.round(v) + 0.5;

// =================================================================== frame, legend, tooltip, table
export interface LegendItem { label: string; color: string; kind: "line" | "dash" | "area" | "dot"; opacity?: number }

function keyStyle(kind: LegendItem["kind"], color: string, opacity?: number): CSSProperties {
  if (kind === "dash") return { borderTopColor: color, opacity };
  return { background: color, opacity };
}

function Legend({ items }: { items: LegendItem[] }) {
  return (
    <ul className="rc-legend" aria-label="Legend">
      {items.map((it) => (
        <li key={it.label}><i className={`rc-key ${it.kind}`} style={keyStyle(it.kind, it.color, it.opacity)} aria-hidden="true" />{it.label}</li>
      ))}
    </ul>
  );
}

interface TipRow { value: ReactNode; label: ReactNode; color?: string; dash?: boolean }
interface TipContent { title: ReactNode; rows: TipRow[] }

function Tip({ x, y = 4, w, content }: { x: number; y?: number; w: number; content: TipContent }) {
  const flip = x > w * 0.56;
  return (
    <div className="rc-tip" aria-hidden="true" style={{ top: y, ...(flip ? { right: w - x + 12 } : { left: x + 12 }) }}>
      <div className="rc-tip-title">{content.title}</div>
      {content.rows.map((r, i) => (
        <div className="rc-tip-row" key={i}>
          {r.color ? <i className={`rc-key ${r.dash ? "dash" : "line"}`} style={keyStyle(r.dash ? "dash" : "line", r.color)} /> : <i className="rc-key none" />}
          <b>{r.value}</b><span>{r.label}</span>
        </div>
      ))}
    </div>
  );
}

interface TableData { columns: string[]; rows: ReactNode[][] }

/** Container size tracked with ResizeObserver (callback ref, so it survives the table/chart toggle). */
function useSize(): [(el: HTMLDivElement | null) => void, { w: number; h: number }] {
  const [size, setSize] = useState({ w: 0, h: 0 });
  const ro = useRef<ResizeObserver | null>(null);
  const ref = useCallback((el: HTMLDivElement | null) => {
    ro.current?.disconnect();
    ro.current = null;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      const w = Math.floor(r.width), h = Math.floor(r.height);
      setSize((s) => (s.w === w && s.h === h ? s : { w, h }));
    };
    measure();
    if (typeof ResizeObserver !== "undefined") {
      ro.current = new ResizeObserver(measure);
      ro.current.observe(el);
    }
  }, []);
  useEffect(() => () => ro.current?.disconnect(), []);
  return [ref, size];
}

/** Active index for crosshair / bar readouts: pointer sets it; ←/→ (Shift = bigger jumps), Home/End, Esc on focus. */
function useActive(n: number) {
  const [idx, setIdx] = useState<number | null>(null);
  const active = idx !== null && idx < n ? idx : null;
  const onKeyDown = (e: ReactKeyboardEvent) => {
    if (!n) return;
    const jump = e.shiftKey ? Math.max(1, Math.round(n / 20)) : 1;
    const cur = active;
    let next: number | null | undefined;
    if (e.key === "ArrowRight") next = cur === null ? n - 1 : Math.min(n - 1, cur + jump);
    else if (e.key === "ArrowLeft") next = cur === null ? n - 1 : Math.max(0, cur - jump);
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = n - 1;
    else if (e.key === "Escape") next = null;
    if (next !== undefined) {
      e.preventDefault();
      setIdx(next);
    }
  };
  const onFocus = (e: ReactFocusEvent<HTMLElement>) => {
    let keyboard = true;
    try { keyboard = e.currentTarget.matches(":focus-visible"); } catch { /* older engines: treat as keyboard */ }
    if (keyboard && n) setIdx((c) => c ?? n - 1);     // keyboard focus shows the latest value, like hovering it
  };
  return { active, setActive: setIdx, onKeyDown, onFocus, onBlur: () => setIdx(null) };
}

/** Common props of every SVG chart. */
export interface FrameProps {
  label: string;                 // accessible title; the chart adds a one-line data summary to it
  height?: number | "fill";      // plot height in px incl. axes, or "fill" = the parent's remaining height (min 200 px)
  width?: number;                // fixed width in px; default: the container's width, tracked live
  caption?: ReactNode;           // small note under the chart (method, units, how to read it)
  tools?: ReactNode;             // extra controls in the chart header, right side (e.g. a LOG toggle)
  busy?: boolean;                // new data loading: keep the previous render, dimmed (no skeleton, no layout jump)
  className?: string;
}

function ChartFrame(props: FrameProps & {
  legend?: LegendItem[]; table: () => TableData; defaultHeight: number;
  children: (w: number, h: number) => ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);
  const [ref, size] = useSize();
  const height = props.height ?? props.defaultHeight;
  const fill = height === "fill";
  const w = props.width ?? size.w;
  const h = fill ? size.h : height;
  const boxStyle: CSSProperties | undefined = fill ? undefined : { height, ...(props.width ? { width: props.width } : {}) };
  return (
    <figure className={`rc ${fill ? "fill" : ""} ${props.busy ? "busy" : ""} ${props.className ?? ""}`} aria-busy={props.busy || undefined}>
      <div className="rc-head">
        {props.legend && props.legend.length >= 2 && <Legend items={props.legend} />}
        <div className="rc-tools">
          {props.tools}
          <button type="button" className="rc-tbtn" aria-pressed={showTable} onClick={() => setShowTable((v) => !v)}
            title={showTable ? "Back to the chart" : "Show the numbers as a table"}>{showTable ? "CHART" : "TABLE"}</button>
        </div>
      </div>
      {showTable ? (
        <DataTable data={props.table()} label={props.label} style={fill ? undefined : { height }} />
      ) : (
        <div className="rc-plot" ref={ref} style={boxStyle}>
          {w > 0 && h > 0 && props.children(w, h)}
        </div>
      )}
      {props.caption && <figcaption className="rc-caption">{props.caption}</figcaption>}
    </figure>
  );
}

function DataTable({ data, label, style }: { data: TableData; label: string; style?: CSSProperties }) {
  return (
    <div className="rc-table-wrap" style={style}>
      <table className="tbl rc-table">
        <caption className="sr-only">{label}</caption>
        <thead><tr>{data.columns.map((c, i) => <th key={i} scope="col" className={i === 0 ? "l" : undefined}>{c}</th>)}</tr></thead>
        <tbody>
          {data.rows.map((r, i) => (
            <tr key={i}>{r.map((c, j) => <td key={j} className={j === 0 ? "l" : undefined}>{c}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function YAxis({ y, x1, x2, fmt }: { y: YScale; x1: number; x2: number; fmt: Fmt }) {
  return (
    <g>
      {y.ticks.map((t) => (
        <g key={t}>
          <line className="rc-grid" x1={x1} x2={x2} y1={crisp(y(t))} y2={crisp(y(t))} />
          <text x={x1 - 6} y={y(t) + 3.5} textAnchor="end">{fmtTick(fmt, t, y.step)}</text>
        </g>
      ))}
    </g>
  );
}

const yLabelWidth = (y: YScale, fmt: Fmt) => Math.max(24, ...y.ticks.map((t) => textW(fmtTick(fmt, t, y.step)))) + 10;

/** Category labels thinned to fit, always keeping the last one. */
function catLabelStep(cats: string[], slot: number): number {
  const lw = Math.max(0, ...cats.map((c) => textW(c))) + 8;
  return Math.max(1, Math.ceil(lw / Math.max(1, slot)));
}

const plotSvgProps = (label: string) => ({ role: "img" as const, "aria-label": label });

// =================================================================== XY engine (FanChart + LineChart)
interface XYLine { name: string; pts: [number, number | null][]; color: string; width?: number; opacity?: number; dash?: string; hoverDot?: boolean }
interface XYArea { pts: [number, number | null, number | null][]; color: string; opacity: number }
interface XYHLine { y: number; color: string; dash?: string; opacity?: number; width?: number; label?: string; from?: number }
interface XYVLine { x: number; label: string }
interface XYDot { x: number; y: number; color: string; label?: string }
interface XYEndLabel { y: number; text: string; strong?: boolean }

interface XYSpec {
  xs: number[];                  // sorted snap positions in x-domain units (ts, or category index)
  cats?: string[];               // categorical x when set (xs = 0..n-1)
  yExt: [number, number] | null;
  log?: boolean; zero?: boolean; fmt: Fmt;
  shade?: { pts: [number, number | null][]; color: string; max: number };
  areas: XYArea[]; lines: XYLine[]; hlines: XYHLine[]; vlines: XYVLine[]; dots: XYDot[];
  endLabels: XYEndLabel[];       // right-margin value labels, highest priority first (colliding ones are dropped)
  tip: (i: number) => TipContent;
}

function linePath(pts: [number, number | null][], X: (v: number) => number, Y: (v: number) => number): string {
  let d = "", pen = false;
  for (const [x, v] of pts) {
    if (!ok(v)) { pen = false; continue; }
    d += `${pen ? "L" : "M"}${f1(X(x))},${f1(Y(v))}`;
    pen = true;
  }
  return d;
}

function areaPath(pts: [number, number | null, number | null][], X: (v: number) => number, Y: (v: number) => number): string {
  let d = "";
  let seg: [number, number, number][] = [];
  const flush = () => {
    if (seg.length > 1) {
      d += seg.map(([x, , hi], i) => `${i ? "L" : "M"}${f1(X(x))},${f1(Y(hi))}`).join("");
      d += seg.slice().reverse().map(([x, lo]) => `L${f1(X(x))},${f1(Y(lo))}`).join("") + "Z";
    }
    seg = [];
  };
  for (const [x, lo, hi] of pts) {
    if (ok(lo) && ok(hi)) seg.push([x, lo, hi]); else flush();
  }
  flush();
  return d;
}

interface XYLayout {
  y: YScale; X: (v: number) => number; ml: number; pw: number; yTop: number; yBot: number; x0: number; x1: number;
  xTicks: { x: number; label: string }[];
  vlines: { x: number; label: string; anchor: Anchor }[];
  ends: (XYEndLabel & { py: number })[];
}

function layoutXY(spec: XYSpec, w: number, h: number): XYLayout {
  const n = spec.xs.length;
  const cat = !!spec.cats;
  const mt = 8 + (spec.vlines.some((v) => v.label) ? 14 : 0);
  const mb = 20;
  const y = makeY(spec.yExt, h - mb, mt, { log: spec.log, zero: spec.zero, count: clamp(Math.floor((h - mt - mb) / 44), 2, 8) });
  const ml = yLabelWidth(y, spec.fmt);
  const mr = spec.endLabels.length ? Math.max(...spec.endLabels.map((e) => textW(e.text))) + 14 : 12;
  const pw = Math.max(20, w - ml - mr);
  let x0 = cat ? -0.5 : spec.xs[0] ?? 0, x1 = cat ? n - 0.5 : spec.xs[n - 1] ?? 1;
  if (x1 <= x0) { x0 -= cat ? 0.5 : DAY; x1 += cat ? 0.5 : DAY; }
  const X = (v: number) => ml + ((v - x0) / (x1 - x0)) * pw;
  const yTop = mt, yBot = h - mb;

  let xTicks: { x: number; label: string }[];
  if (cat) {
    const step = catLabelStep(spec.cats!, pw / Math.max(1, n));
    xTicks = spec.cats!.map((c, i) => ({ x: X(i), label: c })).filter((_, i) => (n - 1 - i) % step === 0);
  } else {
    xTicks = timeTicks(x0, x1, Math.max(2, Math.floor(pw / 80))).map((t) => ({ x: X(t.ts), label: t.label }));
  }
  xTicks = xTicks.filter((t) => t.x - textW(t.label) / 2 >= ml - 8 && t.x + textW(t.label) / 2 <= w - 2);

  // labelled verticals: a marker whose label would collide is dropped whole (an unlabelled line would only confuse)
  let lastEnd = -Infinity;
  const vlines = spec.vlines.filter((v) => v.x >= x0 && v.x <= x1).flatMap((v) => {
    const x = X(v.x);
    const lw = textW(v.label);
    const anchor: Anchor = x + lw / 2 > w - 2 ? "end" : x - lw / 2 < ml ? "start" : "middle";
    const left = anchor === "end" ? x - lw : anchor === "start" ? x : x - lw / 2;
    if (v.label && left <= lastEnd + 4) return [];
    if (v.label) lastEnd = left + lw;
    return [{ x, label: v.label, anchor }];
  });

  // right-margin value labels in priority order; one that would overlap a placed label is dropped
  const placed: number[] = [];
  const ends = spec.endLabels.filter((e) => ok(e.y)).map((e) => ({ ...e, py: clamp(y(e.y), yTop + 4, yBot - 2) })).filter((e) => {
    if (placed.some((p) => Math.abs(p - e.py) < 11)) return false;
    placed.push(e.py);
    return true;
  });
  return { y, X, ml, pw, yTop, yBot, x0, x1, xTicks, vlines, ends };
}

/** Everything that does not change on hover — memoised so the crosshair never re-renders hundreds of marks. */
const XYStatic = memo(function XYStatic({ spec, L, h, uid }: { spec: XYSpec; L: XYLayout; h: number; uid: string }) {
  const { y, X, ml, pw, yTop, yBot, x0 } = L;
  return (
    <>
      <defs><clipPath id={`${uid}-clip`}><rect x={ml} y={yTop} width={pw} height={yBot - yTop} /></clipPath></defs>
      <YAxis y={y} x1={ml} x2={ml + pw} fmt={spec.fmt} />
      <line className="rc-axis" x1={ml} x2={ml + pw} y1={crisp(yBot)} y2={crisp(yBot)} />
      {L.xTicks.map((t) => <text key={`${t.x}-${t.label}`} x={t.x} y={h - 5} textAnchor="middle">{t.label}</text>)}
      <g clipPath={`url(#${uid}-clip)`}>
        {spec.shade && spec.shade.pts.map(([x, v], i, arr) => {
          if (!ok(v) || v <= 0) return null;
          const prev = i > 0 ? arr[i - 1][0] : x, next = i < arr.length - 1 ? arr[i + 1][0] : x;
          const left = Math.round(X((prev + x) / 2)), right = Math.round(X((x + next) / 2));
          if (right <= left) return null;
          return <rect key={i} x={left} y={yTop} width={right - left} height={yBot - yTop} shapeRendering="crispEdges"
            fill={spec.shade!.color} opacity={clamp(v, 0, 1) * spec.shade!.max} />;
        })}
        {spec.areas.map((a, i) => <path key={i} d={areaPath(a.pts, X, y)} fill={a.color} opacity={a.opacity} />)}
        {L.vlines.map((v, i) => <line key={i} className="rc-axis" x1={crisp(v.x)} x2={crisp(v.x)} y1={yTop} y2={yBot} />)}
        {spec.hlines.map((l, i) => {
          const yy = y(l.y);
          if (yy < yTop - 1 || yy > yBot + 1) return null;
          return <line key={i} x1={X(Math.max(l.from ?? x0, x0))} x2={ml + pw} y1={yy} y2={yy} stroke={l.color}
            strokeWidth={l.width ?? 1} strokeDasharray={l.dash} opacity={l.opacity} />;
        })}
        {spec.lines.map((l, i) => (
          <path key={i} d={linePath(l.pts, X, y)} fill="none" stroke={l.color} strokeWidth={l.width ?? 2}
            strokeDasharray={l.dash} opacity={l.opacity} strokeLinejoin="round" strokeLinecap="round" />
        ))}
      </g>
      {spec.hlines.map((l, i) => {
        const yy = y(l.y);
        if (!l.label || yy < yTop + 10 || yy > yBot) return null;
        return <text key={i} className="lbl" x={X(Math.max(l.from ?? x0, x0)) + 5} y={yy - 4}>{l.label}</text>;
      })}
      {L.vlines.map((v, i) => v.label && <text key={i} className="lbl" x={v.x} y={yTop - 5} textAnchor={v.anchor}>{v.label}</text>)}
      {spec.dots.map((d, i) => {
        const yy = y(d.y);
        if (yy < yTop - 1 || yy > yBot + 1) return null;
        return (
          <g key={i}>
            <circle cx={X(d.x)} cy={yy} r={4} fill={d.color} stroke="var(--panel)" strokeWidth={2} />
            {d.label && <text className="strong" x={X(d.x) - 7} y={yy - 7} textAnchor="end">{d.label}</text>}
          </g>
        );
      })}
      {L.ends.map((e, i) => <text key={i} className={e.strong ? "strong" : "lbl"} x={ml + pw + 6} y={e.py + 3.5}>{e.text}</text>)}
    </>
  );
});

function XYPlot({ spec, w, h, label }: { spec: XYSpec; w: number; h: number; label: string }) {
  const uid = "rc" + useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const n = spec.xs.length;
  const { active, setActive, onKeyDown, onFocus, onBlur } = useActive(n);
  const lookups = useMemo(() => spec.lines.map((l) => (l.hoverDot === false ? null : new Map(l.pts.filter((p) => ok(p[1])) as [number, number][]))), [spec.lines]);
  const L = useMemo(() => layoutXY(spec, w, h), [spec, w, h]);
  const { y, X, ml, pw, yTop, yBot, x0, x1 } = L;

  const onMove = (e: ReactPointerEvent<SVGRectElement>) => {
    const r = (e.currentTarget.ownerSVGElement ?? e.currentTarget).getBoundingClientRect();
    const v = x0 + ((e.clientX - r.left - ml) / pw) * (x1 - x0);
    setActive(spec.cats ? clamp(Math.round(v), 0, n - 1) : nearest(spec.xs, v));
  };
  const ax = active !== null ? X(spec.xs[active]) : null;
  const tip = active !== null ? spec.tip(active) : null;

  return (
    <div style={{ position: "relative", width: w, height: h }} tabIndex={0} onKeyDown={onKeyDown} onFocus={onFocus} onBlur={onBlur} {...plotSvgProps(label)}>
      <svg width={w} height={h} aria-hidden="true">
        <XYStatic spec={spec} L={L} h={h} uid={uid} />
        {ax !== null && (
          <g pointerEvents="none">
            <line className="rc-cross" x1={crisp(ax)} x2={crisp(ax)} y1={yTop} y2={yBot} />
            {spec.lines.map((l, i) => {
              const v = lookups[i]?.get(spec.xs[active!]);
              if (!ok(v)) return null;
              const yy = y(v);
              if (yy < yTop || yy > yBot) return null;
              return <circle key={i} cx={ax} cy={yy} r={4} fill={l.color} stroke="var(--panel)" strokeWidth={2} />;
            })}
          </g>
        )}
        <rect className="rc-hit" x={ml} y={yTop} width={pw} height={yBot - yTop}
          onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={() => setActive(null)} />
      </svg>
      {tip && ax !== null && <Tip x={ax} w={w} content={tip} y={yTop} />}
    </div>
  );
}

// =================================================================== FanChart
type NumKey<T> = { [K in keyof T]-?: T[K] extends number ? K : never }[keyof T] & string;

/** One percentile band of a fan: keys of the lower and upper edge in each data point. */
export interface FanBand<T> { lo: NumKey<T>; hi: NumKey<T>; label: string }

/** Monte Carlo bands (McResult.fan), outermost first. */
export const MC_BANDS: FanBand<McBand>[] = [
  { lo: "p5", hi: "p95", label: "5–95%" },
  { lo: "p10", hi: "p90", label: "10–90%" },
  { lo: "p25", hi: "p75", label: "25–75%" },
];

export type QuantForecastPoint = NonNullable<QuantResult["chart"]>["forecast"][number];
/** Quant forecast cone bands (QuantResult.chart.forecast), outermost first. */
export const QUANT_BANDS: FanBand<QuantForecastPoint>[] = [
  { lo: "lo95", hi: "hi95", label: "95% range" },
  { lo: "lo68", hi: "hi68", label: "68% range" },
];

export interface FanChartProps<T extends { ts: number }> extends FrameProps {
  data: T[];                     // forecast points sorted by ts; data[0] is "now" (t = 0)
  median: NumKey<T>;             // key of the median, e.g. "p50"
  bands: FanBand<T>[];           // outermost first, e.g. MC_BANDS / QUANT_BANDS (one hue, more opaque toward the median)
  mean?: NumKey<T>;              // optional dashed mean line (--viz-3)
  history?: [number, number][];  // [ts, close] drawn left of "now" in --text-2
  paths?: number[][];            // sample paths, paths[k][i] ↔ data[i]; thin and faint, clipped to the plot
  spot?: number | null;          // dot at "now" plus a faint level line across the fan
  targets?: number[];            // price levels → dashed --viz-2 lines labelled "T ₹…"
  checkpoints?: { ts: number; label: string }[];   // labelled verticals, e.g. McResult.horizons → "6M", "1Y"
  log?: boolean;                 // log price axis (good for 2–3 year horizons)
  format?: Fmt;                  // default "inr"
}

const fanOpacity = (n: number): number[] =>
  n === 3 ? [0.1, 0.16, 0.26] : n === 2 ? [0.12, 0.24] : n === 1 ? [0.2] : Array.from({ length: n }, (_, i) => 0.08 + (0.2 * i) / Math.max(1, n - 1));

/** Probability fan: history line, nested percentile bands (drawn as non-overlapping rings so each region shows
 *  exactly its opacity), median, optional mean / sample paths, spot, targets and horizon checkpoints. */
export function FanChart<T extends { ts: number }>(props: FanChartProps<T>) {
  const { data, median, bands, mean, history, paths, spot, targets, checkpoints, log } = props;
  const fmt = props.format ?? "inr";
  const get = (d: T, k: NumKey<T>): number | null => {
    const v = d[k] as unknown;
    return ok(v) ? v : null;
  };

  const model = useMemo(() => {
    const hist = (history ?? []).filter(([t, v]) => ok(t) && ok(v));
    const pts = data.filter((d) => ok(d.ts));
    const ops = fanOpacity(bands.length);
    const areas: XYArea[] = [];
    bands.forEach((outer, b) => {
      const inner = bands[b + 1];
      if (inner) {
        areas.push({ pts: pts.map((d) => [d.ts, get(d, inner.hi), get(d, outer.hi)]), color: "var(--viz-1)", opacity: ops[b] });
        areas.push({ pts: pts.map((d) => [d.ts, get(d, outer.lo), get(d, inner.lo)]), color: "var(--viz-1)", opacity: ops[b] });
      } else areas.push({ pts: pts.map((d) => [d.ts, get(d, outer.lo), get(d, outer.hi)]), color: "var(--viz-1)", opacity: ops[b] });
    });
    const lines: XYLine[] = [];
    if (hist.length) lines.push({ name: "History", pts: hist, color: "var(--text-2)", width: 1.5 });
    (paths ?? []).forEach((p, k) => lines.push({
      name: `Path ${k + 1}`, pts: pts.map((d, i) => [d.ts, ok(p[i]) ? p[i] : null]), color: "var(--muted)", width: 1, opacity: 0.2, hoverDot: false,
    }));
    if (mean) lines.push({ name: "Mean", pts: pts.map((d) => [d.ts, get(d, mean)]), color: "var(--viz-3)", width: 1.5, dash: "4 3" });
    lines.push({ name: "Median", pts: pts.map((d) => [d.ts, get(d, median)]), color: "var(--viz-1)", width: 2 });

    const t0 = pts[0]?.ts ?? hist[hist.length - 1]?.[0] ?? 0;
    const hlines: XYHLine[] = [];
    if (ok(spot) && pts.length) hlines.push({ y: spot, color: "var(--muted)", opacity: 0.7, from: t0 });
    (targets ?? []).filter(ok).forEach((t) => hlines.push({ y: t, color: "var(--viz-2)", dash: "5 4", from: t0 }));
    const vlines: XYVLine[] = [];
    if (hist.length && pts.length) vlines.push({ x: t0, label: "now" });
    (checkpoints ?? []).filter((c) => ok(c.ts) && c.ts > t0).forEach((c) => vlines.push({ x: c.ts, label: c.label }));

    const vals: number[] = hist.map((p) => p[1]);
    const outer = bands[0];
    for (const d of pts) for (const k of outer ? [outer.lo, outer.hi, median] : [median]) { const v = get(d, k); if (v !== null) vals.push(v); }
    if (ok(spot)) vals.push(spot);
    // targets widen the scale only when near the data; far ones are labelled at the edge with an arrow instead
    const core = extent(vals);
    const span = core ? core[1] - core[0] : 0;
    const offScale = (t: number) => !!core && (t > core[1] + span * 0.5 || t < core[0] - span * 0.5);
    for (const t of targets ?? []) if (ok(t) && !offScale(t)) vals.push(t);

    const last = pts[pts.length - 1];
    const endLabels: XYEndLabel[] = [];
    if (last) endLabels.push({ y: get(last, median) ?? NaN, text: fmtLabel(fmt, get(last, median)), strong: true });
    for (const t of targets ?? []) if (ok(t)) endLabels.push({ y: t, text: `T ${fmtLabel(fmt, t)}${offScale(t) ? (core && t > core[1] ? " ▲" : " ▼") : ""}` });
    if (last && outer) {
      endLabels.push({ y: get(last, outer.hi) ?? NaN, text: fmtLabel(fmt, get(last, outer.hi)) });
      endLabels.push({ y: get(last, outer.lo) ?? NaN, text: fmtLabel(fmt, get(last, outer.lo)) });
    }

    const xs = Array.from(new Set([...hist.map((p) => p[0]), ...pts.map((d) => d.ts)])).sort((a, b) => a - b);
    const fIdx = new Map(pts.map((d, i) => [d.ts, i]));
    const hIdx = new Map(hist.map((p, i) => [p[0], i]));
    const base = ok(spot) ? spot : pts.length ? get(pts[0], median) : null;
    const tip = (i: number): TipContent => {
      const x = xs[i];
      const j = fIdx.get(x), k = hIdx.get(x);
      const rows: TipRow[] = [];
      let title = dateIST(x);
      if (j !== undefined) {
        const d = pts[j];
        const tt = (d as { t?: unknown }).t;
        if (ok(tt)) title += tt === 0 ? " · now" : ` · +${tt} trading days`;
        const med = get(d, median);
        rows.push({ value: <>{fmtValue(fmt, med)}{ok(base) && ok(med) && j > 0 ? ` (${fracPct(med / base - 1, 1, true)})` : ""}</>, label: "median", color: "var(--viz-1)" });
        for (const b of [...bands].reverse()) rows.push({ value: `${fmtValue(fmt, get(d, b.lo))} – ${fmtValue(fmt, get(d, b.hi))}`, label: b.label });
        if (mean) rows.push({ value: fmtValue(fmt, get(d, mean)), label: "mean", color: "var(--viz-3)", dash: true });
      }
      if (k !== undefined && (j === undefined || !rows.length)) rows.unshift({ value: fmtValue(fmt, hist[k][1]), label: "close", color: "var(--text-2)" });
      return { title, rows };
    };

    const spec: XYSpec = { xs, yExt: extent(vals), log, fmt, areas, lines, hlines, vlines, endLabels, tip,
      dots: ok(spot) && pts.length ? [{ x: t0, y: spot, color: "var(--text)" }] : [] };

    const legend: LegendItem[] = [];
    if (hist.length) legend.push({ label: "History", color: "var(--text-2)", kind: "line" });
    legend.push({ label: "Median", color: "var(--viz-1)", kind: "line" });
    if (mean) legend.push({ label: "Mean", color: "var(--viz-3)", kind: "dash" });
    bands.forEach((b, i) => legend.push({ label: b.label, color: "var(--viz-1)", kind: "area", opacity: ops[i] }));
    if (paths?.length) legend.push({ label: "Sample paths", color: "var(--muted)", kind: "line" });
    if (ok(spot) && pts.length) legend.push({ label: "Spot", color: "var(--text)", kind: "dot" });
    if ((targets ?? []).some(ok)) legend.push({ label: "Target", color: "var(--viz-2)", kind: "dash" });

    let summary = "";
    if (last) {
      summary = `On ${dateIST(last.ts)}: median ${fmtValue(fmt, get(last, median))}`;
      if (outer) summary += `, ${outer.label} range ${fmtValue(fmt, get(last, outer.lo))} to ${fmtValue(fmt, get(last, outer.hi))}`;
      summary += ".";
    }
    if (ok(spot)) summary += ` Spot ${fmtValue(fmt, spot)}.`;

    const table = (): TableData => ({
      columns: ["Date", "Close", "Median", ...bands.map((b) => b.label), ...(mean ? ["Mean"] : [])],
      rows: [
        ...hist.map(([t, v]) => [dateIST(t), fmtValue(fmt, v), "—", ...bands.map(() => "—"), ...(mean ? ["—"] : [])]),
        ...pts.map((d) => [dateIST(d.ts), "—", fmtValue(fmt, get(d, median)),
          ...bands.map((b) => `${fmtValue(fmt, get(d, b.lo))} – ${fmtValue(fmt, get(d, b.hi))}`), ...(mean ? [fmtValue(fmt, get(d, mean))] : [])]),
      ],
    });
    return { spec, legend, summary, table, empty: !pts.length && !hist.length };
  }, [data, median, bands, mean, history, paths, spot, targets, checkpoints, log, fmt]);

  return (
    <ChartFrame {...props} defaultHeight={300} legend={model.empty ? [] : model.legend} table={model.table}>
      {(w, h) => model.empty ? <div className="rc-empty">No data to plot.</div>
        : <XYPlot spec={model.spec} w={w} h={h} label={`${props.label}. ${model.summary}`} />}
    </ChartFrame>
  );
}

// =================================================================== LineChart
export interface LineSeries {
  name: string;
  values: (number | null)[];     // aligned with x; null breaks the line
  tone?: Tone;                   // default by order: 1, 2, 3
  width?: number;                // default 2 (1.5 for tone "text")
  dashed?: boolean;
}
export interface LineBand { name: string; lo: (number | null)[]; hi: (number | null)[]; tone?: Tone; opacity?: number }

export interface LineChartProps extends FrameProps {
  x: number[] | string[];        // epoch seconds (time axis, IST dates) or category labels (e.g. "Mar 2024")
  series: LineSeries[];
  bands?: LineBand[];            // shaded ranges under the lines, e.g. trend ± 2σ (default opacity .14)
  shade?: { name: string; values: (number | null)[]; tone?: Tone; format?: Fmt };   // 0…1 per x → background tint
                                 // (e.g. bull-regime probability; default tone 3, shown as % in the tooltip)
  refs?: { value: number; label: string }[];     // horizontal reference lines (e.g. 0, 50 %)
  marks?: { x: number | string; label: string }[];   // labelled verticals (an x value or category)
  format?: Fmt;                  // y format, default "inr"
  log?: boolean;
  zero?: boolean;                // include 0 in the y range
  endLabels?: boolean;           // label each series' last value at the right edge (default true; collisions dropped)
}

/** Multi-series line chart on a time or category axis, with optional ranges, background shading and references. */
export function LineChart(props: LineChartProps) {
  const { x, series, bands, shade, refs, marks, log, zero } = props;
  const fmt = props.format ?? "inr";
  const showEnds = props.endLabels ?? true;

  const model = useMemo(() => {
    const cat = typeof x[0] === "string";
    const cats = cat ? (x as string[]) : undefined;
    const xs = cat ? cats!.map((_, i) => i) : (x as number[]);
    const at = (i: number) => xs[i];
    const lines: XYLine[] = series.map((s, k) => ({
      name: s.name, pts: s.values.map((v, i) => [at(i), ok(v) ? v : null]), color: toneColor(s.tone, k),
      width: s.width ?? (s.tone === "text" ? 1.5 : 2), dash: s.dashed ? "4 3" : undefined,
    }));
    const areas: XYArea[] = (bands ?? []).map((b) => ({
      pts: b.lo.map((lo, i) => [at(i), ok(lo) ? lo : null, ok(b.hi[i]) ? b.hi[i] : null]), color: toneColor(b.tone ?? 1), opacity: b.opacity ?? 0.14,
    }));
    const shadeColor = toneColor(shade?.tone ?? 3);
    const vals: number[] = [];
    for (const s of series) for (const v of s.values) if (ok(v)) vals.push(v);
    for (const b of bands ?? []) { for (const v of b.lo) if (ok(v)) vals.push(v); for (const v of b.hi) if (ok(v)) vals.push(v); }
    for (const r of refs ?? []) if (ok(r.value)) vals.push(r.value);
    const markX = (m: number | string) => (cat ? cats!.indexOf(String(m)) : Number(m));

    const endLabels: XYEndLabel[] = [];
    if (showEnds) series.forEach((s, k) => {
      const v = s.values[xs.length - 1];       // only series that reach the right edge, or the label points at the wrong line
      if (ok(v)) endLabels.push({ y: v, text: fmtLabel(fmt, v), strong: k === 0 });
    });

    const title = (i: number) => (cat ? cats![i] : dateIST(xs[i]));
    const tip = (i: number): TipContent => {
      const rows: TipRow[] = series.map((s, k) => ({ value: fmtValue(fmt, s.values[i]), label: s.name, color: toneColor(s.tone, k), dash: s.dashed }));
      for (const b of bands ?? []) rows.push({ value: `${fmtValue(fmt, b.lo[i])} – ${fmtValue(fmt, b.hi[i])}`, label: b.name });
      if (shade) rows.push({ value: fmtValue(shade.format ?? "frac", shade.values[i]), label: shade.name });
      return { title: title(i), rows };
    };
    const spec: XYSpec = {
      xs, cats, yExt: extent(vals), log, zero, fmt, areas, lines, tip, dots: [],
      shade: shade ? { pts: shade.values.map((v, i) => [at(i), ok(v) ? v : null]), color: shadeColor, max: 0.22 } : undefined,
      hlines: (refs ?? []).filter((r) => ok(r.value)).map((r) => ({ y: r.value, color: "var(--border-2)", width: 1, label: r.label })),
      vlines: (marks ?? []).map((m) => ({ x: markX(m.x), label: m.label })).filter((m) => ok(m.x) && m.x >= 0),
      endLabels: endLabels.length <= 3 ? endLabels : [],
    };
    const legend: LegendItem[] = [
      ...series.map((s, k) => ({ label: s.name, color: toneColor(s.tone, k), kind: (s.dashed ? "dash" : "line") as LegendItem["kind"] })),
      ...(bands ?? []).map((b) => ({ label: b.name, color: toneColor(b.tone ?? 1), kind: "area" as const, opacity: Math.max(0.25, b.opacity ?? 0.14) })),
      ...(shade ? [{ label: shade.name, color: shadeColor, kind: "area" as const, opacity: 0.3 }] : []),
    ];
    const n = xs.length;
    const summary = n ? `${title(0)} to ${title(n - 1)}. ` + series.map((s) => {
      const e = extent(s.values.filter(ok) as number[]);
      const lastV = [...s.values].reverse().find(ok);
      return e ? `${s.name}: last ${fmtValue(fmt, lastV)}, range ${fmtValue(fmt, e[0])} to ${fmtValue(fmt, e[1])}` : `${s.name}: no data`;
    }).join("; ") + "." : "No data.";
    const table = (): TableData => ({
      columns: [cat ? "" : "Date", ...series.map((s) => s.name), ...(bands ?? []).map((b) => b.name), ...(shade ? [shade.name] : [])],
      rows: xs.map((_, i) => [title(i), ...series.map((s) => fmtValue(fmt, s.values[i])),
        ...(bands ?? []).map((b) => `${fmtValue(fmt, b.lo[i])} – ${fmtValue(fmt, b.hi[i])}`),
        ...(shade ? [fmtValue(shade.format ?? "frac", shade.values[i])] : [])]),
    });
    return { spec, legend, summary, table, empty: !n || !vals.length };
  }, [x, series, bands, shade, refs, marks, log, zero, fmt, showEnds]);

  return (
    <ChartFrame {...props} defaultHeight={260} legend={model.legend} table={model.table}>
      {(w, h) => model.empty ? <div className="rc-empty">No data to plot.</div>
        : <XYPlot spec={model.spec} w={w} h={h} label={`${props.label}. ${model.summary}`} />}
    </ChartFrame>
  );
}

// =================================================================== BarChart
export interface BarSeries { name: string; values: (number | null)[]; tone?: Tone }

export interface BarChartProps extends FrameProps {
  categories: string[];          // x labels, e.g. FinTable.columns ("Jun 2025")
  series: BarSeries[];           // 1–3 series, values aligned with categories; negatives grow down from 0
  format?: Fmt;                  // default "num"
  refs?: { value: number; label: string }[];   // horizontal reference lines
  labelLast?: boolean;           // value label on the last category's bars when it fits (default true)
}

/** Grouped column chart for signed values (quarterly sales / profit, yearly ratios, PIT bins). */
export function BarChart(props: BarChartProps) {
  const { categories, series, refs } = props;
  const fmt = props.format ?? "num";
  const labelLast = props.labelLast ?? true;
  const model = useMemo(() => {
    const vals: number[] = [];
    for (const s of series) for (const v of s.values) if (ok(v)) vals.push(v);
    for (const r of refs ?? []) if (ok(r.value)) vals.push(r.value);
    const legend: LegendItem[] = series.map((s, k) => ({ label: s.name, color: toneColor(s.tone, k), kind: "area" }));
    const n = categories.length;
    const summary = n ? series.map((s) => {
      const last = s.values[n - 1];
      const e = extent(s.values.filter(ok) as number[]);
      return e ? `${s.name}: ${categories[n - 1]} ${fmtValue(fmt, last)}, range ${fmtValue(fmt, e[0])} to ${fmtValue(fmt, e[1])}` : `${s.name}: no data`;
    }).join("; ") + "." : "No data.";
    const table = (): TableData => ({
      columns: ["", ...series.map((s) => s.name)],
      rows: categories.map((c, i) => [c, ...series.map((s) => fmtValue(fmt, s.values[i]))]),
    });
    return { vals, legend, summary, table, empty: !n || !vals.length };
  }, [categories, series, refs, fmt]);

  return (
    <ChartFrame {...props} defaultHeight={220} legend={model.legend} table={model.table}>
      {(w, h) => model.empty ? <div className="rc-empty">No data to plot.</div>
        : <BarPlot {...props} fmt={fmt} labelLast={labelLast} vals={model.vals} w={w} h={h} label={`${props.label}. ${model.summary}`} />}
    </ChartFrame>
  );
}

function BarPlot(p: BarChartProps & { fmt: Fmt; labelLast: boolean; vals: number[]; w: number; h: number; label: string }) {
  const { categories, series, refs, fmt, w, h } = p;
  const n = categories.length, ns = Math.max(1, series.length);
  const { active, setActive, onKeyDown, onFocus, onBlur } = useActive(n);
  const mt = p.labelLast ? 16 : 8, mb = 20;
  const y = makeY(extent(p.vals), h - mb, mt, { zero: true, count: clamp(Math.floor((h - mt - mb) / 44), 2, 6) });
  const ml = yLabelWidth(y, fmt), mr = 8;
  const pw = Math.max(20, w - ml - mr);
  const slot = pw / Math.max(1, n);
  const gap = 2;
  const groupW = Math.min(slot * 0.72, ns * 24 + (ns - 1) * gap);
  const barW = Math.max(1, (groupW - (ns - 1) * gap) / ns);
  const y0 = y(0);
  const step = catLabelStep(categories, slot);
  const bx = (i: number, s: number) => ml + slot * i + (slot - groupW) / 2 + s * (barW + gap);

  const onMove = (e: ReactPointerEvent<SVGRectElement>) => {
    const r = (e.currentTarget.ownerSVGElement ?? e.currentTarget).getBoundingClientRect();
    setActive(clamp(Math.floor((e.clientX - r.left - ml) / slot), 0, n - 1));
  };

  const lastLabels = p.labelLast && n ? series.map((s, k) => {
    const v = s.values[n - 1];
    if (!ok(v)) return null;
    const text = fmtValue(fmt, v);
    if (ns > 1 && textW(text) > barW + gap + 2) return null;
    if (ns === 1 && textW(text) > slot + mr + 4) return null;
    const tip = y(v);
    return { x: bx(n - 1, k) + barW / 2, y: v >= 0 ? tip - 4 : tip + 11, text };
  }) : [];

  const tip: TipContent | null = active === null ? null : {
    title: categories[active],
    rows: series.map((s, k) => ({ value: fmtValue(fmt, s.values[active]), label: s.name, color: toneColor(s.tone, k) })),
  };
  const ax = active !== null ? ml + slot * (active + 0.5) : 0;

  return (
    <div style={{ position: "relative", width: w, height: h }} tabIndex={0} onKeyDown={onKeyDown} onFocus={onFocus} onBlur={onBlur} {...plotSvgProps(p.label)}>
      <svg width={w} height={h} aria-hidden="true">
        <YAxis y={y} x1={ml} x2={ml + pw} fmt={fmt} />
        {active !== null && <rect x={ml + slot * active} y={mt} width={slot} height={h - mb - mt} fill="var(--hover)" opacity={0.7} />}
        {series.map((s, k) => s.values.slice(0, n).map((v, i) => ok(v) && Math.abs(y(v) - y0) >= 0.5 && (
          <path key={`${k}-${i}`} d={barPath(bx(i, k), barW, y0, y(v), 4)} fill={toneColor(s.tone, k)}
            opacity={active === null || active === i ? 1 : 0.6} />
        )))}
        <line className="rc-axis" x1={ml} x2={ml + pw} y1={crisp(y0)} y2={crisp(y0)} />
        {(refs ?? []).filter((r) => ok(r.value)).map((r, i) => (
          <g key={i}>
            <line x1={ml} x2={ml + pw} y1={y(r.value)} y2={y(r.value)} stroke="var(--text-2)" strokeWidth={1} strokeDasharray="5 4" opacity={0.8} />
            <text className="lbl" x={ml + pw - 2} y={y(r.value) - 4} textAnchor="end">{r.label}</text>
          </g>
        ))}
        {lastLabels.map((l, i) => l && <text key={i} className="lbl" x={l.x} y={l.y} textAnchor="middle">{l.text}</text>)}
        {categories.map((c, i) => {
          if ((n - 1 - i) % step !== 0) return null;
          const x = ml + slot * (i + 0.5), lw = textW(c);
          const end = x + lw / 2 > w - 1, start = !end && x - lw / 2 < 0;    // keep edge labels inside the svg
          return <text key={i} x={end ? w - 1 : start ? 0 : x} y={h - 5} textAnchor={end ? "end" : start ? "start" : "middle"}>{c}</text>;
        })}
        <rect className="rc-hit" x={ml} y={mt} width={pw} height={h - mb - mt} style={{ cursor: "default" }}
          onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={() => setActive(null)} />
      </svg>
      {tip && <Tip x={ax} w={w} content={tip} y={mt} />}
    </div>
  );
}

// =================================================================== Histogram
export interface HistMarker { value: number; label: string; kind?: "spot" | "ref" | "target" }

export interface HistogramProps extends FrameProps {
  edges: number[];               // bin edges, length = counts.length + 1 (McResult.terminal.edges)
  counts: number[];
  markers?: HistMarker[];        // labelled verticals: "spot" solid --text, "ref" (default) --text-2, "target" dashed --viz-2
  splitAt?: number | null;       // tint bins below / above this value with --down / --up (e.g. the spot price)
  format?: Fmt;                  // x format, default "inr"
}

/** Distribution of a simulated quantity (e.g. price at the horizon); y = share of paths. */
export function Histogram(props: HistogramProps) {
  const { edges, counts, markers, splitAt } = props;
  const fmt = props.format ?? "inr";
  const valid = edges.length === counts.length + 1 && counts.length > 0 && edges.every(ok);
  const model = useMemo(() => {
    const total = counts.reduce((a, c) => a + (ok(c) ? c : 0), 0) || 1;
    const share = counts.map((c) => (ok(c) ? c / total : 0));
    const cum: number[] = [];
    share.reduce((a, s, i) => (cum[i] = a + s), 0);
    let mode = 0;
    share.forEach((s, i) => { if (s > share[mode]) mode = i; });
    const summary = valid ? `${fx(total, 0)} outcomes between ${fmtValue(fmt, edges[0])} and ${fmtValue(fmt, edges[edges.length - 1])}; ` +
      `most common ${fmtValue(fmt, edges[mode])} to ${fmtValue(fmt, edges[mode + 1])} (${fracPct(share[mode])}).` +
      (markers ?? []).map((m) => ` ${m.label} ${fmtValue(fmt, m.value)}.`).join("") : "No data.";
    const legend: LegendItem[] = ok(splitAt) ? [
      { label: `▼ below ${fmtLabel(fmt, splitAt)}`, color: "var(--down)", kind: "area", opacity: 0.5 },
      { label: `▲ above ${fmtLabel(fmt, splitAt)}`, color: "var(--up)", kind: "area", opacity: 0.5 },
    ] : [];
    const table = (): TableData => ({
      columns: ["From", "To", "Count", "Share", "Cumulative"],
      rows: counts.map((c, i) => [fmtValue(fmt, edges[i]), fmtValue(fmt, edges[i + 1]), fx(c, 0), fracPct(share[i]), fracPct(cum[i])]),
    });
    return { total, share, cum, summary, legend, table };
  }, [edges, counts, markers, splitAt, fmt, valid]);

  return (
    <ChartFrame {...props} defaultHeight={220} legend={model.legend} table={model.table}>
      {(w, h) => !valid ? <div className="rc-empty">No data to plot.</div>
        : <HistPlot {...props} fmt={fmt} share={model.share} cum={model.cum} total={model.total} w={w} h={h} label={`${props.label}. ${model.summary}`} />}
    </ChartFrame>
  );
}

function HistPlot(p: HistogramProps & { fmt: Fmt; share: number[]; cum: number[]; total: number; w: number; h: number; label: string }) {
  const { edges, counts, share, cum, fmt, w, h, splitAt } = p;
  const n = counts.length;
  const { active, setActive, onKeyDown, onFocus, onBlur } = useActive(n);
  const e0 = edges[0], eN = edges[n];
  const span = eN - e0 || 1;

  // markers: keep near-range ones on scale, pin far ones to the edge with an arrow
  let lo = e0, hi = eN;
  for (const m of p.markers ?? []) if (ok(m.value) && m.value >= e0 - span * 0.25 && m.value <= eN + span * 0.25) { lo = Math.min(lo, m.value); hi = Math.max(hi, m.value); }
  const ext = extent(share);
  const mb = 20;
  // marker label rows (greedy, up to 3) decide the top margin
  const ml0 = 40;
  const pw0 = Math.max(20, w - ml0 - 10);
  const X0 = (v: number) => ml0 + ((v - lo) / (hi - lo || 1)) * pw0;
  const rowEnds: number[] = [];
  const marks = (p.markers ?? []).filter((m) => ok(m.value)).sort((a, b) => a.value - b.value).map((m) => {
    const text = m.value < lo ? `◂ ${m.label}` : m.value > hi ? `${m.label} ▸` : m.label;
    const x = X0(clamp(m.value, lo, hi));
    const tw = textW(text);
    let row = rowEnds.findIndex((end) => x - tw / 2 > end + 6);
    if (row === -1) { row = Math.min(2, rowEnds.length); }
    rowEnds[row] = x + tw / 2;
    return { ...m, text, row };
  });
  const rows = marks.length ? Math.max(...marks.map((m) => m.row)) + 1 : 0;
  const rowY = (r: number) => 10 + r * 13;
  const mt = rows ? rowY(rows - 1) + 8 : 8;
  const y = makeY(ext, h - mb, mt, { zero: true, count: clamp(Math.floor((h - mt - mb) / 40), 2, 5) });
  const ml = Math.max(ml0, yLabelWidth(y, "frac"));
  const mr = 10;
  const pw = Math.max(20, w - ml - mr);
  const X = (v: number) => ml + ((v - lo) / (hi - lo || 1)) * pw;
  const binW = X(edges[1]) - X(edges[0]);
  const gap = binW >= 6 ? 2 : binW >= 3 ? 1 : 0;
  const y0 = y(0);
  const xt = linearTicks(lo, hi, Math.max(2, Math.floor(pw / 90)));

  const onMove = (e: ReactPointerEvent<SVGRectElement>) => {
    const r = (e.currentTarget.ownerSVGElement ?? e.currentTarget).getBoundingClientRect();
    const v = lo + ((e.clientX - r.left - ml) / pw) * (hi - lo);
    let i = 0;
    while (i < n - 1 && edges[i + 1] <= v) i++;
    setActive(v < e0 || v > eN ? null : i);
  };
  const fillFor = (i: number) => {
    if (!ok(splitAt)) return { fill: "var(--viz-1)", opacity: 0.8 };
    return { fill: (edges[i] + edges[i + 1]) / 2 < splitAt ? "var(--down)" : "var(--up)", opacity: 0.5 };
  };

  const tip: TipContent | null = active === null ? null : {
    title: `${fmtValue(fmt, edges[active])} – ${fmtValue(fmt, edges[active + 1])}`,
    rows: [
      { value: fracPct(share[active], 1), label: `of outcomes (${fx(counts[active], 0)})` },
      { value: fracPct(cum[active], 1), label: `end at or below ${fmtValue(fmt, edges[active + 1])}` },
      ...(ok(splitAt) ? [{ value: (edges[active] + edges[active + 1]) / 2 < splitAt ? "▼ below" : "▲ above", label: fmtValue(fmt, splitAt) }] : []),
    ],
  };

  return (
    <div style={{ position: "relative", width: w, height: h }} tabIndex={0} onKeyDown={onKeyDown} onFocus={onFocus} onBlur={onBlur} {...plotSvgProps(p.label)}>
      <svg width={w} height={h} aria-hidden="true">
        <YAxis y={y} x1={ml} x2={ml + pw} fmt="frac" />
        {counts.map((_, i) => {
          const x1 = X(edges[i]) + gap / 2, x2 = X(edges[i + 1]) - gap / 2;
          const top = y(share[i]);
          if (y0 - top < 0.5) return null;
          const f = fillFor(i);
          return <path key={i} d={barPath(x1, Math.max(0.6, x2 - x1), y0, top, Math.min(4, (x2 - x1) / 2))} fill={f.fill}
            opacity={active === i ? Math.min(1, f.opacity + 0.3) : active === null ? f.opacity : f.opacity * 0.7} />;
        })}
        <line className="rc-axis" x1={ml} x2={ml + pw} y1={crisp(y0)} y2={crisp(y0)} />
        {xt.ticks.map((t) => {
          const x = X(t);
          const s = fmtTick(fmt, t, xt.step);
          if (x - textW(s) / 2 < ml - 10 || x + textW(s) / 2 > w) return null;
          return <text key={t} x={x} y={h - 5} textAnchor="middle">{s}</text>;
        })}
        {marks.map((m, i) => {
          const x = X(clamp(m.value, lo, hi));
          const kind = m.kind ?? "ref";
          const stroke = kind === "spot" ? "var(--text)" : kind === "target" ? "var(--viz-2)" : "var(--text-2)";
          const tw = textW(m.text);
          const anchor = x + tw / 2 > w - 2 ? "end" : x - tw / 2 < ml - 10 ? "start" : "middle";
          return (
            <g key={i} pointerEvents="none">
              <line x1={x} x2={x} y1={rowY(m.row) + 3} y2={y0} stroke={stroke} strokeWidth={kind === "spot" ? 1.5 : 1}
                strokeDasharray={kind === "target" ? "5 4" : undefined} opacity={kind === "ref" ? 0.75 : 1} />
              <text className={kind === "spot" ? "strong" : "lbl"} x={x} y={rowY(m.row)} textAnchor={anchor}
                style={{ paintOrder: "stroke", stroke: "var(--panel)", strokeWidth: 3 }}>{m.text}</text>
            </g>
          );
        })}
        <rect className="rc-hit" x={ml} y={mt} width={pw} height={h - mb - mt} style={{ cursor: "default" }}
          onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={() => setActive(null)} />
      </svg>
      {tip && active !== null && <Tip x={(X(edges[active]) + X(edges[active + 1])) / 2} w={w} content={tip} y={mt} />}
    </div>
  );
}

// =================================================================== PitHistogram
export interface PitHistogramProps extends Omit<FrameProps, "label"> {
  bins: number[];                // counts per PIT bin over [0, 1] (McBacktest.pit_hist, 10 bins)
  label?: string;
}

/** Probability-integral-transform histogram of a calibration backtest: flat = calibrated. */
export function PitHistogram({ bins, label = "Where the realised price fell inside the forecast distribution (PIT)", ...rest }: PitHistogramProps) {
  const k = bins.length;
  const total = bins.reduce((a, b) => a + (ok(b) ? b : 0), 0) || 1;
  const cats = useMemo(() => bins.map((_, i) => `${fx(i / k, 1)}–${fx((i + 1) / k, 1)}`), [bins, k]);
  const series = useMemo(() => [{ name: "Share of origins", values: bins.map((b) => (ok(b) ? b / total : null)) }], [bins, total]);
  const refs = useMemo(() => (k ? [{ value: 1 / k, label: "flat = calibrated" }] : []), [k]);
  return (
    <BarChart {...rest} label={label} categories={cats} series={series} refs={refs} format="frac" labelLast={false}
      caption={rest.caption ?? "Flat bars = honest bands. U-shape = bands too narrow (over-confident); hump = too wide; slope = biased drift. Few origins make this noisy."} />
  );
}

// =================================================================== CoverageChart
export interface CoverageChartProps extends Omit<FrameProps, "label" | "height"> {
  data: { level: number; hit_rate: number; expected: number }[];   // McBacktest.coverage
  n?: number | null;             // independent windows (McBacktest.effective_n) → binomial noise zone; default ±5 pp
  label?: string;
}

/** Nominal vs actual interval hit rates: bar = how often the price ended inside the band, tick = the ideal,
 *  shaded zone = what sampling noise alone could explain. */
export function CoverageChart({ data, n, label = "Forecast band coverage: actual vs ideal", ...rest }: CoverageChartProps) {
  const rows = useMemo(() => data.filter((d) => ok(d.level) && ok(d.hit_rate) && ok(d.expected)).map((d) => {
    const tol = ok(n) && n > 0 ? Math.min(0.5, 1.96 * Math.sqrt((d.expected * (1 - d.expected)) / n)) : 0.05;
    const verdict = d.hit_rate < d.expected - tol ? "over-confident — bands too narrow" : d.hit_rate > d.expected + tol ? "under-confident — bands too wide" : "consistent with the ideal";
    const glyph = d.hit_rate < d.expected - tol ? "▼" : d.hit_rate > d.expected + tol ? "▲" : "✓";
    return { ...d, tol, verdict, glyph, name: `${fx(d.level * 100, 0)}% band` };
  }), [data, n]);
  const summary = rows.map((r) => `${r.name} held the price ${fracPct(r.hit_rate, 0)} of the time (ideal ${fracPct(r.expected, 0)}), ${r.verdict}`).join("; ") + ".";
  const table = (): TableData => ({
    columns: ["Band", "Actual", "Ideal", "Noise ±", "Verdict"],
    rows: rows.map((r) => [r.name, fracPct(r.hit_rate, 0), fracPct(r.expected, 0), `${fx(r.tol * 100, 0)} pp`, r.verdict]),
  });
  const caption = rest.caption ?? `Bar = share of backtest origins whose realised price ended inside the band; tick = ideal; shaded = ±sampling noise${ok(n) ? ` for ≈${fx(n, 0)} independent windows` : ""}. Bar short of the tick → bands too narrow.`;
  return (
    <ChartFrame {...rest} caption={caption} label={label} height={rows.length * 30 + 24} defaultHeight={0} table={table}>
      {(w, h) => !rows.length ? <div className="rc-empty">No backtest yet.</div> : <CoveragePlot rows={rows} w={w} h={h} label={`${label}. ${summary}`} />}
    </ChartFrame>
  );
}

function CoveragePlot({ rows, w, h, label }: {
  rows: { level: number; hit_rate: number; expected: number; tol: number; verdict: string; glyph: string; name: string }[]; w: number; h: number; label: string;
}) {
  const { active, setActive, onKeyDown, onFocus, onBlur } = useActive(rows.length);
  const ml = 72, mr = 64, rowH = 30, mt = 2;
  const pw = Math.max(20, w - ml - mr);
  const X = (v: number) => ml + clamp(v, 0, 1) * pw;
  const onMove = (e: ReactPointerEvent<SVGRectElement>) => {
    const r = (e.currentTarget.ownerSVGElement ?? e.currentTarget).getBoundingClientRect();
    setActive(clamp(Math.floor((e.clientY - r.top - mt) / rowH), 0, rows.length - 1));
  };
  const a = active !== null ? rows[active] : null;
  return (
    <div style={{ position: "relative", width: w, height: h }} tabIndex={0} onKeyDown={onKeyDown} onFocus={onFocus} onBlur={onBlur} {...plotSvgProps(label)}>
      <svg width={w} height={h} aria-hidden="true">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className="rc-grid" x1={crisp(X(t))} x2={crisp(X(t))} y1={mt} y2={mt + rows.length * rowH} />
            <text x={X(t)} y={h - 5} textAnchor="middle">{`${t * 100}%`}</text>
          </g>
        ))}
        {rows.map((r, i) => {
          const cy = mt + i * rowH + rowH / 2;
          return (
            <g key={r.level} opacity={active === null || active === i ? 1 : 0.6}>
              <text className="lbl" x={ml - 8} y={cy + 3.5} textAnchor="end">{r.name}</text>
              <rect x={X(r.expected - r.tol)} y={cy - 8} width={Math.max(1, X(r.expected + r.tol) - X(r.expected - r.tol))} height={16} rx={2}
                fill="var(--border-2)" opacity={0.6} />
              {r.hit_rate > 0 && <path d={horizBar(X(0), X(r.hit_rate), cy, 5)} fill="var(--viz-1)" />}
              <line x1={X(r.expected)} x2={X(r.expected)} y1={cy - 9} y2={cy + 9} stroke="var(--text)" strokeWidth={2} />
              <text className="strong" x={ml + pw + 8} y={cy + 3.5}>{`${fx(r.hit_rate * 100, 0)}% ${r.glyph}`}</text>
            </g>
          );
        })}
        <rect className="rc-hit" x={0} y={mt} width={w} height={rows.length * rowH} style={{ cursor: "default" }}
          onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={() => setActive(null)} />
      </svg>
      {a && <Tip x={X(a.hit_rate)} w={w} y={mt + (active! + 1) * rowH - 4} content={{
        title: `${a.name} (central)`,
        rows: [
          { value: fracPct(a.hit_rate, 0), label: "held the realised price", color: "var(--viz-1)" },
          { value: fracPct(a.expected, 0), label: "ideal", color: "var(--text)" },
          { value: `±${fx(a.tol * 100, 0)} pp`, label: "sampling noise" },
          { value: `${a.glyph} ${a.verdict}`, label: "" },
        ],
      }} />}
    </div>
  );
}

/** Horizontal bar from x1 to x2 centred on cy, square at the start, 4 px rounded data end. */
function horizBar(x1: number, x2: number, cy: number, half: number): string {
  const w = Math.max(1, x2 - x1);
  const r = Math.min(4, w / 2, half);
  return `M${x1},${cy - half}H${x1 + w - r}Q${x1 + w},${cy - half} ${x1 + w},${cy - half + r}V${cy + half - r}Q${x1 + w},${cy + half} ${x1 + w - r},${cy + half}H${x1}Z`;
}

// =================================================================== ProbBar
export interface ProbBarProps {
  p: number | null | undefined;  // probability of a rise, 0…1
  label?: string;                // visible prefix, default "P(up)"; pass "" to hide (still in the aria-label)
  neutral?: [number, number];    // band shown as neutral grey, default [0.45, 0.55] (the quant model's signal cut-offs)
  width?: number;                // track width in px, default 90
}

/** P(up) diverging around 50 %: fill grows right (▲, --up) or left (▼, --down) from the centre tick. */
export function ProbBar({ p, label = "P(up)", neutral = [0.45, 0.55], width = 90 }: ProbBarProps) {
  const name = label || "P(up)";
  if (!ok(p)) {
    return <span className="rc-prob" role="img" aria-label={`${name} not available`}>{label && <span className="rc-prob-label">{label}</span>}<span className="muted">—</span></span>;
  }
  const q = clamp(p, 0, 1);
  const dir = q >= neutral[1] ? "up" : q <= neutral[0] ? "down" : "flat";
  const glyph = q > 0.5 ? "▲" : q < 0.5 ? "▼" : "•";
  const words = dir === "up" ? "leans up" : dir === "down" ? "leans down" : "no clear lean";
  const pctText = `${fx(q * 100, 0)}%`;
  return (
    <span className="rc-prob" role="img" aria-label={`${name} ${pctText}, ${words}`} title={`${name} ${fx(q * 100, 1)}% — ${words}`}>
      {label && <span className="rc-prob-label" aria-hidden="true">{label}</span>}
      <span className="rc-prob-track" style={{ width }} aria-hidden="true">
        <i className={`rc-prob-fill ${dir} ${q >= 0.5 ? "r" : "l"}`} style={{ width: `${Math.abs(q - 0.5) * 100}%` }} />
      </span>
      <span className={`rc-prob-val ${dir === "flat" ? "flat" : dir}`} aria-hidden="true">{glyph} {pctText}</span>
    </span>
  );
}
