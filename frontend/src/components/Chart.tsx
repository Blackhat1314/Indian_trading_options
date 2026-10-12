import { useEffect, useMemo, useRef, useState } from "react";
import {
  CandlestickSeries, ColorType, CrosshairMode, HistogramSeries, LineSeries, LineStyle, TickMarkType, createChart, createSeriesMarkers,
} from "lightweight-charts";
import type {
  IChartApi, IPriceLine, IPrimitivePaneRenderer, IPrimitivePaneView, ISeriesApi, ISeriesMarkersPluginApi,
  ISeriesPrimitive, SeriesAttachedParameter, SeriesMarker, Time, UTCTimestamp,
} from "lightweight-charts";
import { api } from "../lib/api";
import { fx, toChartTime } from "../lib/format";
import { useLive } from "../lib/store";
import type { ChartData } from "../lib/types";
import { Seg } from "./ui";

/* ---------------------------------------------------------------- zones (FVG / order blocks) */
interface ZoneDraw { t1: number; p1: number; p2: number; fill: string; stroke: string; label: string }

class ZonesPrimitive implements ISeriesPrimitive<Time> {
  zones: ZoneDraw[] = [];
  chart: IChartApi | null = null;
  series: ISeriesApi<"Candlestick"> | null = null;
  private request: (() => void) | null = null;
  private views: IPrimitivePaneView[];

  constructor() {
    const self = this;
    const renderer: IPrimitivePaneRenderer = {
      draw(target) {
        target.useBitmapCoordinateSpace((scope) => {
          const { chart, series } = self;
          if (!chart || !series) return;
          const ctx = scope.context;
          const hpr = scope.horizontalPixelRatio;
          const vpr = scope.verticalPixelRatio;
          const right = scope.bitmapSize.width;
          ctx.font = `${10 * vpr}px "Fira Code", monospace`;
          for (const z of self.zones) {
            const x = chart.timeScale().timeToCoordinate(z.t1 as UTCTimestamp);
            const y1 = series.priceToCoordinate(z.p1);
            const y2 = series.priceToCoordinate(z.p2);
            if (x === null || y1 === null || y2 === null) continue;
            const left = Math.max(0, x * hpr);
            const top = Math.min(y1, y2) * vpr;
            const h = Math.max(1, Math.abs(y2 - y1) * vpr);
            ctx.fillStyle = z.fill;
            ctx.fillRect(left, top, right - left, h);
            ctx.strokeStyle = z.stroke;
            ctx.lineWidth = 1;
            ctx.strokeRect(left + 0.5, top + 0.5, right - left - 1, h - 1);
            if (h > 11 * vpr) {
              ctx.fillStyle = z.stroke;
              ctx.fillText(z.label, left + 4 * hpr, top + 10 * vpr);
            }
          }
        });
      },
    };
    this.views = [{ renderer: () => renderer, zOrder: () => "bottom" }];
  }

  attached(p: SeriesAttachedParameter<Time>) {
    this.chart = p.chart as IChartApi;
    this.series = p.series as ISeriesApi<"Candlestick">;
    this.request = p.requestUpdate;
  }
  detached() {
    this.chart = null;
    this.series = null;
  }
  paneViews() {
    return this.views;
  }
  set(z: ZoneDraw[]) {
    this.zones = z;
    this.request?.();
  }
}

/* ------------------------------------------------------------------------------ chart */
const TFS = {
  intraday: [{ v: "1", label: "1m" }, { v: "5", label: "5m" }, { v: "15", label: "15m" }, { v: "D", label: "1D" }],
  daily: [{ v: "D", label: "1D" }, { v: "15", label: "15m" }, { v: "5", label: "5m" }],
} as const;

type Tf = "1" | "5" | "15" | "D";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** Axis labels. Chart times are already shifted to IST, so read them with UTC getters. */
function tickLabel(time: Time, type: TickMarkType): string {
  const d = new Date((time as number) * 1000);
  const dd = String(d.getUTCDate()).padStart(2, "0");
  switch (type) {
    case TickMarkType.Year: return String(d.getUTCFullYear());
    case TickMarkType.Month: return `${MONTHS[d.getUTCMonth()]} ${String(d.getUTCFullYear()).slice(2)}`;
    case TickMarkType.DayOfMonth: return `${dd} ${MONTHS[d.getUTCMonth()]}`;
    default: return `${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")}`;
  }
}

function ema(values: number[], n: number): (number | null)[] {
  const k = 2 / (n + 1);
  let e: number | null = null;
  return values.map((v, i) => {
    e = e === null ? v : v * k + e * (1 - k);
    return i >= n - 1 ? e : null;
  });
}

export function PriceChart(props: { symbol: string; defaultTf?: Tf; mode?: "intraday" | "daily"; extraLevels?: Record<string, number | undefined> }) {
  const { symbol, mode = "intraday" } = props;
  const [tf, setTf] = useState<Tf>(props.defaultTf ?? (mode === "daily" ? "D" : "5"));
  const [showSmc, setShowSmc] = useState(true);
  const [data, setData] = useState<ChartData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [hover, setHover] = useState<{ o: number; h: number; l: number; c: number; v: number } | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const ref = useRef<{
    chart: IChartApi; candles: ISeriesApi<"Candlestick">; volume: ISeriesApi<"Histogram">; vwap: ISeriesApi<"Line">;
    ema20: ISeriesApi<"Line">; ema50: ISeriesApi<"Line">; zones: ZonesPrimitive; markers: ISeriesMarkersPluginApi<Time>; lines: IPriceLine[];
  } | null>(null);
  const lastBar = useRef<{ t: number; o: number; h: number; l: number; c: number; v: number } | null>(null);
  const lastVol = useRef<number | null>(null);
  const sid = useLive((s) => s.bySymbol[symbol]?.key);

  // create chart once
  useEffect(() => {
    if (!box.current) return;
    const chart = createChart(box.current, {
      autoSize: true,
      localization: { locale: "en-IN" },
      layout: {
        background: { type: ColorType.Solid, color: "#0b1017" }, textColor: "#7a889d",
        fontFamily: '"Fira Code", ui-monospace, monospace', fontSize: 11, attributionLogo: true,
      },
      grid: { vertLines: { color: "#121a25" }, horzLines: { color: "#121a25" } },
      rightPriceScale: { borderColor: "#1b2431", scaleMargins: { top: 0.13, bottom: 0.22 } },
      timeScale: { borderColor: "#1b2431", timeVisible: true, secondsVisible: false, rightOffset: 6, barSpacing: 7, tickMarkFormatter: tickLabel },
      crosshair: { mode: CrosshairMode.Normal, vertLine: { color: "#566377", labelBackgroundColor: "#263244" },
        horzLine: { color: "#566377", labelBackgroundColor: "#263244" } },
    });
    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "#27d17f", downColor: "#ff5263", borderUpColor: "#27d17f", borderDownColor: "#ff5263",
      wickUpColor: "#27d17f", wickDownColor: "#ff5263", priceLineColor: "#ffb000",
    });
    const volume = chart.addSeries(HistogramSeries, { priceScaleId: "vol", priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false });
    chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    const vwap = chart.addSeries(LineSeries, { color: "#45d3f2", lineWidth: 1, priceLineVisible: false, lastValueVisible: true, title: "VWAP" });
    const lineOpts = { lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false } as const;
    const ema20 = chart.addSeries(LineSeries, { ...lineOpts, color: "#ffb000" });
    const ema50 = chart.addSeries(LineSeries, { ...lineOpts, color: "#a78bfa" });
    const zones = new ZonesPrimitive();
    candles.attachPrimitive(zones);
    const markers = createSeriesMarkers(candles, []);
    ref.current = { chart, candles, volume, vwap, ema20, ema50, zones, markers, lines: [] };
    chart.subscribeCrosshairMove((p) => {
      const d = p.seriesData.get(candles) as { open: number; high: number; low: number; close: number } | undefined;
      const v = p.seriesData.get(volume) as { value: number } | undefined;
      setHover(d ? { o: d.open, h: d.high, l: d.low, c: d.close, v: v?.value ?? 0 } : null);
    });
    return () => {
      chart.remove();
      ref.current = null;
    };
  }, []);

  // load data
  useEffect(() => {
    if (!symbol) return;
    let alive = true;
    const load = () =>
      api<ChartData>(`/api/chart/${encodeURIComponent(symbol)}?tf=${tf}&days=${tf === "1" ? 2 : 4}`)
        .then((d) => {
          if (alive) {
            setData(d);
            setErr(null);
          }
        })
        .catch((e) => alive && setErr(String(e.message ?? e)));
    load();
    const id = window.setInterval(load, tf === "D" ? 60000 : 20000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [symbol, tf]);

  // render data
  useEffect(() => {
    const r = ref.current;
    if (!r || !data) return;
    const bars = data.bars;
    r.candles.setData(bars.map((b) => ({ time: toChartTime(b[0]), open: b[1], high: b[2], low: b[3], close: b[4] })));
    r.volume.setData(bars.map((b) => ({ time: toChartTime(b[0]), value: b[5], color: b[4] >= b[1] ? "rgba(39,209,127,.35)" : "rgba(255,82,99,.35)" })));
    r.vwap.setData(tf === "D" ? [] : data.vwap.map(([t, v]) => ({ time: toChartTime(t), value: v })));
    if (tf === "D") {
      const closes = bars.map((b) => b[4]);
      const line = (vals: (number | null)[]) =>
        vals.flatMap((v, i) => (v === null ? [] : [{ time: toChartTime(bars[i][0]), value: v }]));
      r.ema20.setData(line(ema(closes, 20)));
      r.ema50.setData(line(ema(closes, 50)));
    } else {
      r.ema20.setData([]);
      r.ema50.setData([]);
    }
    const last = bars[bars.length - 1];
    lastBar.current = last ? { t: last[0], o: last[1], h: last[2], l: last[3], c: last[4], v: last[5] } : null;
    lastVol.current = null;
    // key levels
    r.lines.forEach((l) => r.candles.removePriceLine(l));
    r.lines = [];
    const lv = { ...data.levels, ...(props.extraLevels ?? {}) };
    const spec: [string, string, string, LineStyle][] = [
      ["pdh", "PDH", "#a78bfa", LineStyle.Dashed], ["pdl", "PDL", "#a78bfa", LineStyle.Dashed],
      ["pdc", "PDC", "#7a889d", LineStyle.Dotted], ["orh", "ORH", "#ffb000", LineStyle.Solid],
      ["orl", "ORL", "#ffb000", LineStyle.Solid], ["entry", "ENTRY", "#45d3f2", LineStyle.Solid],
      ["stop", "STOP", "#ff5263", LineStyle.Dashed], ["target", "TGT", "#27d17f", LineStyle.Dashed],
    ];
    for (const [k, title, color, style] of spec) {
      const v = (lv as Record<string, number | null | undefined>)[k];
      // daily bars: only the trade plan; prior-day / opening-range levels belong to intraday frames
      if (v && (tf !== "D" || ["entry", "stop", "target"].includes(k))) {
        r.lines.push(r.candles.createPriceLine({ price: v, color, lineWidth: 1, lineStyle: style, axisLabelVisible: true, title }));
      }
    }
    // SMC overlays
    if (showSmc && data.smc) {
      const firstT = bars.length ? bars[0][0] : 0;
      const z: ZoneDraw[] = [];
      for (const g of data.smc.fvgs.filter((g) => !g.filled).slice(-12)) {
        if (g.ts < firstT) continue;
        z.push({ t1: toChartTime(g.ts), p1: g.top, p2: g.bottom, label: g.dir > 0 ? "FVG+" : "FVG-",
          fill: g.dir > 0 ? "rgba(39,209,127,.10)" : "rgba(255,82,99,.10)", stroke: g.dir > 0 ? "rgba(39,209,127,.55)" : "rgba(255,82,99,.55)" });
      }
      for (const ob of data.smc.obs.slice(-6)) {
        if (ob.ts < firstT) continue;
        z.push({ t1: toChartTime(ob.ts), p1: ob.top, p2: ob.bottom, label: ob.dir > 0 ? "OB+" : "OB-",
          fill: "rgba(69,211,242,.08)", stroke: "rgba(69,211,242,.6)" });
      }
      r.zones.set(z);
      const mk: SeriesMarker<Time>[] = data.smc.events
        .filter((e) => e.ts >= firstT)
        .slice(-14)
        .map((e) => ({
          time: toChartTime(e.ts), position: e.dir > 0 ? "belowBar" : "aboveBar",
          color: e.kind === "CHoCH" ? "#ffb000" : e.dir > 0 ? "#27d17f" : "#ff5263",
          shape: e.dir > 0 ? "arrowUp" : "arrowDown", text: e.kind,
        }));
      const sw = data.smc.sweeps.filter((s) => s.ts >= firstT).slice(-6).map((s) => ({
        time: toChartTime(s.ts), position: s.dir > 0 ? "belowBar" : "aboveBar", color: "#a78bfa", shape: "circle", text: "SWEEP",
      })) as SeriesMarker<Time>[];
      r.markers.setMarkers([...mk, ...sw].sort((a, b) => (a.time as number) - (b.time as number)));
    } else {
      r.zones.set([]);
      r.markers.setMarkers([]);
    }
    if (bars.length) {
      const n = bars.length;
      const show = tf === "D" ? 140 : tf === "1" ? 240 : 160;
      r.chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - show), to: n + 4 });
    }
  }, [data, showSmc, tf, props.extraLevels]);

  // live updates of the last candle from the quotes stream
  useEffect(() => {
    if (!sid) return;
    return useLive.subscribe((s, prev) => {
      if (s.quotes === prev.quotes || !s.quotes) return;
      const q = s.quotes.q[String(sid)];
      const r = ref.current;
      const lb = lastBar.current;
      if (!q || !r || !lb || !data) return;
      const [ltp, , vol] = q;
      const bucket = tf === "D" ? 86400 : Number(tf) * 60;
      const ts = s.quotes.ts;
      const tStart = tf === "D" ? lb.t : Math.floor((ts - 9 * 3600 - 15 * 60 + 19800) / bucket) * bucket + 9 * 3600 + 15 * 60 - 19800;
      const dv = lastVol.current === null ? 0 : Math.max(0, vol - lastVol.current);
      lastVol.current = vol;
      if (tf !== "D" && tStart > lb.t) {
        lastBar.current = { t: tStart, o: ltp, h: ltp, l: ltp, c: ltp, v: dv };
      } else {
        lb.h = Math.max(lb.h, ltp);
        lb.l = Math.min(lb.l, ltp);
        lb.c = ltp;
        lb.v += dv;
      }
      const b = lastBar.current!;
      r.candles.update({ time: toChartTime(b.t), open: b.o, high: b.h, low: b.l, close: b.c });
      r.volume.update({ time: toChartTime(b.t), value: b.v, color: b.c >= b.o ? "rgba(39,209,127,.35)" : "rgba(255,82,99,.35)" });
    });
  }, [sid, tf, data]);

  const lb = lastBar.current;
  const legend = hover ?? (lb ? { o: lb.o, h: lb.h, l: lb.l, c: lb.c, v: lb.v } : null);
  const tfs = useMemo(() => [...TFS[mode]], [mode]);
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ display: "flex", gap: 6, padding: "4px 6px", borderBottom: "1px solid var(--border)", alignItems: "center", flexWrap: "wrap" }}>
        <Seg label="Timeframe" value={tf} options={tfs as { v: Tf; label: string }[]} onChange={setTf} />
        <button type="button" className="btn ghost" aria-pressed={showSmc} onClick={() => setShowSmc((v) => !v)}
          style={{ height: 22, color: showSmc ? "var(--amber)" : undefined }} title="Toggle Smart Money Concepts overlays">
          SMC {showSmc ? "ON" : "OFF"}
        </button>
        <span className="tiny muted mono" style={{ marginLeft: "auto" }}>
          {data?.smc ? `structure ${data.smc.trend > 0 ? "▲ bullish" : data.smc.trend < 0 ? "▼ bearish" : "• neutral"}` : ""}
        </span>
      </div>
      <div style={{ position: "relative", flex: 1, minHeight: 0 }}>
        <div className="chart-wrap" ref={box} aria-label={`${symbol} price chart`} role="img" />
        <div className="chart-legend">
          <span className="amber b">{symbol}</span>
          {legend && (
            <>
              <span>O <b>{fx(legend.o)}</b></span>
              <span>H <b>{fx(legend.h)}</b></span>
              <span>L <b>{fx(legend.l)}</b></span>
              <span>C <b className={legend.c >= legend.o ? "up" : "down"}>{fx(legend.c)}</b></span>
            </>
          )}
          {tf === "D" ? (
            <>
              <span className="legend-key"><i style={{ background: "#ffb000" }} />EMA20</span>
              <span className="legend-key"><i style={{ background: "#a78bfa" }} />EMA50</span>
            </>
          ) : (
            <>
              <span className="legend-key"><i style={{ background: "#45d3f2" }} />VWAP</span>
              <span className="legend-key"><i style={{ background: "#ffb000" }} />OR</span>
              <span className="legend-key"><i style={{ background: "#a78bfa" }} />PDH/PDL</span>
            </>
          )}
        </div>
        {err && <div className="empty" style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }}>{err}</div>}
      </div>
    </div>
  );
}
