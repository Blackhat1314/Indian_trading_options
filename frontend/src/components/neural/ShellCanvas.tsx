/* The neural shell: every scored stock as a cluster of particles on a slowly turning torus.
 *   position  stocks grouped by sector around the ring (sector names on the rim)
 *   colour    the network's probability vs the average stock — cyan above, red below, steel near it
 *   spike     how far the network pushes the stock away from the average (no edge → a flat shell)
 *   amber / magenta spikes: today's 09:30 neural long / short picks; white: the selected stock
 * Drawn on a 2D canvas with a hand-rolled perspective projection (a few thousand points, no WebGL needed).
 * Stops when off screen or in a background tab; a single still frame under prefers-reduced-motion. */
import { useEffect, useMemo, useRef, useState } from "react";

export interface ShellStock { symbol: string; sector: string; p: number }

interface Props {
  stocks: ShellStock[];
  base: number;                  // the average probability (calibration base rate)
  flat: boolean;                 // no edge at this horizon: draw the probabilities but say so
  selected?: string;
  longs?: Set<string>;
  shorts?: Set<string>;
  onPick?: (symbol: string) => void;
  label: string;                 // what p is, for the tooltip ("P(beat)")
}

const CYAN = [69, 211, 242];
const RED = [255, 82, 99];
const STEEL = [118, 140, 170];
const AMBER = [255, 176, 0];
const MAGENTA = [255, 92, 190];

function mix(a: number[], b: number[], t: number): number[] {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

interface Particle { s: number; th: number; ph: number }
interface Geo {
  order: ShellStock[];
  theta: Float32Array;           // major angle of each stock
  rgb: number[][];
  push: Float32Array;            // −1 … +1: signed distance from the average, scaled
  particles: Particle[];
  sectors: { name: string; th: number }[];
}

function build(stocks: ShellStock[], base: number): Geo {
  const bySec = new Map<string, ShellStock[]>();
  for (const s of stocks) {
    const l = bySec.get(s.sector) ?? [];
    l.push(s);
    bySec.set(s.sector, l);
  }
  const secs = [...bySec.entries()].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
  const order: ShellStock[] = [];
  const sectors: { name: string; th: number }[] = [];
  const gap = secs.length > 1 ? 0.6 : 0;                 // a small gap between sectors, in stock slots
  const slots = stocks.length + gap * secs.length;
  let pos = 0;
  const theta = new Float32Array(stocks.length);
  for (const [name, l] of secs) {
    l.sort((a, b) => a.p - b.p);
    const start = pos;
    for (const s of l) {
      theta[order.length] = ((pos + 0.5) / slots) * Math.PI * 2;
      order.push(s);
      pos += 1;
    }
    sectors.push({ name, th: (((start + pos) / 2) / slots) * Math.PI * 2 });
    pos += gap;
  }
  const dev = order.map((s) => s.p - base);
  const sorted = dev.map(Math.abs).sort((a, b) => a - b);
  const spread = Math.max(0.015, sorted[Math.floor(sorted.length * 0.95)] ?? 0.015);
  const push = new Float32Array(order.length);
  const rgb = order.map((_, i) => {
    const d = Math.max(-1, Math.min(1, dev[i] / spread));
    push[i] = d;
    return d >= 0 ? mix(STEEL, CYAN, d) : mix(STEEL, RED, -d);
  });
  const per = Math.max(14, Math.min(90, Math.round(6500 / Math.max(order.length, 1))));
  const particles: Particle[] = [];
  let seed = 7;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const slot = (Math.PI * 2) / slots;
  order.forEach((_, i) => {
    for (let k = 0; k < per; k++) {
      particles.push({ s: i, th: theta[i] + (rnd() - 0.5) * slot * 0.9, ph: (k / per) * Math.PI * 2 + rnd() * 0.3 });
    }
  });
  return { order, theta, rgb, push, particles, sectors };
}

export function ShellCanvas({ stocks, base, flat, selected, longs, shorts, onPick, label }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const geo = useMemo(() => build(stocks, base), [stocks, base]);
  const [tip, setTip] = useState<{ x: number; y: number; s: ShellStock } | null>(null);
  const hot = useRef<{ x: number; y: number; i: number }[]>([]);
  const mouse = useRef<{ x: number; y: number } | null>(null);
  const stateRef = useRef({ selected, longs, shorts, flat });
  stateRef.current = { selected, longs, shorts, flat };

  useEffect(() => {
    const cv = canvas.current;
    const host = wrap.current;
    if (!cv || !host) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    let w = 0, h = 0, dpr = 1, raf = 0, visible = true, angle = 0.6, last = performance.now();

    const size = () => {
      const r = host.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = Math.max(200, r.width);
      h = Math.max(200, r.height);
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
      cv.style.width = `${w}px`;
      cv.style.height = `${h}px`;
    };

    const draw = () => {
      const { selected: sel, longs: L, shorts: S, flat: isFlat } = stateRef.current;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = "lighter";          // overlapping points glow instead of hiding each other
      const R = Math.min(w * 0.27, h * 0.42);           // major radius
      const r = R * 0.42;                              // tube radius
      const cx = w / 2, cy = h * 0.48;
      const tilt = 0.55;                               // the camera sits ~30° above the ring's plane
      const ct = Math.cos(tilt), st = Math.sin(tilt);
      const ca = Math.cos(angle), sa = Math.sin(angle);
      const D = R * 4.2;
      const project = (x: number, y: number, z: number) => {
        const x1 = x * ca - z * sa, z1 = x * sa + z * ca;          // turn around the vertical axis
        const y2 = y * ct - z1 * st, z2 = y * st + z1 * ct;        // tilt toward the viewer
        const f = D / (D + z2);
        return { X: cx + x1 * f, Y: cy + y2 * f, depth: z2, f };
      };
      const needle = R * 0.42 * (isFlat ? 0.25 : 1);

      // particles: back half first is not needed — depth sets their brightness and size
      for (const pt of geo.particles) {
        const i = pt.s;
        const ring = R + r * Math.cos(pt.ph);
        const x = ring * Math.cos(pt.th), z = ring * Math.sin(pt.th), y = r * Math.sin(pt.ph);
        const q = project(x, y, z);
        const near = 0.5 - q.depth / (2 * (R + r));
        const c = geo.rgb[i];
        const a = (0.22 + 0.7 * Math.max(0, Math.min(1, near))) * (geo.order[i].symbol === sel ? 1 : 0.9);
        const sz = (1.1 + 1.6 * near) * q.f;
        ctx.fillStyle = geo.order[i].symbol === sel ? `rgba(255,255,255,${a})` : `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;
        ctx.fillRect(q.X - sz / 2, q.Y - sz / 2, sz, sz);
      }

      ctx.globalCompositeOperation = "source-over";
      // spikes from the outer rim: length = how far the network pushes the stock from the average
      const hits: { x: number; y: number; i: number }[] = [];
      geo.order.forEach((s, i) => {
        const th = geo.theta[i];
        const isL = L?.has(s.symbol), isS = S?.has(s.symbol), isSel = s.symbol === sel;
        const len = Math.abs(geo.push[i]) * needle + (isL || isS || isSel ? R * 0.12 : 0);
        const b0 = project((R + r) * Math.cos(th), 0, (R + r) * Math.sin(th));
        hits.push({ x: b0.X, y: b0.Y, i });
        if (len < 1.5) return;
        const e = project((R + r + len) * Math.cos(th), -len * 0.35, (R + r + len) * Math.sin(th));
        const near = 0.5 - b0.depth / (2 * (R + r));
        const c = isL ? AMBER : isS ? MAGENTA : isSel ? [255, 255, 255] : geo.rgb[i];
        const grad = ctx.createLinearGradient(b0.X, b0.Y, e.X, e.Y);
        grad.addColorStop(0, `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${0.15 + 0.6 * near})`);
        grad.addColorStop(1, `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},0)`);
        ctx.strokeStyle = grad;
        ctx.lineWidth = isL || isS || isSel ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(b0.X, b0.Y);
        ctx.lineTo(e.X, e.Y);
        ctx.stroke();
        if (isL || isS || isSel) {
          ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.55 + 0.45 * near})`;
          ctx.beginPath();
          ctx.arc(e.X, e.Y, 2.4, 0, Math.PI * 2);
          ctx.fill();
          if (near > 0.35) {
            ctx.font = "600 10.5px 'Fira Code', ui-monospace, monospace";
            const tw = ctx.measureText(s.symbol).width;
            ctx.fillText(s.symbol, Math.min(e.X + 5, w - tw - 4), Math.max(12, e.Y - 4));
          }
        }
      });
      hot.current = hits;

      // sector names on the near half of the rim
      ctx.font = "500 10px 'Fira Sans', system-ui, sans-serif";
      ctx.textAlign = "center";
      for (const sec of geo.sectors) {
        const q = project((R + r * 1.9) * Math.cos(sec.th), r * 0.2, (R + r * 1.9) * Math.sin(sec.th));
        const near = 0.5 - q.depth / (2 * (R + r));
        if (near < 0.45) continue;
        ctx.fillStyle = `rgba(170,182,199,${Math.min(0.8, (near - 0.45) * 2.2)})`;
        const half = ctx.measureText(sec.name).width / 2 + 3;
        ctx.fillText(sec.name, Math.max(half, Math.min(w - half, q.X)), Math.min(h - 18, q.Y));
      }
      ctx.textAlign = "start";
    };

    const frame = (t: number) => {
      const dt = Math.min(64, t - last);
      last = t;
      if (!mouse.current) angle += dt * 0.00009;      // one slow turn every ~70 s; holds still under the pointer
      draw();
      raf = visible && !document.hidden ? requestAnimationFrame(frame) : 0;
    };
    const start = () => {
      if (reduce) { draw(); return; }
      if (!raf && visible && !document.hidden) { last = performance.now(); raf = requestAnimationFrame(frame); }
    };
    size();
    start();
    const ro = new ResizeObserver(() => { size(); draw(); });
    ro.observe(host);
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) start(); });
    io.observe(host);
    const vis = () => start();
    document.addEventListener("visibilitychange", vis);
    const redraw = () => reduce && draw();
    window.addEventListener("neural-shell-redraw", redraw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      document.removeEventListener("visibilitychange", vis);
      window.removeEventListener("neural-shell-redraw", redraw);
    };
  }, [geo]);

  useEffect(() => { window.dispatchEvent(new Event("neural-shell-redraw")); }, [selected, longs, shorts, flat]);

  const nearest = (x: number, y: number) => {
    let best = -1, bd = 18 * 18;
    for (const hpt of hot.current) {
      const d = (hpt.x - x) ** 2 + (hpt.y - y) ** 2;
      if (d < bd) { bd = d; best = hpt.i; }
    }
    return best;
  };
  const onMove = (e: React.PointerEvent) => {
    const r = canvas.current!.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    mouse.current = { x, y };
    const i = nearest(x, y);
    setTip(i >= 0 ? { x, y, s: geo.order[i] } : null);
  };
  const above = stocks.filter((s) => s.p > base).length;
  return (
    <div ref={wrap} className="shell-canvas">
      <canvas ref={canvas} role="img"
        aria-label={`${stocks.length} stocks by sector; ${above} above the average ${label}, ${stocks.length - above} below${flat ? "; no edge at this horizon" : ""}`}
        onPointerMove={onMove} onPointerLeave={() => { mouse.current = null; setTip(null); }}
        onClick={() => tip && onPick?.(tip.s.symbol)} style={{ cursor: tip ? "pointer" : "default" }} />
      {tip && (
        <div className="shell-tip" style={{
          left: Math.max(4, Math.min(tip.x + 14, (wrap.current?.clientWidth ?? 240) - 224)), top: tip.y + 12 }}>
          <b>{tip.s.symbol}</b> <span className="muted">{tip.s.sector}</span>
          <div>{label} <b className={tip.s.p > base ? "cyan" : tip.s.p < base ? "down" : undefined}>{(tip.s.p * 100).toFixed(1)}%</b>
            <span className="muted"> vs {(base * 100).toFixed(1)}% average</span></div>
        </div>
      )}
    </div>
  );
}
