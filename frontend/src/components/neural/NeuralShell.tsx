/* NEURAL shell — the network drawn as what it is doing right now.
 * Left: the selected stock's real inputs (40-day paths of six of its technical features, its fundamentals and
 * news / 1-minute context), wired into the shell. Centre: every scored stock (ShellCanvas). Right: what comes out —
 * the stock's forecasts per horizon and today's 09:30 neural picks. Bottom: how the whole universe is spread.
 * Wires glow with the strength of the input they carry; nothing here is decoration without data behind it. */
import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { Seg } from "../ui";
import { fx } from "../../lib/format";
import type { NeuralResult, NeuralUniverse } from "../../lib/research";
import type { NeuralTop } from "../../lib/types";
import { ShellCanvas } from "./ShellCanvas";
import type { ShellStock } from "./ShellCanvas";

type H = "1" | "5" | "21";
const H_SHORT: Record<string, string> = { "1": "1D", "5": "1W", "21": "1M" };
const ok = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const pc = (p: number | null | undefined, d = 0) => (ok(p) ? `${fx(p * 100, d)}%` : "—");

function Trace({ z }: { z: number[] }) {
  const W = 120, Hh = 26, max = 3;
  const pts = z.map((v, i) => `${((i / Math.max(z.length - 1, 1)) * W).toFixed(1)},${(Hh / 2 - (Math.max(-max, Math.min(max, v)) / max) * (Hh / 2 - 2)).toFixed(1)}`).join(" ");
  const lastV = z[z.length - 1] ?? 0;
  return (
    <svg viewBox={`0 0 ${W} ${Hh}`} preserveAspectRatio="none" className="shell-trace" aria-hidden="true">
      <line x1="0" x2={W} y1={Hh / 2} y2={Hh / 2} className="shell-zero" />
      <polyline points={pts} fill="none" stroke={lastV >= 0 ? "var(--cyan)" : "var(--down)"} strokeWidth="1.3" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

interface Wire { id: string; strength: number; tone: "pos" | "neg" | "violet" | "amber" | "mute" }

export function NeuralShell({ h, setH, universe, result, neural, symbol, onPick }: {
  h: H; setH: (h: H) => void; universe: NeuralUniverse | null; result: NeuralResult | null; neural?: NeuralTop;
  symbol: string; onPick: (s: string) => void;
}) {
  const rankEdge = !!universe?.rank_edge;
  const useBeat = (universe?.rows ?? []).some((r) => ok(r.p_beat));
  const label = useBeat ? "P(beat)" : "P(up)";
  const base = (useBeat ? universe?.base_beat : universe?.base_up) ?? 0.5;
  const flat = useBeat ? !rankEdge : !universe?.edge;
  const stocks: ShellStock[] = useMemo(() => (universe?.rows ?? [])
    .map((r) => ({ symbol: r.symbol, sector: r.sector || "OTHER", p: (useBeat ? r.p_beat : r.p_up) ?? NaN }))
    .filter((r) => ok(r.p)), [universe, useBeat]);
  const longs = useMemo(() => new Set((neural?.long ?? []).map((p) => p.symbol)), [neural]);
  const shorts = useMemo(() => new Set((neural?.short ?? []).map((p) => p.symbol)), [neural]);
  const ready = result?.status === "ready";
  const tr = ready ? result?.trace : null;
  const g = ready ? result?.groups : null;
  const known = (g?.fundamentals ?? []).filter((f) => ok(f.v));
  const minute = g?.minute;
  const news = g?.news;

  // ---- wires: measured from the cards to the canvas edge
  const grid = useRef<HTMLDivElement>(null);
  const [paths, setPaths] = useState<{ d: string; w: Wire }[]>([]);
  const wires: Wire[] = [
    ...(tr?.z ?? []).map((z, i) => {
      const v = z[z.length - 1] ?? 0;
      return { id: `in-${i}`, strength: Math.min(1, Math.abs(v) / 2), tone: (v >= 0 ? "pos" : "neg") as Wire["tone"] };
    }),
    ...(g ? [{ id: "in-fund", strength: known.length / Math.max(g.fundamentals.length, 1), tone: "violet" as const },
      { id: "in-ctx", strength: (news?.news_on ? 0.4 : 0) + (minute?.on ? 0.6 : 0), tone: "violet" as const }] : []),
    ...(result?.horizons ?? []).map((hz) => ({ id: `out-${hz.h}`, strength: String(hz.h) === h ? 1 : 0.45,
      tone: (hz.rank_edge || hz.edge ? "pos" : "mute") as Wire["tone"] })),
    { id: "out-0930", strength: neural?.status === "ok" ? 1 : 0.25, tone: neural?.status === "ok" ? "amber" : "mute" },
  ];
  const wireKey = wires.map((w) => `${w.id}:${w.strength.toFixed(2)}:${w.tone}`).join("|");
  useLayoutEffect(() => {
    const el = grid.current;
    if (!el) return;
    const measure = () => {
      const box = el.getBoundingClientRect();
      const core = el.querySelector<HTMLElement>(".shell-core")?.getBoundingClientRect();
      if (!core || core.width < 50) { setPaths([]); return; }
      const cy = core.top - box.top + core.height / 2;
      const out: { d: string; w: Wire }[] = [];
      for (const w of wires) {
        const card = el.querySelector<HTMLElement>(`[data-wire="${w.id}"]`)?.getBoundingClientRect();
        if (!card) continue;
        const y = card.top - box.top + card.height / 2;
        if (w.id.startsWith("in-")) {
          const x0 = card.right - box.left, x1 = core.left - box.left + core.width * 0.22;
          const y1 = cy + (y - cy) * 0.28;
          out.push({ d: `M${x0},${y} C${x0 + 70},${y} ${x1 - 90},${y1} ${x1},${y1}`, w });
        } else {
          const x0 = core.right - box.left - core.width * 0.2, x1 = card.left - box.left;
          const y0 = cy + (y - cy) * 0.28;
          out.push({ d: `M${x0},${y0} C${x0 + 90},${y0} ${x1 - 70},${y} ${x1},${y}`, w });
        }
      }
      setPaths(out);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wireKey, stocks.length]);

  // ---- the universe's spread (bottom strip)
  const hist = useMemo(() => {
    if (!stocks.length) return null;
    const ps = stocks.map((s) => s.p);
    const lo = Math.min(...ps, base - 0.01), hi = Math.max(...ps, base + 0.01);
    const n = 32;
    const bins = new Array(n).fill(0);
    for (const p of ps) bins[Math.min(n - 1, Math.floor(((p - lo) / (hi - lo)) * n))] += 1;
    return { lo, hi, bins, max: Math.max(...bins) };
  }, [stocks, base]);
  const selP = stocks.find((s) => s.symbol === symbol)?.p;
  const at = (p: number) => (hist ? ((p - hist.lo) / (hist.hi - hist.lo)) * 100 : 0);

  const headNote: ReactNode = !universe ? "loading the network's view…"
    : universe.computing ? "scoring every stock…"
      : !stocks.length ? "no scored stocks yet"
        : flat ? <>flat at {H_SHORT[h]}: the network separates no stock from the average here, so the spikes are kept short</>
          : <>{stocks.length} stocks · {label} at {H_SHORT[h]} · spikes show how far the network pushes each stock</>;

  return (
    <section className="panel shell" style={{ gridArea: "shell" }} aria-labelledby="shell-t">
      <header className="panel-head">
        <span className="panel-code">SHELL</span>
        <h2 className="panel-title" id="shell-t" style={{ margin: 0 }}>Neural shell</h2>
        <span className="panel-sub">{headNote}</span>
        <div className="panel-actions">
          <span className={`chip ${flat ? "amber" : "cyan"}`}>{flat ? "NO EDGE" : rankEdge ? "RANKING EDGE" : "EDGE"}</span>
          <Seg label="Shell horizon" value={h} onChange={setH} options={(["1", "5", "21"] as H[]).map((k) => ({ v: k, label: H_SHORT[k] }))} />
        </div>
      </header>
      <div className="shell-grid" ref={grid}>
        <svg className="shell-wires" aria-hidden="true">
          {paths.map(({ d, w }) => (
            <g key={w.id} className={`wire ${w.tone}`} style={{ opacity: 0.25 + 0.75 * w.strength }}>
              <path d={d} className="wire-base" />
              <path d={d} className="wire-flow" style={{ animationDuration: `${(2.6 - 1.6 * w.strength).toFixed(2)}s` }} />
            </g>
          ))}
        </svg>

        <div className="shell-in">
          <div className="shell-col-head">{ready ? <>Inputs · <b>{result!.symbol}</b></> : "Inputs"}</div>
          {!ready && <p className="shell-empty">Pick a stock to see what the network reads for it.</p>}
          {tr?.z.map((z, i) => (
            <div key={tr.keys[i]} className="shell-card" data-wire={`in-${i}`} title={`${tr.labels[i]}: the last 40 days as the network sees them (z-score vs the stock's own history)`}>
              <div className="shell-card-top"><span>{tr.labels[i]}</span><b className={(z[z.length - 1] ?? 0) >= 0 ? "cyan" : "down"}>{(z[z.length - 1] ?? 0) >= 0 ? "+" : ""}{fx(z[z.length - 1] ?? 0, 2)}σ</b></div>
              <Trace z={z} />
            </div>
          ))}
          {g && (
            <div className="shell-card" data-wire="in-fund" title="Fundamentals the network knew for this stock, as ranks among all stocks (−1 worst … +1 best); only numbers public by that day">
              <div className="shell-card-top"><span>Fundamentals</span><b className="violet">{known.length}/{g.fundamentals.length}</b></div>
              <div className="shell-bars">
                {g.fundamentals.map((f) => (
                  <i key={f.key} title={`${f.label}: ${ok(f.v) ? fx(f.v, 2) : "not known"}`}
                    className={ok(f.v) ? undefined : "off"} style={{ height: ok(f.v) ? `${20 + 40 * (f.v + 1)}%` : "8%" }} />
                ))}
              </div>
            </div>
          )}
          {g && (
            <div className="shell-card" data-wire="in-ctx" title="Yesterday's 1-minute summaries and the news the network saw">
              <div className="shell-card-top"><span>News · 1-minute day</span>
                <b className="violet">{news?.news_on ? `${fx(Math.expm1(news.news_n ?? 0), 0)} headlines` : "no archive"}</b></div>
              <div className="shell-card-sub">{minute ? (minute.on ? `1-minute day: ${minute.items.filter((x) => ok(x.v)).length} summaries` : "no 1-minute bars that day") : "model trained without 1-minute inputs"}</div>
            </div>
          )}
        </div>

        <div className="shell-core">
          {stocks.length ? (
            <ShellCanvas stocks={stocks} base={base} flat={flat} selected={symbol} longs={longs} shorts={shorts} onPick={onPick} label={label} />
          ) : <p className="shell-empty center">{headNote}</p>}
          <div className="shell-legend" aria-hidden="true">
            <span><i className="dot pos" /> above average</span><span><i className="dot neg" /> below</span>
            <span><i className="dot amber" /> 09:30 long</span><span><i className="dot magenta" /> 09:30 short</span>
          </div>
        </div>

        <div className="shell-out">
          <div className="shell-col-head">Outputs</div>
          {(result?.horizons ?? []).map((hz) => (
            <button type="button" key={hz.h} className={`shell-card out ${String(hz.h) === h ? "on" : ""}`} data-wire={`out-${hz.h}`}
              onClick={() => setH(String(hz.h) as H)} title={hz.rank_note ?? hz.edge_note ?? hz.verdict}>
              <div className="shell-card-top"><span>{hz.label}</span><span className={`chip ${hz.rank_edge || hz.edge ? "cyan" : "amber"}`}>{hz.rank_edge ? "rank edge" : hz.edge ? "edge" : "no edge"}</span></div>
              <div className="shell-out-vals">
                <span><small>P(up)</small><b>{pc(hz.p_up)}</b></span>
                {ok(hz.p_beat) && <span><small>P(beat)</small><b className={hz.rank_edge ? "cyan" : undefined}>{pc(hz.p_beat)}</b></span>}
              </div>
            </button>
          ))}
          {!result?.horizons && <p className="shell-empty">The forecasts appear here.</p>}
          <div className="shell-card" data-wire="out-0930">
            <div className="shell-card-top"><span>09:30 picks</span>
              <span className={`chip ${neural?.status === "ok" ? "up" : "amber"}`}>{neural?.status === "ok" ? "live" : neural?.status === "no_edge" ? "no edge" : neural?.status ?? "none"}</span></div>
            {neural?.status === "ok" ? (
              <ul className="shell-picks">
                {([["long", neural.long ?? []], ["short", neural.short ?? []]] as const).flatMap(([side, list]) => list.map((p) => (
                  <li key={p.symbol}>
                    <button type="button" onClick={() => onPick(p.symbol)} className={side === "long" ? "amber" : "magenta"}>{side === "long" ? "▲" : "▼"} {p.symbol}</button>
                    <span className={ok(p.pnl) && p.pnl < 0 ? "down" : "up"} title="since 09:30, in the pick's direction">{ok(p.pnl) ? `${p.pnl > 0 ? "+" : ""}${fx(p.pnl, 2)}%` : "—"}</span>
                  </li>
                )))}
              </ul>
            ) : <div className="shell-card-sub">{neural?.status === "pending" ? "building the daily context…" : neural?.verdict ?? "made at the 09:30 lock when the model has an edge"}</div>}
          </div>
        </div>
      </div>

      {hist && (
        <div className="shell-hist" role="img" aria-label={`Spread of ${label} across ${stocks.length} stocks, from ${pc(hist.lo, 1)} to ${pc(hist.hi, 1)}`}>
          <div className="shell-hist-bars">
            {hist.bins.map((c, i) => {
              const mid = hist.lo + ((i + 0.5) / hist.bins.length) * (hist.hi - hist.lo);
              return <i key={i} className={mid > base ? "pos" : "neg"} style={{ height: `${(c / hist.max) * 100}%` }} />;
            })}
            <span className="shell-mark base" style={{ left: `${at(base)}%` }} title={`average ${pc(base, 1)}`} />
            {ok(selP) && <span className="shell-mark sel" style={{ left: `${at(selP)}%` }} title={`${symbol} ${pc(selP, 1)}`} />}
          </div>
          <div className="shell-hist-axis"><span>{pc(hist.lo, 1)}</span><span>{label} across the universe · average {pc(base, 1)}{ok(selP) ? ` · ${symbol} ${pc(selP, 1)}` : ""}</span><span>{pc(hist.hi, 1)}</span></div>
        </div>
      )}
    </section>
  );
}
