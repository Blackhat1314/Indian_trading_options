/* Basic STOCK page: everything about one stock in plain words — price chart, business quality, trend, what the
 * tested model thinks, today's action, key numbers, pros and cons, and how many shares a planned risk allows. */
import { useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { research } from "../lib/research";
import type { Fundamentals, NeuralResult, QuantResult } from "../lib/research";
import { useLive } from "../lib/store";
import { Badge, Card, Change, Empty, Tip, errText, ok, rupees, useLoad } from "./ui";
import { SETUP, todaySentences } from "./words";

const RANGES: { k: string; days: number }[] = [
  { k: "1M", days: 31 }, { k: "3M", days: 92 }, { k: "6M", days: 183 }, { k: "1Y", days: 366 }, { k: "3Y", days: 1096 }, { k: "5Y", days: 1830 },
];
const PART: Record<string, string> = {
  quality: "Business quality", growth: "Growth", valuation: "Price vs value", health: "Financial health", ownership: "Who owns it",
};
const RATIO_TIP: [RegExp, string][] = [
  [/market cap/i, "The company's total market value: share price × number of shares."],
  [/p\/?e|price to earn/i, "Price ÷ one year's profit per share. Higher = the market expects more growth (or it is expensive)."],
  [/roce/i, "Profit as a % of all the capital the business uses. Above ~15% is usually good."],
  [/roe/i, "Profit as a % of shareholders' money. Above ~15% is usually good."],
  [/dividend/i, "Yearly dividend as a % of today's price."],
  [/book value/i, "Net assets per share. Price far above book = the market values its future, not its assets."],
  [/high|low/i, "The highest and lowest prices of the last 52 weeks."],
  [/face value/i, "The nominal value printed on the share — not related to its market price."],
];

function AreaChart({ closes }: { closes: [number, number][] }) {
  const [range, setRange] = useState("1Y");
  const [hover, setHover] = useState<number | null>(null);
  const box = useRef<SVGSVGElement>(null);
  const pts = useMemo(() => {
    if (!closes.length) return [];
    const last = closes[closes.length - 1][0];
    const days = RANGES.find((r) => r.k === range)?.days ?? 366;
    return closes.filter(([t]) => t >= last - days * 86400);
  }, [closes, range]);
  const W = 800, H = 240, P = 6;
  if (pts.length < 2) return <Empty>Not enough price history yet.</Empty>;
  const vals = pts.map((p) => p[1]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const x = (i: number) => P + (i / (pts.length - 1)) * (W - 2 * P);
  const y = (v: number) => P + (1 - (v - lo) / (hi - lo || 1)) * (H - 2 * P);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
  const area = `${line}L${x(pts.length - 1).toFixed(1)},${H}L${x(0).toFixed(1)},${H}Z`;
  const ret = (vals[vals.length - 1] / vals[0] - 1) * 100;
  const up = ret >= 0;
  const hv = hover !== null ? pts[hover] : null;
  const date = (t: number) => new Date(t * 1000).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric", timeZone: "Asia/Kolkata" });
  const onMove = (e: React.PointerEvent) => {
    const r = box.current!.getBoundingClientRect();
    const i = Math.round(((e.clientX - r.left) / r.width * W - P) / (W - 2 * P) * (pts.length - 1));
    setHover(Math.max(0, Math.min(pts.length - 1, i)));
  };
  return (
    <div className="b-chart">
      <div className="b-chart-bar">
        <div className="b-seg" role="group" aria-label="Chart range">
          {RANGES.map((r) => <button type="button" key={r.k} aria-pressed={range === r.k} onClick={() => setRange(r.k)}>{r.k}</button>)}
        </div>
        <span className="b-chart-ret">{hv ? <>{date(hv[0])}: <b>{rupees(hv[1])}</b></> : <>{range} return <Change v={ret} /></>}</span>
      </div>
      <svg ref={box} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className={up ? "up" : "down"} role="img"
        aria-label={`Price over ${range}: ${up ? "up" : "down"} ${Math.abs(ret).toFixed(1)}%`}
        onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        <defs>
          <linearGradient id="b-fill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="currentColor" stopOpacity="0.18" /><stop offset="1" stopColor="currentColor" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={area} fill="url(#b-fill)" />
        <path d={line} fill="none" stroke="currentColor" strokeWidth="1.8" vectorEffect="non-scaling-stroke" />
        {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={H} className="b-cross" vectorEffect="non-scaling-stroke" />}
      </svg>
      <div className="b-chart-axis"><span>{date(pts[0][0])}</span><span>low {rupees(lo)} · high {rupees(hi)}</span><span>{date(pts[pts.length - 1][0])}</span></div>
    </div>
  );
}

function Quality({ f, loading, error }: { f: Fundamentals | null; loading: boolean; error: unknown }) {
  if (!f) return <Card title="Business quality" id="q">{loading ? <Empty>Loading fundamentals…</Empty> : <Empty>Fundamentals unavailable{error ? `: ${errText(error)}` : ""}.</Empty>}</Card>;
  const s = f.scorecard;
  const pct = s.max ? Math.round((s.total / s.max) * 100) : 0;
  const tone = pct >= 65 ? "good" : pct >= 45 ? "mid" : "weak";
  return (
    <Card title="Business quality" id="q" sub="From the latest results and balance sheet">
      <div className="b-score">
        <span className={`b-score-num ${tone}`}>{pct}<small>/100</small></span>
        <span className="b-score-word">{pct >= 65 ? "A strong business on paper" : pct >= 45 ? "Average on paper" : "Weak on paper"}<small>grade {s.grade}</small></span>
      </div>
      <ul className="b-parts">
        {s.parts.map((p) => (
          <li key={p.key} title={p.checks.map((c) => `${c.pass ? "✓" : c.pass === false ? "✗" : "·"} ${c.label}: ${c.value}`).join("\n")}>
            <span>{PART[p.key] ?? p.label}</span>
            <span className="b-part-track"><i style={{ width: `${(p.score / (p.max || 1)) * 100}%` }} /></span>
            <b>{p.score}/{p.max}</b>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function Trend({ q, closes }: { q: QuantResult | null; closes: [number, number][] }) {
  // plain words only: moving averages and the 52-week range from the closes, plus the quant desk's
  // regime and volatility translated (its own readings name the models, which a beginner doesn't need)
  const lines = useMemo(() => {
    const v = closes.map((c) => c[1]);
    if (v.length < 60) return [];
    const last = v[v.length - 1];
    const sma = (n: number) => v.slice(-n).reduce((a, b) => a + b, 0) / Math.min(n, v.length);
    const out: { text: string; tone: "up" | "down" | "neutral" }[] = [];
    if (v.length >= 200) out.push(last > sma(200)
      ? { text: "Above its 200-day average price — the long-term trend is up.", tone: "up" }
      : { text: "Below its 200-day average price — the long-term trend is down.", tone: "down" });
    out.push(last > sma(50)
      ? { text: "Above its 50-day average — buyers have had the upper hand for the last couple of months.", tone: "up" }
      : { text: "Below its 50-day average — sellers have had the upper hand for the last couple of months.", tone: "down" });
    const yr = v.length > 252 ? (last / v[v.length - 253] - 1) * 100 : null;
    if (yr !== null) out.push({ text: `${yr >= 0 ? "Up" : "Down"} ${Math.abs(yr).toFixed(0)}% over the last year.`, tone: yr >= 0 ? "up" : "down" });
    const hi = Math.max(...v.slice(-252)), lo = Math.min(...v.slice(-252));
    const off = (1 - last / hi) * 100;
    out.push(off < 3
      ? { text: "Trading near its 52-week high.", tone: "up" }
      : { text: `${off.toFixed(0)}% below its 52-week high of ${rupees(hi)} (low ${rupees(lo)}).`, tone: off > 25 ? "down" : "neutral" });
    const r = q?.regime;
    if (r) out.push(r.state === "bull"
      ? { text: "Its recent day-to-day behaviour looks like a rising phase.", tone: "up" }
      : r.state === "bear" ? { text: "Its recent day-to-day behaviour looks like a falling phase.", tone: "down" }
        : { text: "Its recent day-to-day behaviour looks directionless.", tone: "neutral" });
    const vo = q?.vol;
    if (vo && ok(vo.percentile)) out.push(vo.percentile > 0.8
      ? { text: "It is swinging more than usual right now — use a wider stop and fewer shares.", tone: "down" }
      : vo.percentile < 0.2 ? { text: "It is unusually calm right now.", tone: "neutral" }
        : { text: "It is moving about as much as it usually does.", tone: "neutral" });
    return out;
  }, [q, closes]);
  return (
    <Card title="Price trend" id="t" sub="What the chart says, in words">
      {!lines.length ? <Empty>Not enough history yet.</Empty> : (
        <ul className="b-readings">{lines.map((l, i) => <li key={i} className={l.tone}><i aria-hidden="true" />{l.text}</li>)}</ul>
      )}
    </Card>
  );
}

function ModelView({ n }: { n: NeuralResult | null }) {
  const hz = (n?.horizons ?? []).filter((h) => h.h === 5 || h.h === 21);
  const usable = n?.status === "ready" && hz.some((h) => h.rank_edge || h.edge);
  return (
    <Card title="What our model thinks" id="m" sub="The neural model, judged on 4 years it never trained on">
      {!n || n.status !== "ready" ? <Empty>{n?.status === "computing" ? "The model is scoring stocks — try again in a minute." : "No model view for this stock."}</Empty> : !usable ? (
        <p className="b-lead">No reliable view right now. For this horizon the model's tests showed no edge, so it doesn't rank stocks — that's a feature, not a bug.</p>
      ) : (
        <ul className="b-model">
          {hz.map((h) => {
            const top = ok(h.rank_pct) ? Math.max(1, Math.round(100 - h.rank_pct)) : null;
            const has = h.rank_edge || h.edge;
            return (
              <li key={h.h} className={has ? "" : "off"}>
                <span className="b-model-h">{h.h === 21 ? "Next month" : "Next week"}</span>
                {has && top !== null ? (
                  <span className="b-model-text">{top <= 50 ? <>In the <b>top {top}%</b> of stocks</> : <>In the <b>bottom {100 - top + 1}%</b> of stocks</>}
                    <small>{h.rank_edge && ok(h.p_beat) ? `${Math.round(h.p_beat * 100)}% chance to beat the average stock` : ok(h.p_up) ? `${Math.round(h.p_up * 100)}% chance to be higher` : ""}</small></span>
                ) : <span className="b-model-text muted">No reliable view at this horizon</span>}
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}

function RiskCalc({ price, stopHint }: { price: number | null; stopHint: number | null }) {
  const [capital, setCapital] = useState(100000);
  const [riskPct, setRiskPct] = useState(1);
  const [entry, setEntry] = useState<number | "">("");
  const [stop, setStop] = useState<number | "">("");
  const e = entry === "" ? price ?? 0 : entry;
  const s = stop === "" ? stopHint ?? e * 0.97 : stop;
  const perShare = e - s;
  const riskMoney = capital * riskPct / 100;
  const shares = perShare > 0 ? Math.floor(riskMoney / perShare) : 0;
  const cost = shares * e;
  const maxAfford = e > 0 ? Math.floor(capital / e) : 0;
  const n = (v: string) => (v === "" ? "" : Math.max(0, Number(v)));
  return (
    <Card title="How many shares?" id="calc" sub="Size the position so a stop-loss costs only what you planned">
      <div className="b-calc">
        <label>Your capital (₹)<input inputMode="decimal" value={capital} onChange={(ev) => setCapital(Number(n(ev.target.value)) || 0)} /></label>
        <label>Risk per trade (%)<input inputMode="decimal" value={riskPct} onChange={(ev) => setRiskPct(Math.min(10, Number(n(ev.target.value)) || 0))} /></label>
        <label>Buy at (₹)<input inputMode="decimal" placeholder={ok(price) ? price.toFixed(2) : ""} value={entry} onChange={(ev) => setEntry(n(ev.target.value) as number | "")} /></label>
        <label>Stop-loss at (₹)<input inputMode="decimal" placeholder={ok(s) ? s.toFixed(2) : ""} value={stop} onChange={(ev) => setStop(n(ev.target.value) as number | "")} /></label>
      </div>
      {perShare <= 0 ? <p className="b-note warn">The stop must be below the buy price.</p> : (
        <div className="b-calc-out">
          <p className="b-lead">Buy <b>{shares.toLocaleString("en-IN")} shares</b> for about <b>{rupees(cost, true)}</b>. If the stop at {rupees(s)} is hit, you lose about <b>{rupees(shares * perShare, true)}</b> ({riskPct}% of your capital).</p>
          {cost > capital && <p className="b-note warn">That needs more than your capital: with {rupees(capital, true)} you can buy at most {maxAfford.toLocaleString("en-IN")} shares — or choose a stock with a closer stop.</p>}
          <p className="b-note">The stop is {((perShare / e) * 100).toFixed(1)}% below the buy price{stopHint && stop === "" ? " (the setup's own stop)" : stop === "" ? " (a default 3% — set your own)" : ""}. Charges and slippage add a little to every trade.</p>
        </div>
      )}
    </Card>
  );
}

export default function StockPage() {
  const sym = useLive((s) => s.researchSymbol);
  const rank = useLive((s) => s.rank);
  const swing = useLive((s) => s.swing);
  const setMode = useLive((s) => s.setMode);
  const setDesk = useLive((s) => s.setDesk);
  const hist = useLoad(() => research.history(sym), [sym]);
  const fun = useLoad(() => research.screener(sym), [sym]);
  const nn = useLoad(() => research.neural(sym).catch(() => null), [sym]);
  const qt = useLoad(() => research.quant(sym).catch(() => null), [sym]);
  const row = rank?.rows.find((r) => r.symbol === sym) ?? null;
  const sw = swing?.rows.find((r) => r.symbol === sym) ?? null;
  const h = hist.data;
  const closes = h?.closes ?? [];
  const price = row?.ltp ?? h?.spot ?? (closes.length ? closes[closes.length - 1][1] : null);
  const chg = row?.chg ?? (closes.length > 1 ? (closes[closes.length - 1][1] / closes[closes.length - 2][1] - 1) * 100 : null);
  const name = fun.data?.name ?? h?.name ?? sym;
  const today = row ? todaySentences(row) : [];
  const setup = sw?.setup ? SETUP[sw.setup] : null;
  const stats = sw?.setup ? swing?.stats?.[sw.setup] : null;
  // the header already shows the live price; Screener's "Current Price" is from its last page fetch
  const ratios = (fun.data?.ratios ?? []).filter((r) => r.text && !/^current price$/i.test(r.label ?? "")).slice(0, 9);
  const tipFor = (label: string) => RATIO_TIP.find(([re]) => re.test(label))?.[1];
  const openPro = () => { setMode("pro"); setDesk("ai"); };
  return (
    <div className="b-stock">
      <section className="b-stock-head">
        <Badge symbol={sym} size={48} />
        <div className="b-stock-title">
          <h1>{name}</h1>
          <p>{sym}{h?.sector ? ` · ${h.sector}` : fun.data?.sector ? ` · ${fun.data.sector}` : ""}{row ? " · live" : h?.last ? ` · last close ${h.last}` : ""}</p>
        </div>
        <div className="b-stock-price">
          <span>{rupees(price)}</span>
          <Change v={chg} />
        </div>
        <button type="button" className="b-btn" onClick={openPro} title="AI analyst, Monte Carlo, Quant and more in the Pro view">Deeper research</button>
      </section>
      {!!hist.error && !h && <p className="b-note warn">Couldn't load {sym}: {errText(hist.error)}</p>}
      <Card id="chart">{hist.loading && !h ? <Empty>Loading price history…</Empty> : <AreaChart closes={closes} />}</Card>
      <div className="b-grid three">
        <Quality f={fun.data} loading={fun.loading} error={fun.error} />
        <Trend q={qt.data} closes={closes} />
        <ModelView n={nn.data} />
      </div>
      {(today.length > 0 || setup) && (
        <Card title={row ? "Today" : "Setup"} id="today" sub={row ? "Live, from the market feed" : undefined}>
          <ul className="b-readings">
            {today.map((t, i) => <li key={i} className="neutral"><i aria-hidden="true" />{t}</li>)}
            {setup && <li className="up"><i aria-hidden="true" /><span><b>Swing setup — {setup.name}:</b> {setup.text}.
              {stats?.n ? ` Historically +${stats.target_pct.toFixed(0)}% came before −${stats.stop_pct.toFixed(0)}% in ${stats.hit_rate.toFixed(0)}% of ${stats.n.toLocaleString("en-IN")} cases.` : ""}
              {ok(sw?.stop) && ok(sw?.target) ? ` Its stop: ${rupees(sw!.stop)}, target: ${rupees(sw!.target)}.` : ""}</span></li>}
          </ul>
        </Card>
      )}
      <div className="b-grid two">
        <Card title="Key numbers" id="nums" sub={fun.data ? `From Screener.in (${fun.data.view})` : undefined}>
          {!ratios.length ? <Empty>{fun.loading ? "Loading…" : "Not available for this stock."}</Empty> : (
            <dl className="b-ratios">
              {ratios.map((r) => {
                const tip = tipFor(r.label);
                return (
                  <div key={r.key}><dt>{r.label}{tip && <Tip text={tip} />}</dt><dd>{r.text}</dd></div>
                );
              })}
            </dl>
          )}
          {fun.data && (fun.data.pros.length > 0 || fun.data.cons.length > 0) && (
            <div className="b-proscons">
              <div><h3>Strengths</h3><ul>{fun.data.pros.slice(0, 4).map((p, i) => <li key={i}>{p}</li>)}</ul></div>
              <div><h3>Concerns</h3><ul>{fun.data.cons.slice(0, 4).map((p, i) => <li key={i}>{p}</li>)}</ul></div>
            </div>
          )}
        </Card>
        <RiskCalc price={ok(price) ? price : null} stopHint={ok(sw?.stop) && sw!.setup !== "BREAKDOWN" ? sw!.stop! : null} />
      </div>
      {fun.data?.about && <Card title="About the company" id="about"><p className="b-about" style={{ "--lines": 5 } as CSSProperties}>{fun.data.about}</p></Card>}
    </div>
  );
}
