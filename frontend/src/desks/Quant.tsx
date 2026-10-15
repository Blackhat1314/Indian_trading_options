/* QUANT desk — "QuantEdge", the next-move model for one stock, with its report card beside every number.
 * A pooled logistic regression on causal price features (trend, momentum, mean reversion, regime, volatility…)
 * trained on every stock in the lake and scored walk-forward. Short-horizon direction is close to a coin flip for
 * liquid stocks, so each forecast carries its measured out-of-sample AUC and the desk says plainly when there is no
 * reliable edge; the components and contributions show why the model leans the way it does. */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { LineChart, ProbBar, fracPct, inr } from "../components/research/charts";
import type { LineBand, LineSeries } from "../components/research/charts";
import { Disclaimer, ErrorNote, Stat, Stats, Warnings } from "../components/research/common";
import { SymbolPicker } from "../components/research/SymbolPicker";
import { Chg, Empty, Panel, Seg, ZBar } from "../components/ui";
import { ago, compact, dateIST, fx, timeIST } from "../lib/format";
import { research } from "../lib/research";
import type {
  QuantComponent, QuantContribution, QuantHorizon, QuantLeader, QuantLeaderboard, QuantModelInfo, QuantResult, Signal,
} from "../lib/research";
import { useLive } from "../lib/store";
import "./quant.css";

const NO_EDGE_AUC = 0.52;                 // below this the walk-forward ranking is indistinguishable from a coin flip
const H_SHORT: Record<number, string> = { 1: "1D", 5: "1W", 21: "1M" };
const hShort = (h: number) => H_SHORT[h] ?? `${h}D`;

const ok = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const pctOf = (p: number | null | undefined, d = 0) => (ok(p) ? `${fx(p * 100, d)}%` : "—");
const frac01 = (v: number) => Math.min(1, Math.max(0, v > 1 ? v / 100 : v));     // tolerate 0–100 as well as 0–1

function Ret({ v, d = 1 }: { v: number | null | undefined; d?: number }) {
  return ok(v) ? <Chg v={v * 100} d={d} /> : <span className="muted">—</span>;
}

/** The gateway also sends its own verdict next to the AUC (edge = AUC ≥ 0.52 AND every walk-forward fold ≥ 0.50
 *  AND calibration did not shrink the model to its base rate); not yet in research.ts, so read defensively. */
type EdgeExtra = { edge?: boolean | null; edge_note?: string | null; fold_auc?: (number | null)[] | null };
const extra = (x: object | null | undefined): EdgeExtra => (x ?? {}) as EdgeExtra;

function edgeOf(auc: number | null | undefined, edge?: boolean | null): { word: string; short: string; chip: string; none: boolean } {
  if (!ok(auc)) return { word: "edge not measured", short: "n/a", chip: "amber", none: true };
  if (edge === false || auc < NO_EDGE_AUC) return { word: "no reliable edge", short: "no edge", chip: "amber", none: true };
  if (auc < 0.55) return { word: "weak edge", short: "weak", chip: "", none: false };
  return { word: "measurable edge", short: "edge", chip: "cyan", none: false };
}
const cap = (t: string) => t.charAt(0).toUpperCase() + t.slice(1);

/** `noEdge`: the horizon has no reliable edge, so the signal is NEUTRAL whatever P(up) says. */
function SignalChip({ s, noEdge = false }: { s: Signal; noEdge?: boolean }) {
  if (s === "UP") return <span className="chip up" title="P(up) ≥ 55 %">▲ UP</span>;
  if (s === "DOWN") return <span className="chip down" title="P(up) ≤ 45 %">▼ DOWN</span>;
  return <span className="chip" title={noEdge ? "No call: the model has no reliable edge at this horizon" : "P(up) between 45 % and 55 %"}>• NEUTRAL</span>;
}

/** ProbBar's lean band, widened to the whole scale where the horizon has no edge: no "leans up / down" colour. */
const NO_LEAN: [number, number] = [0, 1];

// ------------------------------------------------------------------ fetching (cached across desk switches)
const TTL = 5 * 60_000;
type Cache<T> = Map<string, { at: number; res: T }>;
const resultCache: Cache<QuantResult> = new Map();
const boardCache: Cache<QuantLeaderboard> = new Map();
const loadResult = (sym: string) => research.quant(sym);
const loadBoard = (h: string) => research.leaderboard(Number(h), 15);

interface Fetched<T> { data: T | null; error: unknown; loading: boolean; reload: () => void }

/** GET with a short per-key cache; keeps the previous data while refetching and after a failed refetch. */
function useFetched<T>(key: string, load: (key: string) => Promise<T>, cache: Cache<T>, keep: (r: T) => boolean): Fetched<T> {
  const [st, setSt] = useState<{ key: string; data: T | null; error: unknown; loading: boolean }>({ key: "", data: null, error: null, loading: false });
  const [nonce, setNonce] = useState(0);
  const force = useRef(false);
  useEffect(() => {
    if (!key) return;
    const f = force.current;
    force.current = false;
    const hit = cache.get(key);
    if (!f && hit && Date.now() - hit.at < TTL) {
      setSt({ key, data: hit.res, error: null, loading: false });
      return;
    }
    let alive = true;
    setSt((s) => ({ key, data: s.key === key ? s.data : null, error: null, loading: true }));
    load(key).then(
      (res) => {
        if (!alive) return;
        if (keep(res)) cache.set(key, { at: Date.now(), res }); else cache.delete(key);
        setSt({ key, data: res, error: null, loading: false });
      },
      (e) => { if (alive) setSt((s) => ({ key, data: s.key === key ? s.data : null, error: e, loading: false })); },
    );
    return () => { alive = false; };
  }, [key, nonce, load, cache, keep]);
  const reload = useCallback(() => { force.current = true; setNonce((n) => n + 1); }, []);
  const cur = st.key === key;
  return { data: cur ? st.data : null, error: cur ? st.error : null, loading: !!key && (!cur || st.loading), reload };
}

const keepResult = (r: QuantResult) => r.status === "ready";
const keepBoard = (r: QuantLeaderboard) => r.model.status === "ready";

/** The model's status, polled every 3 s while it trains; `onTrained` runs once when training ends. */
function useModelInfo(resultStatus: string | undefined, onTrained: () => void) {
  const [info, setInfo] = useState<QuantModelInfo | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [trainErr, setTrainErr] = useState<unknown>(null);
  const [starting, setStarting] = useState(false);
  const trained = useRef(onTrained);
  useEffect(() => { trained.current = onTrained; }, [onTrained]);
  const lastRefresh = useRef(0);

  useEffect(() => {
    let alive = true;
    research.quantModel().then((i) => { if (alive) setInfo(i); }, (e) => { if (alive) setError(e); });
    return () => { alive = false; };
  }, []);

  const polling = info?.status === "training" || resultStatus === "training";
  useEffect(() => {
    if (!polling) return;
    let alive = true;
    const t = window.setInterval(() => {
      research.quantModel().then((i) => {
        if (!alive) return;
        setInfo(i);
        setError(null);
        if (i.status !== "training" && Date.now() - lastRefresh.current > 8000) {   // results still say "training" for a moment
          lastRefresh.current = Date.now();
          trained.current();
        }
      }, (e) => { if (alive) setError(e); });
    }, 3000);
    return () => { alive = false; window.clearInterval(t); };
  }, [polling]);

  const train = useCallback(() => {
    setStarting(true);
    setTrainErr(null);
    research.quantTrain().then(setInfo, setTrainErr).finally(() => setStarting(false));
  }, []);
  return { info, error, train, trainErr, starting };
}

// ------------------------------------------------------------------ shared bits
function Training({ info }: { info: QuantModelInfo }) {
  const p = frac01(ok(info.progress) ? info.progress : 0);
  return (
    <div className="qt-training" role="status" aria-live="polite">
      <div className="qt-training-head"><b>Training the quant model…</b><span className="mono">{fx(p * 100, 0)}%</span></div>
      <div className="bar meter" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(p * 100)} aria-label="Training progress">
        <i style={{ width: `${p * 100}%` }} />
      </div>
      <p>
        {info.message || "Building features for every stock in the lake, then testing walk-forward."} The first model takes a minute or
        two; this page checks every 3 seconds and fills in by itself.
      </p>
    </div>
  );
}

/** Loading / error / empty states for panels that need the stock's result; only the NEXT MOVE panel (`main`)
 *  carries the full error with a retry, the others point to it. */
function ResultGate({ q, what, children, main = false }: { q: Fetched<QuantResult>; what: string; children: (r: QuantResult) => ReactNode; main?: boolean }) {
  if (q.data) return <>{children(q.data)}</>;
  if (q.loading) return <Empty>Scoring the stock… {what} appears shortly.</Empty>;
  if (q.error) return main ? <ErrorNote error={q.error} onRetry={q.reload} /> : <Empty>Nothing to show — the quant request failed (see NEXT MOVE).</Empty>;
  return <Empty>Pick a stock to see {what}.</Empty>;
}

// ------------------------------------------------------------------ next move
function MoveCard({ h, stock, symbol }: { h: QuantHorizon; stock: QuantResult["stock_metrics"][number] | undefined; symbol: string }) {
  const ex = extra(h);
  const e = edgeOf(h.oos_auc, ex.edge);
  const flat = !!h.flat;                                 // calibration flattened P(up) to the base rate for every stock
  const weak = e.none || flat;
  const naive = ok(h.base_rate) ? Math.max(h.base_rate, 1 - h.base_rate) : null;
  const usual = ok(h.base_rate) && h.base_rate < 0.5 ? "down" : "up";
  return (
    <article className={`qt-card ${weak ? "weak" : ""}`} aria-label={`${h.label} outlook`}>
      <header className="qt-card-head">
        <span className="qt-card-h">{h.label}</span>
        <SignalChip s={h.signal} noEdge={weak} />
        <span className={`chip ${h.confidence === "high" ? "cyan" : h.confidence === "medium" ? "violet" : ""}`}
          title="Confidence combines how far P(up) is from 50 % with the model's walk-forward AUC at this horizon">{h.confidence.toUpperCase()} CONF.</span>
      </header>
      <ProbBar p={h.p_up} width={150} neutral={weak ? NO_LEAN : undefined} />
      <dl className="qt-kv">
        <div><dt>Expected return</dt><dd><Ret v={h.exp_ret} d={2} /> <small className="muted">±{fracPct(h.ret_sd, 1)} sd</small></dd></div>
        <div><dt>Median price</dt><dd>{inr(h.price_median)}</dd></div>
        <div><dt title="Two in three outcomes should land here">68% range</dt><dd>{inr(h.range68[0])} – {inr(h.range68[1])}</dd></div>
        <div><dt title="Nineteen in twenty outcomes should land here">95% range</dt><dd>{inr(h.range95[0])} – {inr(h.range95[1])}</dd></div>
      </dl>
      <div className="qt-oos" title="Walk-forward: every prediction scored here was made on data the model had not seen">
        <span><span className={`chip ${e.chip}`}>{e.word}</span> AUC <b>{ok(h.oos_auc) ? fx(h.oos_auc, 3) : "—"}</b> <span className="muted">vs 0.500 coin flip</span></span>
        {ok(h.oos_acc) && (
          <span className="muted">hit rate <b>{pctOf(h.oos_acc, 1)}</b>{ok(naive) && ` vs ${pctOf(naive, 1)} always guessing ${usual}`}</span>
        )}
        <span className="muted">
          {stock && ok(stock.auc) ? <>{symbol} alone: AUC {fx(stock.auc, 3)}{ok(stock.acc) && ` · hit ${pctOf(stock.acc, 0)}`} · {fx(stock.n, 0)} tests</> : `Too few tests on ${symbol} alone to score it separately.`}
        </span>
      </div>
      {weak && (
        <p className="qt-noedge" role="note">
          ⚠ {ex.edge_note ? cap(ex.edge_note) : "No reliable edge at this horizon — past forecasts were barely better than a coin flip"}.{" "}
          {flat
            ? <>P(up) is the base rate ({pctOf(h.p_up, 1)}, the share of rises over the whole training sample) for every stock at this horizon: walk-forward calibration flattened the model's
                scores, so it says nothing specific about {symbol}. Expected return is 0 and the median is today's price; the ranges still describe the risk.</>
            : "Read P(up) as ≈ 50/50; the price ranges still describe the risk."}
        </p>
      )}
    </article>
  );
}

function MovesPanel({ q, info, isAdmin }: { q: Fetched<QuantResult>; info: QuantModelInfo | null; isAdmin: boolean }) {
  const r = q.data;
  const status = info?.status === "training" ? "training" : r?.status;
  let body: ReactNode;
  if (!r) body = <ResultGate q={q} what="the next-move forecasts" main>{() => null}</ResultGate>;
  else if (!r.horizons.length && status === "training") body = <Training info={info ?? r.model} />;
  else if (!r.horizons.length && r.status === "missing") {
    body = <Empty>No quant model yet — it trains by itself on first use{isAdmin ? ", or start it with RETRAIN in the model-quality panel" : ""}.</Empty>;
  } else if (!r.horizons.length && r.status === "error") {
    body = <div className="rs-error" role="alert">The quant model failed to train: {r.model.message || "unknown error"}.</div>;
  } else if (!r.horizons.length) body = <Empty>No forecast for {r.symbol} — see the warnings below.</Empty>;
  else {
    body = (
      <div className={`qt-cards ${q.loading ? "busy" : ""}`}>
        {r.horizons.map((h) => <MoveCard key={h.h} h={h} symbol={r.symbol} stock={r.stock_metrics.find((m) => m.h === h.h)} />)}
      </div>
    );
  }
  return (
    <Panel code="NEXT" title="Next move" area="moves"
      sub={r ? `${r.symbol} · ${inr(r.spot)} · data to ${r.asof} · P(up) = calibrated probability the close h days ahead is higher` : ""}>
      {body}
      {r && status === "training" && !!r.horizons.length && <div className="qt-pad"><Training info={info ?? r.model} /></div>}
      {r && <div className="qt-pad"><Warnings items={r.horizons.length ? r.warnings.filter((w) => !/no reliable edge/i.test(w)) : r.warnings} label="Quant warnings" /></div>}
      {!!q.error && r && <ErrorNote error={q.error} onRetry={q.reload} />}
      <div className="qt-pad">
        <Disclaimer>A 55 % P(up) is a small tilt, not a call.</Disclaimer>
      </div>
    </Panel>
  );
}

// ------------------------------------------------------------------ chart
type Range = "3M" | "6M" | "1Y";
const RANGE_BARS: Record<Range, number> = { "3M": 63, "6M": 126, "1Y": 400 };

function chartModel(c: NonNullable<QuantResult["chart"]>, bars: number) {
  const n = c.ts.length;
  const s = Math.max(0, n - bars);
  const ts = c.ts.slice(s);
  const lastTs = ts[ts.length - 1];
  const lastClose = c.close[n - 1] ?? null;
  const fut = c.forecast.filter((f) => f.ts > lastTs).sort((a, b) => a.ts - b.ts);
  const now = c.forecast.find((f) => f.ts === lastTs);
  const gap = (k: number): (number | null)[] => Array.from({ length: k }, () => null);
  const past = (vals: number[]) => [...ts.map((_, i) => (ok(vals[s + i]) ? vals[s + i] : null)), ...gap(fut.length)];
  const cone = (k: "p50" | "lo68" | "hi68" | "lo95" | "hi95") =>
    [...gap(ts.length - 1), now && ok(now[k]) ? now[k] : lastClose, ...fut.map((f) => (ok(f[k]) ? f[k] : null))];
  const series: LineSeries[] = [
    { name: "Close", values: past(c.close), tone: "text" },
    { name: "Kalman trend", values: past(c.trend), tone: 1 },
  ];
  const bands: LineBand[] = [{ name: "Trend channel (± 2 sd)", lo: past(c.lower), hi: past(c.upper), tone: 1, opacity: 0.12 }];
  if (fut.length) {
    series.push({ name: "Forecast median", values: cone("p50"), tone: 2, dashed: true });
    bands.push({ name: "95% range", lo: cone("lo95"), hi: cone("hi95"), tone: 2, opacity: 0.12 });
    bands.push({ name: "68% range", lo: cone("lo68"), hi: cone("hi68"), tone: 2, opacity: 0.22 });
  }
  return {
    x: [...ts, ...fut.map((f) => f.ts)], series, bands, fut: fut.length,
    shade: c.bull_prob.length ? { name: "Bull-regime probability", values: past(c.bull_prob), tone: 3 as const } : undefined,
    marks: fut.length ? [{ x: lastTs, label: "now" }] : [],
  };
}

function ChartPanel({ q }: { q: Fetched<QuantResult> }) {
  const [range, setRange] = useState<Range>("6M");
  const c = q.data?.chart ?? null;
  const m = useMemo(() => (c && c.ts.length >= 2 ? chartModel(c, RANGE_BARS[range]) : null), [c, range]);
  return (
    <Panel code="TRND" title="Trend, regime & 21-day cone" area="chart" pad className="qt-chart"
      sub={q.data ? q.data.symbol : ""}
      actions={<Seg label="Chart range" value={range} onChange={setRange} options={(Object.keys(RANGE_BARS) as Range[]).map((k) => ({ v: k, label: k }))} />}>
      <ResultGate q={q} what="the trend chart">
        {(r) => !m ? <Empty>No chart data for {r.symbol}{r.status === "training" ? " while the model trains" : ""}.</Empty> : (
          <LineChart label={`${r.symbol}: daily close, Kalman trend and the quant model's 21-day forecast range`} x={m.x}
            series={m.series} bands={m.bands} shade={m.shade} marks={m.marks} height="fill" busy={q.loading}
            caption={<>Kalman trend = the noise-filtered price level (shaded: ± 2 standard deviations of the price around the trend over the past year).
              {m.shade ? ` Background tint = probability the stock is in its bull regime${r.regime?.fit_start ? ` (HMM fitted on ${r.regime.fit_start} → ${r.regime.fit_end}, as on the Monte Carlo desk)` : ""}; darker = more likely.` : ""}{" "}
              {m.fut ? "The cone is the model's 68 % / 95 % range for the next 21 trading days around its median — it widens with GARCH volatility." : "No forecast cone (no volatility forecast for this stock)."}</>} />
        )}
      </ResultGate>
    </Panel>
  );
}

// ------------------------------------------------------------------ regime & volatility
function RegimePanel({ q }: { q: Fetched<QuantResult> }) {
  return (
    <Panel code="RGM" title="Regime & volatility" area="regime">
      <ResultGate q={q} what="the regime estimate">
        {(r) => {
          const g = r.regime, v = r.vol;
          const named = g ? g.labels.indexOf(g.state) : -1;      // "sideways" = no state is likely enough to name
          const cur = g ? (named >= 0 ? named : g.probs.reduce((best, p, i) => (p > (g.probs[best] ?? -1) ? i : best), 0)) : 0;
          const pctile = v && ok(v.percentile) ? frac01(v.percentile) : null;
          const fitSpan = g?.fit_start && g.fit_end ? ` from ${g.fit_start} to ${g.fit_end}` : "";
          return (
            <>
              {g ? (
                <>
                  <p className="qt-lead">
                    {named >= 0 ? (
                      <>
                        A hidden-state (HMM) model fitted to {r.symbol}'s daily returns{fitSpan} puts <b>{pctOf(g.probs[cur])}</b> on the{" "}
                        <b className={g.state === "bull" ? "up" : g.state === "bear" ? "down" : undefined}>{g.state}</b> regime today
                      </>
                    ) : (
                      <>
                        A hidden-state (HMM) model fitted to {r.symbol}'s daily returns{fitSpan} is undecided today — <b>sideways</b>:{" "}
                        {g.labels.map((l, k) => `${pctOf(g.probs[k])} ${l}`).join(" vs ")}. It leans <b>{g.labels[cur]}</b>
                      </>
                    )}
                    {ok(g.mu_annual[cur]) && ok(g.vol_annual[cur]) && <>: in that state it has returned <Ret v={g.mu_annual[cur]} /> a year with {fracPct(g.vol_annual[cur], 0)} volatility</>}
                    {ok(g.expected_duration[cur]) && <>, in spells of about {fx(g.expected_duration[cur], 0)} trading days</>}.
                  </p>
                  <table className="tbl qt-tbl">
                    <thead><tr><th className="l">Regime</th><th className="l">Probability today</th><th title="Average annual return while in this state (past data)">Return/yr</th><th>Vol/yr</th><th title="Average length of a spell">Spell</th></tr></thead>
                    <tbody>
                      {g.labels.map((lab, k) => (
                        <tr key={lab} className={k === cur ? "sel" : undefined}>
                          <td className="l">{lab === "bull" ? "▲ " : lab === "bear" ? "▼ " : "• "}{lab}</td>
                          <td className="l"><span className="qt-pbar"><span className="qt-pbar-track" aria-hidden="true"><i style={{ width: `${frac01(g.probs[k] ?? 0) * 100}%` }} /></span><b>{pctOf(g.probs[k])}</b></span></td>
                          <td><Ret v={g.mu_annual[k]} /></td>
                          <td>{fracPct(g.vol_annual[k], 0)}</td>
                          <td>{ok(g.expected_duration[k]) ? `${fx(g.expected_duration[k], 0)}d` : "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {g.fit_end && (
                    <p className="qt-note">
                      States are named by their average return over that window{ok(g.fit_years) ? ` (${fx(g.fit_years, 1)} years)` : ""}, the same fit as the
                      Monte Carlo desk's regime model, so the two desks agree.
                      {g.model_fit_end && ` The quant model's own regime input is a separate HMM fitted once on returns up to ${g.model_fit_end} (kept fixed so its walk-forward test never saw later data), so the value the model uses can differ from this one.`}
                    </p>
                  )}
                </>
              ) : <Empty>No regime estimate for {r.symbol} (too little history).</Empty>}
              {v ? (
                <>
                  <h3 className="qt-h3">Volatility (annualised)</h3>
                  <Stats min={104}>
                    <Stat label="GARCH now" value={fracPct(v.garch_now, 1)} sub={ok(v.long_run) ? (v.garch_now > v.long_run * 1.1 ? "above normal" : v.garch_now < v.long_run * 0.9 ? "below normal" : "near normal") : undefined}
                      tone={ok(v.long_run) && v.garch_now > v.long_run * 1.25 ? "amber" : undefined} title="Today's conditional volatility from a GARCH(1,1)-t fit" />
                    <Stat label="Next 21 days" value={fracPct(v.garch_21d, 1)} sub="GARCH forecast" />
                    <Stat label="Long-run" value={fracPct(v.long_run, 1)} sub="where it reverts" />
                    <Stat label="EWMA" value={fracPct(v.ewma, 1)} sub="RiskMetrics λ 0.94" />
                    <Stat label="Realised 21d" value={fracPct(v.realized_21, 1)} sub="last month" />
                    <Stat label="Percentile" value={pctile !== null ? `${fx(pctile * 100, 0)}th` : "—"} sub="vs its own history" />
                  </Stats>
                  <p className="qt-note">
                    Volatility tends to come in spells and drift back to its long-run level; the forecast ranges on this desk widen or narrow with it.
                    {pctile !== null && ` Today it is higher than ${fx(pctile * 100, 0)} % of ${r.symbol}'s past readings.`}
                  </p>
                </>
              ) : null}
            </>
          );
        }}
      </ResultGate>
    </Panel>
  );
}

// ------------------------------------------------------------------ components & contributions
function toneText(t: QuantComponent["tone"]) {
  if (t === "up") return <span className="up">▲ bullish</span>;
  if (t === "down") return <span className="down">▼ bearish</span>;
  return <span className="flat">• neutral</span>;
}

function ComponentsPanel({ q }: { q: Fetched<QuantResult> }) {
  return (
    <Panel code="FACT" title="What the indicators say" area="comp" sub="today's readings, each in plain words">
      <ResultGate q={q} what="the indicator readings">
        {(r) => !r.components.length ? <Empty>No indicator readings for {r.symbol}.</Empty> : (
          <>
            <table className="tbl qt-tbl">
              <thead>
                <tr>
                  <th className="l">Indicator</th><th>Reading</th>
                  <th className="l" title="How unusual today's reading is for this stock, in standard deviations">vs its history</th>
                  <th className="l">Lean</th><th className="l">What it means</th>
                </tr>
              </thead>
              <tbody>
                {r.components.map((c) => (
                  <tr key={c.key}>
                    <td className="l b">{c.label}</td>
                    <td className="qt-read">{c.display}</td>
                    <td className="l">{ok(c.z) ? <span className="qt-z"><ZBar v={Math.max(-3, Math.min(3, c.z))} max={3} title={`${c.label} z-score`} /><span className="muted">{c.z > 0 ? "+" : ""}{fx(c.z, 1)}σ</span></span> : <span className="muted">—</span>}</td>
                    <td className="l">{toneText(c.tone)}</td>
                    <td className="l qt-wrap">{c.reading}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="qt-note">These are the ingredients, read one at a time. The model weighs them together (see contributions); an indicator that sounds bullish can carry little or even negative weight.</p>
          </>
        )}
      </ResultGate>
    </Panel>
  );
}

function ContributionsPanel({ q }: { q: Fetched<QuantResult> }) {
  const list = q.data?.contributions;
  const top = useMemo(() => [...(list ?? [])].filter((c) => ok(c.contribution))
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 10), [list]);
  const max = Math.max(1e-9, ...top.map((c) => Math.abs(c.contribution)));
  const net = top.reduce((a, c) => a + c.contribution, 0);
  const b21 = q.data?.model.metrics.find((m) => m.h === 21)?.calib_b;
  return (
    <Panel code="WHY" title="Why it leans that way" area="contrib" sub="1-month P(up) · top 10 pushes, in calibrated log-odds">
      <ResultGate q={q} what="the model's reasoning">
        {(r) => r.contributions_flat ? (
          <p className="qt-note warn" role="note">
            The 1-month probability does not depend on these inputs: walk-forward calibration flattened it to the base rate
            {ok(b21) ? ` (spread b = ${fx(b21, 2)})` : ""}, so every stock gets the same P(up) and nothing pushes it up or down.
          </p>
        ) : !top.length ? <Empty>{r.status === "ready" ? "No contributions for this stock." : "Contributions appear once the model is trained."}</Empty> : (
          <>
            <ul className="qt-contrib" aria-label="Contributions to the 1-month log-odds of a rise">
              {top.map((c: QuantContribution) => (
                <li key={c.feature} title={`${c.feature}: standardised reading ${fx(c.value, 2)} × weight = ${fx(c.contribution, 3)} log-odds`}>
                  <span className="qt-c-label">{c.label}</span>
                  <span className="qt-c-track" aria-hidden="true">
                    <i className={c.contribution >= 0 ? "pos" : "neg"} style={{ width: `${(Math.abs(c.contribution) / max) * 50}%` }} />
                  </span>
                  <span className={`qt-c-val ${c.contribution >= 0 ? "up" : "down"}`}>{c.contribution >= 0 ? "▲ +" : "▼ "}{fx(c.contribution, 3)}</span>
                  <span className="qt-c-z muted">z {c.value > 0 ? "+" : ""}{fx(c.value, 1)}</span>
                </li>
              ))}
            </ul>
            <p className="qt-note">
              Each bar = the model's weight × today's standardised reading × the walk-forward calibration spread{ok(b21) ? ` (b = ${fx(b21, 2)})` : ""},
              i.e. how far it moves the displayed 1-month P(up) on the log-odds scale. ▲ pushes toward UP, ▼ toward DOWN; near 50 % a push
              of 0.1 moves P(up) by about 2.5 points. Net of these ten: <b className={net >= 0 ? "up" : "down"}>{net >= 0 ? "▲ +" : "▼ "}{fx(net, 3)}</b>.
            </p>
          </>
        )}
      </ResultGate>
    </Panel>
  );
}

// ------------------------------------------------------------------ model quality
/** Walk-forward metrics with horizons as columns (fits the narrow side column; one row per measure). */
function MetricsTable({ metrics, r }: { metrics: QuantModelInfo["metrics"]; r: QuantResult | null }) {
  const ms = [...metrics].sort((a, b) => a.h - b.h);
  const stock = (h: number) => r?.stock_metrics.find((m) => m.h === h);
  const skill = (v: number, base: number) => (base > 0 ? 1 - v / base : null);
  const rows: { label: string; title: string; cell: (m: QuantModelInfo["metrics"][number]) => ReactNode }[] = [
    { label: "Verdict", title: "From the walk-forward AUC: < 0.52 no reliable edge, 0.52–0.55 weak, ≥ 0.55 measurable",
      cell: (m) => { const x = extra(m), e = edgeOf(m.auc, x.edge); return <span className={`chip ${e.chip}`} title={x.edge_note ?? e.word}>{e.short}</span>; } },
    { label: "AUC, all stocks", title: "Chance the model ranks a random up-move above a random down-move (0.500 = coin flip)",
      cell: (m) => <b className={m.auc >= 0.55 ? "cyan" : m.auc < NO_EDGE_AUC ? "amber" : undefined}>{fx(m.auc, 3)}</b> },
    ...(ms.some((m) => extra(m).fold_auc?.length) ? [{ label: "Worst fold AUC", title: "Lowest AUC of the walk-forward test periods — an edge should hold in each of them",
      cell: (m: QuantModelInfo["metrics"][number]) => {
        const f = (extra(m).fold_auc ?? []).filter(ok);
        return f.length ? <span className={Math.min(...f) < 0.5 ? "amber" : undefined}>{fx(Math.min(...f), 3)}</span> : <span className="muted">—</span>;
      } }] : []),
    { label: r ? `AUC, ${r.symbol} only` : "AUC, this stock", title: "Same walk-forward predictions, this stock only — few tests, so noisy",
      cell: (m) => { const x = stock(m.h); return x && ok(x.auc) ? <span title={`${fx(x.n, 0)} tests`}>{fx(x.auc, 3)}</span> : <span className="muted">—</span>; } },
    { label: "Hit rate", title: "Share of correct up/down calls", cell: (m) => pctOf(m.acc, 1) },
    { label: "Always guessing", title: "Hit rate of always guessing the more common direction — the bar to beat",
      cell: (m) => <span className="muted">{pctOf(Math.max(m.base_rate, 1 - m.base_rate), 1)}</span> },
    { label: "Brier skill", title: "Probability accuracy vs always predicting the historical up-share (positive = better)",
      cell: (m) => <Ret v={skill(m.brier, m.brier_base)} d={2} /> },
    { label: "Log-loss skill", title: "Same idea with log loss, which punishes confident misses harder",
      cell: (m) => <Ret v={skill(m.logloss, m.logloss_base)} d={2} /> },
    { label: "Rank IC", title: "Rank correlation between P(up) and the realised return (0 = none)", cell: (m) => fx(m.ic, 3) },
    { label: "Tests · folds", title: "Out-of-sample predictions scored · walk-forward folds", cell: (m) => <span title={fx(m.n_test, 0)}>{compact(m.n_test)} <small className="muted">· {m.folds}</small></span> },
    ...(r ? [{ label: `Tests, ${r.symbol}`, title: "Out-of-sample predictions on this stock alone",
      cell: (m: QuantModelInfo["metrics"][number]) => fx(stock(m.h)?.n ?? null, 0) }] : []),
  ];
  return (
    <table className="tbl qt-tbl qt-metrics">
      <thead><tr><th className="l">Walk-forward</th>{ms.map((m) => <th key={m.h}>{hShort(m.h)}</th>)}</tr></thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.label}><td className="l" title={row.title}>{row.label}</td>{ms.map((m) => <td key={m.h}>{row.cell(m)}</td>)}</tr>
        ))}
      </tbody>
    </table>
  );
}

function QualityPanel({ info, r, isAdmin, onTrain, trainErr, starting, infoErr }: {
  info: QuantModelInfo | null; r: QuantResult | null; isAdmin: boolean; onTrain: () => void; trainErr: unknown; starting: boolean; infoErr: unknown;
}) {
  const now = Date.now() / 1000;
  const training = info?.status === "training";
  const statusChip = info ? (
    info.status === "ready" ? <span className="chip up">READY</span>
      : info.status === "training" ? <span className="chip cyan">TRAINING</span>
        : info.status === "missing" ? <span className="chip amber">NOT TRAINED</span> : <span className="chip down">ERROR</span>
  ) : null;
  return (
    <Panel code="OOS" title="Model quality" area="quality" sub="walk-forward, out of sample"
      actions={isAdmin ? (
        <button type="button" className="btn" onClick={onTrain} disabled={starting || training}
          title="Rebuild features for every stock in the lake, re-run the walk-forward test and refit (1–2 minutes of CPU)">
          {training ? "TRAINING…" : starting ? "STARTING…" : "RETRAIN"}
        </button>
      ) : undefined}>
      {!info ? (infoErr ? <ErrorNote error={infoErr} /> : <Empty>Loading model status…</Empty>) : (
        <>
          <div className="qt-model-line">
            {statusChip}
            {ok(info.trained_at) && <span>trained {dateIST(info.trained_at)} {timeIST(info.trained_at)} IST <span className="muted">({ago(info.trained_at, now)} ago)</span></span>}
            <span><b>{fx(info.symbols, 0)}</b> stocks · <b>{fx(info.rows, 0)}</b> rows · <b>{info.features.length}</b> features</span>
          </div>
          {training && <div className="qt-pad"><Training info={info} /></div>}
          {info.status === "error" && <div className="rs-error" role="alert">{info.message || "Training failed."}</div>}
          {!!trainErr && <ErrorNote error={trainErr} />}
          {info.metrics.length ? <MetricsTable metrics={info.metrics} r={r} /> : <Empty>No walk-forward metrics yet.</Empty>}
          <p className="qt-note">
            Every number above comes from predictions made on data the model had not seen (expanding window, with a gap so training and test never
            overlap). AUC 0.50 is a coin flip; for daily stock direction 0.52–0.55 is a small real edge and above 0.60 is rare. One stock's own
            figures rest on few tests and are noisy.{!isAdmin && " The model retrains itself daily; manual retraining is for admins."}
          </p>
        </>
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------ leaderboard
type LbH = "1" | "5" | "21";

function LeaderTable({ title, rows, sel, onPick, noEdge }: { title: string; rows: QuantLeader[]; sel: string; onPick: (s: string) => void; noEdge: boolean }) {
  const key = (e: KeyboardEvent<HTMLTableRowElement>, s: string) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPick(s); }
  };
  return (
    <div className="qt-lead-col">
      <h3 className="qt-h3">{title}</h3>
      {!rows.length ? <Empty>None.</Empty> : (
        <table className="tbl qt-tbl">
          <thead><tr><th className="l">#</th><th className="l">Stock</th><th className="l">P(up)</th><th title="Expected return over the horizon">Exp. ret</th><th className="l">Signal</th></tr></thead>
          <tbody>
            {rows.map((x, i) => (
              <tr key={x.symbol} className={x.symbol === sel ? "sel" : undefined} tabIndex={0} onClick={() => onPick(x.symbol)} onKeyDown={(e) => key(e, x.symbol)}
                title={`Analyse ${x.symbol} · last close ${inr(x.close)}`}>
                <td className="l muted">{i + 1}</td>
                <td className="sym">{x.symbol}{x.sector && <small>{x.sector}</small>}</td>
                <td className="l"><ProbBar p={x.p_up} label="" width={44} neutral={noEdge ? NO_LEAN : undefined} /></td>
                <td><Ret v={x.exp_ret} d={2} /></td>
                <td className="l"><SignalChip s={x.signal} noEdge={noEdge} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function LeaderboardPanel({ h, setH, board, sel, onPick }: { h: LbH; setH: (h: LbH) => void; board: Fetched<QuantLeaderboard>; sel: string; onPick: (s: string) => void }) {
  const b = board.data;
  const metric = b?.model.metrics.find((m) => m.h === Number(h));
  const ex = extra(b);
  // flat: calibration gave every stock the same P(up) (the base rate), so the lists would only order noise
  const ps = b ? [...b.up, ...b.down].map((x) => x.p_up).filter(ok) : [];
  const flat = !!b && (!!b.flat || (ps.length > 1 && Math.max(...ps) - Math.min(...ps) < 1e-4));
  const baseP = b && ok(b.base_p) ? b.base_p : ps.length ? ps[0] : null;
  const weak = flat || edgeOf(metric?.auc, ex.edge ?? extra(metric).edge).none;
  return (
    <Panel code="RANK" title="Leaderboard" area="lead"
      sub={b ? (flat ? `no ranking at this horizon · data to ${b.asof}` : `every stock in the lake, ranked by P(up) · data to ${b.asof} · click a row to analyse it`) : ""}
      actions={<Seg label="Leaderboard horizon" value={h} onChange={setH}
        options={(["1", "5", "21"] as LbH[]).map((k) => ({ v: k, label: hShort(Number(k)), title: `${k} trading day${k === "1" ? "" : "s"} ahead` }))} />}>
      {!b ? (board.loading ? <Empty>Scoring every stock with the current model…</Empty>
        : board.error ? <ErrorNote error={board.error} onRetry={board.reload} /> : <Empty>No leaderboard yet.</Empty>)
        : b.model.status === "training" && !b.up.length && !flat ? <Training info={b.model} />
          : flat ? (
            <div className={board.loading ? "qt-busy" : undefined}>
              <p className="qt-note warn" role="note">
                At this horizon the model gives every stock the same probability — the base rate{ok(baseP) ? ` (P(up) ${pctOf(baseP, 1)})` : ""} — because
                walk-forward calibration flattened its scores to it, so there is no ranking.
                {metric ? ` Walk-forward AUC ${fx(metric.auc, 3)}.` : ""}
              </p>
              {!!board.error && <ErrorNote error={board.error} onRetry={board.reload} />}
            </div>
          ) : (
            <div className={board.loading ? "qt-busy" : undefined}>
              <p className={`qt-note ${weak ? "warn" : ""}`}>
                {metric ? <>At this horizon the model's walk-forward AUC is {fx(metric.auc, 3)}{weak && ex.edge_note ? ` (${ex.edge_note})` : ""}. </> : null}
                {weak
                  ? "That is no reliable edge — the top of this list is barely more likely to rise than the bottom. Use it as a list of stocks to research, not to buy."
                  : "A small edge on average across many stocks; any single name can still go the other way. Use it as a screen, not a buy list."}
              </p>
              <div className="qt-lead-grid">
                <LeaderTable title="▲ Highest P(up)" rows={b.up} sel={sel} onPick={onPick} noEdge={weak} />
                <LeaderTable title="▼ Lowest P(up)" rows={b.down} sel={sel} onPick={onPick} noEdge={weak} />
              </div>
              {!!board.error && <ErrorNote error={board.error} onRetry={board.reload} />}
            </div>
          )}
    </Panel>
  );
}

// ------------------------------------------------------------------ desk
export default function Quant() {
  const symbol = useLive((s) => s.researchSymbol);
  const setResearchSymbol = useLive((s) => s.setResearchSymbol);
  const isAdmin = useLive((s) => !!s.me?.admin);
  const q = useFetched(symbol, loadResult, resultCache, keepResult);
  const [lbH, setLbH] = useState<LbH>("21");
  const board = useFetched(lbH, loadBoard, boardCache, keepBoard);
  const reloadQ = q.reload, reloadBoard = board.reload;
  const refreshAll = useCallback(() => {
    resultCache.clear();
    boardCache.clear();
    reloadQ();
    reloadBoard();
  }, [reloadQ, reloadBoard]);
  const model = useModelInfo(q.data?.status, refreshAll);
  const info = model.info ?? q.data?.model ?? board.data?.model ?? null;

  return (
    <div className="desk desk-quant rs-desk">
      <SymbolPicker area="pick" actions={
        <button type="button" className="btn" onClick={q.reload} disabled={!symbol || q.loading} title="Score the stock again with the latest data">
          {q.loading ? "SCORING…" : "REFRESH"}
        </button>
      } />
      <MovesPanel q={q} info={info} isAdmin={isAdmin} />
      <ChartPanel q={q} />
      <RegimePanel q={q} />
      <ComponentsPanel q={q} />
      <ContributionsPanel q={q} />
      <LeaderboardPanel h={lbH} setH={setLbH} board={board} sel={symbol} onPick={setResearchSymbol} />
      <QualityPanel info={info} r={q.data} isAdmin={isAdmin} onTrain={model.train} trainErr={model.trainErr} starting={model.starting} infoErr={model.error} />
    </div>
  );
}
