/* MONTE CARLO desk — thousands of simulated futures for one stock, and the check of whether to believe them.
 * A simulation cannot predict a price: it shows the RANGE that the stock's own behaviour (volatility clustering, fat
 * tails, jumps, regimes) implies around a neutral expected return, so every number here is a probability. The
 * CALIBRATION panel replays the same recipe at past dates on the data available then and scores how often its
 * bands really held the price — the honest answer to "does it predict correctly?". */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { CoverageChart, FanChart, Histogram, LineChart, MC_BANDS, PitHistogram, ProbBar, fracPct, inr } from "../components/research/charts";
import type { HistMarker } from "../components/research/charts";
import { Disclaimer, ErrorNote, Stat, Stats, Warnings, errorText } from "../components/research/common";
import { SymbolPicker, useHistoryInfo } from "../components/research/SymbolPicker";
import { Chg, Empty, Panel, Seg } from "../components/ui";
import { dateIST, fx } from "../lib/format";
import { research } from "../lib/research";
import type { McBacktest, McBacktestRequest, McDrift, McModel, McRequest, McResult } from "../lib/research";
import { getPref, setPref } from "../lib/storage";
import { useLive } from "../lib/store";
import "./montecarlo.css";

// ------------------------------------------------------------------ settings
type HKey = "1M" | "3M" | "6M" | "1Y" | "2Y" | "3Y";
const H_DAYS: Record<HKey, number> = { "1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "3Y": 756 };
const H_KEYS = Object.keys(H_DAYS) as HKey[];
const H_NAME: Record<string, string> = { "1M": "1 month", "3M": "3 months", "6M": "6 months", "1Y": "1 year", "2Y": "2 years", "3Y": "3 years" };
const hName = (label: string, days: number) => H_NAME[label] ?? `${days} trading days`;
const KEY_HORIZONS = new Set(["6M", "1Y", "2Y", "3Y"]);       // the horizons the research desks report on

const PATHS = ["2000", "10000", "25000"] as const;
type PKey = (typeof PATHS)[number];
const PATH_INFO: Record<PKey, { label: string; title: string }> = {
  "2000": { label: "2K", title: "2,000 paths — a quick look; the outer percentiles wobble by a few %" },
  "10000": { label: "10K", title: "10,000 paths — the default; percentiles stable to about 1 %" },
  "25000": { label: "25K", title: "25,000 paths — smoothest tails, slower (capped server-side for 3-year horizons)" },
};
const LOOKBACKS = ["2", "3", "5", "10"] as const;
type LKey = (typeof LOOKBACKS)[number];

const MODEL_INFO: Record<McModel, { label: string; long: string }> = {
  ensemble: { label: "ENSEMBLE", long: "Equal blend of the GARCH, bootstrap, jump and regime models (a quarter of the paths each), so no single model's blind spot dominates. The default." },
  garch: { label: "GARCH", long: "GARCH(1,1) with fat-tailed shocks: calm and stormy spells persist for a while, then volatility drifts back to its long-run level. Starts from today's volatility." },
  bootstrap: { label: "BOOTSTRAP", long: "Replays the stock's own past daily moves in random blocks of about 10 days: keeps its real fat tails and short streaks without assuming a formula." },
  jump: { label: "JUMP", long: "Merton jump-diffusion: ordinary daily wiggles plus rare sudden gaps (results, news shocks) at the rate and size seen in the stock's history." },
  regime: { label: "REGIME", long: "Two hidden states with their own return and volatility that switch over time (a Markov-switching model), starting from the state the stock seems to be in today." },
  gbm: { label: "GBM", long: "Geometric Brownian motion: constant volatility and bell-curve daily moves. A textbook baseline — it understates crash risk." },
};
const MODELS = Object.keys(MODEL_INFO) as McModel[];

const DRIFT_INFO: Record<McDrift, { label: string; long: string }> = {
  capm: { label: "CAPM (default)", long: "Expected return = risk-free rate + β × equity risk premium: what a stock with this market sensitivity earns on average. Neutral — not a view on this stock." },
  historical: { label: "Historical, shrunk", long: "The stock's own past average return, pulled hard toward CAPM: five years of prices pin an average return down only to about ±10–15 %/yr, so raw history is mostly noise." },
  zero: { label: "Zero (risk only)", long: "No growth assumed — the expected price equals today's. A pure “how far could it move” view." },
  quant: { label: "Quant model", long: "The QUANT desk's 1-month view for the first month, fading to CAPM by month 3. Only as good as that model's measured edge (see its AUC on the QUANT desk)." },
};
const DRIFTS = Object.keys(DRIFT_INFO) as McDrift[];

interface McCfg { h: HKey; paths: PKey; model: McModel; drift: McDrift; lookback: LKey }
const CFG_KEY = "td.mc.cfg";
const DEFAULT_CFG: McCfg = { h: "1Y", paths: "10000", model: "ensemble", drift: "capm", lookback: "5" };

function pick<T extends string>(v: unknown, allowed: readonly T[], fallback: T): T {
  return allowed.includes(v as T) ? (v as T) : fallback;
}

function loadCfg(): McCfg {
  try {
    const v = JSON.parse(getPref(CFG_KEY) ?? "{}") as Record<string, unknown>;
    return {
      h: pick(v.h, H_KEYS, DEFAULT_CFG.h), paths: pick(v.paths, PATHS, DEFAULT_CFG.paths),
      model: pick(v.model, MODELS, DEFAULT_CFG.model), drift: pick(v.drift, DRIFTS, DEFAULT_CFG.drift),
      lookback: pick(v.lookback, LOOKBACKS, DEFAULT_CFG.lookback),
    };
  } catch {
    return DEFAULT_CFG;
  }
}

const toReq = (c: McCfg, targets: number[]): McRequest => ({
  horizon_days: H_DAYS[c.h], paths: Number(c.paths), model: c.model, drift: c.drift, lookback_years: Number(c.lookback), targets,
});
const reqKey = (r: McRequest) => JSON.stringify([r.horizon_days, r.paths, r.model, r.drift, r.lookback_years, r.targets]);

// ------------------------------------------------------------------ formatting
const ok = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const pctOf = (p: number | null | undefined, d = 0) => (ok(p) ? `${fx(p * 100, d)}%` : "—");
/** Compact rupees for ranges in narrow tiles: no paise above ₹1,000. */
const inrShort = (v: number | null | undefined) => inr(v, ok(v) && Math.abs(v) >= 1000 ? 0 : ok(v) && Math.abs(v) >= 100 ? 1 : 2);

/** Signed return with ▲▼ and colour (fraction in). */
function Ret({ v, d = 1 }: { v: number | null | undefined; d?: number }) {
  return ok(v) ? <Chg v={v * 100} d={d} /> : <span className="muted">—</span>;
}

/** A loss given as a positive fraction of spot ("losses positive"). */
function Loss({ v }: { v: number | null | undefined }) {
  if (!ok(v)) return <span className="muted">—</span>;
  return v > 0 ? <span className="down">▼ {fx(v * 100, 1)}%</span> : <span className="up">▲ no loss</span>;
}

// ------------------------------------------------------------------ requests (results survive desk switches)
interface Done<Req, Res> { sym: string; req: Req; res: Res }

let simMemo: Done<McRequest, McResult> | null = null;
let btMemo: Done<McBacktestRequest, McBacktest> | null = null;
const rememberSim = (d: Done<McRequest, McResult>) => { simMemo = d; };
const rememberBt = (d: Done<McBacktestRequest, McBacktest>) => { btMemo = d; };

/** One request at a time per kind; a newer run makes older responses irrelevant. */
function useRun<Req, Res>(call: (sym: string, req: Req) => Promise<Res>, initial: Done<Req, Res> | null, remember: (d: Done<Req, Res>) => void) {
  const [done, setDone] = useState<Done<Req, Res> | null>(initial);
  const [err, setErr] = useState<{ sym: string; e: unknown } | null>(null);
  const [busy, setBusy] = useState<{ sym: string; since: number } | null>(null);
  const seq = useRef(0);
  const run = useCallback((sym: string, req: Req) => {
    const id = ++seq.current;
    setErr(null);
    setBusy({ sym, since: Date.now() });
    call(sym, req).then(
      (res) => {
        if (id !== seq.current) return;
        const d = { sym, req, res };
        remember(d);
        setDone(d);
        setBusy(null);
      },
      (e) => {
        if (id !== seq.current) return;
        setErr({ sym, e });
        setBusy(null);
      },
    );
  }, [call, remember]);
  return { done, err, busy, run };
}

/** Seconds since `since` (0 = idle), ticking while active. */
function useElapsed(since: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!since) return;
    setNow(Date.now());
    const t = window.setInterval(() => setNow(Date.now()), 200);
    return () => window.clearInterval(t);
  }, [since]);
  return since ? Math.max(0, now - since) / 1000 : 0;
}

/** Content gate shared by the result panels; only the fan panel (`main`) carries the full error with a retry. */
interface View { r: McResult | null; running: boolean; error: unknown; retry: () => void }
function Gate({ v, children, what, main = false }: { v: View; children: (r: McResult) => ReactNode; what: string; main?: boolean }) {
  if (v.r) return <>{children(v.r)}</>;
  if (v.running) return <Empty>Simulating… {what} appears when the run finishes.</Empty>;
  if (v.error) return main ? <ErrorNote error={v.error} onRetry={v.retry} /> : <Empty>No simulation to show — {errorText(v.error)}</Empty>;
  return <Empty>Press RUN to simulate — {what} appears here.</Empty>;
}

// ------------------------------------------------------------------ target prices
function parseTargets(text: string, spot: number | null): { values: number[]; error: string | null } {
  const tokens = text.replace(/\s+%/g, "%").split(/[\s;]+|,\s+/).map((t) => t.trim()).filter(Boolean);
  const values: number[] = [];
  for (const tok of tokens) {
    const rel = /^([+\-−]?\d+(?:\.\d+)?)\s*%$/.exec(tok);
    if (rel) {
      if (!ok(spot)) return { values: [], error: "Spot price unknown — enter ₹ levels instead of %" };
      const v = spot * (1 + Number(rel[1].replace("−", "-")) / 100);
      if (!(v > 0)) return { values: [], error: `${tok} gives a price below zero` };
      values.push(Math.round(v * 100) / 100);
      continue;
    }
    const v = Number(tok.replace(/[₹,]/g, ""));
    if (!ok(v) || v <= 0) return { values: [], error: `“${tok}” is not a price (try 1650 or +10%)` };
    values.push(v);
  }
  return { values, error: null };
}

function TargetsField({ targets, setTargets, spot }: { targets: number[]; setTargets: (t: number[]) => void; spot: number | null }) {
  const [text, setText] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const full = targets.length >= 6;
  const add = () => {
    if (!text.trim()) return;
    const { values, error } = parseTargets(text, spot);
    if (error) { setErr(error); return; }
    const next = Array.from(new Set([...targets, ...values]));
    if (next.length > 6) { setErr("At most 6 target prices"); return; }
    setTargets(next.sort((a, b) => a - b));
    setText("");
    setErr(null);
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") { e.preventDefault(); add(); }
  };
  return (
    <div className="mc-targets">
      {targets.map((t) => (
        <span key={t} className="chip mc-tchip">
          {inr(t)}{ok(spot) && <small className={t >= spot ? "up" : "down"}>{t >= spot ? "▲" : "▼"}{fx(Math.abs(t / spot - 1) * 100, 0)}%</small>}
          <button type="button" aria-label={`Remove target ${inr(t)}`} title="Remove" onClick={() => setTargets(targets.filter((x) => x !== t))}>×</button>
        </span>
      ))}
      {!full && (
        <>
          <label htmlFor="mc-target-in" className="sr-only">Add target prices (₹ price or ± % from spot)</label>
          <input id="mc-target-in" className="filter-input mc-tin" value={text} placeholder="₹ or ±% · Enter"
            onChange={(e) => { setText(e.target.value); setErr(null); }} onKeyDown={onKey}
            title="Price levels to test, e.g. 1650 or +10% / -15% from spot; several separated by spaces. Up to 6." />
          <button type="button" className="btn ghost mc-tadd" onClick={add} disabled={!text.trim()}>ADD</button>
        </>
      )}
      {err && <span className="err small" role="alert">{err}</span>}
    </div>
  );
}

// ------------------------------------------------------------------ controls
function Controls({ cfg, setCfg, targets, setTargets, spot, error, hasResult, retry, warnings }: {
  cfg: McCfg; setCfg: (p: Partial<McCfg>) => void; targets: number[]; setTargets: (t: number[]) => void; spot: number | null;
  error: unknown; hasResult: boolean; retry: () => void; warnings: string[];
}) {
  return (
    <section className="panel mc-ctrl" style={{ gridArea: "ctrl" }} aria-label="Simulation setup">
      <div className="mc-ctrl-row">
        <div className="mc-field">
          <span className="mc-flabel">Horizon</span>
          <Seg label="Horizon" value={cfg.h} onChange={(h) => setCfg({ h })}
            options={H_KEYS.map((k) => ({ v: k, label: k, title: `${hName(k, H_DAYS[k])} ≈ ${H_DAYS[k]} trading days` }))} />
        </div>
        <div className="mc-field">
          <span className="mc-flabel">Paths</span>
          <Seg label="Number of simulated paths" value={cfg.paths} onChange={(paths) => setCfg({ paths })}
            options={PATHS.map((p) => ({ v: p, label: PATH_INFO[p].label, title: PATH_INFO[p].title }))} />
        </div>
        <div className="mc-field">
          <span className="mc-flabel">Model</span>
          <Seg label="Simulation model" value={cfg.model} onChange={(model) => setCfg({ model })}
            options={MODELS.map((m) => ({ v: m, label: MODEL_INFO[m].label, title: MODEL_INFO[m].long }))} />
        </div>
        <div className="mc-field">
          <label className="mc-flabel" htmlFor="mc-drift">Drift</label>
          <select id="mc-drift" className="mc-select" value={cfg.drift} onChange={(e) => setCfg({ drift: e.target.value as McDrift })}
            title={DRIFT_INFO[cfg.drift].long}>
            {DRIFTS.map((d) => <option key={d} value={d}>{DRIFT_INFO[d].label}</option>)}
          </select>
        </div>
        <div className="mc-field">
          <label className="mc-flabel" htmlFor="mc-lookback">Learn from</label>
          <select id="mc-lookback" className="mc-select" value={cfg.lookback} onChange={(e) => setCfg({ lookback: e.target.value as LKey })}
            title="How many years of the stock's daily history calibrate volatility, tails, jumps and regimes (fewer if the stock has less)">
            {LOOKBACKS.map((l) => <option key={l} value={l}>last {l} years</option>)}
          </select>
        </div>
        <div className="mc-field mc-field-targets">
          <span className="mc-flabel">Target prices <span className="muted">({targets.length}/6)</span></span>
          <TargetsField targets={targets} setTargets={setTargets} spot={spot} />
        </div>
      </div>
      <p className="mc-help">
        <b>{MODEL_INFO[cfg.model].label}:</b> {MODEL_INFO[cfg.model].long} <b>Drift — {DRIFT_INFO[cfg.drift].label}:</b> {DRIFT_INFO[cfg.drift].long}
      </p>
      {!!error && hasResult && <ErrorNote error={error} onRetry={retry} />}
      <Warnings items={warnings} label="Simulation warnings" />
      <Disclaimer />
    </section>
  );
}

// ------------------------------------------------------------------ fan
function FanPanel({ v }: { v: View }) {
  const r = v.r;
  const [log, setLog] = useState<boolean | null>(null);
  const [showPaths, setShowPaths] = useState(true);
  const logOn = log ?? (r ? r.horizon_days >= 504 : false);
  const targets = useMemo(() => (r ? r.targets.map((t) => t.price) : []), [r]);
  const checkpoints = useMemo(() => (r ? r.horizons.map((h) => ({ ts: h.ts, label: h.label })) : []), [r]);
  const tools = (
    <>
      <button type="button" className="rc-tbtn" aria-pressed={showPaths} onClick={() => setShowPaths((s) => !s)}
        title="Show or hide 30 randomly picked simulated paths">PATHS</button>
      <button type="button" className="rc-tbtn" aria-pressed={logOn} onClick={() => setLog(!logOn)}
        title="Logarithmic price axis: equal % moves look equal — better for long horizons">LOG</button>
    </>
  );
  return (
    <Panel code="MC" title="Simulated price paths" area="fan" pad className="mc-fanp"
      sub={r ? `${r.symbol} · ${fx(r.paths, 0)} paths · ${MODEL_INFO[r.model]?.label.toLowerCase() ?? r.model} · ${r.drift} drift · data to ${r.asof} · ${fx(r.elapsed_ms, 0)} ms` : "history + percentile fan"}>
      <Gate v={v} what="the fan chart" main>
        {(r) => (
          <FanChart label={`${r.symbol}: simulated price range over ${r.horizon_days} trading days`} data={r.fan} median="p50" mean="mean"
            bands={MC_BANDS} history={r.history} paths={showPaths ? r.sample_paths : undefined} spot={r.spot} targets={targets}
            checkpoints={checkpoints} log={logOn} height="fill" busy={v.running} tools={tools}
            caption={<>Shaded: where the middle 50 / 80 / 90 % of the {fx(r.paths, 0)} simulated paths are on each date (darker = more likely).
              Median = the middle path, not a prediction; mean (dashed) sits above it because gains compound. Future dates skip weekends but not NSE holidays.</>} />
        )}
      </Gate>
    </Panel>
  );
}

// ------------------------------------------------------------------ outcome + risk (side column)
function OutcomePanel({ v }: { v: View }) {
  const r = v.r;
  const end = r?.horizons[r.horizons.length - 1];
  return (
    <Panel code="END" title={end ? `Price after ${hName(end.label, end.days)}` : "Price at horizon"} sub={end ? dateIST(end.ts) : ""}>
      <Gate v={v} what="the outcome summary">
        {(r) => {
          const t = r.terminal;
          return (
            <>
              <p className="mc-lead">
                Half the paths end above <b>{inr(t.median)}</b> (<Ret v={t.median_ret} />); 9 in 10 end between <b>{inr(t.p5)}</b> and <b>{inr(t.p95)}</b>.
              </p>
              <Stats min={118}>
                <Stat label="Median" value={inr(t.median)} sub={<Ret v={t.median_ret} />} title="The middle outcome: half the paths end above it" />
                <Stat label="Mean" value={inr(t.mean)} sub={<Ret v={t.exp_ret} />} title="Average outcome (expected return) — above the median because gains compound" />
                <Stat label="P(above spot)" value={<ProbBar p={t.prob_up} label="" width={54} />} sub={`spot ${inr(r.spot)}`} title="Share of paths ending above today's price" />
                <Stat label="90% range" value={<span className="mc-range">{inrShort(t.p5)}–{inrShort(t.p95)}</span>} sub="5th–95th pct" />
                <Stat label="50% range" value={<span className="mc-range">{inrShort(t.p25)}–{inrShort(t.p75)}</span>} sub="25th–75th pct" />
                <Stat label="Skew" value={fx(t.skew, 2)} sub={t.skew > 0.1 ? "upside tail" : t.skew < -0.1 ? "downside tail" : "symmetric"}
                  title="Shape of the outcome distribution: positive = big gains rarer but larger than big losses" />
              </Stats>
            </>
          );
        }}
      </Gate>
    </Panel>
  );
}

function RiskPanel({ v }: { v: View }) {
  return (
    <Panel code="RISK" title="Downside risk" sub={v.r ? "losses as % of today's price" : ""}>
      <Gate v={v} what="the risk figures">
        {(r) => {
          const k = r.risk;
          const sep = <span className="sep">·</span>;
          return (
            <>
              <dl className="mc-risk">
                <div title="Value at Risk: the loss at the horizon that only the worst 5 % (1 %) of paths exceed">
                  <dt>Loss at 1-in-20 · 1-in-100 odds (VaR)</dt>
                  <dd><Loss v={k.var95} />{sep}<Loss v={k.var99} /></dd>
                </div>
                <div title="Expected shortfall (CVaR): the average loss inside that worst 5 % (1 %) of paths">
                  <dt>Avg loss in worst 5% · 1% (CVaR)</dt>
                  <dd><Loss v={k.cvar95} />{sep}<Loss v={k.cvar99} /></dd>
                </div>
                <div>
                  <dt>Chance it ends 10% · 20% lower</dt>
                  <dd><span className={k.prob_loss_10 >= 0.25 ? "down" : undefined}>{pctOf(k.prob_loss_10)}</span>{sep}<span className={k.prob_loss_20 >= 0.15 ? "down" : undefined}>{pctOf(k.prob_loss_20)}</span></dd>
                </div>
                <div>
                  <dt>Chance it ends 20% higher</dt>
                  <dd>{pctOf(k.prob_gain_20)}</dd>
                </div>
                <div title="Max drawdown: the worst peak-to-trough fall along the way (not just at the end) — for the typical path, and the level 1 path in 10 exceeds">
                  <dt>Worst dip on the way: typical · 1-in-10</dt>
                  <dd><Loss v={k.max_dd_median} />{sep}<Loss v={k.max_dd_p90} /></dd>
                </div>
                <div title="Share of paths that fall 20 % or more from a previous high at some point before the horizon">
                  <dt>Chance of a 20%+ dip from a peak</dt>
                  <dd className={k.prob_dd_20 >= 0.3 ? "down" : undefined}>{pctOf(k.prob_dd_20)}</dd>
                </div>
              </dl>
              <p className="mc-note">Dips count falls <i>during</i> the horizon — what a holder sits through.</p>
            </>
          );
        }}
      </Gate>
    </Panel>
  );
}

// ------------------------------------------------------------------ horizons table
function HorizonsPanel({ v }: { v: View }) {
  return (
    <Panel code="HZN" title="Price ranges by horizon" area="horiz" sub={v.r ? `percentiles of ${fx(v.r.paths, 0)} paths · spot ${inr(v.r.spot)}` : ""}>
      <Gate v={v} what="the horizon table">
        {(r) => (
          <>
            <table className="tbl mc-tbl">
              <thead>
                <tr>
                  <th className="l">Horizon</th>
                  <th title="1 path in 20 ends below this">5%</th><th title="1 path in 4 ends below this">25%</th>
                  <th title="The middle outcome">Median</th>
                  <th title="1 path in 4 ends above this">75%</th><th title="1 path in 20 ends above this">95%</th>
                  <th title="Share of paths above today's price at that date">P(up)</th>
                  <th title="Median price vs spot">Median Δ</th>
                  <th title="Average outcome vs spot (expected return)">Mean Δ</th>
                  <th title="The median path's growth per year (compound annual)">Median CAGR</th>
                </tr>
              </thead>
              <tbody>
                {r.horizons.map((h) => (
                  <tr key={h.days} className={KEY_HORIZONS.has(h.label) ? "mc-key" : undefined}>
                    <td className="l"><b>{h.label}</b> <small className="muted">{dateIST(h.ts)}</small></td>
                    <td>{inr(h.p5)}</td><td>{inr(h.p25)}</td><td className="b">{inr(h.p50)}</td><td>{inr(h.p75)}</td><td>{inr(h.p95)}</td>
                    <td><ProbBar p={h.prob_up} label="" width={44} /></td>
                    <td><Ret v={h.median_ret} /></td>
                    <td><Ret v={h.exp_ret} /></td>
                    <td><Ret v={h.ann_median_ret} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mc-note">
              Read a row as: “after 1Y, 1 path in 20 is below the 5% column and 1 in 20 above the 95% column.” The spread widens
              roughly with the square root of time. Every model here uses the same drift ({fracPct(r.inputs.drift_annual, 1)}/yr, {r.drift}),
              so medians differ between models only through the shape of risk.
            </p>
          </>
        )}
      </Gate>
    </Panel>
  );
}

// ------------------------------------------------------------------ targets
function TargetsPanel({ v, onQuick }: { v: View; onQuick: (levels: number[]) => void }) {
  const r = v.r;
  const end = r?.horizons[r.horizons.length - 1];
  return (
    <Panel code="TGT" title="Target prices" area="targets" sub={end ? `within ${hName(end.label, end.days)}` : ""}>
      <Gate v={v} what="the target odds">
        {(r) => !r.targets.length ? (
          <div className="mc-empty">
            <p>No target prices yet. Add up to 6 levels — a stop-loss, a profit target — in the setup bar (₹ price or ±% from spot), or:</p>
            <button type="button" className="btn" disabled={v.running}
              onClick={() => onQuick([0.8, 0.9, 1.1, 1.2].map((m) => Math.round(r.spot * m * 100) / 100))}>
              TEST ±10% AND ±20%
            </button>
          </div>
        ) : (
          <>
            <table className="tbl mc-tbl">
              <thead>
                <tr>
                  <th className="l">Level</th><th title="Distance from today's price">From spot</th>
                  <th className="l" title="Chance a daily close reaches the level at least once before the horizon">Touches</th>
                  <th className="l" title="Chance the price is still beyond the level on the last day">Ends beyond</th>
                </tr>
              </thead>
              <tbody>
                {r.targets.map((t) => (
                  <tr key={t.price}>
                    <td className="l b">{t.direction === "up" ? <span className="up">▲</span> : <span className="down">▼</span>} {inr(t.price)}</td>
                    <td><Ret v={t.price / r.spot - 1} /></td>
                    <td className="l"><PBar p={t.prob_touch} /></td>
                    <td className="l"><PBar p={t.prob_end_beyond} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mc-note">
              <b>Touches</b> = a daily close at or beyond the level at any point before the horizon (intraday highs and lows make a touch a
              little more likely); <b>ends beyond</b> = still there on the last day. A level 1 path in 3 touches is far from certain.
            </p>
          </>
        )}
      </Gate>
    </Panel>
  );
}

function PBar({ p }: { p: number | null | undefined }) {
  return (
    <span className="mc-pbar">
      <span className="mc-pbar-track" aria-hidden="true"><i style={{ width: `${ok(p) ? Math.min(100, Math.max(0, p * 100)) : 0}%` }} /></span>
      <b>{pctOf(p)}</b>
    </span>
  );
}

// ------------------------------------------------------------------ terminal histogram
function HistogramPanel({ v }: { v: View }) {
  const r = v.r;
  const end = r?.horizons[r.horizons.length - 1];
  const markers = useMemo<HistMarker[]>(() => {
    if (!r) return [];
    const t = r.terminal;
    return [
      { value: r.spot, label: "Spot", kind: "spot" },
      { value: t.p5, label: "5%" },
      { value: t.median, label: "Median" },
      { value: t.p95, label: "95%" },
      ...r.targets.map((x): HistMarker => ({ value: x.price, label: `T ${inr(x.price, 0)}`, kind: "target" })),
    ];
  }, [r]);
  return (
    <Panel code="DIST" title={end ? `Where the price ends after ${hName(end.label, end.days)}` : "Price distribution at horizon"} area="hist" pad
      sub={r ? `${fx(r.paths, 0)} simulated outcomes` : ""}>
      <Gate v={v} what="the distribution">
        {(r) => {
          const t = r.terminal;
          return (
            <>
              <Histogram label={`${r.symbol}: simulated price after ${r.horizon_days} trading days`} edges={t.edges} counts={t.counts}
                splitAt={r.spot} markers={markers} height={230} busy={v.running}
                caption="Share of simulated paths ending in each price bucket (red = below today's price, green = above). The extreme 0.5 % on each side is cut off to keep the shape readable." />
              <Stats min={104}>
                <Stat label="1st pct" value={inr(t.p1)} sub="1 in 100 below" />
                <Stat label="5th pct" value={inr(t.p5)} />
                <Stat label="Median" value={inr(t.median)} />
                <Stat label="Mean" value={inr(t.mean)} />
                <Stat label="95th pct" value={inr(t.p95)} />
                <Stat label="99th pct" value={inr(t.p99)} sub="1 in 100 above" />
                <Stat label="Std dev" value={inr(t.std)} sub={`${fx((t.std / r.spot) * 100, 0)}% of spot`} />
              </Stats>
            </>
          );
        }}
      </Gate>
    </Panel>
  );
}

// ------------------------------------------------------------------ inputs used
function Row({ k, children, title }: { k: string; children: ReactNode; title?: string }) {
  return <><dt title={title}>{k}</dt><dd>{children}</dd></>;
}

function InputsPanel({ v }: { v: View }) {
  return (
    <Panel code="INPT" title="Inputs used" area="inputs" sub={v.r ? `fitted on ${v.r.inputs.first} → ${v.r.inputs.last}` : ""}>
      <Gate v={v} what="the fitted parameters">
        {(r) => {
          const i = r.inputs;
          const g = i.garch;
          const halfLife = g && g.persistence > 0 && g.persistence < 1 ? Math.log(0.5) / Math.log(g.persistence) : null;
          return (
            <div className="mc-inputs">
              <section>
                <h3>Data</h3>
                <dl className="mc-dl">
                  <Row k="Daily bars">{fx(i.bars, 0)} <span className="muted">({i.first} → {i.last})</span></Row>
                  <Row k="Calibration window" title="History the model parameters were fitted on">{fx(i.lookback_years, 1)} years</Row>
                  <Row k="Start price">{inr(r.spot)} <span className="muted">{r.spot_source === "live" ? "live" : "last close"}</span></Row>
                  <Row k="Random seed" title="Same seed + same data = same paths">{r.seed}</Row>
                </dl>
              </section>
              <section>
                <h3>Return &amp; volatility</h3>
                <dl className="mc-dl">
                  <Row k="Historical volatility">{fracPct(i.sigma_annual, 1)}<span className="muted">/yr</span></Row>
                  <Row k="Historical average" title="Mean log return over the window — shown for context, never simulated">
                    <Ret v={i.mu_annual} /><span className="muted">/yr · context only</span>
                  </Row>
                  <Row k="Drift simulated"><Ret v={i.drift_annual} /><span className="muted">/yr · {r.drift}</span></Row>
                  <Row k="Beta vs NIFTY" title="Blume-adjusted: 0.67 × measured β + 0.33">{ok(i.beta) ? fx(i.beta, 2) : <span className="warn">not measured (β = 1)</span>}</Row>
                  <Row k="Risk-free · ERP">{fracPct(i.rf, 1)} · {fracPct(i.erp, 1)}</Row>
                </dl>
                <p className="mc-note">{i.drift_note}</p>
              </section>
              {g && (
                <section>
                  <h3>GARCH(1,1)-t</h3>
                  <dl className="mc-dl">
                    <Row k="Volatility now">{fracPct(g.current_vol, 1)} <span className="muted">vs long-run {fracPct(g.long_run_vol, 1)}</span></Row>
                    <Row k="Persistence α+β" title="How long volatility shocks last">{fx(g.persistence, 3)}{halfLife !== null && <span className="muted"> · shocks halve in ≈{fx(halfLife, 0)} days</span>}</Row>
                    <Row k="α · β" title="α: reaction to yesterday's move · β: memory of past volatility">{fx(g.alpha, 3)} · {fx(g.beta, 3)}</Row>
                    <Row k="ω">{ok(g.omega) ? g.omega.toExponential(2) : "—"}</Row>
                    <Row k="Tail ν" title="Student-t degrees of freedom: lower = fatter tails (normal ≈ ∞)">{fx(g.nu, 1)} <span className="muted">{g.nu < 6 ? "fat tails" : g.nu < 15 ? "moderate tails" : "near-normal"}</span></Row>
                  </dl>
                </section>
              )}
              {i.jump && (
                <section>
                  <h3>Jumps</h3>
                  <dl className="mc-dl">
                    <Row k="Jumps per year" title="Days with a move beyond 3 robust standard deviations">{fx(i.jump.lambda, 1)}</Row>
                    <Row k="Average jump"><Ret v={i.jump.mu_j} d={2} /></Row>
                    <Row k="Jump size (1 sd)">±{fracPct(i.jump.sigma_j, 1)}</Row>
                  </dl>
                </section>
              )}
              {i.regime && (
                <section>
                  <h3>Regimes</h3>
                  <table className="tbl mc-tbl mc-mini">
                    <thead><tr><th className="l">State</th><th>Today</th><th title="Context only — the simulation uses the common drift">Return/yr</th><th>Vol/yr</th><th title="Average length of a spell in this state">Spell</th></tr></thead>
                    <tbody>
                      {i.regime.labels.map((lab, k) => {
                        const ps = i.regime!.p_stay[k];
                        return (
                          <tr key={lab} className={k === i.regime!.current ? "sel" : undefined}>
                            <td className="l">{lab}{k === i.regime!.current && <span className="muted"> · now</span>}</td>
                            <td>{pctOf(i.regime!.probs[k])}</td>
                            <td><Ret v={i.regime!.mu_annual[k]} /></td>
                            <td>{fracPct(i.regime!.vol_annual[k], 0)}</td>
                            <td>{ok(ps) && ps < 1 ? `${fx(1 / (1 - ps), 0)}d` : "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </section>
              )}
              {(ok(i.bootstrap_block) || !!i.components?.length) && (
                <section>
                  <h3>Model mix</h3>
                  <dl className="mc-dl">
                    {ok(i.bootstrap_block) && <Row k="Bootstrap block">{fx(i.bootstrap_block, 0)} days (mean)</Row>}
                    {!!i.components?.length && (
                      <Row k="Ensemble">{i.components.map((c) => `${MODEL_INFO[c.model]?.label.toLowerCase() ?? c.model} ${pctOf(c.weight)}`).join(" · ")}</Row>
                    )}
                  </dl>
                </section>
              )}
              <Warnings items={r.warnings} label="Simulation warnings" />
            </div>
          );
        }}
      </Gate>
    </Panel>
  );
}

// ------------------------------------------------------------------ calibration backtest
const BT_H: HKey[] = ["1M", "3M", "6M", "1Y"];
const BT_LB = ["1", "2", "3", "5"] as const;
type BtLb = (typeof BT_LB)[number];
const BT_ORIGINS = ["20", "40", "80"] as const;
type BtOrigins = (typeof BT_ORIGINS)[number];
const SPAN_MIN = 0.25;          // years of origins wanted after lookback + horizon, so the test is not a single window

function CalibrationPanel({ symbol, cfg, years }: { symbol: string; cfg: McCfg; years: number | null }) {
  // 3-month forecasts fitted on 2 years: ~11 independent windows in a 5-year lake (a 1-year test has ~2)
  const [bh, setBh] = useState<HKey>("3M");
  const [lb, setLb] = useState<BtLb>("2");
  const [origins, setOrigins] = useState<BtOrigins>("40");
  const bt = useRun(research.backtest, btMemo, rememberBt);
  const shown = bt.done && bt.done.sym === symbol ? bt.done : null;
  const res = shown?.res ?? null;
  const running = bt.busy?.sym === symbol;
  const elapsed = useElapsed(running ? bt.busy!.since : 0);
  const error = bt.err && bt.err.sym === symbol ? bt.err.e : null;

  const need = (l: string, h: HKey) => Number(l) + H_DAYS[h] / 252;
  const fits = (l: string, h: HKey) => years === null || years >= need(l, h) + SPAN_MIN;
  // when the history arrives, step down to a lookback that fits (keeps the user's choice when it does)
  useEffect(() => {
    if (years === null || fits(lb, bh)) return;
    const best = [...BT_LB].reverse().find((l) => fits(l, bh));
    if (best) setLb(best);
  }, [years, bh]); // eslint-disable-line react-hooks/exhaustive-deps
  const feasible = fits(lb, bh);

  const req: McBacktestRequest = { horizon_days: H_DAYS[bh], model: cfg.model, drift: cfg.drift, lookback_years: Number(lb), origins: Number(origins), paths: 2000 };
  const start = () => { if (symbol) bt.run(symbol, req); };
  const differs = res && (res.model !== cfg.model || (res.drift !== cfg.drift && cfg.drift !== "quant"));

  const bar = (
    <>
      <span className="mc-bt-label">Horizon</span>
      <Seg label="Backtest horizon" value={bh} onChange={setBh}
        options={BT_H.map((k) => ({ v: k, label: k, title: `Score forecasts ${hName(k, H_DAYS[k])} ahead` }))} />
      <span className="mc-bt-label">Fit on</span>
      <Seg label="Fit each past forecast on" value={lb} onChange={setLb}
        options={BT_LB.map((l) => ({
          v: l, label: `${l}Y`,
          title: fits(l, bh) ? `Fit each past forecast on the ${l} years before its date` : `Needs ≈${fx(need(l, bh) + SPAN_MIN, 1)} years of history; this stock has ${fx(years, 1)}`,
        }))} />
      <span className="mc-bt-label">Dates</span>
      <Seg label="Number of past dates" value={origins} onChange={setOrigins}
        options={BT_ORIGINS.map((o) => ({ v: o, label: o, title: `${o} past forecast dates (more = slower, but overlapping windows add little information)` }))} />
      <span className="grow" />
      <button type="button" className="btn primary" onClick={start} disabled={!symbol || running || !feasible}
        title={feasible ? `Replay ${MODEL_INFO[cfg.model].label.toLowerCase()} / ${cfg.drift} drift at ${origins} past dates` : "Not enough history for this horizon and fitting window"}>
        {running ? `BACKTESTING ${fx(elapsed, 0)}s` : "BACKTEST THIS MODEL"}
      </button>
    </>
  );

  return (
    <Panel code="CAL" title="Calibration — can these bands be trusted?" area="calib" bar={bar}
      sub={`model ${MODEL_INFO[cfg.model].label.toLowerCase()} · ${cfg.drift === "quant" ? "capm (quant drift can't be replayed)" : cfg.drift} drift · 2,000 paths per date`}>
      {!feasible && (
        <p className="mc-note mc-pad warn">
          {fx(years, 1)} years of history is not enough for a {bh} backtest fitted on {lb} years (needs ≈{fx(need(lb, bh) + SPAN_MIN, 1)}). Pick a shorter horizon or fitting window.
        </p>
      )}
      {res ? (
        <div className={`mc-calib ${running ? "busy" : ""}`}>
          <p className="mc-verdict" role="status">{res.verdict}</p>
          {differs && (
            <p className="mc-note warn">This backtest is of {MODEL_INFO[res.model]?.label.toLowerCase() ?? res.model} / {res.drift} drift; the setup now uses {MODEL_INFO[cfg.model].label.toLowerCase()} / {cfg.drift} — run it again to check that.</p>
          )}
          <Stats min={124}>
            <Stat label="Skill vs random walk" value={<Ret v={res.skill} d={0} />}
              sub="CRPS: + better, − worse" title={`Whole-distribution score (CRPS) ${fx(res.crps, 2)}% of spot vs ${fx(res.crps_naive, 2)}% for a zero-drift random walk with trailing 1-year volatility. Skill = 1 − ratio.`} />
            <Stat label="CRPS" value={`${fx(res.crps, 2)}%`} sub={`random walk ${fx(res.crps_naive, 2)}%`} title="Average distance between the forecast distribution and what happened, in % of spot (lower is better)" />
            <Stat label="Direction hit" value={pctOf(res.direction_hit)} sub="median's direction (coin ≈ 50%)"
              title="How often the simulated median pointed the right way. With a positive drift the median nearly always points up, so this mostly measures how often the stock rose." />
            <Stat label="Independent windows" value={`≈ ${fx(res.effective_n, 0)}`} sub={`${res.origins} dates overlap`}
              title="Forecast windows overlap, so the honest sample size is the history span divided by the horizon" tone={res.effective_n < 5 ? "amber" : undefined} />
            <Stat label="Horizon · fit" value={`${res.horizon_days}d`} sub={`${shown!.req.lookback_years}y fit · ${fx(res.elapsed_ms / 1000, 1)}s`} />
          </Stats>
          <div className="mc-calib-grid">
            <div>
              <h3>How often the bands held the real price</h3>
              <CoverageChart data={res.coverage} n={res.effective_n} />
            </div>
            <div>
              <h3>Where the real price fell in each forecast</h3>
              <PitHistogram bins={res.pit_hist} height={170} />
            </div>
          </div>
          <Replay bt={res} />
          <Warnings items={res.warnings} label="Backtest warnings" />
        </div>
      ) : running ? (
        <Empty>Refitting the model at {origins} past dates on the data available then… {fx(elapsed, 0)} s (usually 5–30 s).</Empty>
      ) : error ? (
        <ErrorNote error={error} onRetry={start} />
      ) : (
        <div className="mc-empty">
          <p>
            A simulation is only worth its calibration. This replays the current model at {origins} past dates, each time using only the data
            available then, and checks how often the real price {bh === "1Y" ? "a year" : hName(bh, H_DAYS[bh])} later landed inside its 50 / 80 / 90 % bands.
            Well-calibrated 90 % bands hold the price about 90 % of the time; a lot less means the fan is too narrow.
          </p>
        </div>
      )}
      {!!error && res && <ErrorNote error={error} onRetry={start} />}
    </Panel>
  );
}

/** Each past forecast's 90 % band at its target date vs the price that actually came. */
function Replay({ bt }: { bt: McBacktest }) {
  const data = useMemo(() => {
    const pts = bt.points.map((p) => ({ ...p, x: (p as { target_ts?: number }).target_ts ?? p.ts })).filter((p) => ok(p.x)).sort((a, b) => a.x - b.x);
    return {
      x: pts.map((p) => p.x),
      series: [
        { name: "Actual price", values: pts.map((p) => p.realized), tone: "text" as const },
        { name: "Forecast median", values: pts.map((p) => p.p50), tone: 1 as const, dashed: true },
      ],
      bands: [{ name: "Forecast 5–95%", lo: pts.map((p) => p.p5), hi: pts.map((p) => p.p95), tone: 1 as const, opacity: 0.18 }],
      outside: pts.filter((p) => p.realized < p.p5 || p.realized > p.p95).length,
      n: pts.length,
    };
  }, [bt]);
  if (data.n < 2) return null;
  return (
    <div>
      <h3>Past forecasts vs what happened</h3>
      <LineChart label={`${bt.symbol}: ${bt.horizon_days}-day forecasts made at ${data.n} past dates vs the realised price`} x={data.x}
        series={data.series} bands={data.bands} height={200} endLabels={false}
        caption={`Plotted at each forecast's target date. The actual price left the 5–95 % band ${data.outside} of ${data.n} times (ideal ≈ ${fx(data.n * 0.1, 0)}).`} />
    </div>
  );
}

// ------------------------------------------------------------------ desk
export default function MonteCarlo() {
  const symbol = useLive((s) => s.researchSymbol);
  const hist = useHistoryInfo(symbol);
  const [cfg, setCfgState] = useState<McCfg>(loadCfg);
  const setCfg = useCallback((p: Partial<McCfg>) => setCfgState((c) => {
    const n = { ...c, ...p };
    setPref(CFG_KEY, JSON.stringify(n));
    return n;
  }), []);
  const [targets, setTargets] = useState<number[]>(() => (simMemo && simMemo.sym === symbol ? simMemo.req.targets : []));
  const sim = useRun(research.montecarlo, simMemo, rememberSim);

  const shown = sim.done && sim.done.sym === symbol ? sim.done : null;
  const running = sim.busy?.sym === symbol;
  const elapsed = useElapsed(running ? sim.busy!.since : 0);
  const error = sim.err && sim.err.sym === symbol ? sim.err.e : null;
  const req = useMemo(() => toReq(cfg, targets), [cfg, targets]);
  const dirty = !!shown && reqKey(shown.req) !== reqKey(req);
  const simRun = sim.run;
  const run = useCallback(() => { if (symbol) simRun(symbol, req); }, [symbol, req, simRun]);

  // auto-run once per stock: when the desk opens on a stock without a cached result, and whenever the stock changes
  const autoFor = useRef<string | null>(simMemo?.sym ?? null);
  useEffect(() => {
    if (!symbol || autoFor.current === symbol) return;
    autoFor.current = symbol;
    setTargets([]);                                   // price levels of another stock mean nothing here
    sim.run(symbol, toReq(cfg, []));
  }, [symbol]); // eslint-disable-line react-hooks/exhaustive-deps

  const quick = (levels: number[]) => {
    setTargets(levels);
    if (symbol) sim.run(symbol, toReq(cfg, levels));
  };

  const v: View = { r: shown?.res ?? null, running, error, retry: run };
  const spot = shown?.res.spot ?? hist.data?.spot ?? null;
  const runLabel = running ? `RUNNING ${fx(elapsed, 1)}s` : dirty ? "RUN ▶ (changed)" : "RUN ▶";

  return (
    <div className="desk desk-mc rs-desk">
      <SymbolPicker area="pick" actions={
        <>
          <span className="mc-runinfo">{cfg.h} · {PATH_INFO[cfg.paths].label} paths · {MODEL_INFO[cfg.model].label.toLowerCase()}</span>
          <button type="button" className="btn primary" onClick={run} disabled={!symbol || running}
            title={dirty ? "Settings changed since the last run — run again to update" : "Run the simulation with these settings"}>{runLabel}</button>
        </>
      } />
      <Controls cfg={cfg} setCfg={setCfg} targets={targets} setTargets={setTargets} spot={spot} error={error}
        hasResult={!!shown} retry={run} warnings={shown?.res.warnings ?? []} />
      <FanPanel v={v} />
      <div className="stack mc-side" style={{ gridArea: "side" }}>
        <OutcomePanel v={v} />
        <RiskPanel v={v} />
      </div>
      <HorizonsPanel v={v} />
      <TargetsPanel v={v} onQuick={quick} />
      <HistogramPanel v={v} />
      <InputsPanel v={v} />
      <CalibrationPanel symbol={symbol} cfg={cfg} years={hist.data ? hist.data.years : null} />
    </div>
  );
}
