/* AI ANALYST desk — one stock read end to end. The gateway gathers a dossier (daily price history and technicals,
 * Screener.in fundamentals through the same parser as the SCREENER desk, the quant model with its measured accuracy,
 * a Monte Carlo range, the market backdrop and recent news) and the AI model set in .env writes a 6-month / 1 / 2 /
 * 3-year view from it. Without a model, over the daily quota or when the call fails, the same dossier feeds a fixed
 * rule-based view, and the desk always says which of the two is on screen. Each horizon's base target is checked
 * against the Monte Carlo 5–95 % band so a view that strays from the stock's own volatility stands out, and the
 * dossier is shown in full so every claim can be traced back to a number. */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, FocusEvent as ReactFocusEvent, KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent, ReactNode } from "react";
import { fmtLabel, inr } from "../components/research/charts";
import { Disclaimer, ErrorNote, Stat, Stats, Warnings, errorText } from "../components/research/common";
import { SymbolPicker, useHistoryInfo } from "../components/research/SymbolPicker";
import { Chg, Panel } from "../components/ui";
import { ApiError } from "../lib/api";
import { dateIST, fx, pct, timeIST } from "../lib/format";
import { research } from "../lib/research";
import type { AiConfig, AiDossier, AiHorizonView, AiJob, AiReport, AiScenario, AiStep, Conf, Stance } from "../lib/research";
import { useLive } from "../lib/store";
import "./ai.css";

const POLL_MS = 1500;
const MAX_POLL_FAILS = 6;            // consecutive network failures before polling pauses (RESUME continues)
const IST = 19800;
const DAY = 86400;
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

type HKey = "6M" | "1Y" | "2Y" | "3Y";
const HORIZONS: { key: HKey; months: number; name: string }[] = [
  { key: "6M", months: 6, name: "6 months" }, { key: "1Y", months: 12, name: "1 year" },
  { key: "2Y", months: 24, name: "2 years" }, { key: "3Y", months: 36, name: "3 years" },
];

// ------------------------------------------------------------------ formatting
const ok = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const texts = (v: unknown): string[] => (Array.isArray(v) ? v.filter((s): s is string => typeof s === "string" && !!s.trim()) : []);
/** Target prices: no paise above ₹1,000. */
const rs = (v: number | null | undefined) => fmtLabel("inr", v);
const probText = (p: number | null | undefined) => (ok(p) ? `${fx(p * 100, 0)}%` : "—");

/** "2026-10-14" → epoch seconds of that day's IST midnight (the lake's bar timestamps). */
function isoToTs(iso: string | null | undefined): number | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso ?? "");
  return m ? Date.UTC(+m[1], +m[2] - 1, +m[3]) / 1000 - IST : null;
}

/** Same IST calendar day `months` later (a month-end day rolls over, e.g. 31 Aug + 6M → 3 Mar). */
function addMonths(ts: number, months: number): number {
  const d = new Date((ts + IST) * 1000);
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + months, d.getUTCDate()) / 1000 - IST;
}

function monthYear(ts: number): string {
  const d = new Date((ts + IST) * 1000);
  return `${MON[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

function shortDate(ts: number): string {
  const d = new Date((ts + IST) * 1000);
  return `${d.getUTCDate()} ${MON[d.getUTCMonth()]} '${String(d.getUTCFullYear()).slice(2)}`;
}

const todayIST = () => dateIST(Date.now() / 1000);
const generatedText = (ts: number) => (ok(ts) && ts > 0 ? `${dateIST(ts)} ${timeIST(ts)} IST` : "—");

function mmss(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

// ------------------------------------------------------------------ stance & confidence
const STANCE: Record<Stance, { glyph: string; text: string; tone: "up" | "down" | "flat" }> = {
  bullish: { glyph: "▲▲", text: "Bullish", tone: "up" },
  moderately_bullish: { glyph: "▲", text: "Moderately bullish", tone: "up" },
  neutral: { glyph: "◆", text: "Neutral", tone: "flat" },
  moderately_bearish: { glyph: "▼", text: "Moderately bearish", tone: "down" },
  bearish: { glyph: "▼▼", text: "Bearish", tone: "down" },
};
const stanceOf = (s: unknown) => (typeof s === "string" && Object.hasOwn(STANCE, s) ? STANCE[s as Stance] : STANCE.neutral);
const confOf = (c: unknown): Conf => (c === "high" || c === "medium" ? c : "low");

function StanceBadge({ stance, big }: { stance: unknown; big?: boolean }) {
  const s = stanceOf(stance);
  return (
    <span className={`ai-stance ${s.tone}${big ? " big" : ""}`} title="Expected return compared with the CAPM-implied return in the dossier">
      <span aria-hidden="true">{s.glyph}</span>{s.text}
    </span>
  );
}

function ConfMeter({ c }: { c: unknown }) {
  const conf = confOf(c);
  const n = conf === "high" ? 3 : conf === "medium" ? 2 : 1;
  return (
    <span className="ai-conf" title="High only when independent kinds of evidence agree and the data is complete">
      <span className="ai-pips" aria-hidden="true">{[0, 1, 2].map((i) => <i key={i} className={i < n ? "on" : ""} />)}</span>
      {conf} confidence
    </span>
  );
}

// ------------------------------------------------------------------ report helpers
/** Spot the report was written against: the dossier's "Spot" fact, else implied by the targets (return_pct is
 *  recomputed server-side from target vs spot, so target / (1 + return) recovers it). */
function reportSpot(rep: AiReport): number | null {
  for (const s of rep.dossier?.sections ?? []) {
    for (const f of s.facts ?? []) {
      if (!/^spot$/i.test((f.label ?? "").trim())) continue;
      const m = /₹\s*([\d,]+(?:\.\d+)?)/.exec(f.value ?? "");
      const v = m ? Number(m[1].replace(/,/g, "")) : NaN;
      if (v > 0) return v;
    }
  }
  const xs = (rep.horizons ?? []).flatMap((h) => h.scenarios ?? [])
    .filter((s) => ok(s.target_price) && ok(s.return_pct) && s.return_pct > -95)
    .map((s) => s.target_price / (1 + s.return_pct / 100))
    .sort((a, b) => a - b);
  return xs.length ? xs[xs.length >> 1] : null;
}

interface HPoint {
  key: HKey; name: string; ts: number;           // ts ≈ the analysis date + the horizon in calendar months
  view: AiHorizonView | null;
  bear: AiScenario | null; base: AiScenario | null; bull: AiScenario | null;
}

function horizonPoints(rep: AiReport, asofTs: number): HPoint[] {
  return HORIZONS.map((h) => {
    const view = (rep.horizons ?? []).find((v) => v?.horizon === h.key) ?? null;
    const sc = (name: AiScenario["name"]) => view?.scenarios?.find((s) => s?.name === name && ok(s.target_price) && s.target_price > 0) ?? null;
    return { key: h.key, name: h.name, ts: addMonths(asofTs, h.months), view, bear: sc("bear"), base: sc("base"), bull: sc("bull") };
  });
}

// ------------------------------------------------------------------ state (results survive desk switches)
interface Shown { sym: string; rep: AiReport; cached: boolean; via: "job" | "saved"; steps: AiStep[] | null }

interface DeskState {
  sym: string;
  shown: Shown | null;
  saved: "loading" | "none" | "ok" | "error";   // the last saved report (research.aiLatest)
  savedErr: unknown;
  job: AiJob | null;
  jobId: string | null;          // job being polled
  stalledId: string | null;      // polling paused after repeated network errors
  starting: boolean;
  startErr: unknown;
  pollErr: unknown;
}

const reportMemo = new Map<string, Shown>();
const jobMemo = new Map<string, string>();         // symbol → id of a job still running (polling resumes on return)
let cfgMemo: AiConfig | null = null;

const initial = (sym: string): DeskState => ({
  sym, shown: reportMemo.get(sym) ?? null, saved: "loading", savedErr: null,
  job: null, jobId: jobMemo.get(sym) ?? null, stalledId: null, starting: false, startErr: null, pollErr: null,
});

/** Keep whichever report is newer (a just-finished run keeps its steps and cached flag). */
const newer = (a: Shown | null, b: Shown): Shown =>
  a && a.sym === b.sym && (a.rep.generated_at ?? 0) >= (b.rep.generated_at ?? 0) ? a : b;

function useAiConfig() {
  const [cfg, setCfg] = useState<AiConfig | null>(cfgMemo);
  const [cfgErr, setCfgErr] = useState<unknown>(null);
  const loadCfg = useCallback(() => {
    research.aiConfig().then(
      (c) => { cfgMemo = c; setCfg(c); setCfgErr(null); },
      (e) => setCfgErr(e),
    );
  }, []);
  useEffect(() => { loadCfg(); }, [loadCfg]);
  return { cfg, cfgErr, loadCfg };
}

/** Saved report, job start and 1.5 s polling for the research symbol; every async result is tagged with the symbol it
 *  was for, so switching stocks mid-flight never shows one stock's report under another's name. */
function useAiDesk(symbol: string, onFinished: () => void) {
  const [st, setSt] = useState<DeskState>(() => initial(symbol));
  const [savedNonce, setSavedNonce] = useState(0);
  const cur = st.sym === symbol ? st : initial(symbol);
  const patch = useCallback((sym: string, p: Partial<DeskState> | ((s: DeskState) => Partial<DeskState>)) => {
    setSt((s) => (s.sym === sym ? { ...s, ...(typeof p === "function" ? p(s) : p) } : s));
  }, []);

  useEffect(() => {
    setSt((s) => (s.sym === symbol ? s : initial(symbol)));
    let alive = true;
    research.aiLatest(symbol).then(
      (rep) => {
        if (!alive) return;
        patch(symbol, (s) => {
          const shown = newer(s.shown, { sym: symbol, rep, cached: false, via: "saved", steps: null });
          reportMemo.set(symbol, shown);
          return { saved: "ok", savedErr: null, shown };
        });
      },
      (e) => {
        if (!alive) return;
        patch(symbol, e instanceof ApiError && e.status === 404 ? { saved: "none", savedErr: null } : { saved: "error", savedErr: e });
      },
    );
    return () => { alive = false; };
  }, [symbol, savedNonce, patch]);

  const finish = useCallback((sym: string, j: AiJob) => {
    jobMemo.delete(sym);
    if (j.status === "done" && j.result) {
      const shown: Shown = { sym, rep: j.result, cached: !!j.cached, via: "job", steps: j.steps ?? null };
      reportMemo.set(sym, shown);
      patch(sym, { job: j, jobId: null, stalledId: null, pollErr: null, shown, saved: "ok", savedErr: null });
    } else {
      patch(sym, { job: j, jobId: null, stalledId: null, pollErr: null });
    }
    onFinished();
  }, [patch, onFinished]);

  const jobId = cur.jobId;
  useEffect(() => {
    if (!jobId) return;
    const sym = symbol;
    let alive = true;
    let timer = 0;
    let fails = 0;
    const tick = () => {
      research.aiJob(jobId).then(
        (j) => {
          if (!alive) return;
          fails = 0;
          if (j.status === "done" || j.status === "error") finish(sym, j);
          else {
            patch(sym, { job: j, pollErr: null });
            timer = window.setTimeout(tick, POLL_MS);
          }
        },
        (e) => {
          if (!alive) return;
          if (e instanceof ApiError && e.status === 404) {   // job expired (kept 1 h) or gateway restarted: show what was saved
            jobMemo.delete(sym);
            patch(sym, { jobId: null, pollErr: null, saved: "loading" });
            setSavedNonce((n) => n + 1);
            return;
          }
          if (e instanceof ApiError && (e.status === 401 || e.status === 403)) {
            jobMemo.delete(sym);
            patch(sym, { jobId: null, pollErr: e });
            return;
          }
          fails += 1;
          if (fails >= MAX_POLL_FAILS) {
            patch(sym, { jobId: null, stalledId: jobId, pollErr: e });
            return;
          }
          patch(sym, { pollErr: e });
          timer = window.setTimeout(tick, POLL_MS * (fails + 1));
        },
      );
    };
    tick();
    return () => { alive = false; window.clearTimeout(timer); };
  }, [symbol, jobId, finish, patch]);

  const start = useCallback((refresh: boolean) => {
    const sym = symbol;
    patch(sym, { starting: true, startErr: null, pollErr: null, stalledId: null });
    research.aiStart(sym, refresh).then(
      (j) => {
        patch(sym, { starting: false });
        if (j.status === "done" || j.status === "error") finish(sym, j);
        else {
          jobMemo.set(sym, j.id);
          patch(sym, { job: j, jobId: j.id });
        }
      },
      (e) => patch(sym, { starting: false, startErr: e }),
    );
  }, [symbol, patch, finish]);

  const resume = useCallback(() => patch(symbol, (s) => ({ jobId: s.stalledId, stalledId: null, pollErr: null })), [symbol, patch]);
  const retrySaved = useCallback(() => {
    patch(symbol, { saved: "loading", savedErr: null });
    setSavedNonce((n) => n + 1);
  }, [symbol, patch]);

  return { cur, start, resume, retrySaved };
}

function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [active]);
  return now;
}

// ------------------------------------------------------------------ AI settings + job progress
function ConfigLine({ cfg, err, onRetry }: { cfg: AiConfig | null; err: unknown; onRetry: () => void }) {
  if (!cfg && err) {
    return (
      <div className="ai-cfg">
        <span className="err">Could not read the AI settings: {errorText(err)}</span>
        <button type="button" className="btn" onClick={onRetry}>RETRY</button>
      </div>
    );
  }
  if (!cfg) return <div className="ai-cfg muted">Reading the AI settings…</div>;
  if (!cfg.enabled) {
    return (
      <div className="ai-cfg off" role="status">
        <span className="chip amber">RULE-BASED</span>
        <span><b>AI not configured</b> — set AI_PROVIDER / AI_API_KEY / AI_MODEL in .env; showing the rule-based view.</span>
        {cfg.reason && <small>{cfg.reason}</small>}
      </div>
    );
  }
  const userLeft = cfg.user_limit - cfg.user_used_today;
  const allLeft = cfg.daily_limit - cfg.used_today;
  return (
    <div className="ai-cfg" role="status">
      <span className="chip violet" title="Analyses are written by this model from the dossier">AI</span>
      <span className="mono"><b>{cfg.provider || "—"}</b> · {cfg.model || "—"}</span>
      <span className="ai-quota" title="Model calls today. Opening a saved report of today uses none.">
        today: you <b>{fx(cfg.user_used_today, 0)}/{fx(cfg.user_limit, 0)}</b> · all users <b>{fx(cfg.used_today, 0)}/{fx(cfg.daily_limit, 0)}</b>
      </span>
      {(userLeft <= 0 || allLeft <= 0) && (
        <span className="warn">⚠ {userLeft <= 0 ? "Your" : "Today's"} AI quota is used up — a new analysis shows the rule-based view; saved reports still open.</span>
      )}
    </div>
  );
}

const STEP_GLYPH: Record<string, string> = { pending: "○", done: "✓", skipped: "–", error: "✕" };

function StepList({ steps, compact }: { steps: AiStep[]; compact?: boolean }) {
  return (
    <ol className={`ai-steps${compact ? " compact" : ""}`}>
      {steps.map((s) => (
        <li key={s.key} className={`ai-step ${s.status}`} title={s.detail ? `${s.label}: ${s.detail}` : s.label}>
          <span className="ai-step-icon" aria-hidden="true">{s.status === "running" ? <span className="ai-spin" /> : STEP_GLYPH[s.status] ?? "•"}</span>
          <span className="ai-step-label">{s.label}<span className="sr-only"> — {s.status}</span></span>
          {s.detail && <span className="ai-step-detail">{s.detail}</span>}
        </li>
      ))}
    </ol>
  );
}

function Progress({ st, symbol, onRetry, onResume }: { st: DeskState; symbol: string; onRetry: () => void; onResume: () => void }) {
  const live = st.starting || !!st.jobId;
  const now = useNow(live);
  const job = st.job;
  if (st.startErr) {
    return <div className="ai-prog"><div className="ai-prog-head err">Could not start the analysis of {symbol}.</div><ErrorNote error={st.startErr} onRetry={onRetry} /></div>;
  }
  if (st.stalledId) {
    return (
      <div className="ai-prog">
        <div className="ai-prog-head err">
          Lost contact with the gateway while following the analysis ({errorText(st.pollErr)}). It may still be running there.
          <button type="button" className="btn" onClick={onResume}>RESUME</button>
        </div>
      </div>
    );
  }
  if (!live && job?.status === "error") {
    return (
      <div className="ai-prog">
        <div className="ai-prog-head err">
          ✕ The analysis of {job.symbol} failed: {job.error || "unknown error"}
          <button type="button" className="btn" onClick={onRetry}>TRY AGAIN</button>
        </div>
        <StepList steps={job.steps ?? []} />
      </div>
    );
  }
  if (!live) {
    return st.pollErr ? <div className="ai-prog"><div className="ai-prog-head muted">{errorText(st.pollErr)}</div></div> : null;
  }
  if (!job) return <div className="ai-prog"><div className="ai-prog-head"><span className="ai-spin" aria-hidden="true" />Starting the analysis of {symbol}…</div></div>;
  const steps = job.steps ?? [];
  const finished = steps.filter((s) => s.status === "done" || s.status === "skipped" || s.status === "error").length;
  const current = steps.find((s) => s.status === "running");
  const where = job.status === "queued" ? "queued" : `step ${Math.min(steps.length, finished + 1)} of ${steps.length}${current ? ` — ${current.label}` : ""}`;
  return (
    <div className="ai-prog">
      <div className="ai-prog-head">
        <span className="ai-spin" aria-hidden="true" />
        {job.status === "queued"
          ? <span><b>Queued</b> — at most two analyses run at once; this one starts when a slot frees up.</span>
          : <span><b>Analysing {job.symbol}</b> · {where}</span>}
        {!!st.pollErr && <span className="warn">connection hiccup — retrying…</span>}
        <span className="ai-elapsed mono" title="Time since the analysis started">{ok(job.started) ? mmss(now / 1000 - job.started) : ""}</span>
      </div>
      <span className="sr-only" aria-live="polite">{`${job.symbol}: ${where}`}</span>
      <StepList steps={steps} />
    </div>
  );
}

// ------------------------------------------------------------------ verdict
const SCORES: { key: keyof AiReport["score"]; label: string; note: string }[] = [
  { key: "technical", label: "Technical", note: "trend & price action" },
  { key: "fundamental", label: "Fundamental", note: "quality, growth, health" },
  { key: "valuation", label: "Valuation", note: "10 = very attractive" },
  { key: "momentum", label: "Momentum", note: "absolute & relative returns" },
  { key: "risk", label: "Risk", note: "10 = riskiest" },
];

function ScoreBars({ score }: { score: AiReport["score"] | null | undefined }) {
  return (
    <div className="ai-scores" role="list" aria-label="Scores out of 10">
      {SCORES.map((s) => {
        const raw = score?.[s.key];
        const v = ok(raw) ? clamp(raw, 0, 10) : null;
        return (
          <div className="ai-score" role="listitem" key={s.key} aria-label={`${s.label} ${v === null ? "not scored" : `${fx(v, 1)} out of 10`} (${s.note})`}>
            <div className="ai-score-top">
              <span className="ai-score-label">{s.label}</span>
              <span className="ai-score-val">{v === null ? "—" : fx(v, 1)}<small>/10</small></span>
            </div>
            <div className="ai-meter" aria-hidden="true"><i style={{ width: `${(v ?? 0) * 10}%` }} /></div>
            <div className="ai-score-note">{s.note}</div>
          </div>
        );
      })}
    </div>
  );
}

/** The four horizons in one table, so the first screen carries every target. */
function Glance({ points }: { points: HPoint[] }) {
  if (!points.some((p) => p.view)) return null;
  return (
    <div className="ai-glance-wrap">
      <table className="tbl ai-glance">
        <caption className="sr-only">Views by horizon at a glance</caption>
        <thead>
          <tr>
            <th className="l" scope="col">Horizon</th><th className="l" scope="col">Stance</th>
            <th scope="col">Base target</th><th scope="col" title="The range the analysis considers likely (roughly the central two thirds)">Likely return</th>
            <th className="l" scope="col" title="Is the base target inside the Monte Carlo 5–95% band?">MC check</th>
          </tr>
        </thead>
        <tbody>
          {points.map((p) => {
            const v = p.view;
            const c = v?.mc_check;
            return (
              <tr key={p.key}>
                <td className="l"><b>{p.key}</b> <small className="muted">≈ {monthYear(p.ts)}</small></td>
                <td className="l">{v ? <><StanceBadge stance={v.stance} /> <small className="muted">{confOf(v.confidence)}</small></> : <span className="muted">no view</span>}</td>
                <td>{p.base ? <>{rs(p.base.target_price)} <Chg v={p.base.return_pct} d={1} /></> : "—"}</td>
                <td>{v ? <><Chg v={v.expected_return_low_pct} d={1} /> <span className="muted">…</span> <Chg v={v.expected_return_high_pct} d={1} /></> : "—"}</td>
                <td className="l">{c ? (c.base_in_band ? <span title={c.note}>✓ inside</span> : <span className="warn" title={c.note}>⚠ outside</span>) : <span className="muted">—</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Verdict({ shown, spot, nowSpot, source, points }: { shown: Shown; spot: number | null; nowSpot: number | null; source: string; points: HPoint[] }) {
  const rep = shown.rep;
  const today = dateIST(rep.generated_at) === todayIST();
  const since = ok(spot) && ok(nowSpot) && spot > 0 ? (nowSpot / spot - 1) * 100 : null;
  const showSince = since !== null && (!today || Math.abs(since) >= 0.5);
  return (
    <Panel code="AI" title="Verdict" area="view" className="ai-report ai-verdict" sub={rep.name && rep.name !== rep.symbol ? `${rep.symbol} · ${rep.name}` : rep.symbol}>
      <div className="ai-vhead">
        <StanceBadge stance={rep.overall_stance} big />
        <ConfMeter c={rep.overall_confidence} />
        <span className="ai-tags">
          {rep.ai
            ? <span className="chip violet" title="Written by an AI model from the dossier below">AI</span>
            : <span className="chip amber" title="No AI model ran: fixed rules over the same dossier (trend, quant signal with measured edge, Screener grade, Monte Carlo median)">RULE-BASED</span>}
          {shown.cached && <span className="chip cyan" title="Today's earlier analysis was returned — no new model call">CACHED</span>}
          {!today && <span className="chip" title="A saved report from an earlier day — ANALYSE for today's view">SAVED {dateIST(rep.generated_at)}</span>}
          {source === "sim" && <span className="chip violet" title="DATA_SOURCE=sim: the price history is synthetic, so technical and statistical conclusions are illustrative only">SIM DATA</span>}
        </span>
      </div>
      <div className="ai-vmeta">
        <span><b>{rep.provider || "—"}</b> · {rep.model || "—"}</span>
        <span>generated <b>{generatedText(rep.generated_at)}</b></span>
        {rep.asof && <span>prices as of <b>{rep.asof}</b></span>}
        {ok(spot) && <span>spot then <b>{inr(spot)}</b></span>}
        {showSince && <span>now <b>{inr(nowSpot)}</b> <Chg v={since} d={1} /></span>}
      </div>
      <p className="ai-summary">{rep.summary?.trim() || "The report has no summary."}</p>
      <div className="ai-pad"><Warnings items={rep.warnings} label="Report warnings" /></div>
      <Glance points={points} />
      <ScoreBars score={rep.score} />
    </Panel>
  );
}

// ------------------------------------------------------------------ horizon chart (history + targets + MC bands)
const CH = 6.4;                       // px per character of the 10.5 px mono axis text (as in charts.tsx)
const textW = (s: string) => s.length * CH;
const crisp = (v: number) => Math.round(v) + 0.5;

function niceStep(span: number, count: number): number {
  const raw = span / Math.max(1, count);
  const p = 10 ** Math.floor(Math.log10(raw));
  const e = raw / p;
  return (e >= 7.07 ? 10 : e >= 3.16 ? 5 : e >= 1.41 ? 2 : 1) * p;
}

function decimals(step: number): number {
  for (let d = 0; d < 6; d++) {
    const s = step * 10 ** d;
    if (Math.abs(s - Math.round(s)) < 1e-6 * Math.max(1, s)) return d;
  }
  return 6;
}

function yearTicks(x0: number, x1: number, maxTicks: number): { ts: number; label: string }[] {
  const y0 = new Date((x0 + IST) * 1000).getUTCFullYear();
  const y1 = new Date((x1 + IST) * 1000).getUTCFullYear();
  const out: { ts: number; label: string }[] = [];
  for (let y = y0; y <= y1; y++) {
    const ts = Date.UTC(y, 0, 1) / 1000 - IST;
    if (ts >= x0 && ts <= x1) out.push({ ts, label: String(y) });
  }
  const k = Math.max(1, Math.ceil(out.length / Math.max(1, maxTicks)));
  return out.filter((_, i) => (out.length - 1 - i) % k === 0);
}

function nearest(xs: number[], v: number): number {
  let lo = 0, hi = xs.length - 1;
  if (hi <= 0) return 0;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= v) lo = mid; else hi = mid;
  }
  return v - xs[lo] <= xs[hi] - v ? lo : hi;
}

interface HModel {
  hist: [number, number][];      // closes from 3 years before the analysis date onwards
  pts: HPoint[];                 // horizons with a base target, in time order
  asofTs: number; spot: number | null;
  x0: number; x1: number; ext: [number, number] | null;
  recent: boolean;               // analysis date within the last few days → label it "now"
  summary: string;
}

function buildModel(history: [number, number][], asofTs: number, spot: number | null, points: HPoint[], symbol: string): HModel {
  const from = asofTs - 3 * 365.25 * DAY;
  const hist = history.filter(([t, v]) => ok(t) && ok(v) && v > 0 && t >= from);
  const pts = points.filter((p) => p.base);
  const vals: number[] = hist.map((p) => p[1]);
  if (ok(spot)) vals.push(spot);
  for (const p of pts) {
    for (const s of [p.bear, p.base, p.bull]) if (s) vals.push(s.target_price);
    const c = p.view?.mc_check;
    if (c) for (const v of [c.p5, c.p95]) if (ok(v) && v > 0) vals.push(v);
  }
  const ext: [number, number] | null = vals.length ? [Math.min(...vals), Math.max(...vals)] : null;
  const x0 = Math.min(hist[0]?.[0] ?? asofTs - 90 * DAY, asofTs - 30 * DAY);
  const x1 = Math.max(pts[pts.length - 1]?.ts ?? asofTs + 365 * DAY, hist[hist.length - 1]?.[0] ?? asofTs);
  const summary = `${symbol}: daily closes and the report's targets. ` + (pts.length ? pts.map((p) => {
    const c = p.view?.mc_check;
    return `${p.key} (about ${monthYear(p.ts)}): base ${rs(p.base!.target_price)} (${pct(p.base!.return_pct, 1, false)})` +
      (p.bear && p.bull ? `, bear ${rs(p.bear.target_price)} to bull ${rs(p.bull.target_price)}` : "") +
      (c ? `, Monte Carlo 5–95% ${rs(c.p5)} to ${rs(c.p95)}, base ${c.base_in_band ? "inside" : "outside"} the band` : "");
  }).join("; ") + "." : "No horizon targets.");
  return { hist, pts, asofTs, spot, x0, x1, ext, recent: Date.now() / 1000 - asofTs < 5 * DAY, summary };
}

interface HLayout {
  X: (t: number) => number; Y: (v: number) => number;
  ml: number; pw: number; mt: number; mb: number;
  yTicks: number[]; tickText: (v: number) => string;
  xTicks: { x: number; label: string }[];
  baseLabels: (string | null)[];
}

function layoutH(m: HModel, w: number, h: number): HLayout {
  const mt = 24, mb = 22;
  const count = clamp(Math.floor((h - mt - mb) / 44), 2, 8);
  let [lo, hi] = m.ext ?? [0, 1];
  if (hi - lo < 1e-9) { const d = Math.abs(hi) * 0.05 || 1; lo -= d; hi += d; }
  const pad = (hi - lo) * 0.05;
  const step = niceStep(hi - lo + 2 * pad, count);
  const a = Math.max(lo >= 0 ? 0 : -Infinity, Math.floor((lo - pad) / step) * step);
  const b = Math.ceil((hi + pad) / step) * step;
  const yTicks: number[] = [];
  for (let k = Math.round(a / step); k * step <= b + step * 1e-9; k++) yTicks.push(+(k * step).toPrecision(12));
  const d = decimals(step);
  const tickText = (v: number) => `₹${fx(v, d)}`;
  const ml = Math.max(24, ...yTicks.map((t) => textW(tickText(t)))) + 10;
  const last = m.pts[m.pts.length - 1];
  const mr = Math.max(16, last ? textW(rs(last.base!.target_price)) + 16 : 0);
  const pw = Math.max(40, w - ml - mr);
  const X = (t: number) => ml + ((t - m.x0) / (m.x1 - m.x0 || 1)) * pw;
  const Y = (v: number) => h - mb - ((v - a) / (b - a || 1)) * (h - mb - mt);
  const xTicks = yearTicks(m.x0, m.x1, Math.max(2, Math.floor(pw / 56)))
    .map((t) => ({ x: X(t.ts), label: t.label }))
    .filter((t) => t.x - textW(t.label) / 2 >= ml - 8 && t.x + textW(t.label) / 2 <= w - 2);
  // base labels sit right of their dot; one that would run into the next horizon's whisker is left to the tooltip
  const baseLabels = m.pts.map((p, i) => {
    const text = rs(p.base!.target_price);
    const next = m.pts[i + 1];
    const limit = next ? X(next.ts) - 9 : w - 2;
    return X(p.ts) + 9 + textW(text) <= limit ? text : null;
  });
  return { X, Y, ml, pw, mt, mb, yTicks, tickText, xTicks, baseLabels };
}

type KeyKind = "line" | "dash" | "area" | "dot";
interface TipRow { value: ReactNode; label: ReactNode; kind?: KeyKind; color?: string; opacity?: number }

const keyStyle = (kind: KeyKind, color: string, opacity?: number): CSSProperties =>
  kind === "dash" ? { borderTopColor: color, opacity } : { background: color, opacity };

function Tip({ x, y, w, title, rows }: { x: number; y: number; w: number; title: ReactNode; rows: TipRow[] }) {
  const flip = x > w * 0.56;
  return (
    <div className="rc-tip" aria-hidden="true" style={{ top: y, ...(flip ? { right: w - x + 12 } : { left: x + 12 }) }}>
      <div className="rc-tip-title">{title}</div>
      {rows.map((r, i) => (
        <div className="rc-tip-row" key={i}>
          {r.kind && r.color ? <i className={`rc-key ${r.kind}`} style={keyStyle(r.kind, r.color, r.opacity)} /> : <i className="rc-key none" />}
          <b>{r.value}</b><span>{r.label}</span>
        </div>
      ))}
    </div>
  );
}

const scenText = (s: AiScenario | null) => (s ? `${rs(s.target_price)} (${pct(s.return_pct, 1, false)})` : "—");

type Active = { kind: "h" | "c"; i: number };

function HorizonPlot({ m, w, h }: { m: HModel; w: number; h: number }) {
  const [active, setActive] = useState<Active | null>(null);
  const L = useMemo(() => layoutH(m, w, h), [m, w, h]);
  const { X, Y, ml, pw, mt, mb } = L;
  const histTs = useMemo(() => m.hist.map((p) => p[0]), [m.hist]);
  const histPath = useMemo(() => m.hist.map(([t, v], i) => `${i ? "L" : "M"}${X(t).toFixed(1)},${Y(v).toFixed(1)}`).join(""), [m.hist, X, Y]);
  const n = m.pts.length;
  const yTop = mt, yBot = h - mb;

  const pick = (px: number): Active | null => {
    let best = -1, bd = Infinity;
    m.pts.forEach((p, i) => {
      const dd = Math.abs(X(p.ts) - px);
      if (dd < bd) { bd = dd; best = i; }
    });
    if (best >= 0 && bd <= 24) return { kind: "h", i: best };
    if (histTs.length && px <= X(histTs[histTs.length - 1]) + 6) {
      const t = m.x0 + ((px - ml) / pw) * (m.x1 - m.x0);
      return { kind: "c", i: nearest(histTs, t) };
    }
    return best >= 0 ? { kind: "h", i: best } : null;
  };
  const onMove = (e: ReactPointerEvent<SVGRectElement>) => {
    const r = (e.currentTarget.ownerSVGElement ?? e.currentTarget).getBoundingClientRect();
    setActive(pick(e.clientX - r.left));
  };
  const onKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (!n) return;
    const i = active?.kind === "h" ? active.i : null;
    let next: number | null | undefined;
    if (e.key === "ArrowRight") next = i === null ? 0 : Math.min(n - 1, i + 1);
    else if (e.key === "ArrowLeft") next = i === null ? n - 1 : Math.max(0, i - 1);
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = n - 1;
    else if (e.key === "Escape") next = null;
    if (next === undefined) return;
    e.preventDefault();
    setActive(next === null ? null : { kind: "h", i: next });
  };
  const onFocus = (e: ReactFocusEvent<HTMLDivElement>) => {
    let keyboard = true;
    try { keyboard = e.currentTarget.matches(":focus-visible"); } catch { /* older engines: treat as keyboard */ }
    if (keyboard && n) setActive((a) => a ?? { kind: "h", i: 0 });
  };

  let tip: { x: number; title: ReactNode; rows: TipRow[] } | null = null;
  let cross: { x: number; y: number | null } | null = null;
  if (active?.kind === "h" && m.pts[active.i]) {
    const p = m.pts[active.i];
    const c = p.view?.mc_check;
    const s = stanceOf(p.view?.stance);
    const x = X(p.ts);
    cross = { x, y: null };
    tip = {
      x, title: `${p.key} · ≈ ${monthYear(p.ts)}`,
      rows: [
        { value: `${s.glyph} ${s.text}`, label: `${confOf(p.view?.confidence)} confidence` },
        { value: scenText(p.bull), label: `bull · ${probText(p.bull?.probability)}`, kind: "line", color: "var(--viz-2)" },
        { value: scenText(p.base), label: `base · ${probText(p.base?.probability)}`, kind: "dot", color: "var(--viz-2)" },
        { value: scenText(p.bear), label: `bear · ${probText(p.bear?.probability)}`, kind: "line", color: "var(--viz-2)" },
        ...(c ? [
          { value: `${rs(c.p5)} – ${rs(c.p95)}`, label: "Monte Carlo 5–95%", kind: "area" as const, color: "var(--viz-1)", opacity: 0.5 },
          { value: rs(c.p50), label: "Monte Carlo median", kind: "line" as const, color: "var(--viz-1)" },
          { value: c.base_in_band ? "✓ inside" : "⚠ outside", label: "base vs the 5–95% band" },
        ] : [{ value: "—", label: "no Monte Carlo check" }]),
      ],
    };
  } else if (active?.kind === "c" && m.hist[active.i]) {
    const [t, v] = m.hist[active.i];
    const x = X(t);
    cross = { x, y: Y(v) };
    tip = { x, title: dateIST(t), rows: [{ value: inr(v), label: "close", kind: "line", color: "var(--text-2)" }] };
  }

  const ax = X(m.asofTs);
  const basePath = [
    ...(ok(m.spot) ? [`M${ax.toFixed(1)},${Y(m.spot).toFixed(1)}`] : []),
    ...m.pts.map((p, i) => `${i || ok(m.spot) ? "L" : "M"}${X(p.ts).toFixed(1)},${Y(p.base!.target_price).toFixed(1)}`),
  ].join("");
  const halo: CSSProperties = { paintOrder: "stroke", stroke: "var(--panel)", strokeWidth: 3 };

  return (
    <div className="ai-hc-abs" tabIndex={0} role="img" aria-label={`Horizon targets. ${m.summary}`}
      onKeyDown={onKeyDown} onFocus={onFocus} onBlur={() => setActive(null)}>
      <svg width={w} height={h} aria-hidden="true">
        {L.yTicks.map((t) => (
          <g key={t}>
            <line className="rc-grid" x1={ml} x2={ml + pw} y1={crisp(Y(t))} y2={crisp(Y(t))} />
            <text x={ml - 6} y={Y(t) + 3.5} textAnchor="end">{L.tickText(t)}</text>
          </g>
        ))}
        <line className="rc-axis" x1={ml} x2={ml + pw} y1={crisp(yBot)} y2={crisp(yBot)} />
        {L.xTicks.map((t) => <text key={t.label} x={t.x} y={h - 5} textAnchor="middle">{t.label}</text>)}
        {m.pts.map((p) => <line key={p.key} className="rc-grid" x1={crisp(X(p.ts))} x2={crisp(X(p.ts))} y1={yTop} y2={yBot} />)}
        <line className="rc-axis" x1={crisp(ax)} x2={crisp(ax)} y1={yTop} y2={yBot} />
        <text className="lbl" x={ax - 4} y={yTop - 8} textAnchor="end">{m.recent ? "now" : `as of ${shortDate(m.asofTs)}`}</text>
        {m.pts.map((p) => {
          const c = p.view?.mc_check;
          if (!c || !ok(c.p5) || !ok(c.p95) || c.p95 < c.p5) return null;
          const x = X(p.ts);
          return (
            <g key={p.key}>
              <rect x={x - 8} y={Y(c.p95)} width={16} height={Math.max(1, Y(c.p5) - Y(c.p95))} rx={2} fill="var(--viz-1)" opacity={0.24} />
              {ok(c.p50) && <line x1={x - 8} x2={x + 8} y1={Y(c.p50)} y2={Y(c.p50)} stroke="var(--viz-1)" strokeWidth={2} />}
            </g>
          );
        })}
        {histPath && <path d={histPath} fill="none" stroke="var(--text-2)" strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />}
        {basePath && <path d={basePath} fill="none" stroke="var(--viz-2)" strokeWidth={1.5} strokeDasharray="4 3" opacity={0.7} />}
        {m.pts.map((p) => {
          if (!p.bear || !p.bull) return null;
          const x = X(p.ts), y1 = Y(p.bull.target_price), y2 = Y(p.bear.target_price);
          return (
            <g key={p.key} stroke="var(--viz-2)" strokeWidth={2} strokeLinecap="round">
              <line x1={x} x2={x} y1={y1} y2={y2} />
              <line x1={x - 5} x2={x + 5} y1={y1} y2={y1} />
              <line x1={x - 5} x2={x + 5} y1={y2} y2={y2} />
            </g>
          );
        })}
        {ok(m.spot) && <circle cx={ax} cy={Y(m.spot)} r={4} fill="var(--text)" stroke="var(--panel)" strokeWidth={2} />}
        {m.pts.map((p, i) => (
          <circle key={p.key} cx={X(p.ts)} cy={Y(p.base!.target_price)} r={active?.kind === "h" && active.i === i ? 6 : 4.5}
            fill="var(--viz-2)" stroke="var(--panel)" strokeWidth={2} />
        ))}
        {m.pts.map((p, i) => L.baseLabels[i] && (
          <text key={p.key} className="strong" x={X(p.ts) + 9} y={Y(p.base!.target_price) + 3.5} style={halo}>{L.baseLabels[i]}</text>
        ))}
        {m.pts.map((p) => <text key={p.key} className="lbl" x={X(p.ts)} y={yTop - 8} textAnchor="middle">{p.key}</text>)}
        {cross && (
          <g pointerEvents="none">
            <line className="rc-cross" x1={crisp(cross.x)} x2={crisp(cross.x)} y1={yTop} y2={yBot} />
            {cross.y !== null && <circle cx={cross.x} cy={cross.y} r={4} fill="var(--text-2)" stroke="var(--panel)" strokeWidth={2} />}
          </g>
        )}
        <rect className="rc-hit" x={ml - 4} y={yTop} width={pw + 12} height={yBot - yTop}
          onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={() => setActive(null)} />
      </svg>
      {tip && <Tip x={tip.x} y={yTop} w={w} title={tip.title} rows={tip.rows} />}
    </div>
  );
}

/** Container size tracked with ResizeObserver (callback ref, so it survives the table/chart toggle). */
function useBox(): [(el: HTMLDivElement | null) => void, { w: number; h: number }] {
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

const LEGEND: { label: string; kind: KeyKind; color: string; opacity?: number }[] = [
  { label: "Close", kind: "line", color: "var(--text-2)" },
  { label: "Spot at analysis", kind: "dot", color: "var(--text)" },
  { label: "Base target", kind: "dot", color: "var(--viz-2)" },
  { label: "Bear–bull range", kind: "line", color: "var(--viz-2)" },
  { label: "Monte Carlo 5–95% · median", kind: "area", color: "var(--viz-1)", opacity: 0.4 },
];

function HorizonChart({ history, asofTs, spot, points, symbol, asof, busy }: {
  history: [number, number][]; asofTs: number; spot: number | null; points: HPoint[]; symbol: string; asof: string; busy?: boolean;
}) {
  const [table, setTable] = useState(false);
  const [ref, size] = useBox();
  const m = useMemo(() => buildModel(history, asofTs, spot, points, symbol), [history, asofTs, spot, points, symbol]);
  const empty = !m.hist.length && !m.pts.length;
  return (
    <figure className={`rc ai-hc${busy ? " busy" : ""}`} aria-busy={busy || undefined}>
      <div className="rc-head">
        <ul className="rc-legend" aria-label="Legend">
          {LEGEND.map((it) => (
            <li key={it.label}><i className={`rc-key ${it.kind}`} style={keyStyle(it.kind, it.color, it.opacity)} aria-hidden="true" />{it.label}</li>
          ))}
        </ul>
        <div className="rc-tools">
          <button type="button" className="rc-tbtn" aria-pressed={table} onClick={() => setTable((v) => !v)}
            title={table ? "Back to the chart" : "Show the numbers as a table"}>{table ? "CHART" : "TABLE"}</button>
        </div>
      </div>
      {table ? (
        <div className="rc-table-wrap">
          <table className="tbl rc-table">
            <caption className="sr-only">{`Horizon targets for ${symbol}`}</caption>
            <thead>
              <tr>
                <th className="l" scope="col">Horizon</th><th scope="col">Date ≈</th><th className="l" scope="col">Stance</th>
                <th scope="col">Bear</th><th scope="col">Base</th><th scope="col">Bull</th>
                <th scope="col">Monte Carlo 5–95%</th><th scope="col">MC median</th><th className="l" scope="col">Base in band</th>
              </tr>
            </thead>
            <tbody>
              {points.map((p) => {
                const c = p.view?.mc_check;
                const s = stanceOf(p.view?.stance);
                return (
                  <tr key={p.key}>
                    <td className="l">{p.key}</td>
                    <td>{monthYear(p.ts)}</td>
                    <td className="l">{p.view ? `${s.glyph} ${s.text}` : "no view"}</td>
                    <td>{scenText(p.bear)} · {probText(p.bear?.probability)}</td>
                    <td>{scenText(p.base)} · {probText(p.base?.probability)}</td>
                    <td>{scenText(p.bull)} · {probText(p.bull?.probability)}</td>
                    <td>{c ? `${rs(c.p5)} – ${rs(c.p95)}` : "—"}</td>
                    <td>{c ? rs(c.p50) : "—"}</td>
                    <td className="l">{c ? (c.base_in_band ? "✓ inside" : "⚠ outside") : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="rc-plot ai-hc-plot" ref={ref}>
          {empty ? <div className="rc-empty">No price history or targets to plot.</div>
            : size.w > 0 && size.h > 0 && <HorizonPlot m={m} w={size.w} h={size.h} />}
        </div>
      )}
      <figcaption className="rc-caption">
        Dot = base target, whisker = bear–bull range, at about 6, 12, 24 and 36 months after the analysis date ({asof || "—"});
        the shaded bar is the Monte Carlo 5–95% range of the same dossier (its 126 / 252 / 504 / 756 trading days) with its median.
        A base target outside its bar departs from what the stock's own volatility implies. Hover or use ←/→ for the numbers.
      </figcaption>
    </figure>
  );
}

// ------------------------------------------------------------------ horizon cards
function RangeStrip({ p, spot }: { p: HPoint; spot: number | null }) {
  const c = p.view?.mc_check ?? null;
  const vals = [p.bear?.target_price, p.base?.target_price, p.bull?.target_price, c?.p5, c?.p95, spot].filter((v): v is number => ok(v) && v > 0);
  if (vals.length < 2) return null;
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.06 || hi * 0.02;
  lo -= pad;
  hi += pad;
  const P = (v: number) => ((v - lo) / (hi - lo)) * 100;
  const label = [
    c ? `Monte Carlo 5–95% ${rs(c.p5)} to ${rs(c.p95)}, median ${rs(c.p50)}` : "no Monte Carlo range",
    p.bear && p.bull ? `bear ${rs(p.bear.target_price)} to bull ${rs(p.bull.target_price)}` : "",
    p.base ? `base ${rs(p.base.target_price)}` : "",
    ok(spot) ? `spot ${rs(spot)}` : "",
  ].filter(Boolean).join("; ");
  return (
    <div className="ai-strip" role="img" aria-label={label} title={`Shaded = Monte Carlo 5–95% (bar = median) · line = bear–bull · dot = base · tick = spot. ${label}`}>
      {c && ok(c.p5) && ok(c.p95) && <i className="ai-strip-mc" style={{ left: `${P(c.p5)}%`, width: `${Math.max(0.5, P(c.p95) - P(c.p5))}%` }} />}
      {c && ok(c.p50) && <i className="ai-strip-med" style={{ left: `${P(c.p50)}%` }} />}
      {p.bear && p.bull && <i className="ai-strip-whisk" style={{ left: `${P(p.bear.target_price)}%`, width: `${Math.max(0.5, P(p.bull.target_price) - P(p.bear.target_price))}%` }} />}
      {ok(spot) && <i className="ai-strip-spot" style={{ left: `${P(spot)}%` }} />}
      {p.base && <i className="ai-strip-base" style={{ left: `${P(p.base.target_price)}%` }} />}
    </div>
  );
}

function McCheck({ p, spot }: { p: HPoint; spot: number | null }) {
  const c = p.view?.mc_check ?? null;
  return (
    <div className={`ai-mc ${c ? (c.base_in_band ? "in" : "out") : "na"}`}>
      <h4 className="ai-h">Monte Carlo check</h4>
      <RangeStrip p={p} spot={spot} />
      {c ? (
        <>
          <div className="ai-mc-verdict">
            {c.base_in_band ? <><span aria-hidden="true">✓ </span>Base target inside the 5–95% band</> : <b><span aria-hidden="true">⚠ </span>Base target outside the 5–95% band</b>}
            {!c.note && <> ({rs(c.p5)} – {rs(c.p95)}, median {rs(c.p50)})</>}
          </div>
          {c.note && <p className="ai-mc-note">{c.note}</p>}
        </>
      ) : (
        <p className="ai-mc-note">No Monte Carlo range for this horizon (the simulation did not run), so the target is unchecked.</p>
      )}
    </div>
  );
}

function TextList({ title, items, empty = "None given." }: { title: string; items: unknown; empty?: string }) {
  const xs = texts(items);
  return (
    <div className="ai-list">
      <h4 className="ai-h">{title}</h4>
      {xs.length ? <ul className="ai-bullets">{xs.map((s, i) => <li key={i}>{s}</li>)}</ul> : <p className="ai-none">{empty}</p>}
    </div>
  );
}

const SCEN_ORDER: AiScenario["name"][] = ["bull", "base", "bear"];

function HorizonCard({ p, spot }: { p: HPoint; spot: number | null }) {
  const v = p.view;
  return (
    <Panel code={p.key} title={p.name} sub={`≈ ${monthYear(p.ts)}`} className="ai-hcard">
      {!v ? (
        <p className="ai-none">No {p.name} view in this report — see the data gaps.</p>
      ) : (
        <>
          <div className="ai-hrow">
            <StanceBadge stance={v.stance} />
            <ConfMeter c={v.confidence} />
          </div>
          <div className="ai-range" title="The outcome range the analysis considers likely (roughly the central two thirds)">
            Likely return <Chg v={v.expected_return_low_pct} d={1} /> <span className="muted">to</span> <Chg v={v.expected_return_high_pct} d={1} />
          </div>
          <div>
            <h4 className="ai-h">Scenarios</h4>
            <ul className="ai-scen">
              {SCEN_ORDER.map((name) => {
                const s = name === "bull" ? p.bull : name === "base" ? p.base : p.bear;
                const prob = ok(s?.probability) ? clamp(s!.probability, 0, 1) : null;
                return (
                  <li key={name} className={`ai-sc ${name}`}>
                    <span className="ai-sc-name">{name.toUpperCase()}</span>
                    <span className="ai-sc-prob" title="Probability of this scenario">
                      <span className="ai-pbar" aria-hidden="true"><i style={{ width: `${(prob ?? 0) * 100}%` }} /></span>
                      <span>{probText(prob)}</span>
                    </span>
                    <span className="ai-sc-target">{s ? rs(s.target_price) : "—"}</span>
                    <span className="ai-sc-ret">{s ? <Chg v={s.return_pct} d={1} /> : "—"}</span>
                    {s?.thesis?.trim() && <p className="ai-sc-thesis">{s.thesis}</p>}
                  </li>
                );
              })}
            </ul>
          </div>
          <McCheck p={p} spot={spot} />
          <TextList title="Drivers" items={v.key_drivers} />
          <TextList title="Risks" items={v.key_risks} />
          <div>
            <h4 className="ai-h">Rationale</h4>
            <p className="ai-text">{v.rationale?.trim() || "—"}</p>
          </div>
        </>
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------ dossier & run details
function Dossier({ dossier }: { dossier: AiDossier | null | undefined }) {
  const secs = useMemo(() => (dossier?.sections ?? [])
    .filter((s) => s && Array.isArray(s.facts))
    .map((s, i) => ({ ...s, id: `${s.key || "s"}-${i}` })), [dossier]);
  const [open, setOpen] = useState<Set<string>>(() => new Set(secs.slice(0, 1).map((s) => s.id)));
  const total = secs.reduce((a, s) => a + s.facts.length, 0);
  const all = secs.length > 0 && secs.every((s) => open.has(s.id));
  const toggle = (id: string, isOpen: boolean) => setOpen((prev) => {
    if (prev.has(id) === isOpen) return prev;
    const next = new Set(prev);
    if (isOpen) next.add(id); else next.delete(id);
    return next;
  });
  return (
    <Panel code="DOS" title="Dossier" area="dos" className="ai-report"
      sub={secs.length ? `what the analysis was given · ${total} facts in ${secs.length} sections` : "what the analysis was given"}
      actions={secs.length ? (
        <button type="button" className="rc-tbtn" onClick={() => setOpen(all ? new Set() : new Set(secs.map((s) => s.id)))}>
          {all ? "COLLAPSE ALL" : "EXPAND ALL"}
        </button>
      ) : undefined}>
      {!secs.length ? <p className="ai-none ai-padded">The report carries no dossier.</p> : (
        <div className="ai-dossier">
          {secs.map((s) => (
            <details key={s.id} className="ai-dsec" open={open.has(s.id)} onToggle={(e) => toggle(s.id, e.currentTarget.open)}>
              <summary><span>{s.title || s.key}</span><small>{s.facts.length}</small></summary>
              {s.facts.length ? (
                <dl className="ai-facts">
                  {s.facts.map((f, i) => <div key={i}><dt>{f.label || "—"}</dt><dd>{f.value || "—"}</dd></div>)}
                </dl>
              ) : <p className="ai-none ai-padded">No facts.</p>}
            </details>
          ))}
        </div>
      )}
    </Panel>
  );
}

function RunDetails({ shown }: { shown: Shown }) {
  const rep = shown.rep;
  const u = rep.usage;
  return (
    <Panel code="LOG" title="Run details" area="run" className="ai-report" sub={shown.cached ? "today's earlier run" : undefined}>
      <Stats min={104}>
        <Stat label="Engine" value={rep.ai ? "AI model" : "Rule-based"} sub={`${rep.provider || "—"} · ${rep.model || "—"}`} />
        <Stat label="Input tokens" value={u ? fx(u.input_tokens, 0) : "—"}
          sub={!u ? (rep.ai ? "not reported" : "no model call") : ok(u.cache_read_input_tokens) ? `${fx(u.cache_read_input_tokens, 0)} cached` : "no cache read"} />
        <Stat label="Output tokens" value={u ? fx(u.output_tokens, 0) : "—"} />
      </Stats>
      {shown.steps?.length ? (
        <>
          <h4 className="ai-h ai-padded">Pipeline</h4>
          <StepList steps={shown.steps} compact />
        </>
      ) : (
        <p className="ai-note">Opened from the report saved {generatedText(rep.generated_at)}. Step-by-step details are kept only for runs started from this desk.</p>
      )}
    </Panel>
  );
}

const VIEWS: { key: "technical_view" | "fundamental_view" | "market_view" | "valuation_view"; title: string }[] = [
  { key: "technical_view", title: "Technical" }, { key: "fundamental_view", title: "Fundamental" },
  { key: "market_view", title: "Market" }, { key: "valuation_view", title: "Valuation" },
];

function Report({ shown, closes, histBusy, nowSpot, source }: {
  shown: Shown; closes: [number, number][]; histBusy: boolean; nowSpot: number | null; source: string;
}) {
  const rep = shown.rep;
  const asofTs = isoToTs(rep.asof) ?? isoToTs(dateIST(ok(rep.generated_at) ? rep.generated_at : Date.now() / 1000))!;
  const spot = useMemo(() => reportSpot(rep), [rep]);
  const points = useMemo(() => horizonPoints(rep, asofTs), [rep, asofTs]);
  const gaps = texts(rep.data_gaps);
  return (
    <>
      <Verdict shown={shown} spot={spot} nowSpot={nowSpot} source={source} points={points} />
      <Panel code="HZN" title="Horizon targets" area="chart" className="ai-report ai-chart-panel" sub="closes · base targets with bear–bull range · Monte Carlo band">
        <HorizonChart history={closes} asofTs={asofTs} spot={spot} points={points} symbol={rep.symbol} asof={rep.asof} busy={histBusy} />
      </Panel>
      <div className="ai-hz ai-report" style={{ gridArea: "hz" }} role="group" aria-label="Views by horizon">
        {points.map((p) => <HorizonCard key={p.key} p={p} spot={spot} />)}
      </div>
      <Panel code="READ" title="The analysis" area="read" className="ai-report">
        <div className="ai-grid2">
          {VIEWS.map((v) => (
            <section key={v.key}>
              <h4 className="ai-h">{v.title}</h4>
              <p className="ai-text">{typeof rep[v.key] === "string" && rep[v.key].trim() ? rep[v.key] : "—"}</p>
            </section>
          ))}
        </div>
      </Panel>
      <Panel code="CHK" title="Catalysts & risks" area="lists" className="ai-report" sub={gaps.length ? `${gaps.length} data gap${gaps.length === 1 ? "" : "s"}` : undefined}>
        <div className="ai-grid2">
          <section><TextList title="Catalysts" items={rep.catalysts} /></section>
          <section><TextList title="Risks" items={rep.risks} /></section>
          <section><TextList title="What would change the view" items={rep.what_would_change_view} /></section>
          <section><TextList title="Data gaps" items={gaps} empty="None reported." /></section>
        </div>
      </Panel>
      <Dossier key={`${rep.symbol}-${rep.generated_at}`} dossier={rep.dossier} />
      <RunDetails shown={shown} />
    </>
  );
}

function EmptyState({ symbol, st, cfg, running, onStart, onRetrySaved }: {
  symbol: string; st: DeskState; cfg: AiConfig | null; running: boolean; onStart: () => void; onRetrySaved: () => void;
}) {
  let body: ReactNode;
  if (!symbol) body = <p>Pick a stock above to analyse it.</p>;
  else if (st.saved === "loading") body = <p>Looking for a saved analysis of <b>{symbol}</b>…</p>;
  else if (running) body = <p>The report appears here when the analysis of <b>{symbol}</b> finishes — the steps above show where it is.</p>;
  else {
    body = (
      <>
        {st.saved === "error" && <ErrorNote error={st.savedErr} onRetry={onRetrySaved} />}
        <p>
          <b>No analysis of {symbol} yet.</b> ANALYSE gathers a dossier for the stock and {cfg?.enabled ? <>asks <b>{cfg.model}</b> to write a</> : "builds the rule-based"} view
          for 6 months, 1, 2 and 3 years from it:
        </p>
        <ul className="ai-bullets">
          <li><b>Price & technicals</b> — up to 5 years of daily bars: returns, CAGR, volatility, drawdowns, moving averages, RSI, 52-week range, support / resistance, beta and relative strength vs NIFTY</li>
          <li><b>Fundamentals</b> — Screener.in through the same parser as the SCREENER desk: ratios, growth, results, balance sheet, cash flow, shareholding and the scorecard</li>
          <li><b>Quant model</b> — next-day / week / month probabilities with their measured walk-forward accuracy (no edge is said plainly)</li>
          <li><b>Monte Carlo</b> — 5,000 simulated 3-year paths (ensemble model, CAPM drift): the statistical anchor every target is checked against</li>
          <li><b>Market & news</b> — NIFTY trend and regime, India VIX, breadth, the sector's move and the last 30 days of headlines</li>
        </ul>
        <p className="muted">
          Missing inputs are listed as data gaps, never guessed. {cfg?.enabled
            ? "A model run usually takes one to a few minutes and uses one of today's AI calls; a report is saved per stock and day."
            : "With no AI model configured the rule-based view takes a few seconds."}
        </p>
        <button type="button" className="btn primary" onClick={onStart}>ANALYSE {symbol}</button>
      </>
    );
  }
  return <Panel code="AI" title="AI analyst" area="main" sub={symbol || undefined}><div className="ai-empty">{body}</div></Panel>;
}

function ReportDisclaimer({ text }: { text: string }) {
  const m = /^(.+?\.)\s+(.*)$/s.exec(text.trim());
  return <p className="rs-disclaimer" role="note">{m ? <><b>{m[1]}</b> {m[2]}</> : text}</p>;
}

// ------------------------------------------------------------------ desk
export default function AiAnalyst() {
  const symbol = useLive((s) => s.researchSymbol);
  const source = useLive((s) => s.source);
  const hist = useHistoryInfo(symbol);
  const { cfg, cfgErr, loadCfg } = useAiConfig();
  const { cur, start, resume, retrySaved } = useAiDesk(symbol, loadCfg);
  const shown = cur.shown;
  const rep = shown?.rep ?? null;
  const running = cur.starting || !!cur.jobId;
  const today = !!rep && dateIST(rep.generated_at) === todayIST();
  const reuse = today && (rep!.ai || !cfg?.enabled);
  const closes = hist.data?.closes ?? [];

  const analyseTitle = !symbol ? "Pick a stock first"
    : reuse ? `Open today's analysis of ${symbol} (no new model call)`
    : cfg?.enabled ? `Analyse ${symbol} with ${cfg.model} — uses one of today's AI calls`
    : `Build the rule-based view of ${symbol} (no AI model configured)`;
  const rerunTitle = cfg?.enabled
    ? `Run a fresh analysis of ${symbol} even though a report is saved — uses one of today's AI calls`
    : `Rebuild the rule-based view of ${symbol} from fresh data`;

  const actions = (
    <>
      <button type="button" className="btn primary" onClick={() => start(false)} disabled={running || !symbol} title={analyseTitle}>
        {running ? "ANALYSING…" : "ANALYSE"}
      </button>
      {rep && <button type="button" className="btn" onClick={() => start(true)} disabled={running} title={rerunTitle}>RE-RUN</button>}
    </>
  );

  let note: ReactNode = null;
  if (rep && !running && !(cur.job?.status === "error")) {
    if (!today) note = <>Showing the analysis saved on <b>{dateIST(rep.generated_at)}</b> (prices as of {rep.asof || "—"}) — ANALYSE for today's view.</>;
    else if (shown?.cached) note = <>Today's earlier analysis — no new model call was made. RE-RUN for a fresh one.</>;
  }

  return (
    <div className={`desk desk-ai rs-desk${rep ? "" : " ai-noreport"}${running && rep ? " ai-busy" : ""}`}>
      <SymbolPicker area="pick" actions={actions} />
      <div className="ai-ctrl" style={{ gridArea: "ctrl" }}>
        <ConfigLine cfg={cfg} err={cfgErr} onRetry={loadCfg} />
        {note && <div className="ai-ctrl-note">{note}</div>}
        <Progress st={cur} symbol={symbol} onRetry={() => start(false)} onResume={resume} />
      </div>
      {shown ? (
        <Report shown={shown} closes={closes} histBusy={hist.loading} nowSpot={hist.data?.spot ?? null} source={source} />
      ) : (
        <EmptyState symbol={symbol} st={cur} cfg={cfg} running={running} onStart={() => start(false)} onRetrySaved={retrySaved} />
      )}
      <div className="ai-disc" style={{ gridArea: "disc" }}>
        {rep?.disclaimer?.trim() ? <ReportDisclaimer text={rep.disclaimer} /> : (
          <Disclaimer>
            The written view comes from an AI model reading the dossier (or from fixed rules when none is set up); it can
            misread or over-weight numbers, so check them in the dossier before acting.
          </Disclaimer>
        )}
      </div>
    </div>
  );
}
