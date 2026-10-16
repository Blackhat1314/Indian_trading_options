/* PAPER TRADE desk — automatic paper trading for the logged-in user, and a journal of WHY every trade was opened
 * and closed. Two strategies trade on simulated fills (no real orders, ever):
 *   INTRADAY — capital × leverage (1–5x). The engine decides at 09:30 and 10:00 IST, long or short, and squares
 *              every trade off by 15:15.
 *   SWING    — capital only. Long-only, days to weeks; entries planned after the close fill at the next open and
 *              the engine manages stops, trailing stops and exits.
 * The account is polled over REST every 3 s while the desk is open (paused while the browser tab is hidden); the
 * public decisions and candidates arrive on the "paper" ws topic; the book and the backtest evidence load on demand. */
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent, ReactNode } from "react";
import { LineChart, fracPct, inr } from "../components/research/charts";
import type { LineSeries } from "../components/research/charts";
import { errorText } from "../components/research/common";
import { Chg, Empty, Kpi, Panel, Seg } from "../components/ui";
import { ApiError } from "../lib/api";
import { toLogin } from "../lib/auth";
import { cls, compact, dateIST, fx, px, signed, timeIST } from "../lib/format";
import { CAPITAL_MAX, CAPITAL_MIN, LEV_MAX, LEV_MIN, paperApi } from "../lib/paper";
import type {
  BookScope, Decision, Evidence, PaperService, PaperSignals, PaperState, Pending, Position, Proposal, PStats, Reason, Regime, Report,
  Side, Stats, StratSettings, Strategy, SwingCandidate, Trade, TradeReasons,
} from "../lib/paper";
import { getPref, setPref } from "../lib/storage";
import { useLive } from "../lib/store";
import "./paper.css";

// =================================================================== formatting
const ok = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const IST = 19800;
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const pad2 = (n: number) => String(n).padStart(2, "0");

/** Signed rupees: +₹1,234 / -₹56.20 (sign, not colour alone, carries the direction). */
function sinr(v: number | null | undefined, d?: number): string {
  if (!ok(v)) return "—";
  return `${v >= 0.005 ? "+" : v <= -0.005 ? "-" : ""}${inr(Math.abs(v), d)}`;
}
/** Signed compact rupees for dense cells: +12.3K, -1.20L. */
function sk(v: number): string {
  return `${v >= 0.5 ? "+" : v <= -0.5 ? "-" : ""}${compact(Math.abs(v)).replace(" ", "")}`;
}
const ymdParts = (n: number) => [Math.floor(n / 10000), Math.floor(n / 100) % 100, n % 100];
/** yyyymmdd → "2026-10-15". */
function ymdText(n: number | null | undefined): string {
  if (!ok(n)) return "—";
  const [y, m, d] = ymdParts(n);
  return `${y}-${pad2(m)}-${pad2(d)}`;
}
/** yyyymmdd → epoch seconds at IST midnight (the research charts read time in IST). */
function ymdTs(n: number): number {
  const [y, m, d] = ymdParts(n);
  return Date.UTC(y, m - 1, d) / 1000 - IST;
}
/** A date that may come as yyyymmdd or as epoch seconds. */
function asofText(v: number | null | undefined): string {
  if (!ok(v)) return "—";
  return v > 19_000_000 && v < 30_000_000 ? ymdText(v) : dateIST(v);
}
const when = (ts: number | null | undefined) => (ok(ts) ? `${dateIST(ts)} ${timeIST(ts)}` : "—");

function fmtFact(v: number | string | null | undefined): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (Number.isInteger(v) || a >= 1000) return fx(v, 0);
  if (a >= 1) return fx(v, 2);
  return fx(v, 4);
}
/** Live max drawdown may be a fraction or rupees; backtests send a (negative) fraction. Always shown as a loss. */
function ddText(v: number | null | undefined): string {
  if (!ok(v)) return "—";
  return Math.abs(v) <= 1 ? fracPct(-Math.abs(v), 1) : sinr(-Math.abs(v), 0);
}
const safeId = (s: string) => s.replace(/[^a-zA-Z0-9_-]/g, "_");

// =================================================================== errors
const isStatus = (e: unknown, s: number) => e instanceof ApiError && e.status === s;
const DOWN = "The paper-trading service is not running — start it with run.py / docker compose.";
function paperErrText(e: unknown): string {
  if (e instanceof DOMException && e.name === "AbortError") return "the gateway did not answer within 10 s";
  if (isStatus(e, 503)) return DOWN;
  if (isStatus(e, 404)) return "This gateway has no paper-trading API yet (HTTP 404) — update and restart the backend.";
  return errorText(e);
}
/** 401: the session ended — off to /login like every other desk. */
function authGate(e: unknown) {
  if (isStatus(e, 401)) toLogin();
}

function ErrLine({ e, retry }: { e: unknown; retry?: () => void }) {
  return (
    <div className="rs-error" role="alert">
      <span>{paperErrText(e)}</span>
      {retry && <button type="button" className="btn" onClick={retry}>Retry</button>}
    </div>
  );
}

// =================================================================== small pieces
function PnL({ v, d }: { v: number | null | undefined; d?: number }) {
  return <span className={`${cls(v, 0.5)} b`}>{sinr(v, d)}</span>;
}
/** A fraction shown as a signed % with ▲▼. */
function Frac({ v, d = 2 }: { v: number | null | undefined; d?: number }) {
  return ok(v) ? <Chg v={v * 100} d={d} /> : <span className="muted">—</span>;
}
function RMult({ v }: { v: number | null | undefined }) {
  return ok(v) ? <span className={cls(v, 0.005)} title="Profit or loss in units of the initial risk (entry to initial stop)">{signed(v, 2)}R</span> : <span className="muted">—</span>;
}
function SideChip({ side }: { side: Side }) {
  return side === "SHORT" ? <span className="chip down">▼ SHORT</span> : <span className="chip up">▲ LONG</span>;
}
function StratChip({ s }: { s: Strategy }) {
  return s === "intraday"
    ? <span className="chip cyan" title="Intraday — decided at 09:30 / 10:00, squared off by 15:15">INTRA</span>
    : <span className="chip violet" title="Swing — long-only, held days to weeks">SWING</span>;
}
function SetupChip({ setup }: { setup: string | undefined }) {
  return setup ? <span className="chip amber pp-setup" title={setup}>{setup}</span> : <span className="muted">—</span>;
}
function exitTone(why: string): string {
  const w = why.toLowerCase();
  if (w.includes("trail")) return "amber";
  if (w.includes("target")) return "up";
  if (w.includes("time") || w.includes("square")) return "cyan";
  if (w.includes("stop") || w.includes("loss")) return "down";
  return "";
}
function ExitChip({ why }: { why: string }) {
  return why ? <span className={`chip ${exitTone(why)}`}>{why.toUpperCase()}</span> : <span className="muted">—</span>;
}
function StatusChip({ s }: { s: Decision["status"] }) {
  if (s === "done") return <span className="chip up">DONE</span>;
  if (s === "skipped") return <span className="chip">SKIPPED</span>;
  return <span className="chip amber">PENDING</span>;
}
function OnOff({ on }: { on: boolean }) {
  return on ? <span className="chip up" title="The engine trades this strategy automatically">● AUTO ON</span>
    : <span className="chip" title="Paused: the engine opens no new trades for this strategy">○ PAUSED</span>;
}

function Lights({ n }: { n: number }) {
  const lit = Math.max(0, Math.min(3, Math.round(ok(n) ? n : 0)));
  return (
    <span className={`pp-lights l${lit}`} role="img" aria-label={`${lit} of 3 market warning lights on`}>
      {[0, 1, 2].map((i) => <i key={i} className={i < lit ? "on" : undefined} />)}
    </span>
  );
}

function RegimeLine({ r, asof }: { r: Regime | null | undefined; asof?: number | null }) {
  if (!r || !ok(r.lights)) return <p className="pp-note">No market-regime reading yet.</p>;
  const lit = Math.max(0, Math.min(3, Math.round(ok(r.lights) ? r.lights : 0)));
  return (
    <div className="pp-regime">
      <Lights n={lit} />
      <div>
        <b className={lit >= 2 ? "down" : lit === 1 ? "amber" : "up"}>{lit === 0 ? "Regime: all clear" : `Regime: ${lit} of 3 warning lights on`}</b>
        <span className="muted">
          {ok(r.risk_scale) && <> · new swing trades risk <b className="pp-strong">{fx(r.risk_scale * 100, 0)}%</b> of normal</>}
          {ok(asof) && <> · as of {asofText(asof)}</>}
        </span>
        {r.text && <div className="pp-regime-text">{r.text}</div>}
      </div>
    </div>
  );
}

/** Expand / collapse button of a table row; the row itself also toggles on click. */
function XBtn({ open, id, label, onToggle }: { open: boolean; id: string; label: string; onToggle: () => void }) {
  return (
    <button type="button" className="pp-x" aria-expanded={open} aria-controls={id} aria-label={label}
      title={open ? "Hide the reasons" : "Show the reasons"} onClick={(e) => { e.stopPropagation(); onToggle(); }}>
      {open ? "▾" : "▸"}
    </button>
  );
}
function DetailRow({ id, cols, children }: { id: string; cols: number; children: ReactNode }) {
  return <tr className="pp-detail" id={id}><td colSpan={cols}><div className="pp-detail-in">{children}</div></td></tr>;
}
function useExpand(): [Set<string>, (id: string) => void] {
  const [s, setS] = useState<Set<string>>(() => new Set());
  const toggle = useCallback((id: string) => setS((o) => {
    const n = new Set(o);
    if (n.has(id)) n.delete(id); else n.add(id);
    return n;
  }), []);
  return [s, toggle];
}
/** A small choice remembered per viewer (localStorage, never required). */
function usePref<T extends string>(key: string, fallback: T, allowed: readonly T[]): [T, (v: T) => void] {
  const [v, setV] = useState<T>(() => {
    const p = getPref(key) as T | null;
    return p && allowed.includes(p) ? p : fallback;
  });
  const set = useCallback((n: T) => { setV(n); setPref(key, n); }, [key]);
  return [v, set];
}

// =================================================================== why a trade was taken
function Drivers({ drivers }: { drivers: Reason[] | undefined }) {
  const list = (drivers ?? []).filter((d) => d && (ok(d.contrib_pct) || ok(d.z)));
  if (!list.length) return null;
  const contrib = list.some((d) => ok(d.contrib_pct));
  const maxC = Math.max(1e-12, ...list.map((d) => (ok(d.contrib_pct) ? Math.abs(d.contrib_pct) : 0)));
  const dc = maxC < 0.001 ? 3 : 2;
  return (
    <div className="pp-drivers">
      <span className="pp-h4">{contrib ? "Model drivers · share of the predicted return" : "Factors · z-score vs the other stocks"}</span>
      <ul aria-label={contrib ? "Model drivers" : "Factor z-scores"}>
        {list.map((d, i) => {
          const c = ok(d.contrib_pct) ? d.contrib_pct : null;
          const v = c ?? (d.z as number);
          const w = c !== null ? (Math.abs(c) / maxC) * 50 : Math.min(50, (Math.abs(v) / 3) * 50);
          const weight = ok(d.weight) ? `w ${Math.abs(d.weight) <= 1 ? `${fx(d.weight * 100, 0)}%` : fx(d.weight, 2)}` : "";
          return (
            <li key={`${d.feature}-${i}`} title={d.feature}>
              <span className="pp-d-label">{d.label || d.feature}</span>
              <span className="pp-d-track" aria-hidden="true"><i className={v >= 0 ? "pos" : "neg"} style={{ width: `${w}%` }} /></span>
              <span className={`pp-d-val ${v >= 0 ? "up" : "down"}`}>
                {v >= 0 ? "▲ " : "▼ "}{c !== null ? fracPct(c, dc, true) : `z ${signed(v, 2)}`}
              </span>
              <span className="pp-d-w muted">{weight || (c !== null && ok(d.z) ? `z ${signed(d.z, 1)}` : "")}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Facts({ facts, title = "Numbers at the decision" }: { facts: Record<string, number | string | null> | undefined; title?: string }) {
  const rows = Object.entries(facts ?? {});
  if (!rows.length) return null;
  return (
    <div className="pp-facts">
      <span className="pp-h4">{title}</span>
      <dl className="pp-kv">
        {rows.map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd>{fmtFact(v)}</dd></Fragment>)}
      </dl>
    </div>
  );
}

function Reasons({ r }: { r: TradeReasons | null | undefined }) {
  if (!r) return <p className="pp-note">No reasons were recorded for this trade.</p>;
  const nums = ok(r.pred) || ok(r.edge) || ok(r.cost) || ok(r.score);
  return (
    <div className="pp-why">
      <div className="pp-why-main">
        <p className="pp-summary"><SetupChip setup={r.setup} /> {r.summary || "—"}</p>
        {nums && (
          <dl className="pp-nums">
            {ok(r.pred) && <div><dt title="The model's predicted return for this trade, before costs">Predicted</dt><dd className={cls(r.pred * 100)}>{fracPct(r.pred, 2, true)}</dd></div>}
            {ok(r.cost) && <div><dt title="Estimated round-trip charges and slippage">Est. cost</dt><dd>{fracPct(r.cost, 2)}</dd></div>}
            {ok(r.edge) && <div><dt title="Predicted return minus the estimated cost">Edge after costs</dt><dd className={cls(r.edge * 100)}>{fracPct(r.edge, 2, true)}</dd></div>}
            {ok(r.score) && <div><dt title="Swing factor score (higher = stronger)">Factor score</dt><dd>{fx(r.score, 2)}</dd></div>}
          </dl>
        )}
        <Drivers drivers={r.drivers} />
      </div>
      <div className="pp-why-side">
        <Facts facts={r.facts} />
        {r.sizing && (
          <div className="pp-sizing">
            <span className="pp-h4">How it was sized</span>
            <p>{r.sizing}</p>
          </div>
        )}
        {ok(r.regime?.lights) && <div className="pp-why-regime"><span className="pp-h4">Market regime then</span><RegimeLine r={r.regime} /></div>}
      </div>
    </div>
  );
}

// =================================================================== data
const EMPTY_STATS: Stats = { trades: 0, win_rate: null, profit_factor: null, avg_pnl: null, avg_r: null, best: null, worst: null, max_dd: null, charges: 0 };

/** Fill in anything the service left out, so a partial payload never takes the desk down. */
function norm(s: PaperState): PaperState {
  const svc: PaperService = s.service ?? { running: false, clock: null, phase: null, mode: "sim", models: { source: "none", trained_to: null } };
  return {
    ...s,
    service: { ...svc, models: svc.models ?? { source: "none", trained_to: null } },
    settings: {
      intraday: s.settings?.intraday ?? { enabled: false, capital: 0, leverage: 1 },
      swing: s.settings?.swing ?? { enabled: false, capital: 0 },
    },
    intraday: { ...s.intraday, positions: s.intraday?.positions ?? [], stats: { ...EMPTY_STATS, ...s.intraday?.stats } },
    swing: {
      ...s.swing, positions: s.swing?.positions ?? [], pending: s.swing?.pending ?? [],
      stats: { ...EMPTY_STATS, ...s.swing?.stats }, regime: s.swing?.regime ?? null,
    },
    equity_curve: s.equity_curve ?? [],
  };
}

/** GET /api/paper/state every 3 s while mounted, paused while the tab is hidden. Writes (settings, reset) hand
 *  their returned state to `apply`, which also discards any poll that started before the write. */
function usePaperState() {
  const [st, setSt] = useState<PaperState | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [at, setAt] = useState(0);
  const gen = useRef(0);
  const inflight = useRef(-1);
  const ctl = useRef<AbortController | null>(null);
  const poll = useCallback(() => {
    if (document.hidden || inflight.current === gen.current) return;
    const g = gen.current;
    inflight.current = g;
    const c = new AbortController();
    ctl.current = c;
    const timer = window.setTimeout(() => c.abort(), 10_000);      // a stalled request must not stop the polling
    paperApi.state(c.signal).then(
      (s) => {
        if (g !== gen.current) return;
        setSt(norm(s));
        setErr(null);
        setAt(Date.now() / 1000);
      },
      (e) => {
        if (g !== gen.current) return;
        authGate(e);
        setErr(e);
      },
    ).finally(() => {
      window.clearTimeout(timer);
      if (inflight.current === g) inflight.current = -1;
    });
  }, []);
  useEffect(() => {
    poll();
    const id = window.setInterval(poll, 3000);
    const onVis = () => { if (!document.hidden) poll(); };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      gen.current += 1;
      ctl.current?.abort();
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [poll]);
  const apply = useCallback((s: PaperState) => {
    gen.current += 1;
    setSt(norm(s));
    setErr(null);
    setAt(Date.now() / 1000);
  }, []);
  return { st, err, at, apply, retry: poll };
}

/** The closed-trade journal for one scope; refetched when the number of closed trades changes. */
function useBook(scope: BookScope, closed: number) {
  const [trades, setTrades] = useState<Trade[] | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    paperApi.book(scope, 200).then(
      (r) => { if (alive) { setTrades(r.trades ?? []); setErr(null); } },
      (e) => { if (alive) { authGate(e); setErr(e); } },
    ).finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [scope, closed, nonce]);
  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { trades, err, loading, reload };
}

/** Today's decisions: the ws "paper" topic keeps the store current; one REST read on open fills it right away. */
function useSignals() {
  const sig = useLive((s) => s.paper);
  const [err, setErr] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    paperApi.signals().then(
      (d) => { if (alive) { useLive.setState({ paper: d }); setErr(null); } },
      (e) => { if (alive) { authGate(e); setErr(e); } },
    ).finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [nonce]);
  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { sig, err, loading, reload };
}

let evidenceMemo: { at: number; data: Evidence } | null = null;     // backtests change rarely: keep across tab switches
function useEvidence() {
  const [data, setData] = useState<Evidence | null>(evidenceMemo?.data ?? null);
  const [err, setErr] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    if (nonce === 0 && evidenceMemo && Date.now() - evidenceMemo.at < 10 * 60_000) return;
    let alive = true;
    setLoading(true);
    paperApi.evidence().then(
      (d) => {
        evidenceMemo = { at: Date.now(), data: d };
        if (alive) { setData(d); setErr(null); }
      },
      (e) => { if (alive) { authGate(e); setErr(e); } },
    ).finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [nonce]);
  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { data, err, loading, reload };
}

// =================================================================== status strip + view tabs
type Tab = "overview" | "positions" | "book" | "signals" | "evidence";
const TABS: { v: Tab; label: string; title: string }[] = [
  { v: "overview", label: "OVERVIEW", title: "Settings, account figures and the equity curve" },
  { v: "positions", label: "POSITIONS", title: "Open paper positions and swing entries waiting for the open" },
  { v: "book", label: "BOOK", title: "Closed trades: why each was opened and why it was closed" },
  { v: "signals", label: "SIGNALS", title: "Today's 09:30 / 10:00 decisions and the swing candidates" },
  { v: "evidence", label: "EVIDENCE", title: "Backtests of both strategies, with a holdout period" },
];
const TAB_IDS = TABS.map((t) => t.v);

const MODEL_SRC: Record<string, string> = { local: "trained here", shipped: "shipped with the app", none: "none" };

function StatusStrip({ st, err, at }: { st: PaperState | null; err: unknown; at: number }) {
  const svc = st?.service;
  const down = isStatus(err, 503) || (!!svc && !svc.running);
  const m = svc?.models;
  return (
    <div className="pp-status" aria-label="Paper service status">
      <span className="pp-brand">PAPER</span>
      <span className="pp-st">
        <span className={`dot ${!svc && !err ? "wait" : down || (!svc && err) ? "off" : "on"}`} aria-hidden="true" />
        {!svc ? (err ? (isStatus(err, 503) ? "SERVICE STOPPED" : "UNAVAILABLE") : "CONNECTING…") : down ? "SERVICE STOPPED" : "ENGINE RUNNING"}
      </span>
      {svc && (
        <span className={`chip ${svc.mode === "live" ? "cyan" : "violet"}`}
          title={svc.mode === "live" ? "Paper fills on live market prices" : "Simulated market data — demonstrates the mechanism only"}>
          {svc.mode === "live" ? "LIVE PRICES" : "SIMULATED MARKET"}
        </span>
      )}
      {m && (
        <span title="The intraday model the engine uses">
          model <b className={m.source === "none" ? "amber" : undefined}>{MODEL_SRC[m.source] ?? m.source}</b>
          {ok(m.trained_to) && <> · data to <b>{asofText(m.trained_to)}</b></>}
        </span>
      )}
      {svc && ok(svc.clock) && <span>engine clock <b>{timeIST(svc.clock, true)}</b>{svc.phase ? <> · <b>{svc.phase}</b></> : null}</span>}
      {st?.user && <span>account <b>{st.user}</b></span>}
      {svc?.note && !down && <span className="pp-st-note">{svc.note}</span>}
      <span className="pp-st-right">{at ? `updated ${timeIST(at, true)} · ` : ""}simulated fills · no real orders</span>
    </div>
  );
}

function SubTabs({ value, onChange, badges }: { value: Tab; onChange: (t: Tab) => void; badges: Partial<Record<Tab, number>> }) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKey = (e: ReactKeyboardEvent<HTMLButtonElement>, i: number) => {
    const n = TABS.length;
    const j = e.key === "ArrowRight" ? (i + 1) % n : e.key === "ArrowLeft" ? (i - 1 + n) % n
      : e.key === "Home" ? 0 : e.key === "End" ? n - 1 : -1;
    if (j < 0) return;
    e.preventDefault();
    onChange(TABS[j].v);
    refs.current[j]?.focus();
  };
  return (
    <div className="pp-tabs" role="tablist" aria-label="Paper trade views">
      {TABS.map((t, i) => (
        <button key={t.v} ref={(el) => { refs.current[i] = el; }} type="button" role="tab" id={`pp-tab-${t.v}`}
          aria-controls="pp-panel" aria-selected={value === t.v} tabIndex={value === t.v ? 0 : -1}
          title={`${t.title} — key ${i + 1}`} onClick={() => onChange(t.v)} onKeyDown={(e) => onKey(e, i)}>
          <span className="pp-tkey" aria-hidden="true">{i + 1}</span>
          {t.label}
          {!!badges[t.v] && <span className="pp-tn">{badges[t.v]}</span>}
        </button>
      ))}
    </div>
  );
}

// =================================================================== OVERVIEW
/** "500000", "5,00,000", "5L", "5 lakh", "1cr", "250k" → rupees. */
function parseAmount(s: string): number | null {
  const t = s.trim().toLowerCase().replace(/[₹,\s_]/g, "");
  if (!t) return null;
  const m = /^(\d+(?:\.\d+)?)(k|l|lac|lacs|lakh|lakhs|cr|crore|crores)?$/.exec(t);
  if (!m) return null;
  const mult = !m[2] ? 1 : m[2] === "k" ? 1e3 : m[2].startsWith("c") ? 1e7 : 1e5;
  return Math.round(Number(m[1]) * mult);
}

type Msg = { ok: boolean; text: string } | null;

function SettingsForm({ kind, s, accountCapital, onState }: {
  kind: Strategy; s: StratSettings; accountCapital: number | undefined; onState: (st: PaperState) => void;
}) {
  const [cap, setCap] = useState<string | null>(null);        // null: showing the saved value
  const [lev, setLev] = useState<number | null>(null);
  const [busy, setBusy] = useState<"" | "save" | "toggle">("");
  const [msg, setMsg] = useState<Msg>(null);
  const intra = kind === "intraday";
  const name = intra ? "intraday" : "swing";
  const savedLev = ok(s.leverage) ? s.leverage : 1;
  const capV = cap === null ? s.capital : parseAmount(cap);
  const levV = lev ?? savedLev;
  const capErr = capV === null ? "Enter an amount, e.g. 500000, 5L or 1cr."
    : capV < CAPITAL_MIN ? `At least ${inr(CAPITAL_MIN, 0)}.` : capV > CAPITAL_MAX ? `At most ${inr(CAPITAL_MAX, 0)} (₹10 crore).` : "";
  const levErr = intra && !(levV >= LEV_MIN && levV <= LEV_MAX) ? "Leverage must be between 1x and 5x." : "";
  const dirty = (cap !== null && capV !== s.capital) || (intra && levV !== savedLev);   // an unreadable amount counts too (UNDO)
  const canSave = dirty && !capErr && !levErr && !busy;
  const bp = ok(capV) ? capV * (intra ? levV : 1) : null;

  const send = (patch: { enabled?: boolean; capital?: number; leverage?: number }, what: "save" | "toggle") => {
    setBusy(what);
    setMsg(null);
    paperApi.settings(intra ? { intraday: patch } : { swing: { enabled: patch.enabled, capital: patch.capital } }).then(
      (next) => {
        onState(next);
        setCap(null);
        setLev(null);
        if (what === "toggle") {
          setMsg({ ok: true, text: patch.enabled ? `Auto-trading on: the engine trades ${name} for you.` : `Paused: the engine opens no new ${name} trades.` });
        } else {
          setMsg({
            ok: true,
            text: intra
              ? `Saved. New intraday trades are sized from ${inr(patch.capital, 0)} × ${fx(patch.leverage, 1)}x = ${inr((patch.capital ?? 0) * (patch.leverage ?? 1), 0)} buying power.`
              : `Saved. New swing trades are sized from ${inr(patch.capital, 0)}.`,
          });
        }
      },
      (e) => { authGate(e); setMsg({ ok: false, text: paperErrText(e) }); },
    ).finally(() => setBusy(""));
  };
  const submit = () => {
    if (!canSave || capV === null) return;
    send(intra ? { capital: capV, leverage: levV } : { capital: capV }, "save");
  };

  const ids = { cap: `pp-cap-${kind}`, capH: `pp-cap-${kind}-h`, lev: `pp-lev-${kind}`, sw: `pp-sw-${kind}` };
  return (
    <section className="pp-sec" aria-label={`${name} settings`}>
      <h3 className="pp-h3">Settings</h3>
      <div className="pp-enable">
        <button type="button" role="switch" aria-checked={s.enabled} aria-labelledby={ids.sw} className={`sw ${s.enabled ? "on" : ""}`}
          disabled={!!busy} onClick={() => send({ enabled: !s.enabled }, "toggle")}>
          <span className="sw-knob" />
        </button>
        <span id={ids.sw}>
          <b>{s.enabled ? "Auto-trading ON" : "Auto-trading OFF"}</b>{" "}
          <span className="muted">{busy === "toggle" ? "saving…" : s.enabled ? `— the engine trades ${name} for you` : "— paused, no new trades"}</span>
        </span>
      </div>
      <form className="pp-form" noValidate onSubmit={(e) => { e.preventDefault(); submit(); }}>
        <div className="field">
          <label htmlFor={ids.cap}>Capital (₹)</label>
          <input id={ids.cap} inputMode="decimal" autoComplete="off" spellCheck={false} value={cap ?? fx(s.capital, 0)}
            onChange={(e) => { setCap(e.target.value); setMsg(null); }} aria-invalid={!!capErr} aria-describedby={ids.capH} />
          <span className={capErr ? "err" : "hint"} id={ids.capH}>
            {capErr || `${inr(capV, 0)} · ${compact(capV)} · min ₹10,000, max ₹10 crore · 5L / 1cr work too`}
          </span>
        </div>
        {intra && (
          <div className="field">
            <label htmlFor={ids.lev}>Leverage</label>
            <div className="pp-lev-row">
              <input id={ids.lev} type="range" min={LEV_MIN} max={LEV_MAX} step={0.5} value={levV}
                onChange={(e) => { setLev(Number(e.target.value)); setMsg(null); }} aria-valuetext={`${fx(levV, 1)} times capital`} />
              <span className="pp-lev-val" aria-hidden="true">{fx(levV, 1)}x</span>
            </div>
            <div className="pp-ticks" aria-hidden="true">{[1, 2, 3, 4, 5].map((t) => <span key={t}>{t}x</span>)}</div>
            {levErr && <span className="err">{levErr}</span>}
          </div>
        )}
        {intra && (
          <div className="pp-bp">
            Buying power = {inr(capV, 0)} capital × {fx(levV, 1)}x leverage = <b>{inr(bp, 0)}</b>
            <span className="muted"> — about the most the engine holds at once across intraday positions (it moves with the account's equity).</span>
          </div>
        )}
        <div className="pp-actions">
          <button type="submit" className="btn primary" disabled={!canSave}>{busy === "save" ? "SAVING…" : "APPLY"}</button>
          {dirty && <button type="button" className="btn ghost" onClick={() => { setCap(null); setLev(null); setMsg(null); }}>UNDO</button>}
          {dirty && <span className="pp-dirty">unsaved changes</span>}
        </div>
      </form>
      {msg && <p className={`pp-msg ${msg.ok ? "ok" : "err"}`} role={msg.ok ? "status" : "alert"}>{msg.text}</p>}
      <p className="pp-note">
        You can change these with positions open: <b>new</b> trades are sized with the new values, open positions keep the size they were opened with.
        {ok(accountCapital) && ok(s.capital) && accountCapital !== s.capital && ` The account still shows ${inr(accountCapital, 0)} until new trades use the new capital.`}
      </p>
    </section>
  );
}

function IntradayCard({ st, onState }: { st: PaperState; onState: (s: PaperState) => void }) {
  const a = st.intraday;
  const s = st.settings.intraday;
  // buying_power = what is still free (leverage × equity − exposure); the limit is exposure + free
  const limit = ok(a.buying_power) && ok(a.gross_exposure) ? a.gross_exposure + a.buying_power : null;
  const used = limit !== null && limit > 0 ? a.gross_exposure / limit : null;
  return (
    <Panel code="INTRA" title="Intraday" area="intra" id="pp-intra" sub="decides 09:30 & 10:00 IST · long or short · flat by 15:15"
      actions={<OnOff on={s.enabled} />}>
      {a.loss_limit_hit && (
        <div className="banner warn pp-banner" role="alert">
          <b>Daily loss limit hit.</b> The engine takes no new intraday trades today; anything still open is squared off by 15:15 as usual.
        </div>
      )}
      <div className="kpis" style={{ "--kpi-min": "128px" } as CSSProperties}>
        <Kpi label="Equity" value={inr(a.equity, 0)} sub={`capital ${inr(a.capital, 0)} · ${fx(a.leverage, 1)}x`} />
        <Kpi label="Total P&L" value={sinr(a.pnl_total, 0)} tone={cls(a.pnl_total, 0.5)}
          sub={ok(a.pnl_total) && ok(a.capital) && a.capital > 0 ? `${fracPct(a.pnl_total / a.capital, 2, true)} on capital` : undefined} />
        <Kpi label="Today" value={sinr(a.pnl_today, 0)} tone={cls(a.pnl_today, 0.5)}
          sub={`booked ${sinr(a.realized_today, 0)} · open ${sinr(a.unrealized, 0)}`} />
        <Kpi label="Exposure" value={inr(a.gross_exposure, 0)}
          sub={used !== null ? `${fx(used * 100, 0)}% used · ${inr(a.buying_power, 0)} buying power free` : `buying power free ${inr(a.buying_power, 0)}`} />
        <Kpi label="Trades today" value={fx(a.trades_today, 0)} sub={`${a.positions.length} open now`} />
      </div>
      {used !== null && (
        <div className="pp-meter" title={`Gross exposure ${inr(a.gross_exposure, 0)} of ${inr(limit, 0)} (capital × leverage at today's equity); ${inr(a.buying_power, 0)} still free`}>
          <div className="bar meter" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(Math.min(1, used) * 100)} aria-label="Buying power in use">
            <i style={{ left: 0, width: `${Math.min(1, Math.max(0, used)) * 100}%` }} />
          </div>
        </div>
      )}
      <SettingsForm kind="intraday" s={s} accountCapital={a.capital} onState={onState} />
    </Panel>
  );
}

function SwingCard({ st, onState }: { st: PaperState; onState: (s: PaperState) => void }) {
  const a = st.swing;
  const s = st.settings.swing;
  const invested = ok(a.invested) && ok(a.equity) && a.equity > 0 ? a.invested / a.equity : null;
  return (
    <Panel code="SWING" title="Swing" area="swing" id="pp-swing" sub="long-only · days to weeks · entries fill at the next open"
      actions={<OnOff on={s.enabled} />}>
      <div className="kpis" style={{ "--kpi-min": "128px" } as CSSProperties}>
        <Kpi label="Equity" value={inr(a.equity, 0)} sub={`capital ${inr(a.capital, 0)}`} />
        <Kpi label="Total P&L" value={sinr(a.pnl_total, 0)} tone={cls(a.pnl_total, 0.5)}
          sub={ok(a.pnl_total) && ok(a.capital) && a.capital > 0 ? `${fracPct(a.pnl_total / a.capital, 2, true)} on capital` : undefined} />
        <Kpi label="Open P&L" value={sinr(a.unrealized, 0)} tone={cls(a.unrealized, 0.5)} sub="unrealised, at last price" />
        <Kpi label="Invested" value={inr(a.invested, 0)} sub={invested !== null ? `${fx(invested * 100, 0)}% of equity` : undefined} />
        <Kpi label="Cash" value={inr(a.cash, 0)} />
        <Kpi label="Positions" value={fx(a.positions.length, 0)} sub={a.pending.length ? `+${a.pending.length} filling at next open` : "none waiting"} />
      </div>
      <RegimeLine r={a.regime} asof={a.regime?.asof} />
      <SettingsForm kind="swing" s={s} accountCapital={a.capital} onState={onState} />
    </Panel>
  );
}

function EquityPanel({ curve }: { curve: PaperState["equity_curve"] }) {
  const last = curve[curve.length - 1];
  const sig = last ? `${curve.length}:${last.date}:${last.intraday}:${last.swing}` : "0";
  const m = useMemo(() => {
    const pts = curve.filter((p) => ok(p.date)).sort((a, b) => a.date - b.date);
    const tot = pts.map((p) => (ok(p.intraday) ? p.intraday : 0) + (ok(p.swing) ? p.swing : 0));
    const series: LineSeries[] = [
      { name: "Total", values: tot, tone: "text" },
      { name: "Intraday", values: pts.map((p) => (ok(p.intraday) ? p.intraday : null)), tone: 1 },
      { name: "Swing", values: pts.map((p) => (ok(p.swing) ? p.swing : null)), tone: 2 },
    ];
    return { x: pts.map((p) => ymdTs(p.date)), series, n: pts.length, first: pts[0], start: tot[0], end: tot[tot.length - 1] };
  }, [sig]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <Panel code="EQ" title="Equity curve" area="curve" id="pp-eq"
      sub={m.n >= 2 ? `end-of-day paper equity · ${sinr(m.end - m.start, 0)} since ${ymdText(m.first?.date)}` : "end-of-day paper equity"}>
      {m.n < 2 ? (
        <Empty>The equity curve starts after the first two trading days close.</Empty>
      ) : (
        <div className="pp-pad">
          <LineChart label="Paper account equity by day (intraday, swing and total)" x={m.x} series={m.series} height={250} format="inr"
            caption="Each point is the account's equity at the close; the gap between Total and the two strategies is simply their sum." />
        </div>
      )}
    </Panel>
  );
}

function ResetPanel({ st, onState }: { st: PaperState; onState: (s: PaperState) => void }) {
  const [which, setWhich] = useState<BookScope>("intraday");
  const [armed, setArmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<Msg>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const armRef = useRef<HTMLButtonElement>(null);
  const wasArmed = useRef(false);
  useEffect(() => {
    if (armed) cancelRef.current?.focus();            // the safe choice gets the focus
    else if (wasArmed.current) armRef.current?.focus();
    wasArmed.current = armed;
  }, [armed]);
  const name = which === "all" ? "both paper accounts" : `the ${which} paper account`;
  const capText = which === "all"
    ? `${inr(st.settings.intraday.capital, 0)} (intraday) and ${inr(st.settings.swing.capital, 0)} (swing)`
    : inr(st.settings[which].capital, 0);
  const go = () => {
    setBusy(true);
    paperApi.reset(which).then(
      (s) => { onState(s); setArmed(false); setMsg({ ok: true, text: `Done — ${name} restarted from ${capText}.` }); },
      (e) => { authGate(e); setMsg({ ok: false, text: paperErrText(e) }); },
    ).finally(() => setBusy(false));
  };
  return (
    <Panel code="RESET" title="Start over" area="reset" id="pp-reset" sub="wipe a paper account and restart it from its capital setting">
      <div className="pp-reset" onKeyDown={(e) => { if (e.key === "Escape" && armed && !busy) { e.stopPropagation(); setArmed(false); } }}>
        {!armed ? (
          <div className="pp-reset-row">
            <Seg label="Account to reset" value={which} onChange={(v) => { setWhich(v); setMsg(null); }}
              options={[{ v: "intraday", label: "INTRADAY" }, { v: "swing", label: "SWING" }, { v: "all", label: "BOTH" }]} />
            <button ref={armRef} type="button" className="btn danger" onClick={() => { setArmed(true); setMsg(null); }}>RESET…</button>
            <span className="muted small">You will be asked to confirm.</span>
          </div>
        ) : (
          <div className="pp-confirm" role="group" aria-labelledby="pp-reset-q">
            <p id="pp-reset-q">
              <b>Reset {name}?</b> This permanently deletes its paper positions, trade book and equity history and restarts it
              from {capText}. It cannot be undone. Paper only — no real orders or money are involved.
            </p>
            <div className="pp-reset-row">
              <button type="button" className="btn danger armed" onClick={go} disabled={busy}>
                {busy ? "RESETTING…" : `YES, RESET ${which === "all" ? "BOTH" : which.toUpperCase()}`}
              </button>
              <button ref={cancelRef} type="button" className="btn ghost" onClick={() => setArmed(false)} disabled={busy}>CANCEL</button>
            </div>
          </div>
        )}
        {msg && <p className={`pp-msg ${msg.ok ? "ok" : "err"}`} role={msg.ok ? "status" : "alert"}>{msg.text}</p>}
      </div>
    </Panel>
  );
}

function OverviewView({ st, onState }: { st: PaperState; onState: (s: PaperState) => void }) {
  return (
    <>
      <IntradayCard st={st} onState={onState} />
      <SwingCard st={st} onState={onState} />
      <EquityPanel curve={st.equity_curve} />
      <ResetPanel st={st} onState={onState} />
    </>
  );
}

// =================================================================== POSITIONS
type StratF = "all" | Strategy;
const STRAT_F: readonly StratF[] = ["all", "intraday", "swing"];

function openedText(p: Position): string {
  if (p.strategy === "intraday") return `${p.decision || "—"} decision · in ${ok(p.entry_ts) ? timeIST(p.entry_ts) : "—"}`;
  return `${ok(p.entry_ts) ? dateIST(p.entry_ts) : "—"}${ok(p.days_held) ? ` · ${fx(p.days_held, 0)} session${p.days_held === 1 ? "" : "s"}` : ""}`;
}

const POS_COLS = 13;
function PosRow({ p, open, onToggle }: { p: Position; open: boolean; onToggle: () => void }) {
  const id = `pp-pos-${safeId(p.id)}`;
  const trailed = ok(p.initial_stop) && ok(p.stop) && Math.abs(p.stop - p.initial_stop) > 1e-9;
  const ref = ok(p.ltp) ? p.ltp : p.entry;
  const away = ok(ref) && ref > 0 && ok(p.stop) ? (p.side === "SHORT" ? (p.stop - ref) / ref : (ref - p.stop) / ref) : null;
  return (
    <>
      <tr className={open ? "sel" : undefined} onClick={onToggle}>
        <td className="pp-xcol"><XBtn open={open} id={id} label={`${p.symbol}: why it was opened`} onToggle={onToggle} /></td>
        <td className="l"><StratChip s={p.strategy} /></td>
        <td className="sym">{p.symbol}</td>
        <td className="l"><SideChip side={p.side} /></td>
        <td>{fx(p.qty, 0)}</td>
        <td>{px(p.entry)}</td>
        <td>{px(p.ltp)}</td>
        <td title={`Initial stop ${px(p.initial_stop)}${away !== null ? ` · current stop ${fx(away * 100, 2)}% from the last price` : ""}`}>
          {px(p.stop)}{trailed && <span className="chip amber pp-trail" title={`Trailing stop — moved from ${px(p.initial_stop)}`}>TRAIL</span>}
        </td>
        <td><PnL v={p.pnl} /></td>
        <td><Frac v={p.pnl_pct} /></td>
        <td><RMult v={p.r_mult} /></td>
        <td className="l"><SetupChip setup={p.reasons?.setup} /></td>
        <td className="l muted">{openedText(p)}</td>
      </tr>
      {open && <DetailRow id={id} cols={POS_COLS}><Reasons r={p.reasons} /></DetailRow>}
    </>
  );
}

function PendingPanel({ pending }: { pending: Pending[] }) {
  const [open, toggle] = useExpand();
  return (
    <Panel code="NEXT" title="Swing entries for the next open" area="pending" id="pp-pending"
      sub="planned after the close · fill at the next session's opening price">
      {!pending.length ? (
        <Empty>No swing entries waiting. New entries are planned after the close and fill at the next open.</Empty>
      ) : (
        <table className="tbl pp-tbl">
          <thead>
            <tr>
              <th className="pp-xcol"><span className="sr-only">Details</span></th><th className="l">Symbol</th><th className="l">Setup</th>
              <th>Score</th><th className="l">Planned</th><th className="l">Status</th><th className="l">Why</th>
            </tr>
          </thead>
          <tbody>
            {pending.map((p, i) => {
              const key = `${p.symbol}-${p.planned_at}-${i}`;
              const id = `pp-pend-${safeId(key)}`;
              const isOpen = open.has(key);
              return (
                <Fragment key={key}>
                  <tr className={isOpen ? "sel" : undefined} onClick={() => toggle(key)}>
                    <td className="pp-xcol"><XBtn open={isOpen} id={id} label={`${p.symbol}: why it will be bought`} onToggle={() => toggle(key)} /></td>
                    <td className="sym">{p.symbol}</td>
                    <td className="l"><SetupChip setup={p.reasons?.setup} /></td>
                    <td>{ok(p.reasons?.score) ? fx(p.reasons.score, 2) : "—"}</td>
                    <td className="l muted">{when(p.planned_at)}</td>
                    <td className="l"><span className="chip cyan">FILLS AT NEXT OPEN</span></td>
                    <td className="l pp-sumcell" title={p.reasons?.summary}>{p.reasons?.summary || "—"}</td>
                  </tr>
                  {isOpen && <DetailRow id={id} cols={7}><Reasons r={p.reasons} /></DetailRow>}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

function PositionsView({ st }: { st: PaperState }) {
  const [f, setF] = usePref<StratF>("td.paper.posF", "all", STRAT_F);
  const [open, toggle] = useExpand();
  const all = [...st.intraday.positions, ...st.swing.positions];
  const rows = all.filter((p) => f === "all" || p.strategy === f);
  const pnl = rows.reduce((a, p) => a + (ok(p.pnl) ? p.pnl : 0), 0);
  const expo = rows.reduce((a, p) => a + Math.abs((ok(p.qty) ? p.qty : 0) * (ok(p.ltp) ? p.ltp : ok(p.entry) ? p.entry : 0)), 0);
  const off = [!st.settings.intraday.enabled && "intraday", !st.settings.swing.enabled && "swing"].filter(Boolean).join(" and ");
  return (
    <>
      <Panel code="POS" title="Open positions" area="open" id="pp-open" className="cap-m pp-scrollp"
        sub={`${rows.length} open · unrealised ${sinr(pnl, 0)} · open a row to see why it was taken`}
        bar={<Seg label="Strategy" value={f} onChange={setF} options={[
          { v: "all", label: <>ALL<span className="n">{all.length}</span></> },
          { v: "intraday", label: <>INTRADAY<span className="n">{st.intraday.positions.length}</span></> },
          { v: "swing", label: <>SWING<span className="n">{st.swing.positions.length}</span></> },
        ]} />}>
        {!rows.length ? (
          <Empty>
            No open paper positions{f !== "all" ? ` for ${f}` : ""}. Intraday trades open at the 09:30 and 10:00 decisions when something clears
            the cost hurdle; swing entries are planned after the close and fill at the next open.
            {off && ` Auto-trading is off for ${off} — switch it on under OVERVIEW.`}
          </Empty>
        ) : (
          <table className="tbl pp-tbl">
            <thead>
              <tr>
                <th className="pp-xcol"><span className="sr-only">Details</span></th><th className="l">Strat</th><th className="l">Symbol</th>
                <th className="l">Side</th><th>Qty</th><th>Entry</th><th>LTP</th><th title="Current stop (TRAIL = trailing stop has moved)">Stop</th>
                <th>P&L</th><th>P&L %</th><th title="Profit or loss in units of the initial risk">R</th><th className="l">Setup</th><th className="l">Opened</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => <PosRow key={p.id} p={p} open={open.has(p.id)} onToggle={() => toggle(p.id)} />)}
            </tbody>
            <tfoot>
              <tr className="pp-tot">
                <td colSpan={8} className="l">Total · {rows.length} position{rows.length === 1 ? "" : "s"} · gross exposure {inr(expo, 0)}</td>
                <td><PnL v={pnl} d={0} /></td>
                <td colSpan={4} />
              </tr>
            </tfoot>
          </table>
        )}
      </Panel>
      <PendingPanel pending={st.swing.pending} />
    </>
  );
}

// =================================================================== BOOK
type SideF = "all" | Side;
type Outcome = "all" | "win" | "loss";
type SortK = "exit_ts" | "pnl" | "pnl_pct" | "r_mult";

const CHARGE_LABEL: Record<string, string> = {
  brokerage: "Brokerage", stt: "STT", exchange: "Exchange charge", sebi: "SEBI fee", stamp: "Stamp duty", gst: "GST", dp: "DP charge",
};
const chargeTotal = (c: Record<string, number> | undefined) => {
  if (!c) return 0;
  if (ok(c.total)) return c.total;
  return Object.entries(c).reduce((a, [, v]) => a + (ok(v) ? v : 0), 0);
};

function Costs({ t }: { t: Trade }) {
  const parts = Object.entries(t.charges ?? {}).filter(([k, v]) => k !== "total" && ok(v) && Math.abs(v) >= 0.005);
  const total = chargeTotal(t.charges);
  const slip = ok(t.slippage) ? Math.abs(t.slippage) : null;
  // slippage either sits inside the fill prices (gross − charges = net) or is deducted on top: show whichever adds up
  const separate = slip !== null && ok(t.pnl_gross) && ok(t.pnl)
    && Math.abs(t.pnl_gross - total - slip - t.pnl) < Math.abs(t.pnl_gross - total - t.pnl);
  return (
    <div className="pp-costs">
      <span className="pp-h4">From gross to net</span>
      <dl className="pp-kv">
        <dt>Gross P&L</dt><dd><PnL v={t.pnl_gross} /></dd>
        {parts.map(([k, v]) => <Fragment key={k}><dt className="sub">− {CHARGE_LABEL[k] ?? k}</dt><dd className="sub">{inr(v)}</dd></Fragment>)}
        <dt>− Charges, total</dt><dd>{inr(total)}</dd>
        {separate && <><dt>− Slippage (modelled)</dt><dd>{inr(slip)}</dd></>}
        <dt className="tot">Net P&L</dt><dd className="tot"><PnL v={t.pnl} /></dd>
      </dl>
      {!separate && slip !== null && (
        <p className="pp-note pp-note-tight">
          Modelled slippage of {inr(slip)} (half the spread plus market impact; stops fill worse) is already in the entry and exit prices.
        </p>
      )}
    </div>
  );
}

function TradeDetail({ t }: { t: Trade }) {
  return (
    <div className="pp-trade">
      <section className="pp-trade-open" aria-label="Why it was opened">
        <h4 className="pp-h3 pp-h3-in">Why it was opened <span className="pp-h3-sub">· {t.decision || "—"} · {when(t.entry_ts)} at {px(t.entry)}</span></h4>
        <Reasons r={t.reasons} />
      </section>
      <section className="pp-trade-close" aria-label="Why it was closed">
        <div>
          <h4 className="pp-h3 pp-h3-in">Why it was closed <span className="pp-h3-sub">· {when(t.exit_ts)} at {px(t.exit)} · held {t.holding || "—"}</span></h4>
          <p className="pp-summary"><ExitChip why={t.why_exit} /> {t.exit_note || "—"}</p>
          <dl className="pp-nums">
            <div><dt>Return</dt><dd><Frac v={t.pnl_pct} /></dd></div>
            <div><dt>R multiple</dt><dd><RMult v={t.r_mult} /></dd></div>
            <div><dt>Quantity</dt><dd>{fx(t.qty, 0)}</dd></div>
          </dl>
        </div>
        <Costs t={t} />
      </section>
    </div>
  );
}

const BOOK_COLS = 14;
function TradeRow({ t, open, onToggle }: { t: Trade; open: boolean; onToggle: () => void }) {
  const id = `pp-trade-${safeId(t.id)}`;
  return (
    <>
      <tr className={open ? "sel" : undefined} onClick={onToggle}>
        <td className="pp-xcol"><XBtn open={open} id={id} label={`${t.symbol}: why it was opened and closed`} onToggle={onToggle} /></td>
        <td className="l muted">{ok(t.exit_ts) ? (t.strategy === "intraday" ? `${dateIST(t.exit_ts)} ${timeIST(t.exit_ts)}` : dateIST(t.exit_ts)) : "—"}</td>
        <td className="l"><StratChip s={t.strategy} /></td>
        <td className="sym">{t.symbol}</td>
        <td className="l"><SideChip side={t.side} /></td>
        <td>{fx(t.qty, 0)}</td>
        <td>{px(t.entry)}</td>
        <td>{px(t.exit)}</td>
        <td><PnL v={t.pnl} /></td>
        <td><Frac v={t.pnl_pct} /></td>
        <td><RMult v={t.r_mult} /></td>
        <td className="muted">{t.holding || "—"}</td>
        <td className="l"><SetupChip setup={t.reasons?.setup} /></td>
        <td className="l"><ExitChip why={t.why_exit} /></td>
      </tr>
      {open && <DetailRow id={id} cols={BOOK_COLS}><TradeDetail t={t} /></DetailRow>}
    </>
  );
}

function StatsRow({ kind, s }: { kind: Strategy; s: Stats }) {
  return (
    <div className="pp-statrow">
      <div className="pp-statrow-h"><StratChip s={kind} /></div>
      {!s.trades ? (
        <p className="pp-note pp-note-mid">No closed {kind} trades yet.</p>
      ) : (
        <div className="kpis" style={{ "--kpi-min": "108px" } as CSSProperties}>
          <Kpi label="Trades" value={fx(s.trades, 0)} />
          <Kpi label="Win rate" value={ok(s.win_rate) ? fracPct(s.win_rate, 1) : "—"} />
          <Kpi label="Profit factor" value={ok(s.profit_factor) ? fx(s.profit_factor, 2) : "—"} sub="wins ÷ losses, ₹" />
          <Kpi label="Avg P&L" value={sinr(s.avg_pnl, 0)} tone={cls(s.avg_pnl, 0.5)} sub="per trade, net" />
          <Kpi label="Avg R" value={ok(s.avg_r) ? `${signed(s.avg_r, 2)}R` : "—"} tone={cls(s.avg_r, 0.005)} />
          <Kpi label="Best" value={sinr(s.best, 0)} tone={cls(s.best, 0.5)} />
          <Kpi label="Worst" value={sinr(s.worst, 0)} tone={cls(s.worst, 0.5)} />
          <Kpi label="Max drawdown" value={ddText(s.max_dd)} tone={ok(s.max_dd) && s.max_dd !== 0 ? "down" : undefined} />
          <Kpi label="Charges paid" value={inr(s.charges, 0)} />
        </div>
      )}
    </div>
  );
}

function SortTh({ k, sort, setSort, children, title }: {
  k: SortK; sort: { k: SortK; d: 1 | -1 }; setSort: (s: { k: SortK; d: 1 | -1 }) => void; children: ReactNode; title?: string;
}) {
  return (
    <th aria-sort={sort.k === k ? (sort.d < 0 ? "descending" : "ascending") : "none"} title={title}>
      <button type="button" className="pp-sortbtn" onClick={() => setSort({ k, d: sort.k === k ? ((-sort.d) as 1 | -1) : -1 })}>{children}</button>
    </th>
  );
}

function BookView({ st }: { st: PaperState | null }) {
  const [scope, setScope] = usePref<BookScope>("td.paper.bookScope", "all", STRAT_F);
  const [side, setSide] = useState<SideF>("all");
  const [res, setRes] = useState<Outcome>("all");
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<{ k: SortK; d: 1 | -1 }>({ k: "exit_ts", d: -1 });
  const [open, toggle] = useExpand();
  const closed = st ? (st.intraday.stats.trades || 0) + (st.swing.stats.trades || 0) : 0;
  const book = useBook(scope, closed);
  const rows = useMemo(() => {
    const s = q.trim().toUpperCase();
    const r = (book.trades ?? []).filter((t) => (scope === "all" || t.strategy === scope) && (side === "all" || t.side === side)
      && (res === "all" || (res === "win" ? t.pnl > 0 : t.pnl <= 0)) && (!s || t.symbol.includes(s)));
    const val = (t: Trade) => { const v = t[sort.k]; return ok(v) ? v : -Infinity; };
    return [...r].sort((a, b) => {
      const x = val(a), y = val(b);
      return x === y ? 0 : (x < y ? -1 : 1) * sort.d;
    });
  }, [book.trades, scope, side, res, q, sort]);
  const tot = useMemo(() => {
    const t = { net: 0, gross: 0, charges: 0, slip: 0, wins: 0, r: 0, nr: 0 };
    for (const x of rows) {
      t.net += ok(x.pnl) ? x.pnl : 0;
      t.gross += ok(x.pnl_gross) ? x.pnl_gross : 0;
      t.charges += chargeTotal(x.charges);
      t.slip += ok(x.slippage) ? Math.abs(x.slippage) : 0;
      if (x.pnl > 0) t.wins += 1;
      if (ok(x.r_mult)) { t.r += x.r_mult; t.nr += 1; }
    }
    return t;
  }, [rows]);
  const n = rows.length;
  return (
    <>
      <Panel code="STATS" title="Closed-trade statistics" area="stats" id="pp-stats" sub="every closed paper trade since the last reset, net of charges">
        {!st ? <Empty>Waiting for the paper account…</Empty> : (
          <>
            <StatsRow kind="intraday" s={st.intraday.stats} />
            <StatsRow kind="swing" s={st.swing.stats} />
          </>
        )}
      </Panel>
      <Panel code="BOOK" title="Trade journal" area="book" id="pp-book" className="cap-m pp-scrollp"
        sub={book.trades ? `${n} shown · latest 200 closed trades · open a row for why it was opened and closed` : "closed trades"}
        actions={<button type="button" className="btn" onClick={book.reload} disabled={book.loading}>{book.loading ? "LOADING…" : "REFRESH"}</button>}
        bar={
          <>
            <Seg label="Strategy" value={scope} onChange={setScope}
              options={[{ v: "all", label: "ALL" }, { v: "intraday", label: "INTRADAY" }, { v: "swing", label: "SWING" }]} />
            <Seg label="Side" value={side} onChange={setSide} options={[{ v: "all", label: "BOTH SIDES" }, { v: "LONG", label: "▲ LONG" }, { v: "SHORT", label: "▼ SHORT" }]} />
            <Seg label="Outcome" value={res} onChange={setRes} options={[{ v: "all", label: "ALL" }, { v: "win", label: "WINNERS" }, { v: "loss", label: "LOSERS" }]} />
            <label className="sr-only" htmlFor="pp-book-q">Filter by symbol</label>
            <input id="pp-book-q" className="filter-input" placeholder="symbol…" value={q} autoComplete="off" spellCheck={false}
              onChange={(e) => setQ(e.target.value)} />
          </>
        }
        foot={n ? (
          <span>
            {n} trade{n === 1 ? "" : "s"} · {tot.wins} won / {n - tot.wins} lost ({fx((tot.wins / n) * 100, 0)}%) · gross {sinr(tot.gross, 0)} ·
            charges {inr(tot.charges, 0)} · modelled slippage {inr(tot.slip, 0)} · net <PnL v={tot.net} d={0} />
          </span>
        ) : undefined}>
        {book.err && !book.trades ? <ErrLine e={book.err} retry={book.reload} /> : !book.trades ? (
          <Empty>Loading the trade journal…</Empty>
        ) : !n ? (
          <Empty>{book.trades.length ? "No trade matches these filters." : "No closed paper trades yet — they appear here, with the reasons, as soon as the engine closes one."}</Empty>
        ) : (
          <div className={book.loading ? "pp-busy" : undefined}>
            {book.err ? <ErrLine e={book.err} retry={book.reload} /> : null}
            <table className="tbl pp-tbl">
              <thead>
                <tr>
                  <th className="pp-xcol"><span className="sr-only">Details</span></th>
                  <SortTh k="exit_ts" sort={sort} setSort={setSort}>Closed</SortTh>
                  <th className="l">Strat</th><th className="l">Symbol</th><th className="l">Side</th><th>Qty</th><th>Entry</th><th>Exit</th>
                  <SortTh k="pnl" sort={sort} setSort={setSort} title="Net of charges">Net P&L</SortTh>
                  <SortTh k="pnl_pct" sort={sort} setSort={setSort}>Return</SortTh>
                  <SortTh k="r_mult" sort={sort} setSort={setSort} title="Profit or loss in units of the initial risk">R</SortTh>
                  <th>Held</th><th className="l">Setup</th><th className="l">Exit reason</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((t) => <TradeRow key={t.id} t={t} open={open.has(t.id)} onToggle={() => toggle(t.id)} />)}
              </tbody>
              <tfoot>
                <tr className="pp-tot">
                  <td colSpan={8} className="l">Total · {n} trade{n === 1 ? "" : "s"} · {tot.wins} W / {n - tot.wins} L</td>
                  <td><PnL v={tot.net} d={0} /></td>
                  <td />
                  <td title="Average R multiple">{tot.nr ? <RMult v={tot.r / tot.nr} /> : "—"}</td>
                  <td colSpan={3} className="l muted">charges {inr(tot.charges, 0)}</td>
                </tr>
              </tfoot>
            </table>
          </div>
        )}
      </Panel>
    </>
  );
}

// =================================================================== SIGNALS
function ProposalRows({ list, idBase }: { list: Proposal[]; idBase: string }) {
  const [open, toggle] = useExpand();
  return (
    <table className="tbl pp-tbl">
      <thead>
        <tr>
          <th className="pp-xcol"><span className="sr-only">Details</span></th><th className="l">Symbol</th><th className="l">Side</th>
          <th className="l">Setup</th><th>Price</th><th>Stop</th><th title="Distance to the stop">Risk</th>
          <th title="Model's predicted return, before costs">Pred.</th><th title="Estimated round-trip cost">Cost</th>
          <th title="Predicted return minus cost">Edge</th>
        </tr>
      </thead>
      <tbody>
        {list.map((p, i) => {
          const key = `${p.symbol}-${p.side}-${i}`;
          const id = `${idBase}-${safeId(key)}`;
          const isOpen = open.has(key);
          const risk = ok(p.price) && p.price > 0 && ok(p.stop) ? Math.abs(p.price - p.stop) / p.price : null;
          return (
            <Fragment key={key}>
              <tr className={isOpen ? "sel" : undefined} onClick={() => toggle(key)}>
                <td className="pp-xcol"><XBtn open={isOpen} id={id} label={`${p.symbol}: what drove the proposal`} onToggle={() => toggle(key)} /></td>
                <td className="sym">{p.symbol}</td>
                <td className="l"><SideChip side={p.side} /></td>
                <td className="l"><SetupChip setup={p.setup} /></td>
                <td>{px(p.price)}</td>
                <td>{px(p.stop)}</td>
                <td className="muted">{risk !== null ? fracPct(risk, 2) : "—"}</td>
                <td className={cls(ok(p.pred) ? p.pred * 100 : null)}>{fracPct(p.pred, 2, true)}</td>
                <td className="muted">{fracPct(p.cost, 2)}</td>
                <td className={`b ${cls(ok(p.edge) ? p.edge * 100 : null)}`}>{fracPct(p.edge, 2, true)}</td>
              </tr>
              {isOpen && (
                <DetailRow id={id} cols={10}>
                  <div className="pp-why">
                    <div className="pp-why-main">{(p.drivers ?? []).length ? <Drivers drivers={p.drivers} /> : <p className="pp-note">No drivers recorded.</p>}</div>
                    <div className="pp-why-side"><Facts facts={p.facts} /></div>
                  </div>
                </DetailRow>
              )}
            </Fragment>
          );
        })}
      </tbody>
    </table>
  );
}

function DecisionCard({ d }: { d: Decision }) {
  const props = d.proposals ?? [];
  return (
    <article className="pp-dec" aria-label={`${d.time} decision`}>
      <header className="pp-dec-head">
        <span className="pp-dec-time">{d.time}</span>
        <StatusChip s={d.status} />
        <span className="muted">{ok(d.at) ? `decided ${timeIST(d.at)} IST` : d.status === "pending" ? "not decided yet" : ""}</span>
        {ok(d.considered) && <span className="muted">· {fx(d.considered, 0)} stocks considered</span>}
      </header>
      {d.note && <p className="pp-dec-note">{d.note}</p>}
      {props.length > 0 ? <ProposalRows list={props} idBase={`pp-prop-${d.time.replace(":", "")}`} />
        : d.status !== "pending" ? <p className="pp-dec-note muted">No proposal cleared the cost hurdle — no trade.</p> : null}
    </article>
  );
}

function CandidateRows({ list }: { list: SwingCandidate[] }) {
  const [open, toggle] = useExpand();
  return (
    <table className="tbl pp-tbl">
      <thead>
        <tr>
          <th className="pp-xcol"><span className="sr-only">Details</span></th><th>#</th><th className="l">Symbol</th><th className="l">Setup</th>
          <th title="Factor score (higher = stronger)">Score</th><th>Price</th><th>Stop</th><th title="Distance to the initial stop">Risk</th>
        </tr>
      </thead>
      <tbody>
        {list.map((c, i) => {
          const key = `${c.symbol}-${i}`;
          const id = `pp-cand-${safeId(key)}`;
          const isOpen = open.has(key);
          const risk = ok(c.price) && c.price > 0 && ok(c.stop) ? Math.abs(c.price - c.stop) / c.price : null;
          return (
            <Fragment key={key}>
              <tr className={isOpen ? "sel" : undefined} onClick={() => toggle(key)}>
                <td className="pp-xcol"><XBtn open={isOpen} id={id} label={`${c.symbol}: factor readings`} onToggle={() => toggle(key)} /></td>
                <td className="muted">{i + 1}</td>
                <td className="sym">{c.symbol}</td>
                <td className="l"><SetupChip setup={c.setup} /></td>
                <td className="amber b">{fx(c.score, 2)}</td>
                <td>{px(c.price)}</td>
                <td>{px(c.stop)}</td>
                <td className="muted">{risk !== null ? fracPct(risk, 1) : "—"}</td>
              </tr>
              {isOpen && <DetailRow id={id} cols={8}>{(c.drivers ?? []).length ? <Drivers drivers={c.drivers} /> : <p className="pp-note">No factor readings recorded.</p>}</DetailRow>}
            </Fragment>
          );
        })}
      </tbody>
    </table>
  );
}

function SignalsView() {
  const { sig, err, loading, reload } = useSignals();
  const decisions = useMemo(() => [...(sig?.intraday?.decisions ?? [])].sort((a, b) => a.time.localeCompare(b.time)), [sig]);
  const sw: PaperSignals["swing"] | null = sig?.swing ?? null;
  const cands = sw?.candidates ?? [];
  const refresh = <button type="button" className="btn" onClick={reload} disabled={loading}>{loading ? "LOADING…" : "REFRESH"}</button>;
  return (
    <>
      <Panel code="DEC" title="Intraday decisions" area="intra" id="pp-dec"
        sub={sig?.date ? `${sig.date} · decides at 09:30 and 10:00 IST` : "decides at 09:30 and 10:00 IST"} actions={refresh}>
        {!sig ? (err ? <ErrLine e={err} retry={reload} /> : <Empty>Loading today's decisions…</Empty>) : (
          <>
            {err ? <ErrLine e={err} retry={reload} /> : null}
            {!decisions.length ? <Empty>No decisions yet today — the first is at 09:30 IST.</Empty> : decisions.map((d) => <DecisionCard key={d.time} d={d} />)}
            <p className="pp-note">
              Proposals are what the model liked at that minute, ranked by edge after estimated costs. The trades the engine actually opened for
              your account (sized from your capital and leverage) are under POSITIONS and BOOK.
            </p>
          </>
        )}
      </Panel>
      <Panel code="CAND" title="Swing candidates" area="swing" id="pp-cand"
        sub={sw ? `for the next open · data as of ${asofText(sw.asof)}` : "for the next open"} actions={refresh}>
        {!sig ? (err ? <ErrLine e={err} retry={reload} /> : <Empty>Loading swing candidates…</Empty>) : (
          <>
            <RegimeLine r={sw?.regime} />
            {!cands.length ? <Empty>No swing candidates right now — none passes the factor and trend filters.</Empty> : <CandidateRows list={cands} />}
            <p className="pp-note">
              Ranked by factor score; open a row for the factor z-scores. Entries the engine has planned for your account are listed under
              POSITIONS → “Swing entries for the next open”.
            </p>
          </>
        )}
      </Panel>
    </>
  );
}

// =================================================================== EVIDENCE
const hasStats = (p: PStats | null | undefined): p is PStats => !!p && ok(p.return);

interface ECol { key: string; label: ReactNode; s: PStats; hold?: boolean; bench?: boolean }

function EvidenceStats({ rep }: { rep: Report }) {
  const holdFrom = ok(rep.holdout_from) ? ymdText(rep.holdout_from) : null;
  const cols: ECol[] = [];
  if (hasStats(rep.all)) cols.push({ key: "all", label: "Whole test", s: rep.all });
  if (hasStats(rep.before_holdout)) cols.push({ key: "pre", label: "Before holdout", s: rep.before_holdout });
  if (hasStats(rep.holdout)) cols.push({ key: "hold", label: <>Holdout<small>not used to design the strategy</small></>, s: rep.holdout, hold: true });
  if (hasStats(rep.benchmark)) cols.push({ key: "bench", label: <>{rep.benchmark.name || "Benchmark"}<small>same period</small></>, s: rep.benchmark, bench: true });
  if (hasStats(rep.holdout_benchmark)) cols.push({ key: "hbench", label: <>Benchmark<small>holdout only</small></>, s: rep.holdout_benchmark, bench: true, hold: true });
  const rows: { label: string; title?: string; cell: (s: PStats, c: ECol) => ReactNode }[] = [
    { label: "Period", cell: (s) => <span className="muted">{ymdText(s.start)} → {ymdText(s.end)}</span> },
    { label: "Trading days", cell: (s) => fx(s.days, 0) },
    { label: "Total return", cell: (s) => <Frac v={s.return} d={1} /> },
    { label: "CAGR", title: "Compound annual growth rate", cell: (s) => <Frac v={s.cagr} d={1} /> },
    { label: "Sharpe", title: "Annualised from daily returns (higher = steadier)", cell: (s) => fx(s.sharpe, 2) },
    { label: "Max drawdown", title: "Worst fall from a previous high", cell: (s) => <span className="down">{ddText(s.max_dd)}</span> },
    { label: "Trades", cell: (s, c) => (c.bench ? <span className="muted">—</span> : fx(s.trades, 0)) },
    { label: "Win rate", cell: (s) => (ok(s.win_rate) ? fracPct(s.win_rate, 1) : <span className="muted">—</span>) },
    { label: "Profit factor", title: "Gross wins ÷ gross losses", cell: (s) => (ok(s.profit_factor) ? fx(s.profit_factor, 2) : <span className="muted">—</span>) },
    { label: "Avg trade, net", title: "Average return per trade after charges", cell: (s) => <Frac v={s.avg_ret} d={2} /> },
    { label: "Charges paid", cell: (s) => (ok(s.charges) ? inr(s.charges, 0) : <span className="muted">—</span>) },
  ];
  return (
    <>
      <div className="pp-hscroll">
        <table className="tbl pp-etbl">
          <caption className="sr-only">Backtest statistics by period</caption>
          <thead>
            <tr>
              <th className="l" scope="col">Measure</th>
              {cols.map((c) => <th key={c.key} scope="col" className={c.hold ? "pp-hold" : undefined}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label}>
                <td className="l" title={r.title}>{r.label}</td>
                {cols.map((c) => <td key={c.key} className={c.hold ? "pp-hold" : undefined}>{r.cell(c.s, c)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="pp-note">
        {holdFrom && !hasStats(rep.holdout)
          ? `This report sets a holdout from ${holdFrom} (data not used to design the strategy) but has no results for it yet — re-run the backtest once there is data after that date.`
          : holdFrom
          ? <><b className="cyan">Holdout</b> = {holdFrom} onwards: data set aside and <b>not used to design or tune the strategy</b> — the fairest preview of how it may behave on new data. “Before holdout” is the period the rules were built on.</>
          : "This report has no holdout period, so every figure is on data the strategy was designed with — read it as an upper bound."}
      </p>
    </>
  );
}

function EvidenceCurve({ rep }: { rep: Report }) {
  const m = useMemo(() => {
    const eq = new Map((rep.equity ?? []).filter((p) => ok(p[0]) && ok(p[1])));
    const be = new Map((rep.benchmark_equity ?? []).filter((p) => ok(p[0]) && ok(p[1])));
    const dates = [...new Set([...eq.keys(), ...be.keys()])].sort((a, b) => a - b);
    const series: LineSeries[] = [{ name: rep.kind === "swing" ? "Swing strategy" : "Intraday strategy", values: dates.map((d) => eq.get(d) ?? null), tone: 1 }];
    if (be.size) series.push({ name: rep.benchmark?.name || "Benchmark", values: dates.map((d) => be.get(d) ?? null), tone: 2, dashed: true });
    return {
      x: dates.map(ymdTs), series, n: dates.length,
      marks: ok(rep.holdout_from) ? [{ x: ymdTs(rep.holdout_from), label: "holdout →" }] : [],
      refs: ok(rep.capital) ? [{ value: rep.capital, label: "starting capital" }] : [],
    };
  }, [rep]);
  return (
    <Panel code="BT" title="Backtest equity" area="curve" id="pp-btcurve"
      sub={`${rep.kind} · ${inr(rep.capital, 0)} start${ok(rep.leverage) ? ` · ${fx(rep.leverage, 1)}x` : ""}${m.series.length > 1 ? " · vs benchmark" : ""}`}>
      {m.n < 2 ? <Empty>No equity curve in this report.</Empty> : (
        <div className="pp-pad">
          <LineChart label={`${rep.kind} strategy backtest equity${m.series.length > 1 ? " versus its benchmark" : ""}`} x={m.x} series={m.series}
            marks={m.marks} refs={m.refs} height={280} format="inr"
            caption={<>Net of modelled charges and slippage.{m.marks.length ? " Right of the “holdout” line is data the strategy was not designed on." : ""}</>} />
        </div>
      )}
    </Panel>
  );
}

function MonthlyGrid({ monthly }: { monthly: { month: number; pnl: number }[] }) {
  const byYear = new Map<number, (number | null)[]>();
  for (const { month, pnl } of monthly) {
    if (!ok(month) || !ok(pnl)) continue;
    const y = Math.floor(month / 100), mo = month % 100;
    if (mo < 1 || mo > 12) continue;
    const arr = byYear.get(y) ?? Array<number | null>(12).fill(null);
    arr[mo - 1] = (arr[mo - 1] ?? 0) + pnl;
    byYear.set(y, arr);
  }
  const years = [...byYear.keys()].sort((a, b) => a - b);
  const max = Math.max(1e-9, ...[...byYear.values()].flat().filter(ok).map(Math.abs));
  const tint = (v: number): CSSProperties => {
    const a = 0.08 + 0.42 * Math.min(1, Math.abs(v) / max);
    return { background: v >= 0 ? `rgba(39, 209, 127, ${a.toFixed(3)})` : `rgba(255, 82, 99, ${a.toFixed(3)})` };
  };
  return (
    <div className="pp-hscroll">
      <table className="tbl pp-mtbl">
        <caption className="sr-only">Net P&L by month, rupees</caption>
        <thead><tr><th className="l" scope="col">Year</th>{MON.map((m) => <th key={m} scope="col">{m}</th>)}<th scope="col">Year</th></tr></thead>
        <tbody>
          {years.map((y) => {
            const arr = byYear.get(y)!;
            const sum = arr.reduce<number>((a, v) => a + (v ?? 0), 0);
            return (
              <tr key={y}>
                <td className="l pp-yr">{y}</td>
                {arr.map((v, i) => (
                  <td key={i} style={v === null ? undefined : tint(v)} title={v === null ? `${MON[i]} ${y}: no trades` : `${MON[i]} ${y}: ${sinr(v, 0)}`}>
                    {v === null ? "" : sk(v)}
                  </td>
                ))}
                <td className={`pp-ytot ${cls(sum, 0.5)}`} title={`${y}: ${sinr(sum, 0)}`}>{sk(sum)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function SetupTable({ rep }: { rep: Report }) {
  const rows = [...(rep.by_setup ?? [])].sort((a, b) => b.pnl - a.pnl);
  const sides = Object.entries(rep.by_side ?? {});
  return (
    <Panel code="SETUP" title="By setup" area="setup" id="pp-bysetup" sub="which kinds of trade made or lost the money">
      {!rows.length && !sides.length ? <Empty>No per-setup breakdown in this report.</Empty> : (
        <>
          {rows.length > 0 && (
            <div className="pp-hscroll">
              <table className="tbl pp-etbl">
                <thead><tr><th className="l" scope="col">Setup</th><th scope="col">Trades</th><th scope="col">Win rate</th><th scope="col">Net P&L</th><th scope="col">Per trade</th></tr></thead>
                <tbody>
                  {rows.map((r) => {
                    const win = !ok(r.win) ? null : r.win <= 1 ? r.win : r.n > 0 ? r.win / r.n : null;   // a share, or a count of winners
                    return (
                      <tr key={r.setup}>
                        <td className="l"><SetupChip setup={r.setup} /></td>
                        <td>{fx(r.n, 0)}</td>
                        <td>{win !== null ? fracPct(win, 1) : "—"}</td>
                        <td><PnL v={r.pnl} d={0} /></td>
                        <td><PnL v={r.n > 0 ? r.pnl / r.n : null} d={0} /></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          {sides.length > 0 && (
            <dl className="pp-kv pp-sides">
              {sides.map(([k, v]) => (
                <Fragment key={k}>
                  <dt>{k.toLowerCase() === "short" ? "▼ Short" : k.toLowerCase() === "long" ? "▲ Long" : k} · {fx(v?.n, 0)} trades</dt>
                  <dd><PnL v={v?.pnl} d={0} /></dd>
                </Fragment>
              ))}
            </dl>
          )}
        </>
      )}
    </Panel>
  );
}

const EV_KINDS: readonly Strategy[] = ["intraday", "swing"];

function EvidenceView() {
  const ev = useEvidence();
  const [kind, setKind] = usePref<Strategy>("td.paper.evKind", "intraday", EV_KINDS);
  const rep = ev.data ? ev.data[kind] : null;
  const cmd = `python -m td.paper.backtest${kind === "intraday" ? " --intraday" : " --swing"}`;
  return (
    <>
      <Panel code="EVID" title="Backtest evidence" area="head" id="pp-evhead" className="pp-wide"
        sub={rep ? `${rep.method} · generated ${when(rep.generated_at)} IST` : "how the two strategies did on past data"}
        actions={
          <>
            <Seg label="Strategy" value={kind} onChange={setKind} options={[{ v: "intraday", label: "INTRADAY" }, { v: "swing", label: "SWING" }]} />
            <button type="button" className="btn" onClick={ev.reload} disabled={ev.loading}>{ev.loading ? "LOADING…" : "REFRESH"}</button>
          </>
        }>
        <div className="pp-pad pp-evhead">
          <p className="rs-disclaimer" role="note">
            <b>Paper trading only — the desk never places real orders.</b> Backtests replay past data and are not a guarantee of future results.
            Brokerage, statutory charges and slippage are modelled, not assumed away, but real fills can still be worse.
          </p>
          {!!ev.err && <ErrLine e={ev.err} retry={ev.reload} />}
          {!ev.data ? (!ev.err && <Empty>Loading the backtest reports…</Empty>) : !rep ? (
            <div className="empty pp-empty">
              No {kind} backtest on the server yet — run <code>python -m td.paper.backtest</code> there (or <code>{cmd}</code> for just this one), then press REFRESH.
            </div>
          ) : (
            <p className="pp-evmeta">
              <span>Capital <b>{inr(rep.capital, 0)}</b></span>
              {ok(rep.leverage) && <span>Leverage <b>{fx(rep.leverage, 1)}x</b></span>}
              {hasStats(rep.all) && <span>Tested <b>{ymdText(rep.all.start)} → {ymdText(rep.all.end)}</b></span>}
              {ok(rep.holdout_from) && <span>Holdout from <b className="cyan">{ymdText(rep.holdout_from)}</b></span>}
              <span>Method <b>{rep.method}</b></span>
              {rep.source && (
                <span title={rep.source === "shipped" ? "No backtest has been run on this server yet; run python -m td.paper.backtest to replace it" : undefined}>
                  Report <b>{rep.source === "shipped" ? "shipped with the app" : "run on this server"}</b>
                </span>
              )}
            </p>
          )}
        </div>
      </Panel>
      {rep && (
        <>
          <EvidenceCurve rep={rep} />
          <Panel code="STAT" title="Headline numbers" area="stats" id="pp-evstats" sub="whole test · before holdout · holdout">
            <EvidenceStats rep={rep} />
          </Panel>
          <Panel code="MON" title="Monthly P&L" area="monthly" id="pp-monthly" sub="net ₹ per calendar month (K = thousand, L = lakh)">
            {(rep.monthly ?? []).length ? <MonthlyGrid monthly={rep.monthly!} /> : <Empty>No monthly breakdown in this report.</Empty>}
          </Panel>
          <SetupTable rep={rep} />
        </>
      )}
    </>
  );
}

// =================================================================== desk
function NoState({ err, retry }: { err: unknown; retry: () => void }) {
  return (
    <Panel code="PAPER" title="Paper account" className="pp-wide">
      {err ? (
        <>
          <ErrLine e={err} retry={retry} />
          {isStatus(err, 503) && <Empty>Nothing to show until the paper service is running. The SIGNALS and EVIDENCE views may still work.</Empty>}
        </>
      ) : <Empty>Loading your paper account…</Empty>}
    </Panel>
  );
}

export default function PaperDesk() {
  const { st, err, at, apply, retry } = usePaperState();
  const [tab, setTab] = usePref<Tab>("td.paper.tab", "overview", TAB_IDS);

  // 1…5 switch views (not while typing, not with modifiers: Alt+digit switches desks)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || !/^[1-5]$/.test(e.key)) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (useLive.getState().helpOpen) return;
      e.preventDefault();
      setTab(TABS[Number(e.key) - 1].v);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setTab]);

  const down = isStatus(err, 503) || (!!st && !st.service.running);
  const badges: Partial<Record<Tab, number>> = st ? {
    positions: st.intraday.positions.length + st.swing.positions.length,
    book: (st.intraday.stats.trades || 0) + (st.swing.stats.trades || 0),
  } : {};

  let body: ReactNode;
  if (tab === "signals") body = <SignalsView />;
  else if (tab === "evidence") body = <EvidenceView />;
  else if (tab === "book") body = <BookView st={st} />;
  else if (!st) body = <NoState err={err} retry={retry} />;
  else if (tab === "positions") body = <PositionsView st={st} />;
  else body = <OverviewView st={st} onState={apply} />;

  return (
    <div className="desk desk-paper">
      <div className="pp-head">
        <StatusStrip st={st} err={err} at={at} />
        <SubTabs value={tab} onChange={setTab} badges={badges} />
      </div>
      {down && (
        <div className="banner warn pp-alert" role="alert">
          <b>The paper service is not running</b> — start it with <code>run.py</code> / <code>docker compose</code>.{" "}
          {isStatus(err, 503)
            ? (st ? `Figures below are as last seen at ${timeIST(at, true)}; nothing trades until it is back.` : "")
            : "Settings and resets still save; no trades are opened, managed or closed until it runs."}
        </div>
      )}
      {!down && !!err && !!st && (
        <div className="banner warn pp-alert" role="status">
          Could not refresh the paper account ({paperErrText(err)}) — showing figures from {timeIST(at, true)}.
        </div>
      )}
      <div className={`pp-view v-${tab}`} role="tabpanel" id="pp-panel" aria-labelledby={`pp-tab-${tab}`}>
        {body}
      </div>
    </div>
  );
}
