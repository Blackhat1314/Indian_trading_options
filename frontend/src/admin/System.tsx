import { useEffect, useState } from "react";
import type { CSSProperties } from "react";
import { Empty, Icon, Kpi, Panel, ZBar } from "../components/ui";
import { api, ApiError } from "../lib/api";
import { ago, cls, dateIST, fx, pct, signed, timeIST } from "../lib/format";
import { useLive } from "../lib/store";

interface SystemInfo {
  status: Record<string, Record<string, unknown>>;
  lake: Record<string, { symbols: number; rows: number; first: number; last: number }>;
  lake_root: string;
  redis: { used_memory_human: string; streams: Record<string, number> };
  gateway: { clients: number; users: string[]; msgs_out: number };
  settings: { source: string; plan: string; universe: string; lock_time: string; top_n: number; limits: Record<string, number> };
}
interface Card {
  date: string; ic: number | null; n_long: number; long_green: number; long_avg: number; universe_avg: number;
  excess_long: number; long_hits_top5: number; long_hits_top20: number; picks: string[];
}
interface Scorecard {
  cards: Card[];
  summary: null | { days: number; mean_ic: number | null; ic_positive_days: number; avg_long: number; avg_universe: number; avg_excess: number; top5_hits_per_day: number; top20_hit_rate: number; pct_green: number };
}

interface BtSummary {
  days: number; universe_size: number; mean_ic: number | null; ic_t: number | null; ic_positive_pct: number | null;
  long_avg_net: number | null; long_green_pct: number | null; short_avg_net: number | null; universe_avg: number;
  top5_hits_per_day: number; random_top5_hits_per_day: number | null; top20_hit_rate: number | null;
  random_top20_rate: number | null; gainer_hits_per_day: number; abstain_long_pct: number;
  factor_ic: Record<string, number | null>;
}
interface Backtest {
  available: boolean; generated?: number; lake?: string; lock?: string; cost_bps?: number;
  factors_available?: string[]; summary?: BtSummary; days?: { date: string; ic: number | null; long_ret: number | null }[];
}

const FACTOR_NAME: Record<string, string> = {
  rvol: "Relative volume", resid: "Relative strength", vwap: "VWAP position", orb: "Opening range",
  oi: "Futures OI", news: "News", sector: "vs sector", obi: "Book imbalance",
};

function BacktestPanel() {
  const [bt, setBt] = useState<Backtest | null>(null);
  useEffect(() => {
    api<Backtest>("/api/research/intraday").then(setBt).catch(() => setBt({ available: false }));
  }, []);
  const s = bt?.summary;
  const synthetic = bt?.lake?.includes("/sim") || bt?.lake?.includes("\\sim");
  const ics = (bt?.days ?? []).map((d) => d.ic ?? 0);
  const maxIc = Math.max(0.05, ...ics.map(Math.abs));
  return (
    <Panel code="BT" title="Backtest — walk-forward" area="bt"
      sub={s ? `${s.days} days · lock ${bt?.lock} · ${bt?.cost_bps} bps costs · run ${bt?.generated ? dateIST(bt.generated) : ""}` : "research/evaluate_intraday.py"}>
      {!bt ? <Empty>Loading…</Empty> : !bt.available || !s ? (
        <Empty>No backtest yet — run <code>python research/evaluate_intraday.py</code> (replays the lake through the same ranker)</Empty>
      ) : (
        <>
          {synthetic && (
            <div className="banner">Synthetic history: minute bars are a Brownian bridge to each day's close, so early moves mean-revert by construction. This run tests the plumbing, not the edge — rerun on the Dhan lake after backfill.</div>
          )}
          <div className="kpis" style={{ "--kpi-min": "122px" } as CSSProperties}>
            <Kpi label="Mean rank IC" value={fx(s.mean_ic, 3)} tone={cls(s.mean_ic ?? 0, 0.005)} sub={`t = ${fx(s.ic_t, 1)} · ${fx(s.ic_positive_pct, 0)}% days > 0`} />
            <Kpi label="Long picks, net" value={pct(s.long_avg_net)} tone={cls(s.long_avg_net)} sub={`green ${fx(s.long_green_pct, 0)}% · universe ${signed(s.universe_avg)}%`} />
            <Kpi label="Short picks, net" value={pct(s.short_avg_net)} tone={cls(s.short_avg_net)} sub="lock → close, mirrored" />
            <Kpi label="Top-5 hits / day" value={fx(s.top5_hits_per_day, 2)} sub={`random ≈ ${fx(s.random_top5_hits_per_day, 2)}`} />
            <Kpi label="Picks in top-20" value={`${fx(s.top20_hit_rate, 0)}%`} sub={`random ≈ ${fx(s.random_top20_rate, 0)}%`} />
            <Kpi label="Day's top gainers" value={fx(s.gainer_hits_per_day, 2)} sub="picks per day in the top-5 by % change" />
          </div>
          <div style={{ padding: "8px 10px", display: "grid", gap: 4 }}>
            <div className="tiny muted mono">FACTOR IC (each factor alone vs lock → close)</div>
            {Object.entries(s.factor_ic).map(([k, v]) => (
              <div key={k} style={{ display: "grid", gridTemplateColumns: "130px 1fr 60px", gap: 8, alignItems: "center", fontSize: 12 }}>
                <span className="muted">{FACTOR_NAME[k] ?? k}</span>
                {v === null ? <span className="tiny muted">not in history — live only</span> : <ZBar v={v} max={0.15} title={`${k} IC`} />}
                <span className={`mono right ${cls(v ?? 0, 0.005)}`}>{v === null ? "—" : signed(v, 3)}</span>
              </div>
            ))}
          </div>
          {ics.length > 1 && (
            <div style={{ padding: "0 10px 10px" }}>
              <div className="tiny muted mono" style={{ marginBottom: 4 }}>DAILY IC (last {ics.length} days)</div>
              <div role="img" aria-label="Daily information coefficient" style={{ display: "flex", alignItems: "center", gap: 2, height: 56, borderTop: "1px dashed var(--border)", borderBottom: "1px dashed var(--border)" }}>
                {bt.days!.map((d) => {
                  const v = d.ic ?? 0;
                  const h = (Math.abs(v) / maxIc) * 26;
                  return (
                    <div key={d.date} title={`${d.date}: IC ${signed(v, 3)}`} style={{ flex: 1, height: 52, display: "flex", flexDirection: "column", justifyContent: "center" }}>
                      <div style={{ height: 26, display: "flex", alignItems: "flex-end" }}>{v > 0 && <i style={{ display: "block", width: "100%", height: h, background: "var(--up)", opacity: 0.75 }} />}</div>
                      <div style={{ height: 26 }}>{v < 0 && <i style={{ display: "block", width: "100%", height: h, background: "var(--down)", opacity: 0.75 }} />}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}
    </Panel>
  );
}

/* DHAN TOKEN: Dhan access tokens last 24 h. The ingestor's TokenKeeper renews the token by itself and every
   process picks the new one up; this panel shows that chain and lets an admin renew or paste one by hand.
   The token never comes back to the browser — only its expiry, source and the last checks. */

interface TokenStatus {
  configured?: boolean; client_id?: string | null; state?: string | null; valid?: boolean | null;
  exp?: unknown; iat?: unknown; expires_in_s?: number | null; source?: string | null;
  auto_renew?: boolean | null; renew_at?: string | null; next_action?: string | null; next_action_at?: unknown;
  message?: string | null; totp_configured?: boolean | null; totp_blocked?: boolean | null;
  last_check?: unknown; last_renew?: unknown; last_error?: unknown; owner_alive?: boolean | null; updated?: unknown;
  profile?: { dataPlan?: string; dataValidity?: string; tokenValidity?: string } | null;
  seen: number;   // browser clock on arrival; with `updated` (server clock) it corrects a PC clock that is off
}

const TOKEN_SOURCE: Record<string, [string, string]> = {
  env: ["ENV", "from .env"], renewed: ["RENEWED", "auto-renewed"], totp: ["TOTP", "generated with TOTP"],
  admin: ["PASTED", "pasted by admin"],
};
const NEXT_ACTION: Record<string, string> = {
  renew: "renew", generate: "new token via TOTP", needs_token: "needs a new token", none: "nothing scheduled",
};
const IST_S = 19800;
const SOON_S = 3 * 3600;   // "expiring soon"
const MARGIN_S = 60;       // the backend already treats a token this close to its expiry as expired
const JWT = /^[\w-]+\.[\w-]+\.[\w-]+$/;

/** Epoch seconds from whatever the backend sent: seconds, milliseconds or an ISO string. */
function epoch(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v) && v > 0) return v > 1e12 ? v / 1000 : v;
  if (typeof v === "string" && v) {
    const t = Date.parse(v);
    return Number.isNaN(t) ? null : t / 1000;
  }
  return null;
}

function text(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "string") return v;
  if (typeof v === "object") {
    const o = v as Record<string, unknown>;
    const m = o.message ?? o.msg ?? o.error ?? o.detail;
    if (typeof m === "string") return m;
  }
  return JSON.stringify(v);
}

function stamp(t: number, now: number): string {
  const d = Math.floor((t + IST_S) / 86400) - Math.floor((now + IST_S) / 86400);
  const day = d === 0 ? "" : d === 1 ? "tomorrow " : d === -1 ? "yesterday " : `${dateIST(t)} `;
  return `${day}${timeIST(t)} IST`;
}

function hms(s: number): string {
  const a = Math.max(0, Math.floor(s));
  const p = (n: number) => String(n).padStart(2, "0");
  const h = Math.floor(a / 3600);
  const m = Math.floor((a % 3600) / 60);
  return h ? `${h}h ${p(m)}m ${p(a % 60)}s` : `${m}m ${p(a % 60)}s`;
}

/** The next "HH:MM" IST after now — the renewal time when an older backend does not say. */
function nextDaily(hhmm: string, now: number): number | null {
  const m = /^(\d{1,2}):(\d{2})$/.exec(hhmm.trim());
  if (!m) return null;
  const t = Math.floor((now + IST_S) / 86400) * 86400 - IST_S + Number(m[1]) * 3600 + Number(m[2]) * 60;
  return t > now ? t : t + 86400;
}

/** FastAPI sends {detail}; a missing endpoint means a gateway older than this page. */
function errMsg(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 404) return "This gateway has no Dhan token endpoints yet — restart it after updating TradeDesk.";
    return e.message && e.message !== "[object Object]" ? e.message : `request failed (HTTP ${e.status})`;
  }
  return e instanceof Error ? e.message : String(e);
}

/** The status itself or a {status, configured} wrapper; {ok: false} is a failure. */
function asStatus(r: unknown): TokenStatus {
  const o = (r && typeof r === "object" ? r : {}) as Record<string, unknown>;
  if (o.ok === false) throw new Error(text(o.error ?? o.detail ?? o.message) || "the gateway refused");
  const inner = o.status && typeof o.status === "object" ? o.status : o.status === null ? {} : null;
  const base = (inner ?? o) as Record<string, unknown>;
  return { ...base, configured: (o.configured ?? base.configured) as boolean | undefined, seen: Date.now() / 1000 };
}
const isStatus = (s: TokenStatus) => s.configured !== undefined || s.exp !== undefined || s.source !== undefined;

/** `now` is server time. The backend's `state` decides; the expiry decides between polls and for older backends. */
function tokenState(st: TokenStatus, now: number, sim: boolean) {
  const exp = epoch(st.exp) ?? (typeof st.expires_in_s === "number" ? (epoch(st.updated) ?? st.seen) + st.expires_in_s : null);
  const left = exp === null ? null : exp - now;
  const state = st.state ?? "";
  const r = (label: string, tone: string, bad = false) => ({ label, tone, bad, exp, left });
  if (state === "missing" || !(st.configured ?? exp !== null)) return sim ? r("NOT SET", "") : r("MISSING", "down", true);
  if (state === "rejected" || (!state && st.valid === false && (left === null || left > MARGIN_S))) return r("REJECTED", "down", true);
  if (state === "expired" || (left !== null && left <= MARGIN_S)) return r("EXPIRED", "down", true);
  if (left !== null ? left < SOON_S : state === "expiring") return r("EXPIRING SOON", "amber");
  if (!state && left === null && !st.valid) return r("UNKNOWN", "");
  return r("VALID", "up");
}

const WHY: Record<string, string> = {
  MISSING: "No Dhan token is set, so there is no live data.",
  EXPIRED: "The Dhan token has expired, so live data has stopped.",
  REJECTED: "Dhan rejected the current token (revoked, or replaced on Dhan Web).",
};

function TokenPanel({ dataSource }: { dataSource?: string }) {
  const [st, setSt] = useState<TokenStatus | null>(null);
  const [loadErr, setLoadErr] = useState("");
  const [tick, setTick] = useState(0);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [pasting, setPasting] = useState(false);
  const [tok, setTok] = useState("");

  useEffect(() => {
    let alive = true;
    const load = () => api<unknown>("/api/admin/dhan-token")
      .then((r) => { if (alive) { setSt(asStatus(r)); setLoadErr(""); } })
      .catch((e: unknown) => { if (alive) setLoadErr(errMsg(e)); });
    load();
    const id = window.setInterval(load, 20000);
    return () => { alive = false; window.clearInterval(id); };
  }, [tick]);
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(id);
  }, []);

  const serverNow = (s: TokenStatus, t = Date.now() / 1000) => t - (s.seen - (epoch(s.updated) ?? s.seen));
  const took = (r: unknown) => {
    const s = asStatus(r);
    if (isStatus(s)) setSt(s);
    return s;
  };
  const closePaste = () => { setPasting(false); setTok(""); };
  const renew = () => {
    const before = st ? tokenState(st, serverNow(st), false).exp : null;
    setBusy(true);
    setMsg(null);
    api<unknown>("/api/admin/dhan-token/renew", { method: "POST" })
      .then((r) => {
        const s = took(r);
        const exp = isStatus(s) ? tokenState(s, serverNow(s), false).exp : null;
        if (exp && (!before || exp > before + 60)) {
          const how = s.source === "totp" ? "A new token was generated with TOTP" : "Renewed";
          setMsg({ ok: true, text: `${how} — valid until ${dateIST(exp)} ${timeIST(exp)} IST.` });
        } else if (text(s.last_error)) setMsg({ ok: false, text: `Not renewed: ${text(s.last_error)}` });
        else setMsg({ ok: true, text: "Renewal requested — the status updates in a moment." });
      })
      .catch((e: unknown) => setMsg({ ok: false, text: errMsg(e) }))
      .finally(() => { setBusy(false); setTick((t) => t + 1); });
  };
  const clean = tok.replace(/\s+/g, "").replace(/^["']+|["']+$/g, "");
  const jwtOk = JWT.test(clean);
  const save = () => {
    if (!jwtOk) return;
    setBusy(true);
    setMsg(null);
    api<unknown>("/api/admin/dhan-token", { method: "POST", body: JSON.stringify({ token: clean }) })
      .then((r) => {
        took(r);
        closePaste();
        setMsg({ ok: true, text: "Saved — Dhan accepted it, and every service switches to it within a minute." });
      })
      .catch((e: unknown) => setMsg({ ok: false, text: errMsg(e) }))
      .finally(() => { setBusy(false); setTick((t) => t + 1); });
  };

  const sim = dataSource === "sim";
  const t = st ? serverNow(st, now) : now;
  const s = st ? tokenState(st, t, sim) : null;
  const prof = st?.profile ?? null;
  const renewAt = st?.renew_at || "08:30";
  const autoOn = st?.auto_renew !== false;
  const action = st?.next_action ?? "";
  const actionAt = action === "needs_token" || action === "none" ? null : epoch(st?.next_action_at);
  const nextRenew = action === "renew" ? actionAt : st && !action && autoOn ? nextDaily(renewAt, t) : null;
  const when = (x: number) => (x <= t + 5 ? "now" : stamp(x, t));
  const iat = epoch(st?.iat);
  const lastCheck = epoch(st?.last_check);
  const lastRenew = epoch(st?.last_renew);
  const lastErr = text(st?.last_error);
  const src = st?.source ? TOKEN_SOURCE[st.source] ?? [st.source.toUpperCase(), st.source] : null;
  const fix = st?.totp_configured
    ? "TOTP is set up, so the app makes a new token by itself within a minute."
    : "Dhan cannot renew a token that is no longer valid — paste a new one (web.dhan.co → DhanHQ Trading APIs), or set DHAN_PIN and DHAN_TOTP_SECRET in .env on the server so the app makes one itself.";
  const ownerDown = dataSource === "dhan" && st?.owner_alive === false;
  return (
    <Panel code="TOK" title="Dhan token" area="tok" className="tok-panel"
      sub={<>{s && <span className={`chip ${s.tone}`}>{s.label}</span>}{nextRenew && !s?.bad ? `next renewal ${when(nextRenew)} · ` : ""}status every 20 s</>}>
      {loadErr && <div className="banner warn" role="alert">Cannot read the token status: {loadErr}</div>}
      {!st ? <Empty>{loadErr ? "No token status from the gateway." : "Loading…"}</Empty> : s && (
        <>
          {s.bad && <div className="banner warn" role="alert">{st.message || `${WHY[s.label]} ${fix}`}</div>}
          {!s.bad && !autoOn && s.label === "EXPIRING SOON" && (
            <div className="banner warn" role="alert">Auto-renew is off — renew now, or the token runs out in {hms(s.left ?? 0)}.</div>
          )}
          {ownerDown && (
            <div className="banner warn">The ingestor is not keeping the token (not running, or an older version), so nothing renews it by itself right now — Renew now still works from here.</div>
          )}
          {lastErr && <div className="banner warn">Last error: {lastErr}</div>}
          <div className="kpis" style={{ "--kpi-min": "150px" } as CSSProperties}>
            <Kpi label="Expires in" tone={s.tone}
              value={s.label === "EXPIRED" ? "EXPIRED" : s.left === null ? "—" : hms(s.left)}
              sub={s.exp === null ? "expiry unknown" : s.left !== null && s.left > 0 ? `at ${stamp(s.exp, t)}` : `${ago(s.exp, t)} ago, at ${stamp(s.exp, t)}`}
              title={prof?.tokenValidity ? `Dhan's profile says valid till ${prof.tokenValidity}` : undefined} />
            <Kpi label="Source" value={src ? src[0] : "—"} sub={src ? `${src[1]}${iat ? ` · ${stamp(iat, t)}` : ""}` : "where the token came from"} />
            <Kpi label="Auto-renew" value={autoOn ? "ON" : "OFF"} tone={autoOn ? "up" : "amber"}
              sub={!autoOn ? "DHAN_TOKEN_AUTO_RENEW=false"
                : s.bad ? (action === "generate" ? `TOTP login ${actionAt ? when(actionAt) : "soon"}` : "waits for a valid token")
                : nextRenew ? `next ${when(nextRenew)}` : `around ${renewAt} IST`} />
            <Kpi label="TOTP recovery" value={st.totp_blocked ? "REFUSED" : st.totp_configured ? "YES" : "NO"}
              tone={st.totp_blocked ? "down" : st.totp_configured ? "up" : ""}
              sub={st.totp_blocked ? "Dhan refused DHAN_PIN / DHAN_TOTP_SECRET — fix, then Renew now"
                : st.totp_configured ? "makes a new token after downtime" : "DHAN_PIN + DHAN_TOTP_SECRET not set"} />
            <Kpi label="Data plan" value={prof?.dataPlan || "—"} tone={!prof?.dataPlan ? "" : /active/i.test(prof.dataPlan) ? "up" : "amber"}
              sub={prof?.dataValidity ? `valid till ${prof.dataValidity}` : "from Dhan's profile"} />
          </div>
          <div className="facts tok-facts">
            <div><span>Client id</span><b>{st.client_id || "—"}</b></div>
            <div><span>Expires at</span><b>{s.exp ? `${dateIST(s.exp)} ${timeIST(s.exp)} IST` : "—"}</b></div>
            <div><span>Dhan says valid till</span><b>{prof?.tokenValidity || "—"}</b></div>
            <div><span>Last checked with Dhan</span><b>{lastCheck ? `${stamp(lastCheck, t)} · ${ago(lastCheck, t)} ago` : "—"}</b></div>
            <div><span>Last renewal</span><b>{lastRenew ? `${stamp(lastRenew, t)} · ${ago(lastRenew, t)} ago` : "—"}</b></div>
            <div><span>Next step</span>
              <b title={st.message || undefined}>{(NEXT_ACTION[action] ?? action.replace(/_/g, " ")) || (s.bad ? "needs a new token" : "—")}{actionAt ? ` · ${when(actionAt)}` : ""}</b></div>
          </div>
          <div className="tok-actions">
            <button type="button" className="btn" disabled={busy} onClick={renew}
              title={!s.bad ? "Ask Dhan for a fresh 24 h token now."
                : st.totp_configured ? "The token has lapsed: log in again with DHAN_PIN + DHAN_TOTP_SECRET."
                : "Dhan only renews a token that is still valid — paste a new one instead."}>
              <Icon name="refresh" size={12} /> Renew now
            </button>
            <button type="button" className="btn ghost" aria-expanded={pasting} aria-controls="tok-form"
              onClick={() => (pasting ? closePaste() : setPasting(true))}>Paste new token</button>
            <span className={`small ${msg ? (msg.ok ? "up" : "down") : ""}`} aria-live="polite">{msg?.text}</span>
          </div>
          {pasting && (
            <form id="tok-form" className="field tok-form" autoComplete="off" onSubmit={(e) => { e.preventDefault(); save(); }}>
              <label htmlFor="tok-input">New access token</label>
              <div className="tok-input-row">
                <input id="tok-input" type="password" value={tok} onChange={(e) => setTok(e.target.value)} autoFocus
                  autoComplete="off" spellCheck={false} data-1p-ignore data-lpignore="true" placeholder="eyJ…"
                  aria-invalid={!!clean && !jwtOk} aria-describedby="tok-hint" />
                <button type="submit" className="btn primary" disabled={busy || !jwtOk}>Save</button>
                <button type="button" className="btn ghost" disabled={busy} onClick={closePaste}>Cancel</button>
              </div>
              <span id="tok-hint" className={clean && !jwtOk ? "err" : "hint"}>
                {clean && !jwtOk
                  ? "That is not a Dhan access token — it has three parts separated by dots (eyJ….….…)."
                  : "From web.dhan.co → DhanHQ Trading APIs → access token (valid 24 h). Dhan checks it before it is used; it is never shown again."}
              </span>
            </form>
          )}
          <div className="banner">
            Dhan tokens last 24 hours. {autoOn
              ? `The app renews its token by itself about once a day, never during market hours (DHAN_TOKEN_RENEW_AT = ${renewAt} IST anchors the cycle; the next renewal is shown above), so a running deployment never needs a manual paste. `
              : "Auto-renew is off (DHAN_TOKEN_AUTO_RENEW=false), so the token has to be replaced by hand every day. "}
            Dhan only renews a token that is still valid: for fully unattended recovery after downtime, set DHAN_PIN and
            DHAN_TOTP_SECRET in .env on the server (they give full account access — keep them there only).
            {sim && " This TradeDesk runs on simulated data (DATA_SOURCE=sim); the token matters only for live Dhan data."}
          </div>
        </>
      )}
    </Panel>
  );
}

export default function SystemDesk() {
  const [sys, setSys] = useState<SystemInfo | null>(null);
  const [sc, setSc] = useState<Scorecard | null>(null);
  const top = useLive((s) => s.top);
  const source = useLive((s) => s.source);
  useEffect(() => {
    const load = () => {
      api<SystemInfo>("/api/system").then(setSys).catch(() => {});
      api<Scorecard>("/api/scorecard").then(setSc).catch(() => {});
    };
    load();
    const id = window.setInterval(load, 4000);
    return () => window.clearInterval(id);
  }, []);
  const ing = sys?.status.ingestor as Record<string, any> | undefined; // eslint-disable-line @typescript-eslint/no-explicit-any
  const ana = sys?.status.analytics as Record<string, any> | undefined; // eslint-disable-line @typescript-eslint/no-explicit-any
  const geo = sys?.status.geo as Record<string, any> | undefined; // eslint-disable-line @typescript-eslint/no-explicit-any
  const gws = Object.entries(sys?.status ?? {}).filter(([k]) => k.startsWith("gateway"));
  const fresh = (u?: number) => (u ? Date.now() / 1000 - u < 10 : false);
  const s = sc?.summary;
  const u = top?.universe_since_lock;
  return (
    <div className="desk desk-system">
      <TokenPanel dataSource={sys?.settings.source} />
      <Panel code="SYS" title="Distributed system" area="sys" sub="one Dhan connection → Redis → any number of users">
        {!sys ? <Empty>Loading…</Empty> : (
          <>
            <div className="kpis" style={{ "--kpi-min": "160px" } as CSSProperties}>
              <Kpi label="Ingestor" value={fresh(ing?.updated) ? "UP" : "DOWN"} tone={fresh(ing?.updated) ? "up" : "down"} sub={`${ing?.mode ?? "?"} · ${ing?.instruments ?? 0} instruments`} />
              <Kpi label="Feed" value={`${fx(ing?.stats?.pps ?? 0, 0)}/s`} sub={`reconnects ${ing?.feed?.reconnects ?? 0}`} />
              <Kpi label="Analytics" value={fresh(ana?.updated) ? "UP" : "DOWN"} tone={fresh(ana?.updated) ? "up" : "down"} sub={`${ana?.ranked ?? 0} ranked · ${ana?.news ?? 0} news`} />
              <Kpi label="Gateways" value={gws.length} sub={`${sys.gateway.clients} clients here`} />
              <Kpi label="Users online" value={sys.gateway.users.length} sub={sys.gateway.users.join(", ") || "—"} />
              <Kpi label="Redis" value={sys.redis.used_memory_human} sub={`ticks stream ${sys.redis.streams["td:ticks"] ?? 0}`} />
              <Kpi label="Data API budget" value={ing?.data_budget_used !== undefined ? `${ing.data_budget_used}/${ing.data_budget_limit}` : "n/a"} sub="calls today (7,000/day)" />
              <Kpi label="Recorder" value={fx(ing?.recorder_rows ?? 0, 0)} sub="ticks written (Parquet)" />
              {ing?.stats?.history && (
                <Kpi label="Auto history" tone={ing.stats.history.stage === "ready" ? "up" : "amber"}
                  value={ing.stats.history.stage === "ready" ? "READY" : `${ing.stats.history.done}/${ing.stats.history.total}`}
                  sub={`${ing.stats.history.stage === "ready" ? "prev close · OI · bars · next 08:40" : ing.stats.history.stage}${ing.stats.history.errors ? ` · ${ing.stats.history.errors} failed` : ""}`}
                  title={ing.stats.history.last_error || "prev close, prev OI, daily + 1-minute bars from Dhan"} />
              )}
              <Kpi label="GEO worker" value={fresh(geo?.updated) ? "UP" : "DOWN"} tone={fresh(geo?.updated) ? "up" : "down"}
                sub={geo ? `${geo.mode} · ${geo.ships} ships · ${geo.events} events · risk ${geo.risk}` : "python -m td.services.geo"} />
            </div>
            <dl className="kv-grid">
              <dt>Data source</dt><dd>{sys.settings.source.toUpperCase()} · plan {sys.settings.plan}</dd>
              <dt>Universe</dt><dd>{sys.settings.universe}</dd>
              <dt>Lock time / picks</dt><dd>{sys.settings.lock_time} · top {sys.settings.top_n}</dd>
              <dt>Dhan limits</dt><dd>{sys.settings.limits.data_per_sec}/s data · {sys.settings.limits.data_per_day}/day · quotes {sys.settings.limits.quote_per_sec}/s</dd>
              <dt>Feed error</dt><dd className={ing?.feed?.last_error ? "down" : "muted"}>{ing?.feed?.last_error || "none"}</dd>
              <dt>Option chains</dt><dd>{(ana?.chains ?? []).join(", ") || "—"}</dd>
              <dt>Lake</dt><dd className="ellipsis" title={sys.lake_root}>{sys.lake_root}</dd>
            </dl>
          </>
        )}
      </Panel>
      <Panel code="SCORE" title="Scorecard — the system's own receipts" area="score"
        sub="locked picks vs what actually happened from lock to close">
        <div className="kpis" style={{ "--kpi-min": "122px" } as CSSProperties}>
          <Kpi label="Days recorded" value={s?.days ?? 0} sub="need 60+ before trusting" />
          <Kpi label="Mean rank IC" value={s?.mean_ic !== undefined && s?.mean_ic !== null ? fx(s.mean_ic, 3) : "—"} tone={cls(s?.mean_ic ?? 0, 0.005)} sub={s ? `${s.ic_positive_days}/${s.days} days positive` : "0.03–0.08 = real edge"} />
          <Kpi label="Top-5 hits / day" value={s ? fx(s.top5_hits_per_day, 2) : "—"} sub="random ≈ 0.125" />
          <Kpi label="Picks in top-20" value={s ? `${fx(s.top20_hit_rate, 0)}%` : "—"} sub="random ≈ 10%" />
          <Kpi label="Picks green" value={s ? `${fx(s.pct_green, 0)}%` : "—"} sub="random ≈ 50%" />
          <Kpi label="Avg excess vs universe" value={s ? `${signed(s.avg_excess)}%` : "—"} tone={cls(s?.avg_excess)} sub="gross, before costs" />
        </div>
        {top?.locked && u && (
          <div className="lockbar">
            <span className="amber b">TODAY (live)</span>
            <span className="kv"><span>Long avg</span><span className={cls(u.long_avg)}>{pct(u.long_avg)}</span></span>
            <span className="kv"><span>Universe</span><span className={cls(u.universe_avg)}>{pct(u.universe_avg)}</span></span>
            <span className="kv"><span>Top-5 hits</span><span>{u.long_hits_top5}</span></span>
            <span className="kv"><span>Top-20 hits</span><span>{u.long_hits_top20}</span></span>
          </div>
        )}
        {!sc?.cards.length ? <Empty>The first scorecard is written automatically at 15:30 on a day the picks were locked.</Empty> : (
          <table className="tbl">
            <thead><tr><th className="l">Date</th><th>IC</th><th>Long avg</th><th>Universe</th><th>Excess</th><th>Top-5</th><th>Top-20</th><th>Green</th><th className="l">Picks</th></tr></thead>
            <tbody>
              {sc.cards.map((c) => (
                <tr key={c.date} style={{ cursor: "default" }}>
                  <td className="l">{c.date}</td>
                  <td className={cls(c.ic ?? 0, 0.005)}>{c.ic !== null ? fx(c.ic, 3) : "—"}</td>
                  <td className={cls(c.long_avg)}>{signed(c.long_avg)}%</td>
                  <td className={cls(c.universe_avg)}>{signed(c.universe_avg)}%</td>
                  <td className={cls(c.excess_long)}>{signed(c.excess_long)}%</td>
                  <td>{c.long_hits_top5}</td><td>{c.long_hits_top20}</td><td>{c.long_green}/{c.n_long}</td>
                  <td className="l muted">{c.picks.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
      <BacktestPanel />
      <Panel code="LAKE" title="Data lake (backfill)" area="lake" sub="Parquet · shared with backfill & research scripts">
        {!sys ? <Empty>Loading…</Empty> : Object.keys(sys.lake).length === 0 ? (
          <Empty>
            {source === "dhan"
              ? "Empty — auto history is downloading ~1 year of daily and 30 days of 1-minute bars (about 3 minutes)."
              : <>Empty — pull it from the HISTORY tab, or run <code>python backfill/backfill.py run --intervals D,1 --years 5</code></>}
          </Empty>
        ) : (
          <table className="tbl">
            <thead><tr><th className="l">Interval</th><th>Symbols</th><th>Rows</th><th>From</th><th>To</th></tr></thead>
            <tbody>
              {Object.entries(sys.lake).map(([iv, x]) => (
                <tr key={iv} style={{ cursor: "default" }}>
                  <td className="l b">{iv}</td><td>{x.symbols}</td><td>{fx(x.rows, 0)}</td>
                  <td>{dateIST(x.first)}</td><td>{dateIST(x.last)} {timeIST(x.last)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <div className="banner">
          Live mode tops this up by itself at start-up and every morning at 08:40 (AUTO_HISTORY). The full 5-year
          archive the research scripts use is pulled from the HISTORY tab.
        </div>
      </Panel>
    </div>
  );
}
