import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { Empty, Icon, Kpi, Panel, Seg } from "../components/ui";
import { api, ApiError } from "../lib/api";
import { dateIST, fx, timeIST } from "../lib/format";
import { getPref, setPref } from "../lib/storage";

/* ADMIN desk (F9): pull Dhan history into the lake through the backfill worker (td.services.backfiller).
   The worker runs in its own process and paces itself around the live desks; this page only sends
   commands and shows its status. */

type MarketHours = "slow" | "pause" | "full";
interface Job {
  intervals: string[]; years: number; rate: number; reserve: number; market_hours: MarketHours;
  symbols: string[]; active: boolean; created: number; by: string;
}
interface Progress {
  total: number; done: number; ok: number; empty: number; failed: number; retried: number; rows: number;
  started: number; current: string; by_interval: Record<string, [number, number]>; last_error: string;
  errors: [number, string, string][]; rate: number; eta_s: number | null;
}
interface IntervalCov { symbols: number; rows: number; first: number | null; last: number | null; bytes: number }
interface Coverage {
  intervals: Record<string, IntervalCov>;
  symbols: Record<string, Record<string, [number, number, number]>>;
  updated: number;
}
interface WorkerStatus {
  worker: { pid: number; host: string; version: string; started: number; updated: number };
  source: string; state: string; reason: string; job: Job | null; progress: Progress | null;
  result: { reason: string; at: number; ok: number; empty: number; failed: number; rows: number; requests: number } | null;
  resume_at: number | null; budget: { used: number; limit: number; reserve: number | null };
  coverage: Coverage | null; log: [number, string, string][];
  limits?: { data_per_sec: number; min_reserve: number };
}
interface AdminResp { status: WorkerStatus | null; source: string; user: string }
interface Plan {
  requests: number; by_interval: Record<string, number>; calls_left_today: number | null; per_day: number;
  days_of_quota: number; run_minutes: number; symbols: number; unknown: string[]; error?: string;
}

const IV_NAME: Record<string, string> = { D: "Daily", "1": "1-minute", "1m": "1-minute" };
const STATE: Record<string, { label: string; tone: string }> = {
  running: { label: "RUNNING", tone: "up" }, paused: { label: "PAUSED", tone: "amber" },
  waiting_quota: { label: "QUOTA WAIT", tone: "amber" }, done: { label: "DONE", tone: "up" },
  idle: { label: "IDLE", tone: "" }, stopped: { label: "STOPPED", tone: "" }, error: { label: "ERROR", tone: "down" },
  unavailable: { label: "UNAVAILABLE", tone: "down" }, blocked: { label: "BLOCKED", tone: "amber" },
  waiting: { label: "WAITING", tone: "amber" }, starting: { label: "STARTING", tone: "" },
};

function adminApi<T>(path: string, init?: RequestInit): Promise<T> {
  const key = getPref("td.adminKey");
  return api<T>(path, { ...init, headers: { ...(init?.headers as Record<string, string>), ...(key ? { "X-Admin-Key": key } : {}) } });
}

const dur = (s: number | null | undefined) => {
  if (s === null || s === undefined) return "—";
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  return `${(s / 3600).toFixed(1)} h`;
};
const mb = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(1)} MB`);

function Meter({ done, total, label }: { done: number; total: number; label: string }) {
  const p = total ? Math.min(100, (done / total) * 100) : 0;
  return (
    <div className="meter-row">
      <span className="muted">{label}</span>
      <div className="bar meter" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={total} aria-valuenow={done}>
        <i style={{ left: 0, width: `${p}%` }} />
      </div>
      <span className="mono right">{fx(done, 0)}/{fx(total, 0)}</span>
    </div>
  );
}

function AdminKeyForm({ onSaved, wrong }: { onSaved: () => void; wrong: boolean }) {
  const [key, setKey] = useState("");
  const [bad, setBad] = useState(false);
  return (
    <form className="form" style={{ maxWidth: 520 }} onSubmit={(e) => {
      e.preventDefault();
      const k = key.trim();
      if (!k) return;
      if (!/^[\x21-\x7e]+$/.test(k)) { setBad(true); return; }
      setBad(false);
      setPref("td.adminKey", k);
      onSaved();
    }}>
      <div className="field" style={{ gridColumn: "1 / -1" }}>
        <label htmlFor="admin-key">Admin key</label>
        <input id="admin-key" type="password" autoFocus value={key} onChange={(e) => setKey(e.target.value)} autoComplete="off" spellCheck={false} />
        <span className={wrong || bad ? "err" : "hint"}>{bad ? "Keys are plain ASCII — check for smart quotes or dashes." : wrong ? "That key was not accepted." : "The ADMIN_KEY from the host's .env. Kept in this browser only."}</span>
      </div>
      <button type="submit" className="btn primary" style={{ justifySelf: "start", height: 30 }}>Unlock</button>
    </form>
  );
}

function PullPanel({ st, reload }: { st: WorkerStatus | null; reload: () => void }) {
  const job = st?.job;
  const [daily, setDaily] = useState(true);
  const [minute, setMinute] = useState(true);
  const [years, setYears] = useState("5");
  const [rate, setRate] = useState("3");
  const [mh, setMh] = useState<MarketHours>("slow");
  const [reserve, setReserve] = useState("2500");
  const [scope, setScope] = useState<"all" | "custom">("all");
  const [symbols, setSymbols] = useState("");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [planErr, setPlanErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const seeded = useRef(false);

  // start from the worker's current job, once
  useEffect(() => {
    if (seeded.current || !st) return;
    seeded.current = true;
    if (!job) return;
    setDaily(job.intervals.includes("D"));
    setMinute(job.intervals.includes("1"));
    setYears(String(job.years));
    setRate(String(job.rate));
    setMh(job.market_hours);
    setReserve(String(job.reserve));
    if (job.symbols.length) {
      setScope("custom");
      setSymbols(job.symbols.join(", "));
    }
  }, [st, job]);

  const minReserve = st?.limits?.min_reserve ?? 2000;
  const limit = st?.budget.limit ?? 7000;
  const reserveN = reserve.trim() === "" ? NaN : Number(reserve);
  const reserveOk = Number.isInteger(reserveN) && reserveN >= minReserve && reserveN < limit;
  const body = useMemo(() => ({
    intervals: [...(daily ? ["D"] : []), ...(minute ? ["1"] : [])],
    years: Number(years), rate: Number(rate), reserve: reserveN, market_hours: mh,
    symbols: scope === "custom" ? symbols.split(/[\s,]+/).map((s) => s.trim().toUpperCase()).filter(Boolean) : [],
  }), [daily, minute, years, rate, reserveN, mh, scope, symbols]);
  const valid = body.intervals.length > 0 && (scope === "all" || body.symbols.length > 0) && reserveOk;
  const workerUp = !!st && st.state !== "unavailable";

  // preview what the pull still needs (debounced; the worker owns the resume state). Only the newest request's
  // answer is shown: an older one arriving late is ignored.
  useEffect(() => {
    setPlan(null);
    setPlanErr("");
    if (!valid || !workerUp) return;
    let current = true;
    const id = window.setTimeout(() => {
      adminApi<Plan>("/api/admin/backfill/plan", { method: "POST", body: JSON.stringify(body) })
        .then((p) => { if (!current) return; if (p.error) setPlanErr(p.error); else setPlan(p); })
        .catch((e: ApiError) => { if (current) setPlanErr(e.message); });
    }, 500);
    return () => { current = false; window.clearTimeout(id); };
  }, [body, valid, workerUp]);

  const running = !!job?.active;
  const send = (path: string, okMsg: string, payload?: object) => {
    setBusy(true);
    setMsg("");
    adminApi<{ ok: boolean }>(path, { method: "POST", body: payload ? JSON.stringify(payload) : undefined })
      .then(() => { setMsg(okMsg); reload(); })
      .catch((e: ApiError) => setMsg(e.message))
      .finally(() => setBusy(false));
  };

  return (
    <Panel code="PULL" title="Download history from Dhan" area="pull" sub="into this machine's lake · data/lake/dhan"
      actions={<button type="button" className="btn ghost" disabled={busy || !workerUp} onClick={() => send("/api/admin/backfill/coverage", "Rescan requested.")} title="Rescan the lake now"><Icon name="refresh" size={12} /> Rescan</button>}>
      <div className="banner">
        Runs in its own worker, so the live desks are not affected: paced under Dhan's 5 requests/s, keeps a reserve of
        today's {fx(st?.budget.limit ?? 7000, 0)} calls for the live app, waits while the live app loads its own history, and
        resumes by itself after midnight IST when the day's quota runs out. Already-downloaded ranges are skipped.
      </div>
      <form className="form" onSubmit={(e) => { e.preventDefault(); if (valid) send("/api/admin/backfill/start", running ? "New settings sent — the pull restarts with them." : "Pull started.", body); }}>
        <fieldset className="field">
          <legend>Data</legend>
          <div style={{ display: "flex", gap: 12 }}>
            <label className="check"><input type="checkbox" checked={daily} onChange={(e) => setDaily(e.target.checked)} /> Daily bars</label>
            <label className="check"><input type="checkbox" checked={minute} onChange={(e) => setMinute(e.target.checked)} /> 1-minute bars</label>
          </div>
        </fieldset>
        <div className="field">
          <span className="field-label">History</span>
          <Seg label="Years of history" value={years} onChange={setYears}
            options={["1", "2", "3", "5"].map((v) => ({ v, label: `${v}y`, title: v === "5" ? "Dhan keeps 5 years of 1-minute bars" : undefined }))} />
        </div>
        <div className="field">
          <span className="field-label">Speed</span>
          <Seg label="Requests per second" value={rate} onChange={setRate}
            options={["1", "2", "3", "4"].map((v) => ({ v, label: `${v}/s`, title: `${v} request${v === "1" ? "" : "s"} per second (Dhan allows 5 in total)` }))} />
        </div>
        <div className="field">
          <span className="field-label">Market hours</span>
          <Seg label="During market hours" value={mh} onChange={setMh}
            options={[{ v: "slow", label: "Slow", title: "1 request/s from 09:00 to 15:40 IST" },
              { v: "pause", label: "Pause", title: "No requests from 09:00 to 15:40 IST" },
              { v: "full", label: "Full", title: "Same speed all day" }]} />
        </div>
        <div className="field">
          <label htmlFor="reserve">Keep for the live app</label>
          <input id="reserve" type="number" required min={minReserve} max={limit - 1} step={100} value={reserve}
            onChange={(e) => setReserve(e.target.value)} aria-invalid={!reserveOk} aria-describedby="reserve-h" />
          <span id="reserve-h" className={reserveOk ? "hint" : "err"}>
            {reserveOk ? "calls/day never used by the pull" : `a whole number from ${fx(minReserve, 0)} (what the live app uses in a day) to ${fx(limit - 1, 0)}`}
          </span>
        </div>
        <div className="field">
          <span className="field-label">Symbols</span>
          <Seg label="Symbols" value={scope} onChange={setScope}
            options={[{ v: "all", label: `All F&O${plan && scope === "all" ? ` (${plan.symbols})` : ""}` }, { v: "custom", label: "Custom" }]} />
        </div>
        {scope === "custom" && (
          <div className="field" style={{ gridColumn: "1 / -1" }}>
            <label htmlFor="syms">Symbols (comma separated)</label>
            <input id="syms" value={symbols} onChange={(e) => setSymbols(e.target.value)} placeholder="RELIANCE, HDFCBANK, NIFTY" spellCheck={false} />
            {plan?.unknown?.length ? <span className="err">Not in the F&O universe: {plan.unknown.join(", ")}</span> : null}
          </div>
        )}
        <div className="plan-line" style={{ gridColumn: "1 / -1" }} aria-live="polite">
          {!workerUp ? <span className="muted">Plan preview needs the backfill worker.</span>
            : !valid ? <span className="err">{!reserveOk ? "Fix the reserve." : `Pick at least one kind of data${scope === "custom" ? " and a symbol" : ""}.`}</span>
              : planErr ? <span className="down">{planErr}</span>
                : !plan ? <span className="muted">Working out what is still missing…</span>
                  : plan.requests === 0 ? <span className="up">Everything asked for is already in the lake (up to the day before yesterday).</span>
                    : (
                      <span>
                        <b className="amber">{fx(plan.requests, 0)}</b> requests still needed
                        {Object.entries(plan.by_interval).map(([k, v]) => ` · ${IV_NAME[k] ?? k} ${fx(v, 0)}`).join("")}
                        {" — "}{plan.calls_left_today !== null ? <>{fx(plan.calls_left_today, 0)} calls left today after the reserve · </> : null}
                        {plan.calls_left_today === null || plan.calls_left_today >= plan.requests ? "fits in today's quota"
                          : plan.calls_left_today === 0 ? `starts after midnight IST${plan.days_of_quota > 1 ? `, spans ${plan.days_of_quota} days of quota` : ""}`
                            : `spans ${plan.days_of_quota} days of quota`} · ≈ {dur(plan.run_minutes * 60)} of pulling
                      </span>
                    )}
        </div>
        <div style={{ gridColumn: "1 / -1", display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <button type="submit" className="btn primary" disabled={busy || !valid || !workerUp}>{running ? "Apply settings" : "Start pull"}</button>
          <button type="button" className="btn danger" disabled={busy || !running} onClick={() => send("/api/admin/backfill/stop", "Stop sent.")}>Stop</button>
          {msg && <span className="small muted" role="status">{msg}</span>}
        </div>
      </form>
    </Panel>
  );
}

function ProgressPanel({ st, err }: { st: WorkerStatus | null; err: ApiError | null }) {
  const p = st?.progress;
  const s = st ? STATE[st.state] ?? { label: st.state.toUpperCase(), tone: "" } : { label: "WORKER DOWN", tone: "down" };
  const b = st?.budget;
  const live = st && (st.state === "running" || st.state === "paused");
  return (
    <Panel code="PROG" title="Progress" area="prog"
      sub={st ? `worker ${st.worker.host}:${st.worker.pid} · v${st.worker.version}` : "backfill worker"}>
      {err && <div className="banner warn" role="alert">Lost contact with the gateway ({err.message}) — the figures below may be stale.</div>}
      {!st ? (
        <Empty>
          The backfill worker is not running. <code>python run.py</code> starts it with the rest of the stack; with Docker:
          {" "}<code>docker compose up -d backfill</code>.
        </Empty>
      ) : (
        <>
          <div className="kpis" style={{ "--kpi-min": "128px" } as CSSProperties}>
            <Kpi label="State" value={s.label} tone={s.tone} title={st.reason || s.label} />
            <Kpi label="Progress" value={p && p.total ? `${fx((p.done / p.total) * 100, 1)}%` : "—"} sub={p ? `${fx(p.done, 0)}/${fx(p.total, 0)} requests` : "no pull yet"} />
            <Kpi label="Rows written" value={p ? fx(p.rows, 0) : "—"} sub={p ? `${fx(p.ok, 0)} with data · ${fx(p.empty, 0)} empty` : ""} />
            <Kpi label="Speed" value={live && p ? `${fx(p.rate, 2)}/s` : "—"} sub={live && p?.eta_s ? `ETA ${dur(p.eta_s)}` : "requests per second"} />
            <Kpi label="Dhan calls today" value={b ? `${fx(b.used, 0)}/${fx(b.limit, 0)}` : "—"} sub={b?.reserve !== null && b?.reserve !== undefined ? `${fx(b.reserve, 0)} kept for the live app` : "shared with the live app"} />
            <Kpi label="Failed" value={p ? fx(p.failed, 0) : "—"} tone={p?.failed ? "down" : ""} sub={p?.retried ? `${p.retried} retried` : "after retries"} title={p?.last_error || undefined} />
          </div>
          {st.reason && <div className={`banner ${["error", "unavailable"].includes(st.state) ? "warn" : ""}`} role="status">{st.reason}</div>}
          <div style={{ padding: "8px 10px", display: "grid", gap: 6 }}>
            {p && Object.entries(p.by_interval).map(([iv, [d, t]]) => <Meter key={iv} label={IV_NAME[iv] ?? iv} done={d} total={t} />)}
            {p?.current && <div className="small mono ellipsis" title={p.current}><span className="muted">NOW </span>{p.current}</div>}
            {st.resume_at && <div className="small"><span className="muted">Resumes </span>{dateIST(st.resume_at)} {timeIST(st.resume_at)} IST</div>}
            {st.job && (
              <div className="small muted">
                Job: {st.job.intervals.map((i) => IV_NAME[i] ?? i).join(" + ")} · {st.job.years}y · {st.job.rate}/s · market hours {st.job.market_hours}
                {" "}· {st.job.symbols.length ? `${st.job.symbols.length} symbols` : "all F&O"} · by {st.job.by} {dateIST(st.job.created)} {timeIST(st.job.created)}
                {st.job.active ? "" : " · finished/stopped"}
              </div>
            )}
            {st.result && (
              <div className="small muted">
                Last run ended {dateIST(st.result.at)} {timeIST(st.result.at)} ({st.result.reason}): {fx(st.result.requests, 0)} requests, {fx(st.result.rows, 0)} rows
                {st.result.failed ? `, ${st.result.failed} failed` : ""}
              </div>
            )}
          </div>
          {p?.errors?.length ? (
            <table className="tbl">
              <thead><tr><th className="l">Failed{p.errors.length > 5 ? ` (last 5 of ${p.errors.length})` : ""}</th><th className="l">Request</th><th className="l">Error</th></tr></thead>
              <tbody>
                {[...p.errors].reverse().slice(0, 5).map(([t, label, m], i) => (
                  <tr key={i} style={{ cursor: "default" }}><td className="l">{timeIST(t, true)}</td><td className="l">{label}</td><td className="l muted ellipsis" title={m}>{m}</td></tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </>
      )}
    </Panel>
  );
}

function LakePanel({ st }: { st: WorkerStatus | null }) {
  const cov = st?.coverage;
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<"sym" | "short">("sym");
  const rows = useMemo(() => {
    const all = Object.entries(cov?.symbols ?? {}).map(([sym, v]) => ({ sym, d: v.D, m: v["1m"] }));
    const f = q.trim().toUpperCase();
    const out = f ? all.filter((r) => r.sym.includes(f)) : all;
    return out.sort((a, b) => (sort === "sym" ? a.sym.localeCompare(b.sym) : (b.m?.[0] ?? Infinity) - (a.m?.[0] ?? Infinity)));
  }, [cov, q, sort]);
  const job = st?.job;
  const target = (iv: "D" | "1", sym: string) => {
    if (!job || !job.intervals.includes(iv) || (job.symbols.length && !job.symbols.includes(sym))) return null;
    const yrs = iv === "1" ? Math.min(job.years, 5) : job.years;
    return Date.now() / 1000 - yrs * 365.25 * 86400 + 20 * 86400;
  };
  const late = (iv: "D" | "1", sym: string, first?: number) => {
    const t = target(iv, sym);
    return t && first && first > t ? "amber" : "";
  };
  return (
    <Panel code="LAKE" title="What is in the lake" area="lake" className="cap-m"
      sub={cov ? `scanned ${timeIST(cov.updated, true)} · amber = starts later than the pull asked (not pulled yet, or listed later)` : "Parquet per symbol"}
      bar={cov ? (
        <>
          <input className="filter-input" placeholder="Filter symbol" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter symbols" spellCheck={false} />
          <Seg label="Sort" value={sort} onChange={setSort} options={[{ v: "sym", label: "A–Z" }, { v: "short", label: "Shortest 1-min first" }]} />
          <span className="grow tiny muted">{rows.length} symbols</span>
        </>
      ) : undefined}>
      {!cov ? <Empty>{st ? "Scanning the lake…" : "Needs the backfill worker."}</Empty> : (
        <>
          <table className="tbl">
            <thead><tr><th className="l">Interval</th><th>Symbols</th><th>Rows</th><th>From</th><th>To</th><th>Size</th></tr></thead>
            <tbody>
              {Object.entries(cov.intervals).map(([iv, x]) => (
                <tr key={iv} style={{ cursor: "default" }}>
                  <td className="l b">{IV_NAME[iv] ?? iv}</td><td>{x.symbols}</td><td>{fx(x.rows, 0)}</td>
                  <td>{x.first ? dateIST(x.first) : "—"}</td><td>{x.last ? dateIST(x.last) : "—"}</td><td>{mb(x.bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="tbl" style={{ marginTop: 6 }}>
            <thead><tr><th className="l">Symbol</th><th>Daily from</th><th>Daily to</th><th>1-min from</th><th>1-min to</th><th>1-min rows</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.sym} style={{ cursor: "default" }}>
                  <td className="sym">{r.sym}</td>
                  <td className={late("D", r.sym, r.d?.[0])}>{r.d ? dateIST(r.d[0]) : "—"}</td><td>{r.d ? dateIST(r.d[1]) : "—"}</td>
                  <td className={late("1", r.sym, r.m?.[0])}>{r.m ? dateIST(r.m[0]) : "—"}</td><td>{r.m ? dateIST(r.m[1]) : "—"}</td>
                  <td>{r.m ? fx(r.m[2], 0) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Panel>
  );
}

function LogPanel({ st }: { st: WorkerStatus | null }) {
  const lines = [...(st?.log ?? [])].reverse();
  return (
    <Panel code="LOG" title="Activity" area="log" className="cap-m" sub="backfill worker log (newest first)">
      {!lines.length ? <Empty>No activity yet.</Empty> : (
        <ol className="log-list">
          {lines.map(([t, lvl, m], i) => (
            <li key={`${t}-${i}`} className={lvl === "ERROR" ? "down" : lvl === "WARNING" ? "amber" : ""}>
              <span className="muted mono">{timeIST(t, true)}</span><span>{m}</span>
            </li>
          ))}
        </ol>
      )}
    </Panel>
  );
}

export default function AdminDesk() {
  const [resp, setResp] = useState<AdminResp | null>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const [tick, setTick] = useState(0);
  const [keyState, setKeyState] = useState<"none" | "checking" | "rejected">("none");
  useEffect(() => {
    let alive = true;
    const load = () =>
      adminApi<AdminResp>("/api/admin/backfill")
        .then((r) => { if (alive) { setResp(r); setErr(null); setKeyState("none"); } })
        .catch((e: ApiError) => {
          if (!alive) return;
          setErr(e instanceof ApiError ? e : new ApiError(0, String(e)));
          setKeyState((k) => (k === "checking" ? "rejected" : k));
        });
    load();
    const id = window.setInterval(load, 2000);
    return () => { alive = false; window.clearInterval(id); };
  }, [tick]);
  const reload = () => setTick((t) => t + 1);

  if (err?.status === 403) {
    const needKey = /admin key/i.test(err.message);
    return (
      <div className="desk desk-admin-locked">
        <Panel code="ADMIN" title="Admin desk" pad>
          {needKey ? (
            <AdminKeyForm wrong={keyState === "rejected"} onSaved={() => { setKeyState("checking"); reload(); }} />
          ) : <Empty>{err.message}</Empty>}
        </Panel>
      </div>
    );
  }
  const st = resp?.status ?? null;
  return (
    <div className="desk desk-admin">
      {!resp && !err ? <Empty>Loading…</Empty> : (
        <>
          <PullPanel st={st} reload={reload} />
          <ProgressPanel st={st} err={err} />
          <LakePanel st={st} />
          <LogPanel st={st} />
        </>
      )}
    </div>
  );
}
