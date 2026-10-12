import { useEffect, useState } from "react";
import type { CSSProperties } from "react";
import { Empty, Kpi, Panel, ZBar } from "../components/ui";
import { api } from "../lib/api";
import { cls, dateIST, fx, pct, signed, timeIST } from "../lib/format";
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
              : <>Empty — run <code>python backfill/backfill.py run --intervals D,1 --years 5</code></>}
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
          Live mode tops this up by itself at start-up and every morning at 08:40 (AUTO_HISTORY). Run{" "}
          <code>backfill.py</code> only for the full 5-year archive the research scripts use.
        </div>
      </Panel>
    </div>
  );
}
