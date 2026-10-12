import { memo, useEffect, useMemo, useRef, useState } from "react";
import { PriceChart } from "../components/Chart";
import { Empty, Panel, Seg, Spark, ZBar } from "../components/ui";
import { api } from "../lib/api";
import { BUILDUP, cls, compact, crore, eventLabel, fx, pct, px, signed } from "../lib/format";
import { useLive } from "../lib/store";
import type { Depth, Pick, RankRow } from "../lib/types";

/* ---------------------------------------------------------------- top picks */
function PickRow({ p, i, side, locked, selected }: { p: Pick; i: number; side: "long" | "short"; locked: boolean; selected: boolean }) {
  const select = useLive((s) => s.select);
  const status = locked ? p.status ?? "LIVE" : "PROV";
  const main = locked ? p.pnl : p.chg;
  return (
    <button type="button" className={`pick ${selected ? "sel" : ""}`} onClick={() => select(p.symbol)}
      aria-label={`${side} pick ${i + 1}: ${p.symbol}`}>
      <span className="pick-n">{i + 1}</span>
      <span style={{ minWidth: 0 }}>
        <span className="pick-sym">{p.symbol}<small>{p.sector}</small><small>score {p.score.toFixed(2)}</small></span>
        <span className="chips pick-chips">
          {p.chips.slice(0, 6).map((c) => (
            <span key={c} className={`chip ${c === "LB" || c === "ORB" || c === "VWAP+" ? "up" : c === "SB" || c === "ORBD" || c === "VWAP-" ? "down" : c.startsWith("RVOL") ? "amber" : c === "RS" ? "cyan" : "violet"}`}>{c}</span>
          ))}
        </span>
        {p.headline && <span className="pick-head-line ellipsis" title={p.headline}>{p.headline}</span>}
      </span>
      <span>{p.spark && p.spark.length > 2 ? <Spark data={p.spark} /> : locked ? <span className="tiny muted mono">entry {px(p.entry)}</span> : null}</span>
      <span>
        <span className={`pick-pnl ${cls(main)}`}>
          {pct(main, 2)}
          <small>{locked ? `@${px(p.entry)} → ${px(p.ltp)}` : `ltp ${px(p.ltp)}`}</small>
        </span>
        <span className={`pick-status ${status}`} style={{ display: "block", marginTop: 3 }}>
          {status === "PROV" ? "PROVISIONAL" : status}
        </span>
      </span>
    </button>
  );
}

function TopPicks() {
  const top = useLive((s) => s.top);
  const ranked = useLive((s) => s.rank?.rows.length ?? 0);
  const selected = useLive((s) => s.selected);
  if (!top) return <Panel code="TOP5" title="Intraday picks" area="picks"><Empty>Waiting for the analytics worker…</Empty></Panel>;
  const u = top.universe_since_lock;
  const mm = Math.floor((top.seconds_to_lock ?? 0) / 60);
  const ss = String((top.seconds_to_lock ?? 0) % 60).padStart(2, "0");
  const sub = top.locked
    ? `locked ${top.lock_time} IST · tracking from entry`
    : top.missed
      ? `missed today's ${top.lock_time} lock (no ranking at lock time) · live ranking, not scored`
      : (top.seconds_to_lock ?? 0) > 0
        ? `provisional · locks at ${top.lock_time} IST (in ${mm}:${ss})`
        : `provisional · locking as soon as the ranking is ready`;
  return (
    <Panel code="TOP5" title="Intraday picks" sub={sub} area="picks" id="picks" className="picks-panel"
      actions={top.locked ? <span className="chip up">LOCKED</span> : <span className="chip amber">LIVE RANKING</span>}
      foot={top.locked && u ? (
        <div className="lockbar" aria-label="Performance since lock">
          <span className="amber b">SINCE LOCK</span>
          <span className="kv"><span>Long avg</span><span className={cls(u.long_avg)}>{pct(u.long_avg)}</span></span>
          <span className="kv"><span>Short avg</span><span className={cls(u.short_avg)}>{pct(u.short_avg)}</span></span>
          <span className="kv"><span>Universe</span><span className={cls(u.universe_avg)}>{pct(u.universe_avg)}</span></span>
          <span className="kv"><span>In top-5</span><span>{u.long_hits_top5}/5</span></span>
          <span className="kv"><span>In top-20</span><span>{u.long_hits_top20}/5</span></span>
          <span className="kv"><span>Leaders</span><span className="muted">{u.leaders.slice(0, 3).map((l) => `${l.symbol} ${signed(l.ret, 1)}%`).join(" · ")}</span></span>
        </div>
      ) : undefined}>
      <div className="picks">
        {(["long", "short"] as const).map((side) => {
          const list = top[side] ?? [];
          return (
            <div className="picks-col" key={side}>
              <div className="picks-head">
                <span className={side === "long" ? "up b" : "down b"}>{side === "long" ? "▲ LONG — continuation up" : "▼ SHORT — continuation down"}</span>
                <span className="muted" style={{ marginLeft: "auto" }}>{list.length}/{5}</span>
              </div>
              {list.length === 0 && <Empty>{ranked === 0
                ? "No ranking yet — waiting for previous closes and history (live mode loads them from Dhan automatically, a few minutes after start; admins see progress under /admin → SYSTEM)."
                : "No stock passes the filters yet — the system abstains rather than force a pick."}</Empty>}
              {list.map((p, i) => <PickRow key={p.symbol} p={p} i={i} side={side} locked={top.locked} selected={selected === p.symbol} />)}
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

/* ---------------------------------------------------------------- ranking */

type SortKey = "score" | "chg" | "gap" | "rvol" | "resid" | "vwap_atr" | "orb" | "oi_chg" | "news" | "turnover" | "score_s";

const Row = memo(function Row({ r, sel, onSel, side }: { r: RankRow; sel: boolean; onSel: (s: string) => void; side: "long" | "short" }) {
  const prev = useRef(r.ltp);
  const [flash, setFlash] = useState("");
  useEffect(() => {
    if (r.ltp !== prev.current) {
      setFlash(r.ltp > prev.current ? "flash-up" : "flash-down");
      prev.current = r.ltp;
      const t = window.setTimeout(() => setFlash(""), 700);
      return () => window.clearTimeout(t);
    }
  }, [r.ltp]);
  const score = side === "long" ? r.score : r.score_s;
  const bu = BUILDUP[r.oi_tag];
  return (
    <tr className={`${sel ? "sel" : ""} ${r.eligible ? "" : "dim"}`} onClick={() => onSel(r.symbol)} aria-selected={sel}>
      <td className="rank-n">{r.rank}</td>
      <td className="sym">{r.symbol}<small>{r.sector}</small></td>
      <td className={flash}>{px(r.ltp)}</td>
      <td className={cls(r.chg)}>{pct(r.chg)}</td>
      <td className={cls(r.gap)}>{signed(r.gap)}%</td>
      <td className={r.rvol >= 2 ? "amber b" : r.rvol >= 1.3 ? "" : "muted"}>{fx(r.rvol, 1)}x</td>
      <td className={cls(r.resid)}>{signed(r.resid)}</td>
      <td className={`opt ${cls(r.vwap_atr, 0.02)}`}>{signed(r.vwap_atr)}</td>
      <td className="opt">{r.orb >= 1 ? <span className="chip up">BRK ▲</span> : r.orb <= -1 ? <span className="chip down">BRK ▼</span> : <span className="muted">{fx(r.or_pos * 100, 0)}%</span>}</td>
      <td>{bu ? <span className={`chip ${bu.cls}`} title={`${bu.long} · OI ${signed(r.oi_chg, 1)}%`}>{bu.label} {signed(r.oi_chg, 1)}%</span> : <span className="muted">—</span>}</td>
      <td title={r.headline ? `${r.headline} (score ${signed(r.news, 2)})` : ""}>{Math.abs(r.news) > 0.05 ? <span className={`chip ${r.news > 0 ? "up" : "down"}`}>{r.news > 0 ? "▲" : "▼"} {eventLabel(r.news_type)}</span> : <span className="muted">—</span>}</td>
      <td className="muted opt">{fx(r.turnover, 0)}</td>
      <td style={{ width: 110 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <ZBar v={score} max={2.5} title="Composite score" />
          <span className={score > 0 ? "amber b" : "muted"} style={{ minWidth: 38 }}>{score.toFixed(2)}</span>
        </div>
      </td>
    </tr>
  );
});

function RankTable() {
  const rank = useLive((s) => s.rank);
  const selected = useLive((s) => s.selected);
  const select = useLive((s) => s.select);
  const [side, setSide] = useState<"long" | "short">("long");
  const [filter, setFilter] = useState<"all" | "elig" | "news">("all");
  const [sector, setSector] = useState("");
  const [sort, setSort] = useState<{ k: SortKey; d: 1 | -1 }>({ k: "score", d: -1 });
  const sectors = useMemo(() => [...new Set((rank?.rows ?? []).map((r) => r.sector))].sort(), [rank?.rows]);
  const rows = useMemo(() => {
    let rs = rank?.rows ?? [];
    if (filter === "elig") rs = rs.filter((r) => r.eligible);
    if (filter === "news") rs = rs.filter((r) => Math.abs(r.news) > 0.05);
    if (sector) rs = rs.filter((r) => r.sector === sector);
    const k = sort.k === "score" && side === "short" ? "score_s" : sort.k;
    return [...rs].sort((a, b) => ((a[k] as number) - (b[k] as number)) * sort.d);
  }, [rank, filter, sector, sort, side]);
  const th = (k: SortKey, label: string, title?: string, opt = false) => (
    <th className={`sortable ${opt ? "opt" : ""}`} title={title} aria-sort={sort.k === k ? (sort.d < 0 ? "descending" : "ascending") : "none"}
      onClick={() => setSort((s) => ({ k, d: s.k === k ? ((-s.d) as 1 | -1) : -1 }))}>{label}</th>
  );
  return (
    <Panel code="RANK" title="Live factor ranking" area="rank" id="rank" className="cap-m"
      sub={rank ? `${rank.rows.length} stocks · minute ${rank.minute.toFixed(0)} · z-scored cross-section` : ""}
      actions={
        <>
          <Seg label="Side" value={side} onChange={setSide} options={[{ v: "long", label: "LONG" }, { v: "short", label: "SHORT" }]} />
          <Seg label="Filter" value={filter} onChange={setFilter} options={[{ v: "all", label: "ALL" }, { v: "elig", label: "ELIGIBLE" }, { v: "news", label: "NEWS" }]} />
          <label className="sr-only" htmlFor="sector-f">Sector</label>
          <select id="sector-f" value={sector} onChange={(e) => setSector(e.target.value)}
            style={{ height: 24, background: "var(--raise)", border: "1px solid var(--border-2)", borderRadius: 3, fontFamily: "var(--font-mono)", fontSize: 11 }}>
            <option value="">ALL SECTORS</option>
            {sectors.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </>
      }>
      {!rank ? <Empty>Waiting for ticks…</Empty> : (
        <table className="tbl">
          <thead>
            <tr>
              <th>#</th><th className="l">Symbol</th><th>LTP</th>
              {th("chg", "Chg")}{th("gap", "Gap")}{th("rvol", "RVOL", "Volume vs the same minute over the last 14 sessions")}
              {th("resid", "RS", "Return minus beta × NIFTY")}{th("vwap_atr", "VWAPσ", "Distance from VWAP in ATR units", true)}
              {th("orb", "OR", "Opening-range position / breakout", true)}{th("oi_chg", "OI", "Futures OI build-up")}
              {th("news", "News", "Decayed catalyst score")}{th("turnover", "₹ Cr", "Turnover today, ₹ crore", true)}{th("score", "Score")}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => <Row key={r.sid} r={r} sel={r.symbol === selected} onSel={select} side={side} />)}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

/* ----------------------------------------------------------------- depth */
const qty = (q: number) => compact(q).replace(" ", "");

function DepthLadder({ sid, symbol }: { sid: number | undefined; symbol: string }) {
  const [d, setD] = useState<Depth | null>(null);
  useEffect(() => {
    if (!sid) return;
    let alive = true;
    const load = () => api<Depth>(`/api/depth/${sid}`).then((x) => alive && setD(x)).catch(() => {});
    load();
    const id = window.setInterval(load, 1000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [sid]);
  const bids = d?.bid ?? [];
  const asks = d?.ask ?? [];
  const maxQ = Math.max(1, ...bids.map((b) => b[1]), ...asks.map((a) => a[1]));
  const avg = (bids.reduce((s, b) => s + b[1], 0) + asks.reduce((s, a) => s + a[1], 0)) / Math.max(1, bids.length + asks.length);
  const tb = bids.reduce((s, b) => s + b[1], 0);
  const ta = asks.reduce((s, a) => s + a[1], 0);
  const imb = tb + ta ? (tb - ta) / (tb + ta) : 0;
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div className="panel-head" style={{ background: "transparent" }}>
        <span className="panel-code">DOM</span><span className="panel-title">20-level depth</span>
        <span className="panel-sub">{symbol}</span>
      </div>
      {!bids.length && !asks.length ? <Empty>{d?.pending ? "Subscribing to 20-level depth…" : "No depth yet"}</Empty> : (
        <>
          <div className="dom" style={{ flex: 1, overflow: "auto" }}>
            <div className="dom-side bid">
              <div className="dom-head dom-row"><span>ORD</span><span>QTY</span><span>BID</span></div>
              {bids.slice(0, 20).map(([p, q, o], i) => (
                <div className="dom-row" key={i}>
                  <span className="fill" style={{ width: `${(q / maxQ) * 100}%` }} />
                  <span className="muted">{o}</span><span className={q > avg * 2.5 ? "dom-wall" : ""}>{qty(q)}</span><span className="up">{fx(p)}</span>
                </div>
              ))}
            </div>
            <div className="dom-side ask">
              <div className="dom-head dom-row"><span>ASK</span><span>QTY</span><span>ORD</span></div>
              {asks.slice(0, 20).map(([p, q, o], i) => (
                <div className="dom-row" key={i}>
                  <span className="fill" style={{ width: `${(q / maxQ) * 100}%` }} />
                  <span className="down">{fx(p)}</span><span className={q > avg * 2.5 ? "dom-wall" : ""}>{qty(q)}</span><span className="muted">{o}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="imb" title="Resting-book imbalance across visible levels (spoofable — weak signal)">
            <span className="muted">IMB</span>
            <div className="bar center" style={{ flex: 1 }}><i className={imb >= 0 ? "pos" : "neg"} style={{ width: `${Math.abs(imb) * 50}%` }} /></div>
            <span className={cls(imb)}>{signed(imb * 100, 0)}%</span>
            <span className="muted tiny">walls = &gt;2.5× avg</span>
          </div>
        </>
      )}
    </div>
  );
}

/* ----------------------------------------------------------- factor view */
const FACTOR_LABEL: Record<string, string> = {
  rvol: "Relative volume", resid: "Relative strength", vwap: "VWAP position", orb: "Opening range",
  oi: "Futures OI build-up", news: "News catalyst", sector: "vs sector", obi: "Book imbalance",
};

function FactorView({ row }: { row: RankRow | undefined }) {
  const weights = useLive((s) => s.rank?.weights);
  if (!row) return <Empty>Select a stock</Empty>;
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "auto" }}>
      <div className="panel-head" style={{ background: "transparent" }}>
        <span className="panel-code">WHY</span><span className="panel-title">Factor breakdown</span>
        <span className="panel-sub">{row.symbol} · rank #{row.rank}</span>
      </div>
      <div style={{ padding: "6px 10px", display: "grid", gap: 5 }}>
        {Object.entries(row.f).map(([k, v]) => (
          <div key={k} style={{ display: "grid", gridTemplateColumns: "118px 1fr 44px 34px", gap: 6, alignItems: "center", fontSize: 12 }}>
            <span className="muted">{FACTOR_LABEL[k] ?? k}</span>
            <ZBar v={v} title={`${k} z-score`} />
            <span className={`mono right ${cls(v, 0.05)}`}>{signed(v)}</span>
            <span className="mono muted tiny right" title="weight">{weights?.[k] ? `${Math.round(weights[k] * 100)}%` : ""}</span>
          </div>
        ))}
      </div>
      <dl className="kv-grid" style={{ borderTop: "1px solid var(--border)" }}>
        <dt>Prev close</dt><dd>{px(row.pc)}</dd>
        <dt>Open / VWAP</dt><dd>{px(row.open)} / {px(row.vwap)}</dd>
        <dt>Day range</dt><dd>{px(row.low)} – {px(row.high)}</dd>
        <dt>Opening range</dt><dd>{px(row.orl)} – {px(row.orh)}</dd>
        <dt>ATR (14d)</dt><dd>{fx(row.atr_pct)}%</dd>
        <dt>Turnover</dt><dd>{crore(row.turnover)}</dd>
      </dl>
      {row.headline && <div className="banner" style={{ borderTop: "1px solid var(--border)" }}><b className="amber">{row.news_type}</b> · {row.headline}</div>}
    </div>
  );
}

export default function IntradayDesk() {
  const rank = useLive((s) => s.rank);
  const top = useLive((s) => s.top);
  const selected = useLive((s) => s.selected);
  const select = useLive((s) => s.select);
  const bySymbol = useLive((s) => s.bySymbol);
  useEffect(() => {
    if (!selected || (rank && !rank.rows.some((r) => r.symbol === selected))) {
      const first = top?.long?.[0]?.symbol ?? rank?.rows?.[0]?.symbol;
      if (first) select(first);
    }
  }, [selected, rank, top, select]);
  const row = rank?.rows.find((r) => r.symbol === selected);
  const pick = [...(top?.long ?? []), ...(top?.short ?? [])].find((p) => p.symbol === selected);
  const extra = useMemo(() => (pick && top?.locked ? { entry: pick.entry, stop: pick.invalidation, target: pick.target } : undefined),
    [pick?.entry, pick?.invalidation, pick?.target, top?.locked]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="desk desk-intraday">
      <TopPicks />
      <Panel code="GP" title="Chart" area="chart" className="chart-panel" sub={row ? `${selected} · ${pct(row.chg)} · RVOL ${fx(row.rvol, 1)}x` : selected}>
        {selected ? <PriceChart symbol={selected} extraLevels={extra} /> : <Empty>Select a stock</Empty>}
      </Panel>
      <RankTable />
      <section className="panel" style={{ gridArea: "side" }}>
        <div className="split" style={{ height: "100%" }}>
          <DepthLadder sid={bySymbol[selected]?.key} symbol={selected} />
          <FactorView row={row} />
        </div>
      </section>
    </div>
  );
}
