/* Basic HOME: what is the market doing, what does the system like today (with its record), where to look next. */
import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { api } from "../lib/api";
import { research } from "../lib/research";
import { useLive } from "../lib/store";
import type { Pick } from "../lib/types";
import { SCANS } from "./scans";
import { Badge, Card, Change, Empty, StockLink, ok, rupees, useLoad } from "./ui";
import { chipText, statusText } from "./words";

interface ScoreCard { date: string; long_avg: number; universe_avg: number; picks: string[] }

function MarketToday() {
  const market = useLive((s) => s.market);
  const clock = useLive((s) => s.clock);
  if (!market) return <Card title="Market today" id="mkt"><Empty>Waiting for market data…</Empty></Card>;
  const idx = (n: string) => market.indices.find((i) => i.name === n);
  const nifty = idx("NIFTY"), bank = idx("BANKNIFTY"), vix = idx("INDIAVIX");
  const b = market.breadth;
  const med = b?.median_chg ?? 0;
  const mood = med >= 0.4 ? "A strong day" : med <= -0.4 ? "A weak day" : "A mixed day";
  const live = clock?.phase === "OPEN";
  return (
    <Card title={live ? "Market today" : "Last session"} id="mkt" className="b-market">
      <div className="b-index-row">
        {[nifty, bank].filter(Boolean).map((i) => (
          <div className="b-index" key={i!.name}>
            <span className="b-index-name">{i!.name === "NIFTY" ? "NIFTY 50" : "BANK NIFTY"}</span>
            <span className="b-index-val">{i!.ltp.toLocaleString("en-IN", { maximumFractionDigits: 2 })}</span>
            <Change v={i!.chg} />
          </div>
        ))}
      </div>
      {b && (
        <>
          <p className="b-lead"><b>{mood}.</b> {b.adv} of the {b.n} stocks we follow are up, {b.dec} are down
            {ok(b.above_vwap_pct) ? `, and ${Math.round(b.above_vwap_pct)}% trade above today's average price` : ""}.</p>
          <div className="b-breadth" role="img" aria-label={`${b.adv} up, ${b.dec} down`}>
            <i className="up" style={{ flex: b.adv || 0.0001 }} /><i className="flat" style={{ flex: b.unch || 0.0001 }} /><i className="down" style={{ flex: b.dec || 0.0001 }} />
          </div>
        </>
      )}
      {vix && ok(vix.ltp) && (
        <p className="b-note">Fear gauge (India VIX) {vix.ltp.toFixed(1)}: {vix.ltp < 13 ? "calm — moves are usually small" : vix.ltp < 18 ? "normal" : "nervous — expect bigger swings, size positions smaller"}.</p>
      )}
    </Card>
  );
}

function PickRow({ p }: { p: Pick }) {
  const st = statusText(p.status ?? (p.pnl === undefined ? "PROV" : undefined), p.hit_at);
  const reasons = (p.chips ?? []).slice(0, 2).map(chipText);
  return (
    <li>
      <StockLink symbol={p.symbol}>
        <Badge symbol={p.symbol} />
        <span className="b-row-main">
          <b>{p.symbol}</b>
          <small>{reasons.length ? reasons.join(" · ") : p.sector}</small>
        </span>
        <span className="b-row-end">
          <span>{rupees(p.ltp ?? p.entry)}</span>
          {ok(p.pnl) ? <small><Change v={p.pnl} /> since 9:30</small> : <small><Change v={p.chg} /> today</small>}
          <small className={`b-status ${st.tone}`}>{st.text}</small>
        </span>
      </StockLink>
    </li>
  );
}

function TodayPicks() {
  const top = useLive((s) => s.top);
  const sc = useLoad(() => api<{ cards: ScoreCard[] }>("/api/scorecard"), []);
  const rec = useMemo(() => {
    const cards = (sc.data?.cards ?? []).filter((c) => (c.picks ?? []).length).slice(0, 30);
    if (!cards.length) return null;
    return { won: cards.filter((c) => c.long_avg > c.universe_avg).length, n: cards.length };
  }, [sc.data]);
  const picks = top?.long ?? [];
  return (
    <Card title="Today's strongest" id="picks" className="b-picks"
      sub={top?.locked ? `Picked at ${top.lock_time.slice(0, 5)} by the intraday ranker` : top?.closed ? `No picks today — ${top.closed}` : "Final at 9:30 — until then this list can change"}>
      {!top ? <Empty>Waiting for the analytics…</Empty> : !picks.length ? (
        <Empty>{top.closed ? "Markets are closed today." : "No stock passes the filters right now — the system prefers no idea to a bad one."}</Empty>
      ) : <ul className="b-rows">{picks.map((p) => <PickRow key={p.symbol} p={p} />)}</ul>}
      <p className="b-note">
        {rec ? <>Track record: these picks beat the average stock on <b>{rec.won} of the last {rec.n} days</b>{rec.n < 60 ? " — too few days to call it skill yet" : ""}. </> : null}
        Stops and targets are set for the day; read <i>Learn</i> before acting on any idea.
      </p>
    </Card>
  );
}

function ReadyIdeas() {
  const rank = useLive((s) => s.rank);
  const fno = useLive((s) => s.fno);
  const swing = useLive((s) => s.swing);
  const setScan = useLive((s) => s.setBasicScan);
  const setTab = useLive((s) => s.setBasicTab);
  const data = { rank: rank?.rows ?? [], fno: fno?.all ?? [], swing: swing?.rows ?? [], swingStats: swing?.stats ?? {} };
  return (
    <Card title="Ready-made ideas" id="ideas" sub="Simple filters over today's market — tap one to see the stocks">
      <div className="b-chips">
        {SCANS.map((s) => {
          const n = s.rows(data).length;
          return (
            <button type="button" key={s.id} className="b-chip" onClick={() => { setScan(s.id); setTab("ideas"); }} title={s.desc}>
              {s.name}<span>{n}</span>
            </button>
          );
        })}
      </div>
    </Card>
  );
}

function Sectors() {
  const market = useLive((s) => s.market);
  const rows = useMemo(() => [...(market?.sectors ?? [])].sort((a, b) => b.median - a.median), [market]);
  if (!rows.length) return <Card title="Sectors" id="sec" className="b-sectors"><Empty>Waiting for market data…</Empty></Card>;
  const max = Math.max(0.5, ...rows.map((r) => Math.abs(r.median)));
  const show = rows.length > 9 ? [...rows.slice(0, 5), ...rows.slice(-4)] : rows;
  return (
    <Card title="Sectors" id="sec" className="b-sectors" sub="Typical move of the stocks in each sector today">
      <ul className="b-bars">
        {show.map((r) => (
          <li key={r.sector}>
            <span className="b-bar-name">{r.name || r.sector}</span>
            <span className="b-bar-track"><i className={r.median >= 0 ? "up" : "down"}
              style={{ width: `${(Math.abs(r.median) / max) * 50}%`, [r.median >= 0 ? "left" : "right"]: "50%" } as CSSProperties} /></span>
            <Change v={r.median} />
          </li>
        ))}
      </ul>
    </Card>
  );
}

function MostActive() {
  const market = useLive((s) => s.market);
  const [k, setK] = useState<"gainers" | "losers" | "active">("gainers");
  const rows = (market?.movers?.[k] ?? []).slice(0, 6);
  return (
    <Card title="Biggest movers" id="mov" action={
      <div className="b-seg" role="group" aria-label="Movers">
        {(["gainers", "losers", "active"] as const).map((x) => (
          <button type="button" key={x} aria-pressed={k === x} onClick={() => setK(x)}>{x === "active" ? "Most traded" : x === "gainers" ? "Up most" : "Down most"}</button>
        ))}
      </div>}>
      {!rows.length ? <Empty>Waiting for market data…</Empty> : (
        <ul className="b-rows compact">
          {rows.map((r) => (
            <li key={r.symbol}>
              <StockLink symbol={r.symbol}>
                <Badge symbol={r.symbol} size={28} />
                <span className="b-row-main"><b>{r.symbol}</b><small>{r.sector}</small></span>
                <span className="b-row-end"><span>{rupees(r.ltp)}</span><small><Change v={r.chg} /></small></span>
              </StockLink>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function MonthLeaders() {
  const lb = useLoad(() => research.neuralLeaderboard(21, 5), []);
  const b = lb.data;
  if (!b || !b.ranked || !b.top.length) return null;
  const beat = b.ranked_by === "beat";
  return (
    <Card title="Strongest for the next month" id="month"
      sub={`From the tested neural model · data to ${b.as_of ?? "—"}`}>
      <ul className="b-rows compact">
        {b.top.map((x, i) => (
          <li key={x.symbol}>
            <StockLink symbol={x.symbol}>
              <span className="b-rank">{i + 1}</span>
              <Badge symbol={x.symbol} size={28} />
              <span className="b-row-main"><b>{x.symbol}</b><small>{x.sector}</small></span>
              <span className="b-row-end">
                <span>{rupees(x.close)}</span>
                <small>{beat && ok(x.p_beat) ? `${Math.round(x.p_beat * 100)}% chance to beat the average stock` : ok(x.p_up) ? `${Math.round(x.p_up * 100)}% chance to be higher` : ""}</small>
              </span>
            </StockLink>
          </li>
        ))}
      </ul>
      <p className="b-note">A ranking, not a promise: across 4 years of tests the model put stocks in a better-than-chance order a month ahead. Any single stock can still go the other way — spread ideas, and use stops.</p>
    </Card>
  );
}

export default function Home() {
  return (
    <div className="b-home">
      <div className="b-grid two">
        <div className="b-col"><MarketToday /><Sectors /></div>
        <TodayPicks />
      </div>
      <ReadyIdeas />
      <div className="b-grid two">
        <MostActive />
        <MonthLeaders />
      </div>
    </div>
  );
}
