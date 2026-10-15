import { Empty, Kpi, Panel } from "../components/ui";
import { cls, crore, fx, heatColor, pct, px, signed } from "../lib/format";
import { useLive } from "../lib/store";
import type { MoverRow } from "../lib/types";

/** Crore figures in thousands once they get long (23,856 Cr -> 23.9K). */
const kcr = (v: number) => {
  const a = Math.abs(v);
  return a >= 100000 ? `${(v / 1000).toFixed(0)}K` : a >= 10000 ? `${(v / 1000).toFixed(1)}K` : fx(v, 0);
};

function BreadthPanel() {
  const m = useLive((s) => s.market);
  if (!m) return <Panel code="BRTH" title="Breadth" area="breadth"><Empty>Waiting…</Empty></Panel>;
  const b = m.breadth;
  if (!b?.n) {
    // n = 0: the analytics worker runs but no prices have come in (feed down, Dhan token expired, before the open)
    return (
      <Panel code="BRTH" title="Market breadth" area="breadth">
        <Empty>No prices yet — breadth appears once the market feed delivers quotes (check the Dhan token and market hours).</Empty>
      </Panel>
    );
  }
  const hist = b.hist ?? [];
  const bins = b.hist_bins ?? [];
  const total = Math.max(1, b.adv + b.dec + b.unch);
  const maxH = Math.max(1, ...hist);
  return (
    <Panel code="BRTH" title="Market breadth" area="breadth" sub={`${b.n} F&O stocks · median ${pct(b.median_chg)}`}>
      <div className="kpis">
        <Kpi label="Advance / Decline" value={<><span className="up">{b.adv}</span> / <span className="down">{b.dec}</span></>} sub={`ratio ${fx(b.ad_ratio)} · ${b.unch} flat`} />
        <Kpi label="Above VWAP" value={`${fx(b.above_vwap_pct, 0)}%`} tone={b.above_vwap_pct >= 50 ? "up" : "down"} sub={`${b.above_vwap} stocks`} />
        <Kpi label="At day high / low" value={<><span className="up">{b.near_high}</span> / <span className="down">{b.near_low}</span></>} sub="within 0.2%" />
        <Kpi label="PDH / PDL breaks" value={<><span className="up">{b.above_pdh}</span> / <span className="down">{b.below_pdl}</span></>} sub="above prev-day high / below prev-day low" />
        <Kpi label="Turnover up / down" value={<><span className="up">{kcr(b.up_turnover)}</span> / <span className="down">{kcr(b.down_turnover)}</span></>} sub="₹ Cr traded in green vs red stocks" />
        <Kpi label="Median change" value={pct(b.median_chg)} tone={cls(b.median_chg)} sub={`${b.n} stocks · equal-weight`} />
      </div>
      <div style={{ padding: "8px 10px" }}>
        <div className="adbar" role="img" aria-label={`${b.adv} advancing, ${b.dec} declining`}>
          <i style={{ width: `${(b.adv / total) * 100}%`, background: "var(--up)" }} />
          <i style={{ width: `${(b.unch / total) * 100}%`, background: "#3a4659" }} />
          <i style={{ width: `${(b.dec / total) * 100}%`, background: "var(--down)" }} />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${hist.length}, 1fr)`, gap: 3, alignItems: "end", height: 70, marginTop: 10 }}
          role="img" aria-label="Distribution of % change">
          {hist.map((h, i) => (
            <div key={i} title={`${bins[i] ?? ""}%: ${h} stocks`} style={{ display: "flex", flexDirection: "column", justifyContent: "flex-end", height: "100%" }}>
              <span className="tiny mono muted center">{h}</span>
              <div style={{ height: `${(h / maxH) * 52}px`, background: i < 5 ? "rgba(255,82,99,.65)" : "rgba(39,209,127,.65)", borderRadius: 1 }} />
            </div>
          ))}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${hist.length}, 1fr)`, gap: 3 }}>
          {bins.map((l) => <span key={l} className="tiny mono muted center">{l}</span>)}
        </div>
      </div>
    </Panel>
  );
}

function Heatmap() {
  const m = useLive((s) => s.market);
  const select = useLive((s) => s.select);
  const setDesk = useLive((s) => s.setDesk);
  return (
    <Panel code="HEAT" title="Sector heatmap" area="heat" sub="tiles sized equally · colour = % change (text shows the value)">
      {!m ? <Empty>Waiting…</Empty> : !m.sectors?.length ? <Empty>No prices yet — sectors fill in once quotes arrive.</Empty> : (
        <div className="heat">
          {m.sectors.map((s) => (
            <div className="heat-sector" key={s.sector}>
              <div className="heat-sector-head">
                <span>{s.name}</span>
                <span className={cls(s.chg)}>{pct(s.chg)}</span>
              </div>
              <div className="heat-tiles">
                {s.members.slice(0, 18).map((mm) => (
                  <button type="button" key={mm.symbol} className="heat-tile" style={{ background: heatColor(mm.chg) }}
                    title={`${mm.symbol} ${pct(mm.chg)} · ${crore(mm.turnover)}`}
                    onClick={() => { select(mm.symbol); setDesk("intraday"); }}>
                    <b>{mm.symbol}</b><span>{signed(mm.chg, 1)}%</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

function MoverTable({ title, rows, metric }: { title: string; rows: MoverRow[]; metric: "chg" | "turnover" | "rvol" }) {
  const select = useLive((s) => s.select);
  const setDesk = useLive((s) => s.setDesk);
  return (
    <section className="panel" style={{ minHeight: 0 }}>
      <header className="panel-head"><span className="panel-title">{title}</span></header>
      <div className="panel-body">
        <table className="tbl">
          <thead><tr><th className="l">Symbol</th><th>LTP</th><th>Chg</th><th>{metric === "turnover" ? "₹ Cr" : "RVOL"}</th></tr></thead>
          <tbody>
            {!rows?.length && <tr><td colSpan={4} className="muted center">No prices yet</td></tr>}
            {(rows ?? []).map((r) => (
              <tr key={r.sid} onClick={() => { select(r.symbol); setDesk("intraday"); }}>
                <td className="sym">{r.symbol}</td><td>{px(r.ltp)}</td><td className={cls(r.chg)}>{pct(r.chg)}</td>
                <td className={metric === "rvol" && r.rvol >= 2 ? "amber" : ""}>{metric === "turnover" ? fx(r.turnover, 0) : `${fx(r.rvol, 1)}x`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Movers() {
  const m = useLive((s) => s.market);
  return (
    <section style={{ gridArea: "movers", minHeight: 0 }} className="stack">
      {!m ? <Panel title="Movers"><Empty>Waiting…</Empty></Panel> : (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gridTemplateRows: "1fr 1fr", gap: 6, height: "100%", minHeight: 0 }}>
          <MoverTable title="▲ Top gainers" rows={m.movers.gainers} metric="chg" />
          <MoverTable title="▼ Top losers" rows={m.movers.losers} metric="chg" />
          <MoverTable title="Most active (turnover)" rows={m.movers.active} metric="turnover" />
          <MoverTable title="Volume shockers (RVOL)" rows={m.movers.rvol} metric="rvol" />
        </div>
      )}
    </section>
  );
}

export default function MarketDesk() {
  return (
    <div className="desk desk-market">
      <BreadthPanel />
      <Heatmap />
      <Movers />
    </div>
  );
}
