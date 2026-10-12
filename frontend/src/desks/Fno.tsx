import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, RefObject } from "react";
import { Empty, Kpi, Panel, Seg, Spark } from "../components/ui";
import { api } from "../lib/api";
import { BUILDUP, cls, compact, fx, pct, px, signed, timeIST } from "../lib/format";
import { live } from "../lib/socket";
import { useLive } from "../lib/store";
import type { ChainSnap, FnoRow } from "../lib/types";

function BuildTable({ tag, rows, count }: { tag: "LB" | "SB" | "SC" | "LU"; rows: FnoRow[]; count: number }) {
  const select = useLive((s) => s.select);
  const setDesk = useLive((s) => s.setDesk);
  const b = BUILDUP[tag];
  return (
    <section className="panel">
      <header className="panel-head">
        <span className={`chip ${b.cls}`}>{tag}</span>
        <span className="panel-title">{b.long}</span>
        <span className="panel-sub">{tag === "LB" ? "price ▲ OI ▲" : tag === "SB" ? "price ▼ OI ▲" : tag === "SC" ? "price ▲ OI ▼" : "price ▼ OI ▼"}</span>
        <span className="panel-actions mono muted">{count} stocks</span>
      </header>
      <div className="panel-body">
        {rows.length === 0 ? <Empty>None</Empty> : (
          <table className="tbl">
            <thead><tr><th className="l">Future</th><th>LTP</th><th>Chg</th><th title="Open-interest change vs yesterday">ΔOI</th><th title="Open interest value, ₹ crore">OI ₹Cr</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.sid} onClick={() => { select(r.symbol); setDesk("intraday"); }} title={`${r.symbol} · ${r.sector} — open on the intraday desk`}>
                  <td className="sym">{r.symbol}</td>
                  <td>{px(r.ltp)}</td>
                  <td className={cls(r.chg)}>{pct(r.chg)}</td>
                  <td className={cls(r.oi_chg, 0.05)}>{signed(r.oi_chg, 1)}%</td>
                  <td className="muted">{fx(r.oi_value_cr, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

function Buildup() {
  const fno = useLive((s) => s.fno);
  return (
    <Panel code="OI" title="Futures OI build-up" area="build" sub="stock futures · price vs open-interest change since yesterday">
      {!fno ? <Empty>Waiting…</Empty> : (
        <div className="quad">
          <BuildTable tag="LB" rows={fno.groups.LB} count={fno.counts.LB} />
          <BuildTable tag="SC" rows={fno.groups.SC} count={fno.counts.SC} />
          <BuildTable tag="SB" rows={fno.groups.SB} count={fno.counts.SB} />
          <BuildTable tag="LU" rows={fno.groups.LU} count={fno.counts.LU} />
        </div>
      )}
    </Panel>
  );
}

/** Track an element's width so SVG charts draw at real pixel size (text never stretches). */
function useWidth<T extends HTMLElement>(): [RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

function GexChart({ c }: { c: ChainSnap }) {
  const [box, W] = useWidth<HTMLDivElement>();
  const H = 104;
  const rows = c.gex;
  const max = Math.max(1, ...rows.map((r) => Math.abs(r.gex)));
  const bw = rows.length ? W / rows.length : 0;
  const xOf = (k: number) => {
    const i = rows.findIndex((r) => r.strike >= k);
    return (i < 0 ? rows.length - 1 : i) * bw + bw / 2;
  };
  const sx = xOf(c.spot);
  const fx0 = c.gamma_flip ? xOf(c.gamma_flip) : null;
  return (
    <div ref={box} style={{ width: "100%" }}>
      {rows.length > 0 && W > 0 && (
        <svg className="svg-chart" width={W} height={H + 16} role="img"
          aria-label="Gamma exposure by strike: bars above the line are call-dominated, below put-dominated">
          <line x1="0" x2={W} y1={H / 2} y2={H / 2} stroke="#2a3547" />
          {rows.map((r, i) => {
            const h = (Math.abs(r.gex) / max) * (H / 2 - 6);
            return <rect key={r.strike} x={i * bw + 1} width={Math.max(1, bw - 2)} y={r.gex >= 0 ? H / 2 - h : H / 2} height={h}
              fill={r.gex >= 0 ? "rgba(39,209,127,.7)" : "rgba(255,82,99,.7)"}><title>{`${r.strike}: ${compact(r.gex)}`}</title></rect>;
          })}
          <line className="spot-line" x1={sx} x2={sx} y1="0" y2={H} />
          <text x={sx > W - 90 ? sx - 4 : sx + 4} y="10" textAnchor={sx > W - 90 ? "end" : "start"} style={{ fill: "var(--amber)" }}>spot {fx(c.spot, 0)}</text>
          {fx0 !== null && (
            <>
              <line x1={fx0} x2={fx0} y1={H / 2 - 8} y2={H / 2 + 8} stroke="var(--violet)" />
              <text x={fx0 > W - 90 ? fx0 - 4 : fx0 + 4} y={H - 3} textAnchor={fx0 > W - 90 ? "end" : "start"} style={{ fill: "var(--violet)" }}>γ-flip {c.gamma_flip}</text>
            </>
          )}
          <text x="2" y={H + 13}>{rows[0].strike}</text>
          <text x={W - 2} y={H + 13} textAnchor="end">{rows[rows.length - 1].strike}</text>
        </svg>
      )}
    </div>
  );
}

function Chain() {
  const chainName = useLive((s) => s.chainName);
  const setChain = useLive((s) => s.setChain);
  const c = useLive((s) => s.chains[chainName]);
  const [names, setNames] = useState<string[]>(["NIFTY", "BANKNIFTY", "FINNIFTY"]);
  useEffect(() => {
    api<{ chains: string[] }>("/api/meta").then((m) => setNames(m.chains)).catch(() => {});
  }, []);
  useEffect(() => {
    live.sub([`chain:${chainName}`]);
  }, [chainName]);
  const maxOi = useMemo(() => Math.max(1, ...(c?.rows ?? []).flatMap((r) => [r.ce_oi, r.pe_oi])), [c]);
  // bring the ATM strike into view whenever the underlying (or its ATM) changes
  const atmRow = useRef<HTMLTableRowElement>(null);
  const atmStrike = c?.atm;
  useEffect(() => {
    const row = atmRow.current;
    const body = row?.closest(".chain-scroll");
    if (!row || !body) return;
    // scroll only the chain panel (scrollIntoView would also move the desk grid)
    body.scrollTop += row.getBoundingClientRect().top - body.getBoundingClientRect().top - body.clientHeight / 2;
  }, [chainName, atmStrike]);
  const pcrSeries = (c?.pcr_hist ?? []).map((p) => p[1]).filter((x): x is number => x !== null);
  return (
    <Panel code="OMON" title="Option chain" area="chain"
      sub={c ? `${c.name} · expiry ${c.expiry ?? "—"} · updated ${c.ts ? timeIST(c.ts, true) : "—"}` : chainName}
      actions={<Seg label="Underlying" value={chainName} onChange={setChain} options={names.map((n) => ({ v: n, label: n }))} />}>
      {!c ? <Empty>Loading {chainName} chain… (Dhan allows one refresh per underlying every 3 s)</Empty> : (
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
          <div className="kpis" style={{ "--kpi-min": "170px" } as CSSProperties}>
            <Kpi label="Spot" value={fx(c.spot)} sub={`ATM ${c.atm}`} />
            <Kpi label="PCR (OI)" value={fx(c.pcr_oi)} tone={c.pcr_oi && c.pcr_oi > 1 ? "up" : "down"} sub={<span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>vol {fx(c.pcr_vol)} <Spark data={pcrSeries.slice(-60)} width={50} height={14} color="var(--amber)" /></span>} />
            <Kpi label="Max pain" value={c.max_pain ?? "—"} sub={`${signed(c.max_pain ? ((c.max_pain / c.spot) - 1) * 100 : null)}% vs spot`} />
            <Kpi label="ATM IV" value={`${fx(c.atm_iv)}%`} sub={`25Δ skew ${signed(c.skew_25d)}`} />
            <Kpi label="Expected move" value={`±${fx(c.expected_move_pct)}%`} sub={`straddle ${fx(c.straddle)}`} />
            <Kpi label="Walls" value={<><span className="down">{c.call_wall}</span> / <span className="up">{c.put_wall}</span></>} sub="call / put max OI" />
            <Kpi label="Net GEX" value={compact(c.total_gex)} tone={c.total_gex >= 0 ? "up" : "down"} sub={c.gamma_flip ? `flip ≈ ${c.gamma_flip}` : "no flip in range"} />
            <Kpi label="OI change" value={<><span className="down">{compact(c.ce_oi_chg_total)}</span> / <span className="up">{compact(c.pe_oi_chg_total)}</span></>} sub="calls / puts" />
          </div>
          <div style={{ padding: "6px 8px", borderBottom: "1px solid var(--border)", flexShrink: 0 }}>
            <div className="tiny muted mono" style={{ marginBottom: 2 }}>GAMMA EXPOSURE BY STRIKE (call γ·OI − put γ·OI)</div>
            <GexChart c={c} />
          </div>
          <div className="chain-scroll" style={{ flex: 1, minHeight: 180, overflow: "auto" }}>
          <table className="tbl chain-tbl" style={{ fontSize: 12 }}>
            <thead>
              <tr>
                <th>OI</th><th>Chg OI</th><th>IV</th><th>Δ</th><th>LTP</th>
                <th className="center" style={{ color: "var(--amber)" }}>STRIKE</th>
                <th className="l">LTP</th><th className="l">Δ</th><th className="l">IV</th><th className="l">Chg OI</th><th className="l">OI</th>
              </tr>
            </thead>
            <tbody>
              {c.rows.map((r) => {
                const itmCall = r.strike < c.spot;
                const atm = r.strike === c.atm;
                return (
                  <tr key={r.strike} ref={atm ? atmRow : undefined} style={atm ? { boxShadow: "inset 0 1px 0 var(--amber), inset 0 -1px 0 var(--amber)" } : undefined}>
                    <td style={{ position: "relative", background: itmCall ? "rgba(255,176,0,.04)" : undefined }}>
                      <span style={{ position: "absolute", right: 0, top: 3, bottom: 3, width: `${(r.ce_oi / maxOi) * 100}%`, background: "rgba(255,82,99,.16)" }} />
                      <span style={{ position: "relative" }}>{compact(r.ce_oi)}</span>
                    </td>
                    <td className={cls(r.ce_oi_chg)}>{compact(r.ce_oi_chg)}</td>
                    <td className="muted">{fx(r.ce_iv, 1)}</td>
                    <td className="muted">{fx(r.ce_delta, 2)}</td>
                    <td>{fx(r.ce_ltp)}</td>
                    <td className="center b" style={{ color: atm ? "var(--amber)" : "var(--text)", background: "#0f1520" }}>{r.strike}</td>
                    <td className="l">{fx(r.pe_ltp)}</td>
                    <td className="l muted">{fx(r.pe_delta, 2)}</td>
                    <td className="l muted">{fx(r.pe_iv, 1)}</td>
                    <td className={`l ${cls(r.pe_oi_chg)}`}>{compact(r.pe_oi_chg)}</td>
                    <td className="l" style={{ position: "relative", background: !itmCall ? "rgba(255,176,0,.04)" : undefined }}>
                      <span style={{ position: "absolute", left: 0, top: 3, bottom: 3, width: `${(r.pe_oi / maxOi) * 100}%`, background: "rgba(39,209,127,.16)" }} />
                      <span style={{ position: "relative" }}>{compact(r.pe_oi)}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          </div>
        </div>
      )}
    </Panel>
  );
}

export default function FnoDesk() {
  return (
    <div className="desk desk-fno">
      <Buildup />
      <Chain />
    </div>
  );
}
