import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { getPref, setPref } from "../lib/storage";
import { PriceChart } from "../components/Chart";
import { Empty, Kpi, Panel, Seg } from "../components/ui";
import { cls, fx, pct, px, signed } from "../lib/format";
import { useLive } from "../lib/store";
import type { SwingRow } from "../lib/types";

const SETUPS = ["ALL", "BREAKOUT", "VCP", "PULLBACK", "MOMENTUM", "BREAKDOWN"] as const;
type SetupF = (typeof SETUPS)[number];

const SETUP_SHORT: Record<string, string> = { BREAKOUT: "BO", VCP: "VCP", PULLBACK: "PB", MOMENTUM: "MOM", BREAKDOWN: "BD" };

const SETUP_RULE: Record<string, string> = {
  BREAKOUT: "Close above the prior 20-day high on ≥1.5× average volume, trend score ≥ 2.",
  VCP: "Volatility contraction (10-day ATR < 0.75× 50-day ATR) within 5% of the 55-day high, uptrend.",
  PULLBACK: "Full uptrend (close > EMA20 > EMA50 > EMA200), low tags the 20 EMA, RSI 40–60.",
  MOMENTUM: "Full uptrend with RSI above 60 — trend continuation.",
  BREAKDOWN: "Close below the prior 20-day low on ≥1.5× volume in a weak trend (short setup).",
};

function setupChip(s: string) {
  const c = s === "BREAKDOWN" ? "down" : s === "BREAKOUT" ? "up" : s === "VCP" ? "violet" : s === "PULLBACK" ? "cyan" : "amber";
  return s ? <span className={`chip ${c}`}>{s}</span> : <span className="muted">—</span>;
}

function Scanner({ sel, onSel }: { sel: string; onSel: (s: string) => void }) {
  const swing = useLive((s) => s.swing);
  const [f, setF] = useState<SetupF>("ALL");
  const [onlySetups, setOnlySetups] = useState(true);
  const rows = useMemo(() => {
    let r = swing?.rows ?? [];
    if (onlySetups) r = r.filter((x) => x.setup);
    if (f !== "ALL") r = r.filter((x) => x.setup === f);
    return r;
  }, [swing, f, onlySetups]);
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const r of swing?.rows ?? []) if (r.setup) c[r.setup] = (c[r.setup] ?? 0) + 1;
    return c;
  }, [swing]);
  return (
    <Panel code="SWNG" title="Swing scanner" area="scan" className="cap-m"
      sub={swing ? `${rows.length} shown · daily bars + today's live bar · 3–10 day horizon` : ""}
      bar={
        <>
          <Seg label="Setup" value={f} onChange={setF}
            options={SETUPS.map((s) => ({
              v: s, title: s === "ALL" ? "All setups" : `${s}: ${SETUP_RULE[s]}`,
              label: s === "ALL" ? "ALL" : <>{SETUP_SHORT[s]}<span className="n">{counts[s] ?? 0}</span></>,
            }))} />
          <Seg label="Universe" value={onlySetups ? "setups" : "all"} onChange={(v) => setOnlySetups(v === "setups")}
            options={[{ v: "setups", label: "WITH SETUP" }, { v: "all", label: "ALL STOCKS" }]} />
        </>
      }>
      {!swing ? <Empty>Computing swing features from the data lake…</Empty> : (
        <table className="tbl">
          <thead>
            <tr>
              <th className="l">Symbol</th><th className="l">Setup</th><th>Score</th><th>Close</th><th>Today</th>
              <th title="Close>EMA20>EMA50>EMA200">Trend</th><th title="3-month return vs NIFTY, percentile">RS%ile</th>
              <th>RSI</th><th>ATR%</th><th title="Projected volume vs 20-day average">Vol×</th>
              <th title="Distance from 55-day high">55H</th><th className="l">SMC</th>
              <th title="Historical hit-rate of this setup across the universe">Hit%</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.symbol} className={r.symbol === sel ? "sel" : ""} onClick={() => onSel(r.symbol)}>
                <td className="sym">{r.symbol}<small>{r.sector}</small></td>
                <td className="l">{setupChip(r.setup)}</td>
                <td className="amber b">{fx(r.score, 0)}</td>
                <td>{px(r.close)}</td>
                <td className={cls(r.chg)}>{pct(r.chg)}</td>
                <td title={`trend score ${r.trend}/3`} aria-label={`trend ${r.trend} of 3`}>
                  <span className="up">{"●".repeat(r.trend)}</span><span className="muted">{"○".repeat(3 - r.trend)}</span>
                </td>
                <td className={r.rs_rank >= 70 ? "up" : r.rs_rank <= 30 ? "down" : ""}>{fx(r.rs_rank, 0)}</td>
                <td className={r.rsi >= 70 ? "amber" : r.rsi <= 30 ? "cyan" : ""}>{fx(r.rsi, 0)}</td>
                <td className="muted">{fx(r.atr_pct, 1)}</td>
                <td className={(r.vol_ratio ?? 0) >= 1.5 ? "amber b" : ""}>{fx(r.vol_ratio, 1)}</td>
                <td className={cls(r.dist_hh55)}>{signed(r.dist_hh55, 1)}%</td>
                <td className="l">{r.smc_event ? <span className={`chip ${r.smc_event_dir === 1 ? "up" : "down"}`}>{r.smc_event} {r.smc_event_dir === 1 ? "▲" : "▼"} <span className="muted">{r.smc_age}d</span></span> : <span className="muted">—</span>}</td>
                <td>{r.stats ? `${fx(r.stats.hit_rate, 0)}%` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

function Detail({ r }: { r: SwingRow | undefined }) {
  const swing = useLive((s) => s.swing);
  if (!r) return <Panel code="SETUP" title="Setup receipts" area="detail"><Empty>Select a stock</Empty></Panel>;
  const st = r.stats;
  const rules = swing?.rules;
  return (
    <Panel code="SETUP" title="Setup receipts" area="detail" sub={`${r.symbol} · ${r.setup || "no active setup"}`}>
      <div className="kpis" style={{ "--kpi-min": "100px" } as CSSProperties}>
        <Kpi label="Setup" value={r.setup || "—"} sub={r.setup ? "rule-based, no look-ahead" : "watch only"} />
        <Kpi label="Hit rate" value={st ? `${fx(st.hit_rate, 1)}%` : "—"}
          tone={st && st.hit_rate >= 40 ? "up" : undefined}
          sub={st ? `+${st.target_pct}% before −${st.stop_pct}% in ${st.horizon}d · n=${st.n}` : ""} />
        <Kpi label="Stopped first" value={st ? `${fx(st.stop_rate, 1)}%` : "—"} sub={st ? `avg ${st.horizon}d return ${signed(st.avg_fwd)}%` : ""} />
        <Kpi label="Entry" value={px(r.close)} sub="last close" />
        <Kpi label="Stop" value={px(r.stop)} tone="down"
          sub={r.stop ? `risk ${fx((Math.abs(r.close - r.stop) / r.close) * 100, 1)}%` : "—"} />
        <Kpi label="Target" value={px(r.target)} tone="up"
          sub={r.target ? `2R · ${signed(((r.target - r.close) / r.close) * 100, 1)}%` : "—"} />
      </div>
      <div style={{ padding: "8px 10px", fontSize: 12.5, color: "var(--text-2)" }}>
        {r.setup ? <p style={{ margin: "0 0 6px" }}><b className="amber">Rule:</b> {SETUP_RULE[r.setup]}</p> : null}
        <p style={{ margin: 0 }} className="muted small">
          Receipts are computed on the whole universe in the lake (target {rules?.target}% / stop {rules?.stop}% / {rules?.horizon} days;
          same-day touches count as losses). A setup with a hit-rate near the base rate has no edge — treat it as a filter, not a signal.
          {useLive.getState().source === "sim" && " Simulated data: these numbers only demonstrate the mechanism."}
        </p>
      </div>
    </Panel>
  );
}

export default function SwingDesk() {
  const swing = useLive((s) => s.swing);
  const [sel, setSel] = useState<string>(() => getPref("td.swingSel") ?? "");
  useEffect(() => {
    if (!sel && swing?.rows.length) setSel(swing.rows[0].symbol);
  }, [sel, swing]);
  const pick = (s: string) => {
    setSel(s);
    setPref("td.swingSel", s);
  };
  const row = swing?.rows.find((r) => r.symbol === sel);
  const extra = useMemo(() => (row?.stop ? { entry: row.close, stop: row.stop, target: row.target } : undefined),
    [row?.close, row?.stop, row?.target]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="desk desk-swing">
      <Scanner sel={sel} onSel={pick} />
      <Panel code="GP" title="Daily chart" area="chart" className="chart-panel" sub={row ? `${sel} · ${row.setup || "no setup"} · EMA20 ${px(row.e20)} · EMA50 ${px(row.e50)}` : sel}>
        {sel ? <PriceChart symbol={sel} mode="daily" defaultTf="D" extraLevels={extra} /> : <Empty>Select a stock</Empty>}
      </Panel>
      <Detail r={row} />
    </div>
  );
}
