/* Basic PRACTICE: the paper-trading account in plain words — virtual money, real prices, real charges. */
import { useState } from "react";
import { paperApi } from "../lib/paper";
import type { PaperState } from "../lib/paper";
import { useLive } from "../lib/store";
import { Card, Change, Empty, errText, ok, rupees, useLoad } from "./ui";

function Book({ title, sub, acc, enabled, onToggle, busy }: {
  title: string; sub: string; acc: { capital: number; equity: number; pnl_total: number; positions: unknown[]; stats: { trades: number; win_rate: number | null; worst: number | null } };
  enabled: boolean; onToggle: () => void; busy: boolean;
}) {
  const ret = acc.capital ? (acc.equity / acc.capital - 1) * 100 : 0;
  return (
    <Card title={title} sub={sub} action={
      <button type="button" className={`b-btn ${enabled ? "ghost" : ""}`} disabled={busy} onClick={onToggle}>{enabled ? "Pause" : "Start practising"}</button>}>
      <div className="b-stats">
        <div><span>Virtual money</span><b>{rupees(acc.equity, true)}</b><small>started with {rupees(acc.capital, true)}</small></div>
        <div><span>Total result</span><b className={acc.pnl_total >= 0 ? "b-up" : "b-down"}>{acc.pnl_total >= 0 ? "+" : "−"}{rupees(Math.abs(acc.pnl_total), true)}</b><small><Change v={ret} /></small></div>
        <div><span>Trades so far</span><b>{acc.stats.trades}</b><small>{ok(acc.stats.win_rate) ? `${Math.round(acc.stats.win_rate * (acc.stats.win_rate <= 1 ? 100 : 1))}% were winners` : "none closed yet"}</small></div>
        <div><span>Open now</span><b>{acc.positions.length}</b><small>{ok(acc.stats.worst) ? `worst trade ${rupees(acc.stats.worst, true)}` : "—"}</small></div>
      </div>
      {!enabled && <p className="b-note">Paused — no new practice trades are taken.</p>}
    </Card>
  );
}

export default function Practice() {
  const st = useLoad(() => paperApi.state(), []);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const setMode = useLive((s) => s.setMode);
  const setDesk = useLive((s) => s.setDesk);
  const s: PaperState | null = st.data;
  const toggle = (k: "swing" | "intraday") => {
    if (!s) return;
    setBusy(true);
    setErr(null);
    paperApi.settings({ [k]: { enabled: !s.settings[k].enabled } }).then(() => st.reload()).catch((e) => setErr(errText(e))).finally(() => setBusy(false));
  };
  return (
    <div className="b-practice">
      <div className="b-intro">
        <h1>Practice with virtual money</h1>
        <p>The system trades a virtual account on real prices, with India's real charges and slippage, so you can watch how ideas play out — wins, losses and all — before risking a rupee. Nothing here places a real order.</p>
      </div>
      <ol className="b-steps">
        <li><b>Start with swing practice.</b> Fewer, slower trades held for days — easier to follow and learn from than intraday.</li>
        <li><b>Watch for a few weeks.</b> Note how often trades lose, and how big the worst ones are. That is what real trading feels like.</li>
        <li><b>Only then consider real money</b> — small, with a stop on every trade and 1% risk per idea.</li>
      </ol>
      {err && <p className="b-note warn">{err}</p>}
      {st.error && !s ? <Empty>Practice account unavailable: {errText(st.error)}</Empty> : !s ? <Empty>Loading your practice account…</Empty> : (
        <div className="b-grid two">
          <Book title="Swing practice" sub="Holds for days to weeks, buys only" acc={s.swing} enabled={s.settings.swing.enabled} busy={busy} onToggle={() => toggle("swing")} />
          <Book title="Intraday practice" sub="In and out the same day, long or short" acc={s.intraday} enabled={s.settings.intraday.enabled} busy={busy} onToggle={() => toggle("intraday")} />
        </div>
      )}
      <p className="b-note">Want every trade, its reasons and the charges? <button type="button" className="b-inline" onClick={() => { setMode("pro"); setDesk("paper"); }}>Open the full practice desk</button> in the Pro view.</p>
    </div>
  );
}
