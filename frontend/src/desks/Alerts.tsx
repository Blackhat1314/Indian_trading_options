import { useCallback, useEffect, useState } from "react";
import { Empty, Icon, Panel } from "../components/ui";
import { api } from "../lib/api";
import { pct, px, timeIST } from "../lib/format";
import { useLive } from "../lib/store";
import type { Alert, AlertFire, Pick } from "../lib/types";

const TYPE_INFO: Record<string, { label: string; value?: string; placeholder?: string }> = {
  PRICE_ABOVE: { label: "Price crosses above", value: "Price (₹)", placeholder: "1450" },
  PRICE_BELOW: { label: "Price crosses below", value: "Price (₹)", placeholder: "1380" },
  CHG_ABOVE: { label: "% change ≥", value: "Percent", placeholder: "3" },
  CHG_BELOW: { label: "% change ≤", value: "Percent", placeholder: "-3" },
  RVOL_ABOVE: { label: "Relative volume ≥", value: "Multiple", placeholder: "2.5" },
  VWAP_CROSS_UP: { label: "Reclaims VWAP" },
  VWAP_CROSS_DOWN: { label: "Loses VWAP" },
  ORB_UP: { label: "Opening-range breakout ▲" },
  ORB_DOWN: { label: "Opening-range breakdown ▼" },
  OI_BUILDUP: { label: "Futures OI build-up type", value: "LB / SB / SC / LU", placeholder: "LB" },
  NEWS: { label: "New catalyst headline" },
  TOP_PICK: { label: "Enters intraday top list" },
};

type NewAlert = { symbol: string; type: string; value: number | string | null; note: string; repeat: boolean };

/** One-click alerts on the locked picks' own plan levels (stop / target / VWAP). */
function QuickAlerts({ existing, onMade }: { existing: Alert[]; onMade: () => void }) {
  const top = useLive((s) => s.top);
  const [busy, setBusy] = useState(false);
  if (!top?.locked) return null;
  const rows = [
    ...top.long.map((p) => ({ p, long: true })),
    ...top.short.map((p) => ({ p, long: false })),
  ];
  if (!rows.length) return null;
  const has = (a: NewAlert) => existing.some((x) => x.active && x.symbol === a.symbol && x.type === a.type && (a.value === null || x.value === a.value));
  const plan = (p: Pick, long: boolean): { key: string; label: string; a: NewAlert }[] => [
    { key: "stop", label: `STOP ${px(p.invalidation)}`, a: { symbol: p.symbol, type: long ? "PRICE_BELOW" : "PRICE_ABOVE", value: p.invalidation, note: `${long ? "long" : "short"} pick invalidated`, repeat: false } },
    { key: "tgt", label: `TGT ${px(p.target)}`, a: { symbol: p.symbol, type: long ? "PRICE_ABOVE" : "PRICE_BELOW", value: p.target, note: `${long ? "long" : "short"} pick target`, repeat: false } },
    { key: "vwap", label: long ? "LOSES VWAP" : "RECLAIMS VWAP", a: { symbol: p.symbol, type: long ? "VWAP_CROSS_DOWN" : "VWAP_CROSS_UP", value: null, note: "pick lost its VWAP side", repeat: false } },
  ];
  const create = async (list: NewAlert[]) => {
    setBusy(true);
    try {
      for (const a of list.filter((x) => !has(x))) await api("/api/alerts", { method: "POST", body: JSON.stringify(a) });
      onMade();
    } finally {
      setBusy(false);
    }
  };
  const allStops = rows.map(({ p, long }) => plan(p, long)[0].a);
  return (
    <div style={{ borderTop: "1px solid var(--border)" }}>
      <div className="panel-bar" style={{ borderBottom: "1px solid var(--border)" }}>
        <span className="panel-title">Quick alerts · locked picks</span>
        <button type="button" className="btn grow" disabled={busy || allStops.every(has)} onClick={() => create(allStops)}>
          <Icon name="bell" size={13} /> Arm all stops
        </button>
      </div>
      <table className="tbl">
        <tbody>
          {rows.map(({ p, long }) => (
            <tr key={`${long}-${p.symbol}`} style={{ cursor: "default" }}>
              <td className="sym" style={{ width: 1 }}><span className={long ? "up" : "down"}>{long ? "▲" : "▼"}</span> {p.symbol}</td>
              <td className="l">
                <span className="chips">
                  {plan(p, long).map(({ key, label, a }) => {
                    const on = has(a);
                    return (
                      <button type="button" key={key} className={`chip ${on ? "up" : ""}`} disabled={busy || on} aria-pressed={on}
                        style={{ cursor: on ? "default" : "pointer", height: 22 }} onClick={() => create([a])}
                        title={on ? "Alert already armed" : `Alert when ${TYPE_INFO[a.type].label.toLowerCase()}${a.value !== null ? ` ${a.value}` : ""}`}>
                        {on ? "✓ " : "+ "}{label}
                      </button>
                    );
                  })}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function AlertsDesk() {
  const universe = useLive((s) => s.universe);
  const selected = useLive((s) => s.selected);
  const toasts = useLive((s) => s.toasts);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [log, setLog] = useState<AlertFire[]>([]);
  const [form, setForm] = useState({ symbol: selected || "", type: "PRICE_ABOVE", value: "", note: "", repeat: false });
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    api<{ alerts: Alert[]; log: AlertFire[] }>("/api/alerts").then((d) => {
      setAlerts(d.alerts);
      setLog(d.log);
    }).catch(() => {});
  }, []);
  useEffect(() => {
    load();
    const id = window.setInterval(load, 5000);
    return () => window.clearInterval(id);
  }, [load]);
  useEffect(() => {
    if (toasts.some((t) => t.kind === "alert")) load();
  }, [toasts, load]);

  const info = TYPE_INFO[form.type];
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    const sym = form.symbol.trim().toUpperCase();
    if (!universe.some((i) => i.symbol === sym)) return setErr(`Unknown symbol "${sym}" — pick one from the list`);
    if (info.value && form.value.trim() === "") return setErr(`${info.value} is required for this alert`);
    const value = info.value ? (form.type === "OI_BUILDUP" ? form.value.trim().toUpperCase() : Number(form.value)) : null;
    if (typeof value === "number" && Number.isNaN(value)) return setErr("Value must be a number");
    setBusy(true);
    try {
      await api("/api/alerts", { method: "POST", body: JSON.stringify({ symbol: sym, type: form.type, value, note: form.note, repeat: form.repeat }) });
      setForm((f) => ({ ...f, value: "", note: "" }));
      load();
    } catch (x) {
      setErr(String((x as Error).message));
    } finally {
      setBusy(false);
    }
  };
  const remove = async (id: string) => {
    await api(`/api/alerts/${id}`, { method: "DELETE" });
    load();
  };
  return (
    <div className="desk desk-alerts">
      <Panel code="ALRT" title="Create alert" area="make" sub="evaluated every second on the live stream — data only, no orders">
        <form className="form" onSubmit={submit} noValidate>
          <div className="field">
            <label htmlFor="al-sym">Symbol</label>
            <input id="al-sym" list="al-syms" value={form.symbol} onChange={(e) => setForm({ ...form, symbol: e.target.value.toUpperCase() })} placeholder="RELIANCE" required />
            <datalist id="al-syms">{universe.filter((i) => i.kind === "EQ").map((i) => <option key={i.key} value={i.symbol} />)}</datalist>
          </div>
          <div className="field">
            <label htmlFor="al-type">Condition</label>
            <select id="al-type" value={form.type} onChange={(e) => setForm({ ...form, type: e.target.value, value: "" })}>
              {Object.entries(TYPE_INFO).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
            </select>
          </div>
          {info.value && (
            <div className="field">
              <label htmlFor="al-val">{info.value}</label>
              <input id="al-val" value={form.value} inputMode={form.type === "OI_BUILDUP" ? "text" : "decimal"} placeholder={info.placeholder}
                onChange={(e) => setForm({ ...form, value: e.target.value })} aria-describedby={err ? "al-err" : undefined} />
            </div>
          )}
          <div className="field">
            <label htmlFor="al-note">Note (optional)</label>
            <input id="al-note" value={form.note} onChange={(e) => setForm({ ...form, note: e.target.value })} placeholder="why this matters" />
          </div>
          <label className="check"><input type="checkbox" checked={form.repeat} onChange={(e) => setForm({ ...form, repeat: e.target.checked })} /> Re-arm after firing</label>
          <div className="field">
            <button type="submit" className="btn primary" disabled={busy} style={{ height: 30 }}><Icon name="plus" /> {busy ? "Saving…" : "Create alert"}</button>
          </div>
          {err && <div className="field" style={{ gridColumn: "1 / -1" }}><span className="err" id="al-err" role="alert">{err}</span></div>}
        </form>
        <QuickAlerts existing={alerts} onMade={load} />
        <div style={{ borderTop: "1px solid var(--border)" }}>
          <div className="panel-bar" style={{ borderBottom: "1px solid var(--border)" }}>
            <span className="panel-title">Your alerts</span>
            <span className="tiny muted mono grow">{alerts.filter((a) => a.active).length} active · {alerts.length} total</span>
          </div>
          {alerts.length === 0 ? <Empty>No alerts yet</Empty> : (
            <table className="tbl">
              <thead><tr><th className="l">Symbol</th><th className="l">Condition</th><th>Value</th><th className="l">Note</th><th>Fired</th><th>Status</th><th /></tr></thead>
              <tbody>
                {alerts.map((a) => (
                  <tr key={a.id} style={{ cursor: "default" }}>
                    <td className="sym">{a.symbol}</td>
                    <td className="l">{TYPE_INFO[a.type]?.label ?? a.type}</td>
                    <td>{a.value ?? "—"}</td>
                    <td className="l muted ellipsis" style={{ maxWidth: 160 }}>{a.note || "—"}</td>
                    <td>{a.fired}</td>
                    <td>{a.active ? <span className="chip up">ACTIVE</span> : <span className="chip">DONE</span>}</td>
                    <td><button type="button" className="btn icon-btn ghost danger" onClick={() => remove(a.id)} aria-label={`Delete alert ${a.symbol}`}><Icon name="trash" size={13} /></button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </Panel>
      <Panel code="LOG" title="Fired alerts" area="log" sub="latest 100">
        {log.length === 0 ? <Empty>Nothing has fired yet</Empty> : (
          <table className="tbl">
            <thead><tr><th className="l">Time</th><th className="l">Symbol</th><th className="l">Message</th><th>LTP</th><th>Chg</th></tr></thead>
            <tbody>
              {log.map((f, i) => (
                <tr key={i} style={{ cursor: "default" }}>
                  <td className="l muted">{timeIST(f.ts, true)}</td>
                  <td className="sym">{f.symbol}</td>
                  <td className="l" style={{ whiteSpace: "normal", fontFamily: "var(--font-ui)" }}>{f.msg}</td>
                  <td>{px(f.ltp)}</td>
                  <td>{pct(f.chg)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </div>
  );
}
