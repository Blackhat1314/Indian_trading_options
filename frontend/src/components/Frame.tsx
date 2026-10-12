import { useEffect, useMemo, useRef, useState } from "react";
import { fx, pct, cls, timeIST } from "../lib/format";
import { useLive, type DeskId } from "../lib/store";
import { Icon } from "./ui";

export const DESKS: { id: DeskId; key: string; label: string; cmd: string }[] = [
  { id: "intraday", key: "F1", label: "INTRADAY", cmd: "TOP5" },
  { id: "market", key: "F2", label: "MARKET", cmd: "MKT" },
  { id: "fno", key: "F3", label: "F&O", cmd: "FNO" },
  { id: "swing", key: "F4", label: "SWING", cmd: "SWING" },
  { id: "news", key: "F5", label: "NEWS", cmd: "NEWS" },
  { id: "alerts", key: "F6", label: "ALERTS", cmd: "ALRT" },
  { id: "system", key: "F7", label: "SYSTEM", cmd: "SYS" },
  { id: "geo", key: "F8", label: "GEO", cmd: "GEO" },
  { id: "admin", key: "F9", label: "ADMIN", cmd: "ADMIN" },
];

/** The ADMIN desk only shows where the gateway allows it (see ADMIN_KEY in .env). */
export function useDesks() {
  const mode = useLive((s) => s.adminMode);
  return DESKS.filter((d) => d.id !== "admin" || mode === "open" || mode === "key");
}

const PHASE_LABEL: Record<string, string> = {
  OPEN: "MARKET OPEN", PRE_OPEN: "PRE-OPEN", PRE_OPEN_MATCH: "PRE-OPEN", POST_CLOSE: "POST-CLOSE", CLOSED: "CLOSED",
};

function CommandBar() {
  const universe = useLive((s) => s.universe);
  const desks = useDesks();
  const { select, setDesk, setChain, setHelp } = useLive.getState();
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [hi, setHi] = useState(0);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA");
      if ((e.key === "/" && !typing) || (e.key.toLowerCase() === "k" && (e.ctrlKey || e.metaKey))) {
        e.preventDefault();
        input.current?.focus();
        input.current?.select();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const matches = useMemo(() => {
    const s = q.trim().toUpperCase();
    if (!s) return [];
    const fns = desks.filter((d) => d.cmd.startsWith(s) || d.label.startsWith(s)).map((d) => ({ kind: "fn" as const, text: d.cmd, sub: `${d.key} · ${d.label} desk`, desk: d.id }));
    const syms = universe
      .filter((i) => i.kind !== "FUT" && i.kind !== "MACRO" && (i.symbol.startsWith(s) || (s.length > 2 && i.symbol.includes(s))))
      .slice(0, 8)
      .map((i) => ({ kind: "sym" as const, text: i.symbol, sub: i.kind === "IDX" ? "INDEX" : i.sector, desk: undefined }));
    const extra = s.startsWith("CHAIN") ? [{ kind: "fn" as const, text: s.includes(" ") ? s : "CHAIN NIFTY", sub: "option chain", desk: "fno" as DeskId }] : [];
    const help = "HELP".startsWith(s) ? [{ kind: "fn" as const, text: "HELP", sub: "keyboard & commands", desk: undefined }] : [];
    return [...extra, ...fns, ...syms, ...help].slice(0, 10);
  }, [q, universe, desks]);

  const run = (text: string) => {
    const s = text.trim().toUpperCase();
    if (!s) return;
    const fn = desks.find((d) => d.cmd === s || d.label === s);
    if (fn) setDesk(fn.id);
    else if (s === "HELP" || s === "?") setHelp(true);
    else if (s.startsWith("CHAIN")) {
      const name = s.split(/\s+/)[1] ?? "NIFTY";
      setChain(name);
      setDesk("fno");
    } else {
      const inst = universe.find((i) => i.symbol === s) ?? universe.find((i) => i.symbol.startsWith(s) && i.kind !== "FUT");
      if (inst) {
        if (inst.kind === "IDX") {
          setChain(inst.symbol);
          setDesk("fno");
        } else {
          select(inst.symbol);
          const d = useLive.getState().desk;
          if (!["intraday", "swing"].includes(d)) setDesk("intraday");
        }
      }
    }
    setQ("");
    setOpen(false);
    input.current?.blur();
  };

  return (
    <div className="cmd-wrap">
      <form className="cmd" role="search" onSubmit={(e) => { e.preventDefault(); run(matches[hi]?.text ?? q); }}>
        <span className="cmd-prompt" aria-hidden="true">&gt;</span>
        <label htmlFor="cmd" className="sr-only">Command or symbol</label>
        <input id="cmd" ref={input} value={q} autoComplete="off" spellCheck={false}
          placeholder="Symbol or function (TOP5, FNO, SWING, CHAIN BANKNIFTY) — press /"
          onChange={(e) => { setQ(e.target.value); setOpen(true); setHi(0); }}
          onFocus={() => setOpen(true)} onBlur={() => window.setTimeout(() => setOpen(false), 120)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setHi((h) => Math.min(matches.length - 1, h + 1)); }
            if (e.key === "ArrowUp") { e.preventDefault(); setHi((h) => Math.max(0, h - 1)); }
            if (e.key === "Escape") { setQ(""); input.current?.blur(); }
          }}
          aria-autocomplete="list" aria-controls="cmd-list" aria-expanded={open && matches.length > 0} role="combobox" />
        <button className="cmd-go" type="submit">GO</button>
      </form>
      {open && matches.length > 0 && (
        <div className="suggest" id="cmd-list" role="listbox">
          {matches.map((m, i) => (
            <button key={m.kind + m.text} type="button" role="option" aria-selected={i === hi} className="suggest-item"
              onMouseDown={(e) => { e.preventDefault(); run(m.text); }}>
              <span>{m.kind === "fn" ? <span className="amber">{m.text}</span> : m.text}</span>
              <span className="muted tiny">{m.sub}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** Live geo-risk 0–100 from the GEO worker; click to open the GEO desk. */
function GeoRiskChip() {
  const risk = useLive((s) => s.geo["geo:risk"] as { value: number; level: string; drivers: string[] } | undefined);
  const setDesk = useLive((s) => s.setDesk);
  if (!risk) return null;
  const tone = risk.level === "SEVERE" ? "down" : risk.level === "HIGH" ? "amber" : risk.level === "ELEVATED" ? "warn" : "up";
  return (
    <button type="button" className={`geo-chip ${tone}`} onClick={() => setDesk("geo")}
      title={`Geo-risk ${risk.value.toFixed(0)}/100 (${risk.level}) — main drivers: ${risk.drivers.join(", ")}. Open the GEO desk (F8).`}>
      GEO RISK <b>{risk.value.toFixed(0)}</b> <span>{risk.level}</span>
    </button>
  );
}

export function TopBar() {
  const market = useLive((s) => s.market);
  const clock = useLive((s) => s.clock);
  const conn = useLive((s) => s.conn);
  const source = useLive((s) => s.source);
  const latency = useLive((s) => s.latencyMs);
  const setHelp = useLive((s) => s.setHelp);
  const order = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "INDIAVIX"];
  const indices = [...(market?.indices ?? [])].sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark">TD</span>
        <span>TRADEDESK<span className="brand-sub"> · NSE TERMINAL</span></span>
      </div>
      <CommandBar />
      <div className="tickers" aria-label="Indices">
        {indices.map((i) => (
          <div className="tick" key={i.name} title={`O ${fx(i.o)} · H ${fx(i.h)} · L ${fx(i.l)} · prev ${fx(i.pc)}`}>
            <span className="tick-name">{i.name === "INDIAVIX" ? "INDIA VIX" : i.name}</span>
            <span className="tick-row">
              <span className="tick-val">{fx(i.ltp)}</span>
              <span className={`tick-chg ${cls(i.chg)}`}>{pct(i.chg)}</span>
            </span>
          </div>
        ))}
      </div>
      <div className="top-right">
        <GeoRiskChip />
        {source && <span className={`src-badge ${source}`} title={source === "sim" ? "Simulated market — set DATA_SOURCE=dhan for live data" : "Live Dhan data"}>{source === "sim" ? "SIMULATED" : "DHAN LIVE"}</span>}
        {clock && <span className={`phase ${clock.phase}`}>{PHASE_LABEL[clock.phase] ?? clock.phase}</span>}
        <span className="clock" aria-label="Market time IST">{clock ? timeIST(clock.ts, true) : "--:--:--"}</span>
        <span className="conn" title={conn === "open" ? "Connected to gateway" : conn === "denied" ? "Access key required" : "Reconnecting"}>
          <span className={`dot ${conn === "open" ? "on" : conn === "connecting" ? "wait" : "off"}`} />
          <span>{conn === "open" ? (latency !== null ? `${latency}ms` : "LIVE") : conn.toUpperCase()}</span>
        </span>
        <button type="button" className="btn icon-btn ghost" onClick={() => setHelp(true)} aria-label="Help and shortcuts"><Icon name="help" /></button>
      </div>
    </header>
  );
}

export function Tabs() {
  const desk = useLive((s) => s.desk);
  const setDesk = useLive((s) => s.setDesk);
  const unread = useLive((s) => s.unreadAlerts);
  const desks = useDesks();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const idx = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"].indexOf(e.key);
      const alt = e.altKey && /^[1-9]$/.test(e.key) ? Number(e.key) - 1 : -1;
      const i = idx >= 0 ? idx : alt;
      if (i >= 0 && i < desks.length) {
        e.preventDefault();
        setDesk(desks[i].id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setDesk, desks]);
  return (
    <nav className="tabs" role="tablist" aria-label="Desks">
      {desks.map((d) => (
        <button key={d.id} role="tab" type="button" className="tab" aria-selected={desk === d.id} onClick={() => setDesk(d.id)}>
          <span className="fkey">{d.key}</span>
          {d.label}
          {d.id === "alerts" && unread > 0 && <span className="tab-badge">{unread}</span>}
        </button>
      ))}
    </nav>
  );
}

export function StatusBar() {
  const clock = useLive((s) => s.clock);
  const rank = useLive((s) => s.rank);
  const nEq = useLive((s) => s.universe.filter((i) => i.kind === "EQ").length);
  const user = useLive((s) => s.user);
  const top = useLive((s) => s.top);
  const b = useLive((s) => s.market?.breadth);
  const lastMsg = useLive((s) => s.lastMsg);
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(id);
  }, []);
  const stale = lastMsg && now - lastMsg > 5;
  return (
    <footer className="statusbar" aria-label="Status">
      <span>USER <b>{user || "—"}</b></span><span className="sep">│</span>
      <span>DATE <b>{clock?.date ?? "—"}</b></span><span className="sep">│</span>
      <span>UNIVERSE <b>{nEq}</b> F&O stocks · ranked <b className={rank?.rows.length ? "" : "down"}>{rank?.rows.length ?? 0}</b></span><span className="sep">│</span>
      {b && <><span>A/D <b className="up">{b.adv}</b>/<b className="down">{b.dec}</b></span><span className="sep">│</span></>}
      <span>PICKS <b>{top?.locked ? `LOCKED @ ${top.lock_time}` : top ? `provisional · locks ${top.lock_time}` : "—"}</b></span>
      <span className="sep">│</span>
      <span className={stale ? "down" : ""}>{stale ? `STALE ${Math.round(now - lastMsg)}s` : "FEED OK"}</span>
      <span style={{ marginLeft: "auto" }}>Research tool · not investment advice</span>
    </footer>
  );
}

export function Toasts() {
  const toasts = useLive((s) => s.toasts);
  const dismiss = useLive((s) => s.dismiss);
  return (
    <div className="toasts" aria-live="polite" aria-atomic="false">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`} role="status">
          <div><b>{t.title}</b>{t.body}</div>
          <button type="button" className="btn icon-btn ghost" onClick={() => dismiss(t.id)} aria-label="Dismiss"><Icon name="x" size={12} /></button>
        </div>
      ))}
    </div>
  );
}

export function HelpModal() {
  const open = useLive((s) => s.helpOpen);
  const setHelp = useLive((s) => s.setHelp);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setHelp(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setHelp]);
  if (!open) return null;
  return (
    <div className="modal-back" onClick={() => setHelp(false)}>
      <div className="panel modal" role="dialog" aria-modal="true" aria-labelledby="help-t" onClick={(e) => e.stopPropagation()}>
        <header className="panel-head">
          <span className="panel-code">HELP</span><h2 className="panel-title" id="help-t" style={{ margin: 0 }}>Keyboard & commands</h2>
          <div className="panel-actions"><button className="btn icon-btn ghost" onClick={() => setHelp(false)} aria-label="Close"><Icon name="x" /></button></div>
        </header>
        <dl className="help-grid">
          <dt><kbd>/</kbd> or <kbd>Ctrl</kbd>+<kbd>K</kbd></dt><dd>Focus the command line</dd>
          <dt><kbd>F1</kbd>…<kbd>F9</kbd> / <kbd>Alt</kbd>+<kbd>1</kbd>…<kbd>9</kbd></dt><dd>Switch desk (F9 ADMIN where enabled)</dd>
          <dt>RELIANCE <kbd>GO</kbd></dt><dd>Load a symbol in the chart & depth</dd>
          <dt>TOP5 · MKT · FNO · SWING · NEWS · ALRT · SYS · GEO · ADMIN</dt><dd>Open a desk</dd>
          <dt>CHAIN BANKNIFTY</dt><dd>Option chain for an underlying</dd>
          <dt>Click a row</dt><dd>Select the stock everywhere</dd>
        </dl>
        <div className="banner" style={{ borderTop: "1px solid var(--border)" }}>
          Picks are a transparent factor ranking for research. The SCORECARD shows how they actually performed — check it before trusting them. Not investment advice.
        </div>
      </div>
    </div>
  );
}
