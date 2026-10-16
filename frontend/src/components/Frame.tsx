import { useEffect, useMemo, useRef, useState } from "react";
import { fx, pct, cls, timeIST } from "../lib/format";
import { logout } from "../lib/auth";
import { useLive, type DeskId } from "../lib/store";
import { Icon } from "./ui";

export const DESKS: { id: DeskId; key: string; label: string; cmd: string }[] = [
  { id: "intraday", key: "F1", label: "INTRADAY", cmd: "TOP5" },
  { id: "market", key: "F2", label: "MARKET", cmd: "MKT" },
  { id: "fno", key: "F3", label: "F&O", cmd: "FNO" },
  { id: "swing", key: "F4", label: "SWING", cmd: "SWING" },
  { id: "news", key: "F5", label: "NEWS", cmd: "NEWS" },
  { id: "alerts", key: "F6", label: "ALERTS", cmd: "ALRT" },
  { id: "geo", key: "F7", label: "GEO", cmd: "GEO" },
  // research desks: any NSE stock, daily history + fundamentals (not the live feed)
  { id: "ai", key: "F8", label: "AI ANALYST", cmd: "AI" },
  { id: "screener", key: "F9", label: "SCREENER", cmd: "SCRN" },
  { id: "montecarlo", key: "F10", label: "MONTE CARLO", cmd: "MC" },
  { id: "quant", key: "Alt+Q", label: "QUANT", cmd: "QUANT" },
  { id: "paper", key: "Alt+P", label: "PAPER TRADE", cmd: "PAPER" },
];
// SYSTEM and the history pulls live on the separate admin page (/admin), for admins only.

const PHASE_LABEL: Record<string, string> = {
  OPEN: "MARKET OPEN", PRE_OPEN: "PRE-OPEN", PRE_OPEN_MATCH: "PRE-OPEN", POST_CLOSE: "POST-CLOSE", CLOSED: "CLOSED",
};

function CommandBar() {
  const universe = useLive((s) => s.universe);
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
    const fns = DESKS.filter((d) => d.cmd.startsWith(s) || d.label.startsWith(s)).map((d) => ({ kind: "fn" as const, text: d.cmd, sub: `${d.key} · ${d.label} desk`, desk: d.id }));
    const syms = universe
      .filter((i) => i.kind !== "FUT" && i.kind !== "MACRO" && (i.symbol.startsWith(s) || (s.length > 2 && i.symbol.includes(s))))
      .slice(0, 8)
      .map((i) => ({ kind: "sym" as const, text: i.symbol, sub: i.kind === "IDX" ? "INDEX" : i.sector, desk: undefined }));
    const extra = s.startsWith("CHAIN") ? [{ kind: "fn" as const, text: s.includes(" ") ? s : "CHAIN NIFTY", sub: "option chain", desk: "fno" as DeskId }] : [];
    const help = "HELP".startsWith(s) ? [{ kind: "fn" as const, text: "HELP", sub: "keyboard & commands", desk: undefined }] : [];
    return [...extra, ...fns, ...syms, ...help].slice(0, 10);
  }, [q, universe]);

  const run = (text: string) => {
    const s = text.trim().toUpperCase();
    if (!s) return;
    const fn = DESKS.find((d) => d.cmd === s || d.label === s);
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
          if (["ai", "screener", "montecarlo", "quant"].includes(d)) useLive.getState().setResearchSymbol(inst.symbol);
          else if (!["intraday", "swing"].includes(d)) setDesk("intraday");
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
      title={`Geo-risk ${risk.value.toFixed(0)}/100 (${risk.level}) — main drivers: ${risk.drivers.join(", ")}. Open the GEO desk (F7).`}>
      GEO RISK <b>{risk.value.toFixed(0)}</b> <span>{risk.level}</span>
    </button>
  );
}

/** Who is logged in, the admin page (admins) and log out. Nothing in local mode (no accounts). */
function UserMenu() {
  const me = useLive((s) => s.me);
  if (!me || !me.user) return null;
  if (me.via === "local") {
    // no accounts yet: the admin page leads to the first-admin setup (on the host) or explains .env
    return <a className="btn admin-link" href="/admin" title="Set up accounts and open the admin page">ADMIN</a>;
  }
  return (
    <div className="user-menu">
      <span className="user-chip" title={me.via === "key" ? "Signed in with an access key" : `Signed in as ${me.user} (${me.role})`}>
        {me.user}{me.role === "admin" ? <b> · admin</b> : null}
      </span>
      {me.admin && <a className="btn admin-link" href="/admin" title="Users, history pulls, system health">ADMIN</a>}
      <button type="button" className="btn ghost" onClick={() => void logout()}>{me.via === "key" ? "Forget key" : "Log out"}</button>
    </div>
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
        <UserMenu />
        <button type="button" className="btn icon-btn ghost" onClick={() => setHelp(true)} aria-label="Help and shortcuts"><Icon name="help" /></button>
      </div>
    </header>
  );
}

export function Tabs() {
  const desk = useLive((s) => s.desk);
  const setDesk = useLive((s) => s.setDesk);
  const unread = useLive((s) => s.unreadAlerts);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // F-keys, Alt+1…9 / Alt+0 (10th desk), Alt+Q and Alt+P; e.code, because Alt+key types other characters on macOS
      const t = e.target as HTMLElement | null;
      const typing = !!t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA");
      if (typing && !/^F\d+$/.test(e.key)) return;          // Alt+letter / Alt+digit type characters in fields
      const idx = DESKS.findIndex((d) => d.key === e.key || (e.altKey && d.key === `Alt+${e.code.replace(/^Key/, "")}`));
      const digit = /^Digit[0-9]$/.test(e.code) ? Number(e.code.slice(5)) : -1;
      const alt = e.altKey && digit >= 0 ? (digit === 0 ? 9 : digit - 1) : -1;
      const i = idx >= 0 ? idx : alt;
      if (i >= 0 && i < DESKS.length) {
        e.preventDefault();
        setDesk(DESKS[i].id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setDesk]);
  return (
    <nav className="tabs" role="tablist" aria-label="Desks">
      {DESKS.map((d) => (
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
          <dt><kbd>F1</kbd>…<kbd>F10</kbd> / <kbd>Alt</kbd>+<kbd>1</kbd>…<kbd>0</kbd></dt><dd>Switch desk</dd>
          <dt><kbd>Alt</kbd>+<kbd>Q</kbd></dt><dd>QUANT desk</dd>
          <dt><kbd>Alt</kbd>+<kbd>P</kbd></dt><dd>PAPER TRADE desk (sub-views: <kbd>1</kbd>…<kbd>5</kbd>)</dd>
          <dt>RELIANCE <kbd>GO</kbd></dt><dd>Load a symbol in the chart & depth</dd>
          <dt>TOP5 · MKT · FNO · SWING · NEWS · ALRT · GEO · AI · SCRN · MC · QUANT · PAPER</dt><dd>Open a desk</dd>
          <dt>ADMIN (top right)</dt><dd>Users, history pulls and system health — admins only</dd>
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
