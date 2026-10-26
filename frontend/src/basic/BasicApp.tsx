/* BASIC mode — the simplified view for beginners: plain words, today's ideas, one page per stock, practice and a
 * glossary. Same live data and models as the terminal (PRO), a light and calm layout. Switch back any time. */
import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { logout } from "../lib/auth";
import { research } from "../lib/research";
import type { SearchHit } from "../lib/research";
import { useLive, type BasicTab } from "../lib/store";
import { timeIST } from "../lib/format";
import "./basic.css";

const Home = lazy(() => import("./Home"));
const Ideas = lazy(() => import("./Ideas"));
const StockPage = lazy(() => import("./StockPage"));
const Practice = lazy(() => import("./Practice"));
const Learn = lazy(() => import("./Learn"));

const NAV: { id: BasicTab; label: string }[] = [
  { id: "home", label: "Home" },
  { id: "ideas", label: "Ideas" },
  { id: "stock", label: "Stock" },
  { id: "practice", label: "Practice" },
  { id: "learn", label: "Learn" },
];

const PHASE: Record<string, string> = {
  OPEN: "Market open", PRE_OPEN: "Pre-open", PRE_OPEN_MATCH: "Pre-open", POST_CLOSE: "Market closed", CLOSED: "Market closed",
};

function Search() {
  const universe = useLive((s) => s.universe);
  const openStock = useLive((s) => s.openStock);
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [open, setOpen] = useState(false);
  const [hi, setHi] = useState(0);
  const box = useRef<HTMLInputElement>(null);
  const quick = useMemo(() => {
    const s = q.trim().toUpperCase();
    if (!s) return [];
    return universe.filter((i) => i.kind === "EQ" && i.symbol.startsWith(s)).slice(0, 5)
      .map((i) => ({ symbol: i.symbol, name: i.symbol, sector: i.sector, in_universe: true, has_history: true }));
  }, [q, universe]);
  useEffect(() => {
    const s = q.trim();
    if (s.length < 2) { setHits([]); return; }
    const t = window.setTimeout(() => { research.search(s, 8).then(setHits).catch(() => setHits([])); }, 200);
    return () => window.clearTimeout(t);
  }, [q]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.key === "/" && !(t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA"))) { e.preventDefault(); box.current?.focus(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const list = useMemo(() => {
    const seen = new Set<string>();
    return [...quick, ...hits].filter((h) => (seen.has(h.symbol) ? false : (seen.add(h.symbol), true))).slice(0, 8);
  }, [quick, hits]);
  const go = (sym: string) => { openStock(sym); setQ(""); setOpen(false); box.current?.blur(); };
  return (
    <div className="b-search" role="combobox" aria-expanded={open && list.length > 0} aria-haspopup="listbox" aria-owns="b-search-list">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
        <circle cx="11" cy="11" r="7" /><path d="M20 20l-3.5-3.5" />
      </svg>
      <input ref={box} value={q} placeholder="Search any NSE stock — name or symbol" aria-label="Search a stock"
        aria-autocomplete="list" aria-controls="b-search-list"
        onChange={(e) => { setQ(e.target.value); setOpen(true); setHi(0); }}
        onFocus={() => setOpen(true)} onBlur={() => window.setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") { e.preventDefault(); setHi((h) => Math.min(h + 1, list.length - 1)); }
          else if (e.key === "ArrowUp") { e.preventDefault(); setHi((h) => Math.max(h - 1, 0)); }
          else if (e.key === "Enter") { const h = list[hi] ?? null; if (h) go(h.symbol); else if (q.trim()) go(q.trim().toUpperCase()); }
          else if (e.key === "Escape") { setOpen(false); box.current?.blur(); }
        }} />
      {open && list.length > 0 && (
        <ul className="b-search-list" id="b-search-list" role="listbox">
          {list.map((h, i) => (
            <li key={h.symbol} role="option" aria-selected={i === hi}>
              <button type="button" className={i === hi ? "on" : undefined} onMouseDown={(e) => e.preventDefault()} onClick={() => go(h.symbol)}>
                <b>{h.symbol}</b><span>{h.name !== h.symbol ? h.name : ""}</span><small>{h.sector}</small>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function BasicApp() {
  const tab = useLive((s) => s.basicTab);
  const setTab = useLive((s) => s.setBasicTab);
  const setMode = useLive((s) => s.setMode);
  const clock = useLive((s) => s.clock);
  const conn = useLive((s) => s.conn);
  const me = useLive((s) => s.me);
  useEffect(() => { document.documentElement.classList.add("basic-root"); return () => document.documentElement.classList.remove("basic-root"); }, []);
  return (
    <div className="basic">
      <header className="b-top">
        <div className="b-top-row">
          <button type="button" className="b-brand" onClick={() => setTab("home")} aria-label="TradeDesk home">
            <span className="b-brand-mark">TD</span><span>TradeDesk</span>
          </button>
          <Search />
          <div className="b-top-right">
            {clock && (
              <span className={`b-phase ${clock.phase === "OPEN" ? "open" : ""}`} title="Market time (IST)">
                <i aria-hidden="true" />{PHASE[clock.phase] ?? clock.phase} · {timeIST(clock.ts)}
              </span>
            )}
            <button type="button" className="b-btn ghost" onClick={() => setMode("pro")} title="The full trading terminal">Pro view</button>
            {me?.user && (
              <button type="button" className="b-btn ghost" onClick={() => void logout()}>{me.via === "key" ? "Forget key" : "Log out"}</button>
            )}
          </div>
        </div>
        <nav className="b-nav" aria-label="Sections">
          {NAV.map((n) => (
            <button type="button" key={n.id} className={tab === n.id ? "on" : undefined} aria-current={tab === n.id ? "page" : undefined}
              onClick={() => setTab(n.id)}>{n.label}</button>
          ))}
        </nav>
      </header>
      {conn === "closed" && <div className="b-banner" role="status">Reconnecting to live prices…</div>}
      <main className="b-main">
        <Suspense fallback={<div className="b-empty">Loading…</div>}>
          {tab === "home" && <Home />}
          {tab === "ideas" && <Ideas />}
          {tab === "stock" && <StockPage />}
          {tab === "practice" && <Practice />}
          {tab === "learn" && <Learn />}
        </Suspense>
      </main>
      <footer className="b-foot">
        A research and learning tool, not investment advice. Ideas are filters and model outputs with their track record —
        they can be wrong, and past results do not promise future ones. No orders are placed from this app.
      </footer>
    </div>
  );
}
