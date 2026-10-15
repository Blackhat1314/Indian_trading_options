/* Research symbol picker — one box shared by the AI ANALYST, SCREENER, MONTE CARLO and QUANT desks.
 * Searches any NSE stock through the gateway (F&O universe + Dhan's scrip master); picking sets the shared, persisted
 * `researchSymbol`, so switching desks keeps the stock. It also shows what daily history the gateway holds for the
 * stock (bars, years, last bar, spot) because every model's reliability depends on that.
 *
 * Usage:
 *   <div className="desk desk-mc rs-desk">
 *     <SymbolPicker area="pick" actions={<button className="btn primary" onClick={run}>RUN</button>} />
 *     ...
 *   const sym = useLive((s) => s.researchSymbol);
 *   const hist = useHistoryInfo(sym);          // same cached request the picker makes — no duplicate fetch
 */
import { useCallback, useEffect, useId, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { fx } from "../../lib/format";
import { research } from "../../lib/research";
import type { HistoryInfo, SearchHit } from "../../lib/research";
import { getPref, setPref } from "../../lib/storage";
import { useLive } from "../../lib/store";
import { Icon, Spark } from "../ui";
import { inr } from "./charts";
import { errorText } from "./common";
import "./research.css";

/** Symbols the gateway accepts (research_api validates the same pattern). */
export const SYMBOL_RE = /^[A-Z0-9&._-]{1,40}$/;

// ------------------------------------------------------------------ history (shared, cached)
const HIST_TTL = 2 * 60_000;           // short: HistoryInfo carries the spot price
const histCache = new Map<string, { at: number; p: Promise<HistoryInfo> }>();

function loadHistory(symbol: string, fetch: boolean, force: boolean): Promise<HistoryInfo> {
  const key = `${symbol}|${fetch ? 1 : 0}`;
  const hit = histCache.get(key);
  if (!force && hit && Date.now() - hit.at < HIST_TTL) return hit.p;
  const p = research.history(symbol, fetch);
  histCache.set(key, { at: Date.now(), p });
  p.catch(() => {
    if (histCache.get(key)?.p === p) histCache.delete(key);     // never cache failures
  });
  return p;
}

export interface HistoryState {
  data: HistoryInfo | null;      // null while loading the first time or after an error
  error: string | null;
  loading: boolean;
  reload: () => void;            // bypass the cache
}

/** HistoryInfo for a symbol; requests are shared and cached for 2 minutes across desks and components.
 *  fetch = let the gateway pull missing daily history from Dhan (dhan mode, budget permitting). */
export function useHistoryInfo(symbol: string, fetch = true): HistoryState {
  const [state, setState] = useState<{ key: string; data: HistoryInfo | null; error: string | null; loading: boolean }>(
    { key: "", data: null, error: null, loading: false });
  const [nonce, setNonce] = useState(0);
  const force = useRef(false);
  useEffect(() => {
    if (!symbol) return;
    const key = `${symbol}|${fetch}`;
    let alive = true;
    setState((s) => ({ key, data: s.key === key ? s.data : null, error: null, loading: true }));
    const f = force.current;
    force.current = false;
    loadHistory(symbol, fetch, f).then(
      (data) => { if (alive) setState({ key, data, error: null, loading: false }); },
      (e) => { if (alive) setState({ key, data: null, error: errorText(e), loading: false }); },
    );
    return () => { alive = false; };
  }, [symbol, fetch, nonce]);
  const reload = useCallback(() => { force.current = true; setNonce((n) => n + 1); }, []);
  const current = state.key === `${symbol}|${fetch}`;
  return { data: current ? state.data : null, error: current ? state.error : null, loading: !!symbol && (!current || state.loading), reload };
}

// ------------------------------------------------------------------ recent symbols (per viewer)
const RECENT_KEY = "td.rsym.recent";
function readRecent(): string[] {
  try {
    const v = JSON.parse(getPref(RECENT_KEY) ?? "[]");
    return Array.isArray(v) ? v.filter((s) => typeof s === "string" && SYMBOL_RE.test(s)).slice(0, 8) : [];
  } catch {
    return [];
  }
}
function pushRecent(sym: string): string[] {
  const list = [sym, ...readRecent().filter((s) => s !== sym)].slice(0, 8);
  setPref(RECENT_KEY, JSON.stringify(list));
  return list;
}

// ------------------------------------------------------------------ picker
export interface SymbolPickerProps {
  area?: string;                 // CSS grid-area when the picker is a direct child of the desk grid
  actions?: ReactNode;           // desk controls at the right end of the first row (RUN button, horizon Seg…)
  history?: boolean;             // show the daily-history line (default true)
  fetchHistory?: boolean;        // let the gateway pull missing history from Dhan (default true)
  placeholder?: string;
}

export function SymbolPicker({ area, actions, history = true, fetchHistory = true, placeholder }: SymbolPickerProps) {
  const symbol = useLive((s) => s.researchSymbol);
  const setResearchSymbol = useLive((s) => s.setResearchSymbol);
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [hitsFor, setHitsFor] = useState("");
  const [searchErr, setSearchErr] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [hi, setHi] = useState(0);
  const [recent, setRecent] = useState<string[]>(readRecent);
  const [inputErr, setInputErr] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const id = useId();
  const listId = `${id}-list`;
  const hist = useHistoryInfo(symbol, fetchHistory);

  // remember stocks that actually resolved (also when picked from the command bar), never typos
  const loaded = hist.data?.symbol;
  useEffect(() => { if (loaded) setRecent(pushRecent(loaded)); }, [loaded]);

  const term = q.trim();
  useEffect(() => {
    if (!term) return;
    let alive = true;
    const t = window.setTimeout(() => {
      research.search(term, 12).then(
        (r) => { if (alive) { setHits(r); setHitsFor(term); setHi(0); setSearchErr(null); } },
        (e) => { if (alive) { setHits([]); setHitsFor(term); setSearchErr(errorText(e)); } },
      );
    }, 200);
    return () => { alive = false; window.clearTimeout(t); };
  }, [term]);

  const shown = term && hitsFor === term ? hits : [];
  const searching = !!term && hitsFor !== term;

  const pick = (raw: string) => {
    const s = raw.trim().toUpperCase();
    if (!s) return;
    if (!SYMBOL_RE.test(s)) {
      setInputErr("Use the NSE symbol: letters, digits and & . _ - only");
      return;
    }
    setInputErr(null);
    setResearchSymbol(s);
    setQ("");
    setOpen(false);
    input.current?.blur();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setHi((h) => Math.min(Math.max(0, shown.length - 1), h + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHi((h) => Math.max(0, h - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      pick(open && shown[hi] ? shown[hi].symbol : q);
    } else if (e.key === "Escape") {
      if (open && term) setOpen(false);
      else { setQ(""); input.current?.blur(); }
    }
  };

  const showList = open && !!term;
  const activeId = showList && shown[hi] ? `${id}-opt-${hi}` : undefined;
  const d = hist.data;
  const closes = d?.closes ?? [];
  const spark = closes.slice(-250).map((c) => c[1]);

  return (
    <div className="rs-picker" style={area ? { gridArea: area } : undefined} role="region" aria-label="Research symbol">
      <div className="rs-row">
        <div className="rs-search">
          <div className="rs-box">
            <Icon name="search" size={13} />
            <label htmlFor={`${id}-in`} className="sr-only">Search any NSE stock</label>
            <input id={`${id}-in`} ref={input} value={q} autoComplete="off" spellCheck={false}
              placeholder={placeholder ?? "Any NSE stock — name or symbol"}
              role="combobox" aria-autocomplete="list" aria-expanded={showList} aria-controls={listId} aria-activedescendant={activeId}
              onChange={(e) => { setQ(e.target.value); setOpen(true); setHi(0); setInputErr(null); }}
              onFocus={() => setOpen(true)} onBlur={() => window.setTimeout(() => setOpen(false), 120)}
              onKeyDown={onKeyDown} />
          </div>
          {showList && (
            <ul className="rs-suggest" id={listId} role="listbox" aria-label="Matching stocks">
              {shown.map((h, i) => (
                <li key={h.symbol} id={`${id}-opt-${i}`} role="option" aria-selected={i === hi} className="rs-opt"
                  onMouseDown={(e) => { e.preventDefault(); pick(h.symbol); }} onMouseEnter={() => setHi(i)}>
                  <span className="ellipsis"><span className="rs-opt-sym">{h.symbol}</span><span className="rs-opt-name">{h.name}</span></span>
                  <span className="rs-opt-tags">
                    {h.sector && <span>{h.sector}</span>}
                    {!h.has_history && <span className="chip amber" title="No daily bars stored yet — the first run pulls them from Dhan (dhan mode)">no history yet</span>}
                  </span>
                </li>
              ))}
              {searching && !shown.length && <li className="rs-opt-note" role="presentation">Searching…</li>}
              {!searching && searchErr && <li className="rs-opt-note err" role="presentation">{searchErr}</li>}
              {!searching && !searchErr && !shown.length && (
                <li className="rs-opt-note" role="presentation">No match — press Enter to try “{term.toUpperCase()}” anyway.</li>
              )}
            </ul>
          )}
        </div>
        <div className="rs-current" title={d ? `${d.name}${d.sector ? ` · ${d.sector}` : ""}` : symbol}>
          <span className="rs-sym">{symbol || "—"}</span>
          {d?.name && d.name !== symbol && <span className="rs-name">{d.name}</span>}
          {d?.sector && <span className="chip">{d.sector}</span>}
        </div>
        {d && d.spot !== null && (
          <span className="rs-spot" title={d.spot_source === "live" ? "Live price" : "Last daily close"}>
            {inr(d.spot)}<small>{d.spot_source === "live" ? "live" : "last close"}</small>
          </span>
        )}
        {spark.length > 1 && <span title="Daily closes, last year" aria-hidden="true"><Spark data={spark} width={84} height={22} /></span>}
        {recent.filter((s) => s !== symbol).length > 0 && (
          <div className="rs-recent" role="group" aria-label="Recent symbols">
            {recent.filter((s) => s !== symbol).slice(0, 5).map((s) => (
              <button key={s} type="button" className="chip" onClick={() => pick(s)} title={`Research ${s}`}>{s}</button>
            ))}
          </div>
        )}
        {actions && <div className="rs-actions">{actions}</div>}
      </div>
      {(history || inputErr) && (
        <div className="rs-meta" aria-live="polite">
          {inputErr && <span className="err">{inputErr}</span>}
          {history && <HistoryLine state={hist} />}
        </div>
      )}
    </div>
  );
}

function HistoryLine({ state }: { state: HistoryState }) {
  const d = state.data;
  if (state.error) {
    return (
      <>
        <span className="err">{state.error}</span>
        <button type="button" className="btn" onClick={state.reload}>Retry</button>
      </>
    );
  }
  if (!d) return <span>{state.loading ? "Loading daily history… (a first look at a stock may pull it from Dhan)" : ""}</span>;
  return (
    <>
      <span><b>{fx(d.bars, 0)}</b> daily bars</span>
      <span><b>{fx(d.years, 1)}</b> years</span>
      {d.first && d.last && <span>{d.first} → <b>{d.last}</b></span>}
      {d.fetched && <span className="chip cyan" title="Missing daily bars were just pulled from Dhan into the data lake">updated from Dhan</span>}
      {d.bars === 0 && !d.warning && <span className="warn">⚠ no daily history stored for this stock</span>}
      {d.warning && <span className="warn">⚠ {d.warning}</span>}
      {state.loading && <span>refreshing…</span>}
    </>
  );
}
