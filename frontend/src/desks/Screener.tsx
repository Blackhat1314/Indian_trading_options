/* SCREENER desk — a company's fundamentals as Screener.in publishes them, plus a transparent checklist over them.
 * The gateway fetches the Screener.in page politely (one request at a time, cached for hours) and the AI analyst
 * reads the very same parsed numbers, so what you see here is what the analyst reasons over. Nothing is filled in:
 * a missing figure shows as "—" and every derived metric says how it was computed or why it is missing. The
 * scorecard is a list of simple published rules with each check's value, not a rating or a recommendation. */
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { BarChart, LineChart, inr } from "../components/research/charts";
import type { BarSeries, LineSeries, Tone } from "../components/research/charts";
import { Disclaimer, ErrorNote, Stat, Stats, Warnings, errorText } from "../components/research/common";
import { SYMBOL_RE, SymbolPicker } from "../components/research/SymbolPicker";
import { Empty, Icon, Panel, Seg } from "../components/ui";
import { ApiError } from "../lib/api";
import { ago, dateIST, fx, timeIST } from "../lib/format";
import { research } from "../lib/research";
import type { FinRow, FinTable, Fundamentals, ScorePart } from "../lib/research";
import { getPref, setPref } from "../lib/storage";
import { useLive } from "../lib/store";
import "./screener.css";

type View = "auto" | "consolidated" | "standalone";
const VIEWS: View[] = ["auto", "consolidated", "standalone"];
const VIEW_KEY = "td.scr.view";
const TAB_KEY = "td.scr.tab";

// ------------------------------------------------------------------ formatting
const ok = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const MINUS = "−";

/** Indian-grouped number with a real minus sign; integers (crore figures) without decimals. */
function num(v: number | null | undefined, d?: number): string {
  if (!ok(v)) return "—";
  const a = Math.abs(v);
  const s = fx(a, d ?? (Number.isInteger(v) || a >= 1000 ? 0 : 2));
  return v < 0 && /[1-9]/.test(s) ? MINUS + s : s;
}

/** Signed number for changes: "+6.4", "−24.6". */
function sgn(v: number | null | undefined, d = 1): string {
  if (!ok(v)) return "—";
  const s = num(v, d);
  return v > 0 && /[1-9]/.test(s) ? `+${s}` : s;
}

const pctText = (v: number | null | undefined, d?: number) => (ok(v) ? `${num(v, d ?? (Number.isInteger(v) ? 0 : 1))}%` : "—");
const tone = (v: number | null | undefined) => (ok(v) && v < 0 ? "down" : "");
const signTone = (v: number | null | undefined) => (!ok(v) ? "muted" : v > 0 ? "up" : v < 0 ? "down" : "");
const arrow = (v: number | null | undefined) => (!ok(v) || v === 0 ? "" : v > 0 ? "▲ " : "▼ ");

/** Only plain web links from the scraped page become hrefs (never javascript: or relative URLs). */
const safeUrl = (u: string | null | undefined) => (u && /^https?:\/\/[^\s]+$/i.test(u.trim()) ? u.trim() : null);

function unitShort(unit: string | null | undefined): string {
  if (!unit) return "";
  if (/crore/i.test(unit)) return "₹ Cr";
  return unit;
}

function fetchedText(ts: number): string {
  if (!ok(ts) || ts <= 0) return "—";
  return `${dateIST(ts)} ${timeIST(ts)} IST (${ago(ts, Date.now() / 1000)} ago)`;
}

function Ext({ href, children, title }: { href: string | null; children: ReactNode; title?: string }) {
  if (!href) return null;
  return <a className="scr-link" href={href} target="_blank" rel="noopener noreferrer" title={title}>{children}<span aria-hidden="true"> ↗</span></a>;
}

// ------------------------------------------------------------------ data (results survive desk switches)
const memo = new Map<string, { at: number; f: Fundamentals }>();
const MEMO_FRESH = 10 * 60_000;      // the gateway caches Screener pages for hours; re-ask it after 10 min

function remember(key: string, f: Fundamentals) {
  memo.delete(key);
  memo.set(key, { at: Date.now(), f });
  while (memo.size > 24) memo.delete(memo.keys().next().value as string);
}

interface FState { key: string; data: Fundamentals | null; error: unknown; loading: boolean }

function useFundamentals(symbol: string, view: View) {
  const key = `${symbol}|${view}`;
  const [st, setSt] = useState<FState>(() => ({ key, data: memo.get(key)?.f ?? null, error: null, loading: !!symbol && !memo.has(key) }));
  const seq = useRef(0);
  const load = useCallback((refresh: boolean) => {
    if (!symbol) return;
    const id = ++seq.current;
    setSt((s) => ({ key, data: s.key === key ? s.data : memo.get(key)?.f ?? null, error: null, loading: true }));
    research.screener(symbol, refresh, view).then(
      (f) => {
        remember(key, f);
        if (id === seq.current) setSt({ key, data: f, error: null, loading: false });
      },
      (e) => { if (id === seq.current) setSt((s) => ({ key, data: s.key === key ? s.data : null, error: e, loading: false })); },
    );
  }, [symbol, view, key]);

  useEffect(() => {
    if (!symbol) return;
    const m = memo.get(key);
    if (m && Date.now() - m.at < MEMO_FRESH) {
      seq.current++;
      setSt({ key, data: m.f, error: null, loading: false });
      return;
    }
    load(false);
  }, [key, symbol, load]);

  const current = st.key === key;
  return {
    data: current ? st.data : memo.get(key)?.f ?? null,
    error: current ? st.error : null,
    loading: !!symbol && (!current || st.loading),
    refresh: () => load(true),
    retry: () => load(false),
  };
}

function readView(): View {
  const v = getPref(VIEW_KEY);
  return VIEWS.includes(v as View) ? (v as View) : "auto";
}

// ------------------------------------------------------------------ table helpers
function findRow(t: FinTable | null | undefined, ...names: string[]): FinRow | null {
  if (!t) return null;
  for (const n of names) {
    const r = t.rows.find((x) => x.label.trim().toLowerCase() === n.toLowerCase());
    if (r) return r;
  }
  return null;
}

const valuesFor = (t: FinTable, r: FinRow | null, from: number) =>
  t.columns.slice(from).map((_, i) => (r && ok(r.values[from + i]) ? r.values[from + i] : null));

/** Sales (or a lender's Revenue) and net profit for the last `last` columns of a statement. */
function salesProfit(t: FinTable | null | undefined, last: number) {
  if (!t || !t.columns.length) return null;
  const sales = findRow(t, "Sales", "Revenue");
  const profit = findRow(t, "Net Profit");
  if (!sales && !profit) return null;
  const from = Math.max(0, t.columns.length - last);
  const series: BarSeries[] = [];
  if (sales) series.push({ name: sales.label, values: valuesFor(t, sales, from), tone: 1 });
  if (profit) series.push({ name: "Net profit", values: valuesFor(t, profit, from), tone: 2 });
  return { categories: t.columns.slice(from), series, salesName: sales?.label ?? "Sales" };
}

const KEY_ROWS = new Set([
  "sales", "revenue", "operating profit", "financing profit", "net profit", "eps in rs", "total liabilities",
  "total assets", "cash from operating activity", "free cash flow", "net cash flow", "promoters", "roce %", "roe %",
]);

function colTitle(c: string): string | undefined {
  if (/^ttm$/i.test(c.trim())) return "Trailing twelve months: the sum of the last four reported quarters";
  const m = /(\d{1,2})\s*m$/i.exec(c.trim());
  if (m) return `Changed financial year: this period covers ${m[1]} months`;
  return undefined;
}

// ------------------------------------------------------------------ tabs
interface TabOpt<T extends string> { v: T; label: string; empty?: boolean; title?: string }

function Tabs<T extends string>({ value, options, onChange, label, idBase }: {
  value: T; options: TabOpt<T>[]; onChange: (v: T) => void; label: string; idBase: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKey = (e: KeyboardEvent<HTMLButtonElement>, i: number) => {
    const n = options.length;
    const j = e.key === "ArrowRight" ? (i + 1) % n : e.key === "ArrowLeft" ? (i - 1 + n) % n
      : e.key === "Home" ? 0 : e.key === "End" ? n - 1 : -1;
    if (j < 0) return;
    e.preventDefault();
    onChange(options[j].v);
    refs.current[j]?.focus();
  };
  return (
    <div className="scr-tabs" role="tablist" aria-label={label}>
      {options.map((o, i) => (
        <button key={o.v} ref={(el) => { refs.current[i] = el; }} type="button" role="tab"
          id={`${idBase}-tab-${o.v}`} aria-controls={`${idBase}-panel`} aria-selected={value === o.v}
          tabIndex={value === o.v ? 0 : -1} className={o.empty ? "empty" : undefined}
          title={o.empty ? `${o.label}: not on the Screener.in page for this view` : o.title}
          onClick={() => onChange(o.v)} onKeyDown={(e) => onKey(e, i)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ company header + top ratios
function CompanyPanel({ f, view, error }: { f: Fundamentals; view: View; error: unknown }) {
  const [more, setMore] = useState(false);
  const site = safeUrl(f.website);
  const page = safeUrl(f.url) ?? "https://www.screener.in/";
  const nse = f.nse_code ? f.nse_code.toUpperCase() : null;
  const nseUrl = nse && SYMBOL_RE.test(nse) ? `https://www.nseindia.com/get-quotes/equity?symbol=${encodeURIComponent(nse)}` : null;
  const about = (f.about ?? "").trim();
  const long = about.length > 260;
  const badges = (
    <span className="scr-badges">
      <span className="chip" title={view === "auto" ? "AUTO: consolidated when Screener.in has it, else standalone" : undefined}>
        {(f.view ?? "—").toUpperCase()}{view === "auto" ? " · AUTO" : ""}
      </span>
      {f.stale ? (
        <span className="chip amber" title="Screener.in could not be reached; this is the last copy the gateway fetched">STALE</span>
      ) : f.cached ? (
        <span className="chip" title="Served from the gateway's disk cache; Refresh asks Screener.in again">CACHED</span>
      ) : (
        <span className="chip cyan" title="Fetched from Screener.in just now">FRESH</span>
      )}
    </span>
  );
  return (
    <Panel code="FUND" title="Company" area="co" id="scr-co" actions={badges}
      sub={`${f.symbol} · fetched ${fetchedText(f.fetched_at)}`}
      foot={
        <span>
          Data: <Ext href={page} title="Open this company on Screener.in">screener.in</Ext> — compiled from company
          filings by Screener.in; ₹ figures in crore unless marked. Fetched {fetchedText(f.fetched_at)}.
        </span>
      }>
      <div className="scr-head">
        <div className="scr-title">
          <h3 className="scr-name">{f.name || f.symbol}</h3>
          <span className="scr-codes">
            {nse && <span>NSE <b>{nse}</b></span>}
            {f.bse_code && <span>BSE <b>{f.bse_code}</b></span>}
          </span>
        </div>
        <div className="scr-tags">
          {f.sector && <span className="chip" title="Sector (Screener.in)">{f.sector}</span>}
          {f.industry && f.industry !== f.sector && <span className="chip violet" title="Industry (Screener.in)">{f.industry}</span>}
          <span className="scr-links">
            <Ext href={page} title="The Screener.in page these numbers come from">Screener.in</Ext>
            <Ext href={site} title="Company website (as listed on Screener.in)">Website</Ext>
            <Ext href={nseUrl} title="NSE quote page">NSE</Ext>
          </span>
        </div>
        {about && (
          <p className={`scr-about ${more || !long ? "open" : ""}`}>
            {about}
            {long && <button type="button" className="scr-more" onClick={() => setMore((m) => !m)} aria-expanded={more}>{more ? "less" : "more"}</button>}
          </p>
        )}
      </div>
      {error ? <ErrorNote error={error} /> : null}
      <RatioTiles f={f} />
      <div className="scr-pad"><Warnings items={f.warnings} label="Screener notes" /></div>
    </Panel>
  );
}

function RatioTiles({ f }: { f: Fundamentals }) {
  const ratios = f.ratios ?? [];
  const byKey = new Map(ratios.map((r) => [r.key, r]));
  const derived = new Map((f.derived ?? []).map((d) => [d.key, d]));
  const price = byKey.get("current_price")?.value;
  if (!ratios.length) return <p className="scr-note">Screener.in's top ratio box was not on the page.</p>;

  const valueText = (r: Fundamentals["ratios"][number]): string => {
    const v = r.value, u = (r.unit ?? "").trim();
    if (!ok(v)) return r.text?.trim() || "—";
    if (/cr/i.test(u)) return Math.abs(v) >= 1e5 ? `₹${num(v / 1e5, 2)} L Cr` : `₹${num(v, 0)} Cr`;   // lakh crore
    if (u.startsWith("₹")) return inr(v);
    if (u === "%") return pctText(v, 2);
    return num(v, Math.abs(v) >= 100 ? 0 : 2);
  };
  const sub = (key: string): ReactNode => {
    const ref = byKey.get(key)?.value;
    if ((key === "high" || key === "low") && ok(price) && ok(ref) && ref > 0) {
      const d = (price / ref - 1) * 100;
      return <span className={signTone(d)}>price {sgn(d)}%</span>;
    }
    if (key === "stock_pe") {
      const ey = derived.get("earnings_yield")?.value;
      return ok(ey) ? `earnings yield ${pctText(ey, 1)}` : undefined;
    }
    if (key === "book_value") {
      const pb = derived.get("price_to_book")?.value;
      return ok(pb) ? `P/B ${num(pb, 2)}×` : undefined;
    }
    return undefined;
  };
  return (
    <Stats min={150}>
      {ratios.map((r) => (
        <Stat key={r.key} label={r.label} value={valueText(r)} sub={sub(r.key)} tone={tone(r.value)}
          title={`${r.label}: ${r.text || valueText(r)} (Screener.in)`} />
      ))}
    </Stats>
  );
}

// ------------------------------------------------------------------ scorecard
const GRADE_TONE: Record<string, string> = { A: "up", B: "up", C: "amber", D: "down", E: "down" };
/** "₹ 79,020 Cr" never breaks inside a figure when a long check value wraps. */
const keepUnits = (s: string | null | undefined) => (s ?? "").replace(/₹ /g, "₹\u00a0").replace(/ (Cr|pp|%)(?=\b|$|,)/g, "\u00a0$1");

function Meter({ v, max, label }: { v: number; max: number; label: string }) {
  const w = max > 0 ? Math.max(0, Math.min(1, v / max)) * 100 : 0;
  return <span className="scr-meter" role="img" aria-label={`${label}: ${v} of ${max}`}><i style={{ width: `${w}%` }} /></span>;
}

function Scorecard({ f }: { f: Fundamentals }) {
  const sc = f.scorecard;
  const parts: ScorePart[] = sc?.parts ?? [];
  const checks = parts.reduce((n, p) => n + (p.checks?.length ?? 0), 0);
  const na = checks - (sc?.max ?? 0);
  const grade = sc?.grade ?? "N/A";
  return (
    <Panel code="SCORE" title="Scorecard" area="score" id="scr-score" sub={`${checks} rule checks on the page's numbers`}
      foot={
        <span>
          Grade = share of the checks that could be evaluated which passed: A ≥ 80 %, B ≥ 65 %, C ≥ 50 %, D ≥ 35 %, else
          E; N/A with fewer than 6. Checks with missing data or that don't apply (–, e.g. debt for a bank) are left out.
          A checklist, not a rating.
        </span>
      }>
      {!parts.length ? <Empty>No scorecard — Screener.in had too little data.</Empty> : (
        <>
          <div className="scr-grade-row">
            <div className={`scr-grade ${GRADE_TONE[grade] ?? "muted"}`} aria-label={`Grade ${grade}`}>{grade}</div>
            <div className="scr-grade-info">
              <div className="scr-total"><b>{sc.total}</b> of {sc.max} checks passed</div>
              <Meter v={sc.total} max={sc.max} label="Checks passed" />
              <div className="scr-na">{na > 0 ? `${na} not evaluated (missing data or not applicable)` : "every check had data"}</div>
            </div>
          </div>
          {parts.map((p) => (
            <section key={p.key} className="scr-part" aria-label={`${p.label}: ${p.score} of ${p.max}`}>
              <header>
                <h3>{p.label}</h3>
                <Meter v={p.score} max={p.max} label={p.label} />
                <span className="scr-pscore">{p.max ? `${p.score}/${p.max}` : "n/a"}</span>
              </header>
              <ul className="scr-checks">
                {(p.checks ?? []).map((c, i) => {
                  const state = c.pass === true ? "pass" : c.pass === false ? "fail" : "na";
                  return (
                    <li key={i} className={state}>
                      <span className="scr-glyph" role="img" aria-label={state === "pass" ? "passed" : state === "fail" ? "failed" : "not evaluated"}>
                        {state === "pass" ? "✓" : state === "fail" ? "✗" : "–"}
                      </span>
                      <span className="scr-clabel">{c.label}</span>
                      <span className="scr-cval">{keepUnits(c.value) || "—"}</span>
                      <span className="scr-crule">{c.rule}</span>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
        </>
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------ derived metrics
const DERIVED_GROUPS: { title: string; keys: string[] }[] = [
  { title: "Growth", keys: ["sales_cagr_3y", "sales_cagr_5y", "profit_cagr_3y", "profit_cagr_5y", "eps_cagr_3y", "qtr_sales_yoy", "qtr_profit_yoy"] },
  { title: "Margins & returns", keys: ["opm_latest", "opm_trend_3y", "roe", "roce", "roa"] },
  { title: "Balance sheet & cash", keys: ["debt_to_equity", "interest_coverage", "fcf_latest", "cash_conversion_3y", "gross_npa", "net_npa"] },
  { title: "Valuation", keys: ["earnings_yield", "peg", "price_to_book", "pe_vs_peers"] },
  { title: "Ownership & payout", keys: ["promoter_holding", "promoter_change_4q", "fii_dii_change_4q", "dividend_payout"] },
];
const CHANGE_KEY = /cagr|yoy|change|trend/;

type Derived = Fundamentals["derived"][number];

function derivedText(d: Derived): string {
  const v = d.value;
  if (!ok(v)) return "—";
  const change = CHANGE_KEY.test(d.key);
  switch (d.unit) {
    case "%": return change ? `${sgn(v, 1)}%` : pctText(v, 1);
    case "pp": return `${sgn(v, 2)} pp`;
    case "x": return `${num(v, 2)}×`;
    case "₹ Cr": return `₹${num(v, 0)} Cr`;
    default: return `${num(v, 2)}${d.unit ? ` ${d.unit}` : ""}`;
  }
}

function DerivedPanel({ f }: { f: Fundamentals }) {
  const groups = useMemo(() => {
    const list = f.derived ?? [];
    const used = new Set<string>();
    const out = DERIVED_GROUPS.map((g) => {
      const items = g.keys.map((k) => list.find((d) => d.key === k)).filter((d): d is Derived => !!d);
      items.forEach((d) => used.add(d.key));
      return { title: g.title, items };
    });
    out.push({ title: "Other", items: list.filter((d) => !used.has(d.key)) });
    return out.filter((g) => g.items.length);
  }, [f]);
  const missing = (f.derived ?? []).filter((d) => !ok(d.value)).length;
  return (
    <Panel code="DRV" title="Derived metrics" area="derived" id="scr-drv"
      sub={missing ? `${missing} of ${(f.derived ?? []).length} not computable — the note says why` : "computed from the statements below"}>
      {!groups.length ? <Empty>No metrics could be derived.</Empty> : (
        <table className="tbl scr-dtbl">
          <caption className="sr-only">Metrics derived from {f.name}'s Screener.in statements</caption>
          <thead><tr><th className="l" scope="col">Metric</th><th scope="col">Value</th><th className="l" scope="col">Basis</th></tr></thead>
          {groups.map((g) => (
            <tbody key={g.title}>
              <tr className="scr-grp"><td colSpan={3} className="l">{g.title}</td></tr>
              {g.items.map((d) => {
                const change = CHANGE_KEY.test(d.key) || d.key === "fcf_latest";
                return (
                  <tr key={d.key}>
                    <td className="l">{d.label}</td>
                    <td className={!ok(d.value) ? "muted" : change ? signTone(d.value) : tone(d.value)}>
                      {change && ok(d.value) ? arrow(d.value) : ""}{derivedText(d)}
                    </td>
                    <td className="l scr-basis">{d.note || ""}</td>
                  </tr>
                );
              })}
            </tbody>
          ))}
        </table>
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------ pros / cons / compounded growth
function ProsConsPanel({ f }: { f: Fundamentals }) {
  const pros = f.pros ?? [], cons = f.cons ?? [], growth = f.growth ?? [];
  return (
    <Panel code="P/C" title="Pros, cons & growth" area="pc" id="scr-pc" sub="Screener.in's automated read of the numbers">
      <div className="scr-pc">
        <section aria-label="Pros">
          <h3 className="up">▲ Pros</h3>
          {pros.length ? <ul>{pros.map((p, i) => <li key={i}>{p}</li>)}</ul> : <p className="scr-note">None listed.</p>}
        </section>
        <section aria-label="Cons">
          <h3 className="down">▼ Cons</h3>
          {cons.length ? <ul>{cons.map((c, i) => <li key={i}>{c}</li>)}</ul> : <p className="scr-note">None listed.</p>}
        </section>
      </div>
      {growth.length ? (
        <div className="scr-growth">
          {growth.map((g) => (
            <table key={g.title} className="scr-gbox">
              <caption>{g.title}</caption>
              <tbody>
                {(g.rows ?? []).map((r) => (
                  <tr key={r.period}>
                    <th scope="row">{r.period}</th>
                    <td className={tone(r.value)}>{pctText(r.value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ))}
        </div>
      ) : <p className="scr-note">Screener.in's compounded growth boxes were not on the page.</p>}
    </Panel>
  );
}

// ------------------------------------------------------------------ charts
function SalesProfitPanel({ f, kind }: { f: Fundamentals; kind: "q" | "a" }) {
  const t = kind === "q" ? f.quarters : f.profit_loss;
  const data = useMemo(() => salesProfit(t, 13), [t]);
  const d = useMemo(() => new Map((f.derived ?? []).map((x) => [x.key, x.value])), [f]);
  const title = kind === "q" ? "Quarterly sales vs net profit" : "Annual sales vs net profit";

  let caption: ReactNode = null;
  if (data && kind === "q") {
    const n = data.categories.length - 1;
    const s = data.series.find((x) => x.tone === 1)?.values[n];
    const p = data.series.find((x) => x.tone === 2)?.values[n];
    const margin = ok(s) && ok(p) && s > 0 ? (p / s) * 100 : null;
    caption = (
      <>
        ₹ crore. {data.categories[n]}: {data.salesName.toLowerCase()} YoY <span className={signTone(d.get("qtr_sales_yoy"))}>{sgn(d.get("qtr_sales_yoy"))}%</span>,
        net profit YoY <span className={signTone(d.get("qtr_profit_yoy"))}>{sgn(d.get("qtr_profit_yoy"))}%</span>
        {ok(margin) && <>, net margin {num(margin, 1)}%</>}.
      </>
    );
  } else if (data) {
    caption = (
      <>
        ₹ crore; TTM = last four quarters. {data.salesName} CAGR 3y {pctText(d.get("sales_cagr_3y"), 1)} · 5y {pctText(d.get("sales_cagr_5y"), 1)};
        profit CAGR 3y {pctText(d.get("profit_cagr_3y"), 1)} · 5y {pctText(d.get("profit_cagr_5y"), 1)}.
      </>
    );
  }
  return (
    <Panel code={kind === "q" ? "QTR" : "ANN"} title={title} area={kind === "q" ? "qbar" : "abar"} pad
      sub={t ? `${t.columns.length} ${kind === "q" ? "quarters" : "periods"} · ${f.view}` : undefined}>
      {data ? (
        <BarChart label={`${f.symbol}: ${title.toLowerCase()} (₹ crore)`} categories={data.categories} series={data.series}
          format="cr" height={220} caption={caption} />
      ) : (
        <Empty>{t ? "No sales or net profit row in this table." : `No ${kind === "q" ? "quarterly results" : "annual profit & loss"} on Screener.in for this view.`}</Empty>
      )}
    </Panel>
  );
}

const SH_SERIES: { re: RegExp; name: string; tone: Tone }[] = [
  { re: /^promoter/i, name: "Promoters", tone: 1 },
  { re: /^fii/i, name: "FIIs", tone: 2 },
  { re: /^dii/i, name: "DIIs", tone: 3 },
  { re: /^public/i, name: "Public", tone: "text" },
];

function ShareholdingPanel({ f }: { f: Fundamentals }) {
  const sh = f.shareholding ?? { quarterly: null, yearly: null };
  const [mode, setMode] = useState<"q" | "y">("q");
  const t = (mode === "q" ? sh.quarterly ?? sh.yearly : sh.yearly ?? sh.quarterly) ?? null;
  const shownMode = t === sh.quarterly ? "q" : "y";
  const series = useMemo<LineSeries[]>(() => {
    if (!t) return [];
    return SH_SERIES.flatMap((s) => {
      const r = t.rows.find((x) => s.re.test(x.label.trim()));
      return r ? [{ name: s.name, values: t.columns.map((_, i) => (ok(r.values[i]) ? r.values[i] : null)), tone: s.tone }] : [];
    });
  }, [t]);
  const d = new Map((f.derived ?? []).map((x) => [x.key, x.value]));
  const tools = sh.quarterly && sh.yearly ? (
    <Seg label="Shareholding period" value={shownMode} onChange={setMode}
      options={[{ v: "q", label: "QTR", title: "Quarter by quarter" }, { v: "y", label: "YR", title: "Year by year" }]} />
  ) : null;
  return (
    <Panel code="SHP" title="Shareholding trend" area="shp" pad sub={t ? `% of shares · ${t.columns.length} ${shownMode === "q" ? "quarters" : "periods"}` : undefined}>
      {t && series.length ? (
        <LineChart label={`${f.symbol}: shareholding by holder group (%)`} x={t.columns} series={series} format="pct" height={220}
          tools={tools}
          caption={
            <>
              Promoter change over 4 quarters <span className={signTone(d.get("promoter_change_4q"))}>{sgn(d.get("promoter_change_4q"), 2)} pp</span>;
              FII + DII <span className={signTone(d.get("fii_dii_change_4q"))}>{sgn(d.get("fii_dii_change_4q"), 2)} pp</span>.
              {!series.some((s) => s.name === "Promoters") && " No promoter group listed (widely held)."}
            </>
          } />
      ) : <Empty>No shareholding pattern on Screener.in for this company.</Empty>}
    </Panel>
  );
}

// ------------------------------------------------------------------ statements
type TabKey = "quarters" | "profit_loss" | "balance_sheet" | "cash_flow" | "ratios_table" | "shareholding";
const TABS: { v: TabKey; label: string; title: string }[] = [
  { v: "quarters", label: "Quarterly", title: "Quarterly results" },
  { v: "profit_loss", label: "Profit & Loss", title: "Annual profit & loss, plus trailing twelve months" },
  { v: "balance_sheet", label: "Balance Sheet", title: "Year-end balance sheets" },
  { v: "cash_flow", label: "Cash Flow", title: "Annual cash flow statement" },
  { v: "ratios_table", label: "Ratios", title: "Working-capital days and return ratios by year" },
  { v: "shareholding", label: "Shareholding", title: "Shareholding pattern, quarterly or yearly" },
];

/** Decimals for one row or column, so its figures line up: 2 when any value below 1,000 has a fraction, else 0. */
function decimalsOf(values: (number | null | undefined)[]): number {
  return values.some((v) => ok(v) && Math.abs(v) < 1000 && !Number.isInteger(v)) ? 2 : 0;
}

function cellText(v: number | null | undefined, pct: boolean, d: number): string {
  if (!ok(v)) return "—";
  const s = num(v, Math.abs(v) >= 1000 ? 0 : d);
  return pct ? `${s}%` : s;
}

function FinTableView({ t, caption }: { t: FinTable; caption: string }) {
  const wrap = useRef<HTMLDivElement>(null);
  // the newest periods are on the right: start scrolled there when the table is wider than the panel
  useLayoutEffect(() => {
    const el = wrap.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [t]);
  const ttm = t.columns.map((c) => /^ttm$/i.test(c.trim()));
  return (
    <div className="scr-fin" ref={wrap} tabIndex={0} role="region" aria-label={`${caption} — scrolls sideways`}>
      <table className="tbl scr-fintbl">
        <caption className="sr-only">{caption}{t.unit ? ` (${t.unit})` : ""}</caption>
        <thead>
          <tr>
            <th className="l" scope="col">{unitShort(t.unit)}</th>
            {t.columns.map((c, j) => <th key={j} scope="col" className={ttm[j] ? "ttm" : undefined} title={colTitle(c)}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {t.rows.map((r, i) => {
            const key = KEY_ROWS.has(r.label.trim().toLowerCase());
            const d = decimalsOf(r.values);
            return (
              <tr key={i} className={key ? "key" : undefined}>
                <td className="l">{r.label}</td>
                {t.columns.map((_, j) => {
                  const v = r.values[j];
                  const c = [ttm[j] ? "ttm" : "", !ok(v) ? "nil" : v < 0 ? "neg" : ""].filter(Boolean).join(" ");
                  return <td key={j} className={c || undefined}>{cellText(v, r.is_pct, d)}</td>;
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function StatementsPanel({ f }: { f: Fundamentals }) {
  const idBase = useId();
  const [tab, setTabState] = useState<TabKey>(() => {
    const v = getPref(TAB_KEY);
    return TABS.some((t) => t.v === v) ? (v as TabKey) : "quarters";
  });
  const setTab = (v: TabKey) => { setTabState(v); setPref(TAB_KEY, v); };
  const [shMode, setShMode] = useState<"q" | "y">("q");
  const sh = f.shareholding ?? { quarterly: null, yearly: null };
  const tableFor = (k: TabKey): FinTable | null =>
    k === "shareholding" ? (shMode === "q" ? sh.quarterly ?? sh.yearly : sh.yearly ?? sh.quarterly) ?? null : f[k] ?? null;
  const t = tableFor(tab);
  const label = TABS.find((x) => x.v === tab)?.label ?? "";
  const shown = tab === "shareholding" && t ? (t === sh.quarterly ? "quarterly" : "yearly") : null;

  const bar = (
    <>
      <Tabs label="Financial statements" idBase={idBase} value={tab} onChange={setTab}
        options={TABS.map((x) => ({ ...x, empty: !tableFor(x.v) && !(x.v === "shareholding" && (sh.quarterly || sh.yearly)) }))} />
      {tab === "shareholding" && sh.quarterly && sh.yearly && (
        <span className="grow">
          <Seg label="Shareholding period" value={shMode} onChange={setShMode}
            options={[{ v: "q", label: "QUARTERLY" }, { v: "y", label: "YEARLY" }]} />
        </span>
      )}
    </>
  );
  return (
    <Panel code="FIN" title="Financial statements" className="scr-full" id="scr-fin" bar={bar}
      sub={`${f.view} · ${t ? unitShort(t.unit) || "as published" : "—"} · negatives in red with −, oldest → newest`}>
      <div role="tabpanel" id={`${idBase}-panel`} aria-labelledby={`${idBase}-tab-${tab}`}>
        {t ? (
          <FinTableView t={t} caption={`${f.name}: ${t.title || label}${shown ? ` (${shown})` : ""}, ${f.view}`} />
        ) : (
          <Empty>
            No {label.toLowerCase()} table on Screener.in for the {f.view} view.
            {f.view === "consolidated" ? " Try STANDALONE." : ""}
          </Empty>
        )}
      </div>
    </Panel>
  );
}

// ------------------------------------------------------------------ peers
function PeersPanel({ f, onPick }: { f: Fundamentals; onPick: (s: string) => void }) {
  const p = f.peers;
  if (!p || !p.rows?.length || !p.columns?.length) return null;
  const me = new Set([f.symbol.toUpperCase(), (f.nse_code ?? "").toUpperCase()].filter(Boolean));
  const isMedian = (r: { name: string; symbol: string | null }) => r.symbol === null && /^median/i.test(r.name.trim());
  const body = p.rows.filter((r) => !isMedian(r));
  const foot = p.rows.filter(isMedian);
  const pctCol = p.columns.map((c) => /%/.test(c));
  const dec = p.columns.map((_, j) => decimalsOf(p.rows.map((r) => r.values[j])));
  const cell = (v: number | null | undefined, j: number) => cellText(v, pctCol[j], dec[j]);
  return (
    <Panel code="PEER" title="Peer comparison" className="scr-full" id="scr-peers"
      sub={`Screener.in's peer set${f.industry ? ` · ${f.industry}` : ""} · click a peer to research it`}>
      <div className="scr-fin scr-peers">
        <table className="tbl scr-fintbl">
          <caption className="sr-only">Peers of {f.name} on Screener.in</caption>
          <thead>
            <tr><th className="l" scope="col">Company</th>{p.columns.map((c, j) => <th key={j} scope="col">{c}</th>)}</tr>
          </thead>
          <tbody>
            {body.map((r, i) => {
              const self = !!r.symbol && me.has(r.symbol.toUpperCase());
              const pickable = !!r.symbol && !self && SYMBOL_RE.test(r.symbol.toUpperCase());
              return (
                <tr key={i} className={self ? "sel" : undefined}>
                  <td className="l">
                    {pickable ? (
                      <button type="button" className="scr-peer" onClick={() => onPick(r.symbol!.toUpperCase())}
                        title={`Research ${r.symbol} on these desks`}>{r.name}</button>
                    ) : <span title={r.symbol ? undefined : "No NSE symbol on Screener.in (BSE-only listing)"}>{r.name}</span>}
                    {r.symbol && <small>{r.symbol}</small>}
                  </td>
                  {p.columns.map((_, j) => <td key={j} className={!ok(r.values[j]) ? "nil" : (r.values[j] as number) < 0 ? "neg" : undefined}>{cell(r.values[j], j)}</td>)}
                </tr>
              );
            })}
          </tbody>
          {foot.length > 0 && (
            <tfoot>
              {foot.map((r, i) => (
                <tr key={i}>
                  <td className="l">{r.name}</td>
                  {p.columns.map((_, j) => <td key={j} className={!ok(r.values[j]) ? "nil" : (r.values[j] as number) < 0 ? "neg" : undefined}>{cell(r.values[j], j)}</td>)}
                </tr>
              ))}
            </tfoot>
          )}
        </table>
      </div>
    </Panel>
  );
}

// ------------------------------------------------------------------ loading / error states
function StatePanel({ symbol, loading, error, retry, view }: { symbol: string; loading: boolean; error: unknown; retry: () => void; view: View }) {
  if (!symbol) return <Panel title="Screener" className="scr-state"><Empty>Pick a stock above to load its fundamentals.</Empty></Panel>;
  if (loading) {
    return (
      <Panel code="FUND" title="Fundamentals" className="scr-state" sub={symbol}>
        <div className="scr-msg" role="status">
          <p><b>Fetching {symbol} from Screener.in…</b></p>
          <p>A first look at a company takes a few seconds: the gateway asks Screener.in politely, one page at a time
            {view === "auto" ? ", and falls back to the standalone view when there are no consolidated statements" : ""}.
            Later views come from its cache.</p>
        </div>
      </Panel>
    );
  }
  if (error instanceof ApiError && error.status === 404) {
    return (
      <Panel code="FUND" title="Not on Screener.in" className="scr-state" sub={symbol}>
        <div className="scr-msg" role="alert">
          <p><b>{errorText(error)}</b></p>
          <p>Screener.in lists companies by their NSE symbol (BSE-only companies by their BSE code). Indices, ETFs and
            some SME or newly listed stocks have no page there. Check the symbol, or look it up on <Ext href="https://www.screener.in/">screener.in</Ext>.</p>
        </div>
      </Panel>
    );
  }
  if (error) {
    return (
      <Panel code="FUND" title="Fundamentals" className="scr-state" sub={symbol}>
        <ErrorNote error={error} onRetry={retry} />
        <p className="scr-note">
          {error instanceof ApiError && error.status === 502
            ? "Screener.in did not answer and the gateway holds no earlier copy of this company. Try again in a minute."
            : "The fundamentals could not be loaded."}
        </p>
      </Panel>
    );
  }
  return <Panel title="Screener" className="scr-state"><Empty>No data.</Empty></Panel>;
}

// ------------------------------------------------------------------ desk
export default function Screener() {
  const symbol = useLive((s) => s.researchSymbol);
  const setResearchSymbol = useLive((s) => s.setResearchSymbol);
  const [view, setViewState] = useState<View>(readView);
  const setView = (v: View) => { setViewState(v); setPref(VIEW_KEY, v); };
  const fs = useFundamentals(symbol, view);
  const f = fs.data;
  const busy = fs.loading && !!f;

  const actions = (
    <>
      <Seg label="Statement view" value={view} onChange={setView}
        options={[
          { v: "auto", label: "AUTO", title: "Consolidated (company + subsidiaries) when Screener.in has it, else standalone" },
          { v: "consolidated", label: "CONSOLIDATED", title: "Group figures: the company and its subsidiaries" },
          { v: "standalone", label: "STANDALONE", title: "The listed company on its own" },
        ]} />
      <button type="button" className="btn" onClick={fs.refresh} disabled={!symbol || fs.loading}
        title="Fetch the page from Screener.in again (at most once every 10 minutes per company)">
        <Icon name="refresh" size={12} />{fs.loading ? "LOADING…" : "REFRESH"}
      </button>
    </>
  );

  return (
    <div className={`desk desk-scr rs-desk ${busy ? "scr-busy" : ""}`} aria-busy={fs.loading || undefined}>
      <SymbolPicker area="pick" history={false} fetchHistory={false} actions={actions} />
      {f ? (
        <>
          <CompanyPanel f={f} view={view} error={fs.error} />
          <Scorecard f={f} />
          <DerivedPanel f={f} />
          <ProsConsPanel f={f} />
          <SalesProfitPanel f={f} kind="q" />
          <SalesProfitPanel f={f} kind="a" />
          <ShareholdingPanel f={f} />
          <StatementsPanel f={f} />
          <PeersPanel f={f} onPick={setResearchSymbol} />
        </>
      ) : (
        <StatePanel symbol={symbol} loading={fs.loading} error={fs.error} retry={fs.retry} view={view} />
      )}
      <div className="scr-full">
        <Disclaimer>
          Fundamentals are Screener.in's compilation of company filings and can lag a results announcement or carry
          errors; check the filing before acting. The scorecard applies fixed published rules to these numbers — a
          checklist, not a rating or a recommendation.
        </Disclaimer>
      </div>
    </div>
  );
}
