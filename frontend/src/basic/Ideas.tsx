/* Basic IDEAS: ready-made filters as cards (what each finds, in one line), and the stocks of the one you open. */
import { useEffect, useMemo, useRef } from "react";
import { useLive } from "../lib/store";
import { SCANS } from "./scans";
import type { ScanData } from "./scans";
import { Badge, Card, Change, Empty, StockLink, rupees } from "./ui";

export default function Ideas() {
  const rank = useLive((s) => s.rank);
  const fno = useLive((s) => s.fno);
  const swing = useLive((s) => s.swing);
  const sel = useLive((s) => s.basicScan);
  const setSel = useLive((s) => s.setBasicScan);
  const data: ScanData = useMemo(() => ({ rank: rank?.rows ?? [], fno: fno?.all ?? [], swing: swing?.rows ?? [], swingStats: swing?.stats ?? {} }),
    [rank, fno, swing]);
  const counts = useMemo(() => Object.fromEntries(SCANS.map((s) => [s.id, s.rows(data).length])), [data]);
  const scan = SCANS.find((s) => s.id === sel) ?? SCANS[0];
  const rows = useMemo(() => scan.rows(data), [scan, data]);
  const note = scan.note?.(data) ?? null;
  const results = useRef<HTMLDivElement>(null);
  const first = useRef(true);
  useEffect(() => {
    // a click on a scan card brings its results into view when they sit below the fold (not on first render)
    if (first.current) { first.current = false; return; }
    const el = results.current;
    if (el && el.getBoundingClientRect().top > window.innerHeight - 160) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [sel]);
  const loading = scan.when === "swing" ? !swing : scan.id === "lb" || scan.id === "sb" ? !fno : !rank;
  const groups: { when: "today" | "swing"; title: string; sub: string }[] = [
    { when: "today", title: "For today", sub: "Built from live prices — they change through the day" },
    { when: "swing", title: "For the next two weeks", sub: "Daily-chart setups, each with its historical record" },
  ];
  return (
    <div className="b-ideas">
      <div className="b-intro">
        <h1>Ideas</h1>
        <p>Each idea is a simple filter, not a prediction. Use it to find stocks worth a closer look, then open the stock page and size the trade before acting.</p>
      </div>
      {groups.map((g) => (
        <section key={g.when} aria-labelledby={`g-${g.when}`}>
          <h2 className="b-section-title" id={`g-${g.when}`}>{g.title} <span>{g.sub}</span></h2>
          <div className="b-scan-grid">
            {SCANS.filter((s) => s.when === g.when).map((s) => (
              <button type="button" key={s.id} className={`b-scan ${s.id === scan.id ? "on" : ""}`} aria-pressed={s.id === scan.id} onClick={() => setSel(s.id)}>
                <span className="b-scan-head"><b>{s.name}</b><span className="b-count">{counts[s.id] ?? 0}</span></span>
                <span className="b-tags">{s.tags.map((t) => <i key={t}>{t}</i>)}</span>
                <span className="b-scan-desc">{s.desc}</span>
              </button>
            ))}
          </div>
        </section>
      ))}
      <div ref={results} className="b-results">
        <Card title={scan.name} id="res" sub={scan.desc}>
          {note && <p className="b-note boxed">{note}</p>}
          {loading ? <Empty>Loading live data…</Empty> : !rows.length ? (
            <Empty>No stock matches right now{scan.when === "today" ? " — try again during market hours" : ""}.</Empty>
          ) : (
            <ul className="b-rows">
              {rows.map((r) => (
                <li key={r.symbol}>
                  <StockLink symbol={r.symbol}>
                    <Badge symbol={r.symbol} />
                    <span className="b-row-main"><b>{r.symbol}</b><small>{r.why}{r.extra ? ` · ${r.extra}` : ""}</small></span>
                    <span className="b-row-end"><span>{rupees(r.price)}</span><small><Change v={r.chg} /> today</small><small className="b-muted">{r.sector}</small></span>
                  </StockLink>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  );
}
