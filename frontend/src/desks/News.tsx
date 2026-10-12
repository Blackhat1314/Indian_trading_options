import { useMemo, useState } from "react";
import { Empty, Panel, Seg } from "../components/ui";
import { cls, eventLabel, pct, signed, timeIST } from "../lib/format";
import { useLive } from "../lib/store";

export default function NewsDesk() {
  const news = useLive((s) => s.news);
  const rank = useLive((s) => s.rank);
  const select = useLive((s) => s.select);
  const setDesk = useLive((s) => s.setDesk);
  const [f, setF] = useState<"all" | "stock" | "pos" | "neg">("all");
  const [q, setQ] = useState("");
  const items = useMemo(() => {
    let r = [...news].sort((a, b) => b.ts - a.ts);
    if (f === "stock") r = r.filter((n) => n.symbols.length);
    if (f === "pos") r = r.filter((n) => n.sign > 0);
    if (f === "neg") r = r.filter((n) => n.sign < 0);
    if (q) {
      const s = q.toUpperCase();
      r = r.filter((n) => n.title.toUpperCase().includes(s) || n.symbols.some((x) => x.includes(s)));
    }
    return r;
  }, [news, f, q]);
  const board = useMemo(() => (rank?.rows ?? []).filter((r) => Math.abs(r.news) > 0.05).sort((a, b) => Math.abs(b.news) - Math.abs(a.news)), [rank]);
  const types = useMemo(() => {
    const c: Record<string, number> = {};
    for (const n of news) c[n.type] = (c[n.type] ?? 0) + 1;
    return Object.entries(c).sort((a, b) => b[1] - a[1]);
  }, [news]);
  const go = (s: string) => {
    select(s);
    setDesk("intraday");
  };
  return (
    <div className="desk desk-news">
      <Panel code="NEWS" title="Catalyst feed" area="feed" className="cap-m" sub={`${items.length} headlines · classified by event type, direction and novelty`}
        actions={
          <>
            <label className="sr-only" htmlFor="news-q">Search news</label>
            <input id="news-q" value={q} onChange={(e) => setQ(e.target.value)} placeholder="filter…"
              style={{ height: 24, width: 140, background: "#05080c", border: "1px solid var(--border-2)", borderRadius: 3, padding: "0 6px", fontFamily: "var(--font-mono)", fontSize: 11.5 }} />
            <Seg label="News filter" value={f} onChange={setF} options={[{ v: "all", label: "ALL" }, { v: "stock", label: "STOCK" }, { v: "pos", label: "▲ POS" }, { v: "neg", label: "▼ NEG" }]} />
          </>
        }>
        {items.length === 0 ? <Empty>No headlines yet. Live mode needs the Dhan Pro Traders plan (News Headlines API).</Empty> : items.map((n) => (
          <article className="news-item" key={n.id}>
            <span className="news-time">{timeIST(n.ts)}</span>
            <div>
              <div className="news-title">{n.title}</div>
              <div className="news-meta">
                <span className={`chip ${n.sign > 0 ? "up" : n.sign < 0 ? "down" : ""}`}>{n.sign > 0 ? "▲" : n.sign < 0 ? "▼" : "•"} {n.type.replace(/_/g, " ")}</span>
                {!n.novel && <span className="chip" title="Similar story already seen in the last 24h — down-weighted">REPEAT</span>}
                {n.symbols.slice(0, 6).map((s) => (
                  <button type="button" key={s} className={`chip news-sym ${n.impacts[s] > 0 ? "up" : n.impacts[s] < 0 ? "down" : ""}`} onClick={() => go(s)}>{s}</button>
                ))}
                <span className="tiny muted">{n.source} · weight {n.weight.toFixed(2)}</span>
              </div>
            </div>
          </article>
        ))}
      </Panel>
      <section className="stack" style={{ gridArea: "board", minHeight: 0 }}>
        <Panel code="CATL" title="Catalyst leaderboard" sub="decayed news score feeding the intraday ranker">
          {board.length === 0 ? <Empty>No active catalysts</Empty> : (
            <table className="tbl">
              <thead><tr><th className="l">Symbol</th><th className="l">Event</th><th>Score</th><th>Chg</th><th>RVOL</th></tr></thead>
              <tbody>
                {board.map((r) => (
                  <tr key={r.sid} onClick={() => go(r.symbol)} title={r.headline ?? ""}>
                    <td className="sym">{r.symbol}</td>
                    <td className="l"><span className="chip">{eventLabel(r.news_type)}</span></td>
                    <td className={cls(r.news)}>{signed(r.news)}</td>
                    <td className={cls(r.chg)}>{pct(r.chg)}</td>
                    <td className={r.rvol >= 2 ? "amber" : ""}>{r.rvol.toFixed(1)}x</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
        <Panel code="TYPE" title="Event mix today">
          <div style={{ padding: 8, display: "flex", flexWrap: "wrap", gap: 4 }}>
            {types.map(([t, c]) => <span key={t} className="chip">{t.replace(/_/g, " ")} <b className="amber">{c}</b></span>)}
          </div>
        </Panel>
      </section>
    </div>
  );
}
