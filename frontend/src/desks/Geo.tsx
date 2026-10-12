import { useEffect, useMemo, useRef, useState } from "react";
import type React from "react";
import { Empty, Panel, Spark } from "../components/ui";
import { GeoMap, worldview, type Focus } from "../geo/GeoMap";
import type {
  GeoAir, GeoChokepoint, GeoConflict, GeoFires, GeoImpact, GeoMacro, GeoRisk, GeoShips, GeoSites, GeoStatus,
} from "../geo/types";
import { BASE_GEO_TOPICS, GLOBAL_TOPICS, ROOT_COLOR, SATELLITE, STATUS_COLOR, VIEWS, type ViewId, type Views } from "../geo/views";
import { cls, fx, pct, signed } from "../lib/format";
import { live } from "../lib/socket";
import { getPref, setPref } from "../lib/storage";
import { useLive } from "../lib/store";

const yesterday = () => new Date(Date.now() - 86400000).toISOString().slice(0, 10);
const agoTxt = (ts: number) => {
  const s = Math.max(0, Date.now() / 1000 - ts);
  return s < 3600 ? `${Math.round(s / 60)}m` : s < 86400 ? `${Math.round(s / 3600)}h` : `${Math.round(s / 86400)}d`;
};

function loadViews(): Views {
  const def = Object.fromEntries(VIEWS.map((v) => [v.id, v.on])) as Views;
  try {
    return { ...def, ...(JSON.parse(getPref("td.geo.views") ?? "{}") as Partial<Views>) };
  } catch {
    return def;
  }
}

/* ------------------------------------------------------------------ switches */
function Switch({ on, onChange, id }: { on: boolean; onChange: (v: boolean) => void; id: string }) {
  return (
    <button type="button" role="switch" aria-checked={on} aria-labelledby={`${id}-l`} className={`sw ${on ? "on" : ""}`}
      onClick={() => onChange(!on)}>
      <span className="sw-knob" aria-hidden="true" />
    </button>
  );
}

const STATE_CLASS: Record<string, string> = { LIVE: "up", SIM: "violet", CLIENT: "cyan", PAUSED: "", WAITING: "",
  "NEEDS KEY": "amber", ERROR: "down", DISABLED: "" };

function ViewSwitches({ views, setView, status, satId, setSatId, satDate, setSatDate }: {
  views: Views; setView: (id: ViewId, v: boolean) => void; status?: GeoStatus;
  satId: string; setSatId: (s: string) => void; satDate: string; setSatDate: (s: string) => void;
}) {
  const sat = SATELLITE.find((s) => s.id === satId) ?? SATELLITE[0];
  return (
    <Panel code="VIEW" title="Views" sub={status ? `${status.mode === "sim" ? "simulated" : "live"} feeds` : ""} className="geo-switches">
      <ul className="sw-list">
        {VIEWS.map((v) => {
          const st = v.status ? status?.layers[v.status] : undefined;
          return (
            <li key={v.id} className={views[v.id] ? "" : "off"}>
              <Switch id={`v-${v.id}`} on={views[v.id]} onChange={(on) => setView(v.id, on)} />
              <div className="sw-text" title={v.hint}>
                <span id={`v-${v.id}-l`}>{v.label}</span>
                {st && views[v.id] && (
                  <span className={`chip ${STATE_CLASS[st.state] ?? ""}`} title={`${st.source}${st.detail ? ` — ${st.detail}` : ""}`}>
                    {st.state}{st.n != null && st.state !== "NEEDS KEY" ? ` · ${st.n}` : ""}
                  </span>
                )}
              </div>
            </li>
          );
        })}
      </ul>
      {views.satellite && (
        <div className="sat-ctl">
          <label className="tiny muted mono" htmlFor="sat-src">IMAGERY</label>
          <select id="sat-src" value={satId} onChange={(e) => setSatId(e.target.value)}>
            {SATELLITE.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
          </select>
          {sat.dated && (
            <>
              <label className="tiny muted mono" htmlFor="sat-date">DATE (UTC)</label>
              <input id="sat-date" type="date" value={satDate} max={new Date().toISOString().slice(0, 10)}
                onChange={(e) => e.target.value && setSatDate(e.target.value)} />
            </>
          )}
          <span className="tiny muted">Tiles load from NASA / EOX in your browser. Today's pass may still be processing — use yesterday.</span>
        </div>
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------------ risk */
function RiskGauge({ v }: { v: number }) {
  const color = v >= 65 ? "var(--down)" : v >= 45 ? "#ff8a3d" : v >= 25 ? "var(--amber)" : "var(--up)";
  return (
    <div className="gauge" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(v)} aria-label="Geo-risk">
      <div className="gauge-track">
        {[25, 45, 65].map((t) => <i key={t} className="gauge-tick" style={{ left: `${t}%` }} />)}
        <div className="gauge-fill" style={{ width: `${v}%`, background: color }} />
      </div>
      <div className="gauge-scale tiny muted mono"><span>0 LOW</span><span>25</span><span>45</span><span>65 SEVERE</span><span>100</span></div>
    </div>
  );
}

function RiskPanel({ risk }: { risk?: GeoRisk }) {
  if (!risk) return <Panel code="GPR" title="Geo-risk"><Empty>Waiting for the GEO worker…</Empty></Panel>;
  const hist = risk.history.map((h) => h[1]);
  const gpr = risk.gpr;
  const ind = gpr.monthly?.GPRC_IND ?? [];
  return (
    <Panel code="GPR" title="Geopolitical risk" sub="India-weighted, live">
      <div className="risk-head">
        <div>
          <div className={`risk-num ${risk.value >= 65 ? "down" : risk.value >= 45 ? "warn" : risk.value >= 25 ? "amber" : "up"}`}>{risk.value.toFixed(0)}</div>
          <div className="risk-level mono">{risk.level}</div>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <RiskGauge v={risk.value} />
          {hist.length > 2 && <div className="tiny muted mono" style={{ marginTop: 4 }}>last {Math.round((risk.history.at(-1)![0] - risk.history[0][0]) / 3600)}h
            <Spark data={hist} width={150} height={20} color="var(--amber)" /></div>}
        </div>
      </div>
      <div className="risk-comps">
        {Object.entries(risk.components).map(([k, v]) => (
          <div key={k} className="risk-comp">
            <span className="muted">{k}</span>
            <div className="bar"><i style={{ left: 0, width: `${v}%`, background: v >= 65 ? "var(--down)" : v >= 35 ? "var(--amber)" : "var(--up)" }} /></div>
            <span className="mono right">{v.toFixed(0)}</span>
            <span className="mono muted tiny right">{Math.round((risk.weights[k] ?? 0) * 100)}%</span>
          </div>
        ))}
      </div>
      <div className="facts" title="Caldara & Iacoviello newspaper-based Geopolitical Risk index; 100 = 1985–2019 average">
        <div><span>GPR daily (Caldara–Iacoviello)</span><b>{gpr.latest ? fx(gpr.latest[1], 0) : "—"}<small>{gpr.latest?.[0]}</small></b></div>
        <div><span>GPR 30-day avg (100 = normal)</span><b>{gpr.ma30 != null ? fx(gpr.ma30, 0) : "—"}</b></div>
        <div><span>India GPR (monthly)</span><b>{ind.length ? fx(ind.at(-1)![1], 2) : "—"}<small>{ind.at(-1)?.[0]}</small></b></div>
        {gpr.series.length > 10 && <div><span>GPR, last year</span><Spark data={gpr.series.map((x) => x[1])} width={120} height={18} color="var(--violet)" /></div>}
      </div>
      <table className="tbl">
        <thead><tr><th className="l">Hot region (24h)</th><th>Events</th><th>Severity</th></tr></thead>
        <tbody>
          {risk.regions.slice(0, 6).map((r) => (
            <tr key={r.id} style={{ cursor: "default" }} title={r.top ? `Top: ${r.top.label} — ${r.top.place}` : ""}>
              <td className="l">{r.name}{r.india >= 0.9 && <span className="chip amber" style={{ marginLeft: 6 }}>INDIA</span>}</td>
              <td>{r.n}</td><td className={r.weighted > 100 ? "down" : ""}>{fx(r.sev, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

/* ------------------------------------------------------------------ impact */
function ImpactPanel({ impact }: { impact?: GeoImpact }) {
  const select = useLive((s) => s.select);
  const setDesk = useLive((s) => s.setDesk);
  const sigs = impact?.signals ?? [];
  return (
    <Panel code="IMPACT" title="Market impact" sub="rule-based · check before acting">
      {sigs.length === 0 ? <Empty>No geo signal strong enough to map to stocks right now.</Empty> : (
        <ul className="impact-list">
          {sigs.map((s) => (
            <li key={s.id}>
              <div className="impact-title">
                <span className={`sev sev${s.sev}`} aria-label={`severity ${s.sev} of 3`}>{"●".repeat(s.sev)}{"○".repeat(3 - s.sev)}</span>
                <b>{s.title}</b>
              </div>
              <div className="chips" style={{ margin: "4px 0" }}>
                {s.symbols.slice(0, 10).map((x) => (
                  <button type="button" key={x.s} className={`chip ${x.dir > 0 ? "up" : "down"}`} style={{ cursor: "pointer" }}
                    onClick={() => { select(x.s); setDesk("intraday"); }} title={`Open ${x.s} on the intraday desk`}>
                    {x.dir > 0 ? "▲" : "▼"} {x.s}
                  </button>
                ))}
              </div>
              <p className="tiny muted" style={{ margin: 0 }}>{s.why}</p>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------------ chokepoints */
function ChokepointsPanel({ rows, onFocus }: { rows: GeoChokepoint[]; onFocus: (lon: number, lat: number, z: number) => void }) {
  const asOf = rows.find((r) => r.pw)?.pw?.as_of;
  return (
    <Panel code="CHOKE" title="Chokepoints" sub={asOf ? `transits vs normal · PortWatch to ${asOf}` : "live vessels + conflict nearby"}>
      {rows.length === 0 ? <Empty>Waiting…</Empty> : (
        <table className="tbl">
          <thead><tr><th className="l">Strait</th><th className="l">Status</th><th title="Vessels inside the circle now (tankers)">Now</th>
            <th title="Average daily transits, last 7 days vs the usual day over the past year">7d vs normal</th><th title="Conflict events within 150 km, 72 h">Conf.</th></tr></thead>
          <tbody>
            {rows.map((c) => (
              <tr key={c.id} onClick={() => onFocus(c.lon, c.lat, 6)} title={[c.why, ...c.reasons].join(" · ")}>
                <td className="l b">{c.name}</td>
                <td className="l"><span className="chip" style={{ color: STATUS_COLOR[c.status], borderColor: STATUS_COLOR[c.status] }}>{c.status}</span></td>
                <td>{c.ships_now}<span className="muted"> ({c.tankers_now})</span></td>
                <td>
                  {c.pw?.ratio != null ? (
                    <span className="ratio">
                      <span className="bar"><i style={{ left: 0, width: `${Math.min(100, c.pw.ratio * 66)}%`, background: c.pw.ratio < 0.5 ? "var(--down)" : c.pw.ratio < 0.8 ? "var(--amber)" : "var(--up)" }} /></span>
                      <span className={c.pw.ratio < 0.8 ? "down" : ""}>{Math.round(c.pw.ratio * 100)}%</span>
                    </span>
                  ) : <span className="muted">—</span>}
                </td>
                <td className={c.conflict_n ? "amber" : "muted"}>{c.conflict_n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------------ conflict feed */
function ConflictPanel({ data, onFocus }: { data?: GeoConflict; onFocus: (lon: number, lat: number, z: number) => void }) {
  const [region, setRegion] = useState("india");
  const [sort, setSort] = useState<"time" | "sev">("time");
  const events = useMemo(() => {
    let ev = data?.events ?? [];
    if (region === "india") {
      ev = ev.filter((e) => e.lon > 30 && e.lon < 100 && e.lat > 0 && e.lat < 42);
    }
    return [...ev].sort((a, b) => (sort === "time" ? b.ts - a.ts : b.sev - a.sev)).slice(0, 200);
  }, [data, region, sort]);
  return (
    <Panel code="WAR" title="Conflict events" sub={data ? `${data.n} in 24 h` : ""}
      actions={
        <>
          <select aria-label="Region" value={region} onChange={(e) => setRegion(e.target.value)} className="mini-select">
            <option value="india">India's neighbourhood</option><option value="all">World</option>
          </select>
          <select aria-label="Sort" value={sort} onChange={(e) => setSort(e.target.value as "time" | "sev")} className="mini-select">
            <option value="time">Newest</option><option value="sev">Severity</option>
          </select>
        </>
      }>
      {events.length === 0 ? <Empty>No events.</Empty> : (
        <ul className="ev-list">
          {events.map((e) => (
            <li key={e.id} onClick={() => onFocus(e.lon, e.lat, 6)}>
              <span className="ev-dot" style={{ background: ROOT_COLOR[e.root] ?? "#ff5263" }} aria-hidden="true" />
              <div style={{ minWidth: 0 }}>
                <div className="ev-title"><b>{e.label}</b> <span className="muted">· {e.place || `${e.lat.toFixed(1)}, ${e.lon.toFixed(1)}`}</span></div>
                <div className="tiny muted ellipsis">
                  {agoTxt(e.ts)} ago · {[e.a1, e.a2].filter(Boolean).join(" → ") || "actors n/a"} · {e.articles} art. · sev {e.sev}
                  {e.url && <> · <a href={e.url} target="_blank" rel="noopener noreferrer" onClick={(x) => x.stopPropagation()}>source ↗</a></>}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------------ fires at energy sites */
function FiresPanel({ data, onFocus }: { data?: GeoFires; onFocus: (lon: number, lat: number, z: number) => void }) {
  const sites = (data?.sites ?? []).filter((s) => s.status !== "QUIET");
  return (
    <Panel code="FIRE" title="Fires at energy sites" sub={data ? `${data.n} hotspots · 36 h window` : ""}>
      {sites.length === 0 ? <Empty>No satellite hotspots at tracked refineries, terminals or fields.</Empty> : (
        <table className="tbl">
          <thead><tr><th className="l">Site</th><th className="l">Status</th><th title="Hotspots in the last 36 h">Hot</th>
            <th title="Total fire radiative power, MW">MW</th><th title="Usual detections per day at this site">Usual</th><th /></tr></thead>
          <tbody>
            {sites.map((s) => (
              <tr key={s.id} onClick={() => onFocus(s.lon, s.lat, 9)} title={`${s.kind} · ${s.owner}`}>
                <td className="l"><b>{s.name}</b> <small className="muted">{s.country}</small>
                  {Object.keys(s.symbols).length > 0 && <span className="muted tiny"> · {Object.keys(s.symbols).join(", ")}</span>}</td>
                <td className="l"><span className="chip" style={{ color: STATUS_COLOR[s.status], borderColor: STATUS_COLOR[s.status] }}>{s.status}</span></td>
                <td>{s.hotspots}</td><td>{fx(s.frp, 0)}</td><td className="muted">{s.baseline ?? "—"}</td>
                <td><a href={worldview(s.lon, s.lat, new Date((s.last ?? Date.now() / 1000) * 1000).toISOString().slice(0, 10))} target="_blank"
                  rel="noopener noreferrer" onClick={(x) => x.stopPropagation()} title="Open the satellite pass in NASA Worldview">IMG ↗</a></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="panel-foot">Refineries and gas fields flare every day; ANOMALY = far above the site's own usual count or a very hot cluster.</div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ airspace */
function AirPanel({ data, onFocus }: { data?: GeoAir; onFocus: (lon: number, lat: number, z: number) => void }) {
  const mil = (data?.rows ?? []).filter((r) => r[8] === 1).length;
  return (
    <Panel code="AIR" title="Airspace watch" sub={data ? `${mil} military aircraft tracked` : ""}>
      {!data ? <Empty>Waiting…</Empty> : (
        <table className="tbl">
          <thead><tr><th className="l">Airspace</th><th title="Aircraft above 3,000 ft now">Now</th><th title="Usual for this hour">Usual</th><th>Ratio</th><th className="l">Status</th></tr></thead>
          <tbody>
            {data.regions.map((r) => (
              <tr key={r.id} onClick={() => onFocus(r.lon, r.lat, 5)}>
                <td className="l">{r.name}</td><td>{r.count ?? "—"}</td><td className="muted">{r.baseline != null ? fx(r.baseline, 0) : "—"}</td>
                <td className={r.ratio != null && r.ratio < 0.6 ? "down" : ""}>{r.ratio != null ? `${Math.round(r.ratio * 100)}%` : "—"}</td>
                <td className="l"><span className="chip" style={{ color: STATUS_COLOR[r.status], borderColor: STATUS_COLOR[r.status] }}>{r.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------------ oil & supply / macro */
function SeriesRow({ label, value, chg, chg5, spark, unit, src }: { label: string; value: number | null | undefined; chg?: number | null;
  chg5?: number | null; spark?: number[]; unit?: string; src?: string }) {
  const sub = [unit, src, chg5 != null ? `5d ${signed(chg5, 1)}%` : ""].filter(Boolean).join(" · ");
  return (
    <div className="mrow" title={sub}>
      <span className="mrow-l">{label}{sub && <small>{sub}</small>}</span>
      <span className="mono right b">{value != null ? fx(value, value > 1000 ? 0 : 2) : "—"}</span>
      <span className={`mono right ${cls(chg)}`}>{chg != null ? pct(chg) : ""}</span>
      <span className="right">{spark && spark.length > 2 ? <Spark data={spark} width={56} height={16} color="var(--text-2)" /> : null}</span>
    </div>
  );
}

function OilPanel({ macro, chokes, fires }: { macro?: GeoMacro; chokes: GeoChokepoint[]; fires?: GeoFires }) {
  const d = macro?.daily ?? {};
  const l = macro?.live ?? {};
  const hormuz = chokes.find((c) => c.id === "hormuz");
  const anomalies = (fires?.sites ?? []).filter((s) => s.status === "ANOMALY");
  return (
    <Panel code="OIL" title="Oil & supply">
      <div className="mrows">
        <SeriesRow label="MCX crude" value={l.CRUDEOIL?.ltp} chg={l.CRUDEOIL?.chg} unit="₹/bbl" src={l.CRUDEOIL?.src} />
        <SeriesRow label="MCX nat. gas" value={l.NATURALGAS?.ltp} chg={l.NATURALGAS?.chg} unit="₹/mmBtu" src={l.NATURALGAS?.src} />
        {["DCOILBRENTEU", "DCOILWTICO", "DHHNGSP"].map((k) => d[k] && (
          <SeriesRow key={k} label={d[k].label.replace(/ \(.*\)/, "")} value={d[k].last[1]} chg={d[k].chg1d} chg5={d[k].chg5d}
            spark={d[k].spark} unit={d[k].label.match(/\((.*)\)/)?.[1]} src={`FRED ${d[k].last[0]?.slice(5)}`} />
        ))}
      </div>
      <div className="facts">
        <div><span>Hormuz tankers / day</span><b>{hormuz?.pw ? `${fx(hormuz.pw.last7_tanker, 0)} vs ${fx(hormuz.pw.base_tanker, 0)} usual` : "—"}</b></div>
        <div><span>Tankers in Hormuz now</span><b>{hormuz ? hormuz.tankers_now : "—"}</b></div>
        <div><span>Unusual fires at energy sites</span>
          <b className={anomalies.length ? "down" : ""}>{anomalies.length ? anomalies.map((a) => a.name).join(", ") : "none"}</b></div>
      </div>
    </Panel>
  );
}

function MacroPanel({ macro }: { macro?: GeoMacro }) {
  const d = macro?.daily ?? {};
  const l = macro?.live ?? {};
  return (
    <Panel code="MACRO" title="India macro">
      <div className="mrows">
        <SeriesRow label="USD/INR fut." value={l.USDINR?.ltp} chg={l.USDINR?.chg} src={l.USDINR?.src} />
        {d.DEXINUS && <SeriesRow label="USD/INR spot" value={d.DEXINUS.last[1]} chg={d.DEXINUS.chg1d} chg5={d.DEXINUS.chg5d} spark={d.DEXINUS.spark} src={`FRED ${d.DEXINUS.last[0]?.slice(5)}`} />}
        <SeriesRow label="MCX gold" value={l.GOLD?.ltp} chg={l.GOLD?.chg} unit="₹/10g" src={l.GOLD?.src} />
        <SeriesRow label="MCX silver" value={l.SILVER?.ltp} chg={l.SILVER?.chg} unit="₹/kg" src={l.SILVER?.src} />
        <SeriesRow label="India VIX" value={macro?.india_vix} />
        {d.DGS10 && <SeriesRow label="US 10Y yield" value={d.DGS10.last[1]} chg5={d.DGS10.chg5d} spark={d.DGS10.spark} unit="%" />}
        {d.DTWEXBGS && <SeriesRow label="Dollar index" value={d.DTWEXBGS.last[1]} chg={d.DTWEXBGS.chg1d} chg5={d.DTWEXBGS.chg5d} spark={d.DTWEXBGS.spark} />}
      </div>
      <table className="tbl">
        <thead><tr><th className="l">Upcoming</th><th>Date · IST</th></tr></thead>
        <tbody>
          {(macro?.calendar ?? []).slice(0, 7).map((e) => (
            <tr key={e.date + e.title} style={{ cursor: "default" }}>
              <td className="l"><span className={`chip ${e.kind === "RBI" ? "amber" : e.kind === "FED" ? "cyan" : ""}`}>{e.kind}</span> {e.title}</td>
              <td>{e.date.slice(5)}{e.approx ? "*" : ""} <span className="muted">{e.time.replace(" IST", "")}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="panel-foot">* usual release pattern — confirm on the MOSPI calendar. Add dates in data/config/macro_calendar.json.</div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ desk */
export default function GeoDesk() {
  const [views, setViews] = useState<Views>(loadViews);
  const [satId, setSatIdState] = useState(() => getPref("td.geo.sat") ?? "viirs");
  const [satDate, setSatDate] = useState(yesterday);
  const [focus, setFocus] = useState<Focus | null>(null);
  const focusKey = useRef(0);
  const geo = useLive((s) => s.geo);

  const setView = (id: ViewId, on: boolean) => {
    const next = { ...views, [id]: on };
    setViews(next);
    setPref("td.geo.views", JSON.stringify(next));
  };
  const setSatId = (s: string) => {
    setSatIdState(s);
    setPref("td.geo.sat", s);
  };

  // subscribe only to what is switched on (off = the gateway stops sending it, and idle feeds pause)
  const topics = useMemo(() => {
    const t = new Set(BASE_GEO_TOPICS);
    VIEWS.forEach((v) => views[v.id] && v.topics.forEach((x) => t.add(x)));
    return [...t];
  }, [views]);
  const prev = useRef<string[]>([]);
  useEffect(() => {
    live.sub(topics);
    const drop = prev.current.filter((t) => !topics.includes(t) && !GLOBAL_TOPICS.includes(t));
    if (drop.length) live.unsub(drop);
    prev.current = topics;
  }, [topics]);
  useEffect(() => () => live.unsub(prev.current.filter((t) => !GLOBAL_TOPICS.includes(t))), []);

  const onFocus = (lon: number, lat: number, zoom: number) => setFocus({ lon, lat, zoom, key: ++focusKey.current });

  const sites = geo["geo:sites"] as GeoSites | undefined;
  const status = geo["geo:status"] as GeoStatus | undefined;
  const ships = views.ships ? (geo["geo:ships"] as GeoShips | undefined) : undefined;
  const chokes = ((geo["geo:chokepoints"] as { rows: GeoChokepoint[] } | undefined)?.rows) ?? [];
  const conflict = views.conflict ? (geo["geo:conflict"] as GeoConflict | undefined) : undefined;
  const fires = views.fires || views.oil ? (geo["geo:fires"] as GeoFires | undefined) : undefined;
  const air = views.air ? (geo["geo:air"] as GeoAir | undefined) : undefined;
  const risk = geo["geo:risk"] as GeoRisk | undefined;
  const macro = geo["geo:macro"] as GeoMacro | undefined;
  const impact = geo["geo:impact"] as GeoImpact | undefined;

  const left = [views.risk && <RiskPanel key="risk" risk={risk} />, views.impact && <ImpactPanel key="impact" impact={impact} />].filter(Boolean);
  const bottomTabs = [
    views.chokepoints && { id: "ch", label: "CHOKEPOINTS", badge: chokes.filter((c) => c.status !== "NORMAL").length,
      el: <ChokepointsPanel rows={chokes} onFocus={onFocus} /> },
    views.conflict && { id: "cf", label: "CONFLICT", badge: conflict?.n ?? 0, el: <ConflictPanel data={conflict} onFocus={onFocus} /> },
    views.fires && { id: "fi", label: "FIRES @ ENERGY", badge: (fires?.sites ?? []).filter((s) => s.status === "ANOMALY").length,
      el: <FiresPanel data={fires} onFocus={onFocus} /> },
  ].filter(Boolean) as { id: string; label: string; badge: number; el: React.ReactNode }[];
  const [tab, setTabState] = useState(() => getPref("td.geo.tab") ?? "ch");
  const setTab = (t: string) => {
    setTabState(t);
    setPref("td.geo.tab", t);
  };
  const activeTab = bottomTabs.find((t) => t.id === tab) ?? bottomTabs[0];
  const bottom = bottomTabs.length ? [
    <div key="tabs" className="geo-tabbed">
      <div className="geo-tabs" role="tablist" aria-label="Event tables">
        {bottomTabs.map((t) => (
          <button key={t.id} type="button" role="tab" aria-selected={t.id === activeTab?.id} onClick={() => setTab(t.id)}>
            {t.label}{t.badge > 0 && <span className="tab-count">{t.badge}</span>}
          </button>
        ))}
      </div>
      <div className="geo-tab-body">{activeTab?.el}</div>
    </div>,
  ] : [];
  const right = [
    views.oil && <OilPanel key="oil" macro={macro} chokes={chokes} fires={fires} />,
    views.macro && <MacroPanel key="mac" macro={macro} />,
    views.air && <AirPanel key="air" data={air} onFocus={onFocus} />,
  ].filter(Boolean);

  return (
    <div className={`desk desk-geo ${right.length ? "" : "no-right"} ${bottom.length ? "" : "no-bottom"}`}>
      <div className="geo-left stack">
        <ViewSwitches views={views} setView={setView} status={status} satId={satId} setSatId={setSatId}
          satDate={satDate} setSatDate={setSatDate} />
        {left}
      </div>
      <section className="panel geo-center" aria-label="Map">
        <GeoMap views={views} satId={satId} satDate={satDate} focus={focus} sites={sites} ships={ships}
          chokepoints={views.chokepoints || views.oil ? chokes : []} conflict={conflict} fires={fires}
          air={air} />
      </section>
      {bottom.length > 0 && <div className="geo-bottom">{bottom}</div>}
      {right.length > 0 && <div className="geo-right stack">{right}</div>}
    </div>
  );
}
