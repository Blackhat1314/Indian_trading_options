import { Suspense, lazy, useEffect, useState } from "react";
import { setPref } from "./lib/storage";
import { HelpModal, StatusBar, Tabs, Toasts, TopBar } from "./components/Frame";
import { Empty } from "./components/ui";
import { api, ApiError } from "./lib/api";
import { live } from "./lib/socket";
import { useLive, type AdminMode, type DeskId } from "./lib/store";
import type { Instrument } from "./lib/types";

const Intraday = lazy(() => import("./desks/Intraday"));
const Market = lazy(() => import("./desks/Market"));
const Fno = lazy(() => import("./desks/Fno"));
const Swing = lazy(() => import("./desks/Swing"));
const News = lazy(() => import("./desks/News"));
const Alerts = lazy(() => import("./desks/Alerts"));
const System = lazy(() => import("./desks/System"));
const Geo = lazy(() => import("./desks/Geo"));
const Admin = lazy(() => import("./desks/Admin"));

const BASE_TOPICS = ["clock", "market", "top", "quotes", "geo:risk"];
const DESK_TOPICS: Record<DeskId, string[]> = {
  intraday: ["rank"],
  market: [],
  fno: ["fno"],
  swing: ["swing"],
  news: ["news", "rank"],
  alerts: [],
  system: [],
  geo: [],          // the GEO desk subscribes per switched-on view
  admin: [],        // polls /api/admin/backfill instead (admin-only, never on the shared bus)
};

/** Shown when the gateway requires an access key (ACCESS_KEYS is set on the host). */
function AccessKeyModal() {
  const [key, setKey] = useState("");
  const save = (e: React.FormEvent) => {
    e.preventDefault();
    if (!key.trim()) return;
    setPref("td.key", key.trim());
    const u = new URL(location.href);
    u.searchParams.delete("key");
    location.replace(u.toString());
  };
  return (
    <div className="modal-back" role="dialog" aria-modal="true" aria-labelledby="ak-title">
      <form className="panel modal" onSubmit={save} style={{ width: "min(420px, 100%)" }}>
        <header className="panel-head"><span className="panel-code">KEY</span><h2 className="panel-title" id="ak-title" style={{ margin: 0 }}>Access key required</h2></header>
        <div style={{ padding: 12, display: "grid", gap: 10 }}>
          <p className="small muted" style={{ margin: 0 }}>This desk is shared privately. Paste the access key the host gave you (or open the link with <code>?key=…</code>).</p>
          <div className="field">
            <label htmlFor="ak">Access key</label>
            <input id="ak" autoFocus value={key} onChange={(e) => setKey(e.target.value)} autoComplete="off" spellCheck={false} />
          </div>
          <button type="submit" className="btn primary" style={{ justifySelf: "start", height: 30 }}>Connect</button>
        </div>
      </form>
    </div>
  );
}

export default function App() {
  const desk = useLive((s) => s.desk);
  const conn = useLive((s) => s.conn);
  const setUniverse = useLive((s) => s.setUniverse);
  const setAdminMode = useLive((s) => s.setAdminMode);

  useEffect(() => {
    live.sub(BASE_TOPICS);
    live.connect();
    const loadUniverse = () =>
      api<Instrument[]>("/api/universe").then((u) => {
        if (u.length) setUniverse(u);
        else window.setTimeout(loadUniverse, 2000);
      }).catch((e: ApiError) => {
        if (e.status === 401) useLive.getState().setConn("denied");
        else window.setTimeout(loadUniverse, 3000);
      });
    loadUniverse();
    const loadMeta = () =>
      api<{ admin?: AdminMode }>("/api/meta").then((m) => setAdminMode(m.admin ?? "off"))
        .catch((e: ApiError) => { if (e.status !== 401) window.setTimeout(loadMeta, 5000); });
    loadMeta();
  }, [setUniverse, setAdminMode]);

  useEffect(() => {
    const want = DESK_TOPICS[desk];
    live.sub(want);
    const others = Object.entries(DESK_TOPICS).filter(([d]) => d !== desk).flatMap(([, t]) => t)
      .filter((t) => !want.includes(t) && !BASE_TOPICS.includes(t));
    live.unsub(others);
  }, [desk]);

  return (
    <div className="app">
      <TopBar />
      <Tabs />
      <main style={{ minHeight: 0, display: "flex", flexDirection: "column" }} role="tabpanel" aria-label={`${desk} desk`}>
        {conn === "closed" && <div className="banner warn" role="status">Disconnected from the gateway — reconnecting…</div>}
        <div style={{ flex: 1, minHeight: 0, display: "grid" }}>
          <Suspense fallback={<Empty>Loading desk…</Empty>}>
            {desk === "intraday" && <Intraday />}
            {desk === "market" && <Market />}
            {desk === "fno" && <Fno />}
            {desk === "swing" && <Swing />}
            {desk === "news" && <News />}
            {desk === "alerts" && <Alerts />}
            {desk === "system" && <System />}
            {desk === "geo" && <Geo />}
            {desk === "admin" && <Admin />}
          </Suspense>
        </div>
      </main>
      <StatusBar />
      <Toasts />
      <HelpModal />
      {conn === "denied" && <AccessKeyModal />}
    </div>
  );
}
