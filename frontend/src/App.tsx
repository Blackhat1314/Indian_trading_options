import { Suspense, lazy, useEffect } from "react";
import { HelpModal, StatusBar, Tabs, Toasts, TopBar } from "./components/Frame";
import { Empty } from "./components/ui";
import { api, ApiError } from "./lib/api";
import { live } from "./lib/socket";
import { getMe, toLogin } from "./lib/auth";
import { useLive, type DeskId } from "./lib/store";
import type { Instrument } from "./lib/types";

const Intraday = lazy(() => import("./desks/Intraday"));
const Market = lazy(() => import("./desks/Market"));
const Fno = lazy(() => import("./desks/Fno"));
const Swing = lazy(() => import("./desks/Swing"));
const News = lazy(() => import("./desks/News"));
const Alerts = lazy(() => import("./desks/Alerts"));
const Geo = lazy(() => import("./desks/Geo"));
const AiAnalyst = lazy(() => import("./desks/AiAnalyst"));
const Screener = lazy(() => import("./desks/Screener"));
const MonteCarlo = lazy(() => import("./desks/MonteCarlo"));
const Quant = lazy(() => import("./desks/Quant"));

const BASE_TOPICS = ["clock", "market", "top", "quotes", "geo:risk"];
const DESK_TOPICS: Record<DeskId, string[]> = {
  intraday: ["rank"],
  market: [],
  fno: ["fno"],
  swing: ["swing"],
  news: ["news", "rank"],
  alerts: [],
  geo: [],          // the GEO desk subscribes per switched-on view
  ai: [],           // research desks: REST only (daily history, fundamentals, models)
  screener: [],
  montecarlo: [],
  quant: [],
};

export default function App() {
  const desk = useLive((s) => s.desk);
  const conn = useLive((s) => s.conn);
  const connError = useLive((s) => s.connError);
  const setUniverse = useLive((s) => s.setUniverse);
  const setMe = useLive((s) => s.setMe);

  useEffect(() => {
    live.sub(BASE_TOPICS);
    live.connect();
    const loadUniverse = () =>
      api<Instrument[]>("/api/universe").then((u) => {
        if (u.length) setUniverse(u);
        else window.setTimeout(loadUniverse, 2000);
      }).catch((e: ApiError) => {
        if (e.status === 401) toLogin();
        else window.setTimeout(loadUniverse, 3000);
      });
    loadUniverse();
    const loadMe = () =>
      getMe().then((m) => {
        if (m.login_required && !m.user) toLogin();
        else setMe(m);
      }).catch(() => window.setTimeout(loadMe, 5000));
    loadMe();
  }, [setUniverse, setMe]);

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
        {conn === "denied" && connError && <div className="banner warn" role="alert">{connError}</div>}
        <div style={{ flex: 1, minHeight: 0, display: "grid" }}>
          <Suspense fallback={<Empty>Loading desk…</Empty>}>
            {desk === "intraday" && <Intraday />}
            {desk === "market" && <Market />}
            {desk === "fno" && <Fno />}
            {desk === "swing" && <Swing />}
            {desk === "news" && <News />}
            {desk === "alerts" && <Alerts />}
            {desk === "geo" && <Geo />}
            {desk === "ai" && <AiAnalyst />}
            {desk === "screener" && <Screener />}
            {desk === "montecarlo" && <MonteCarlo />}
            {desk === "quant" && <Quant />}
          </Suspense>
        </div>
      </main>
      <StatusBar />
      <Toasts />
      <HelpModal />
    </div>
  );
}
