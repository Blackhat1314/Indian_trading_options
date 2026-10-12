import { Suspense, lazy, useEffect, useState } from "react";
import { Empty } from "../components/ui";
import { getMe, logout, toLogin, type Me } from "../lib/auth";
import { live } from "../lib/socket";
import { getPref, setPref } from "../lib/storage";
import { useLive } from "../lib/store";

/* /admin — a separate page for admins: USERS (accounts), HISTORY (Dhan history pulls) and SYSTEM (service
   health, scorecard, backtest, lake). The terminal stays at /. */

const Users = lazy(() => import("./Users"));
const History = lazy(() => import("./History"));
const System = lazy(() => import("./System"));

type Tab = "users" | "history" | "system";
const TABS: { id: Tab; key: string; label: string; sub: string }[] = [
  { id: "users", key: "F1", label: "USERS", sub: "accounts and roles" },
  { id: "history", key: "F2", label: "HISTORY", sub: "pull 5 years of Dhan data" },
  { id: "system", key: "F3", label: "SYSTEM", sub: "services, scorecard, lake" },
];
const isTab = (t: string | null): t is Tab => TABS.some((x) => x.id === t);

export default function AdminApp() {
  const [me, setMe] = useState<Me | null>(null);
  const [err, setErr] = useState("");
  const hash = location.hash.slice(1);
  const [tab, setTab] = useState<Tab>(isTab(hash) ? hash : isTab(getPref("td.adminTab")) ? (getPref("td.adminTab") as Tab) : "users");

  useEffect(() => {
    // checked again every 20 s: a demoted, disabled or logged-out admin must not keep a stale admin page
    const check = () => getMe().then((m) => {
      if (!m.user || m.via !== "session") {
        toLogin("/admin");
        return;
      }
      setMe(m);
      useLive.getState().setMe(m);
    }).catch(() => setErr("Cannot reach the gateway — is TradeDesk running?"));
    check();
    const id = window.setInterval(check, 20000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (!me?.admin) return;
    live.sub(["clock", "top"]);          // the SYSTEM tab shows today's locked picks
    live.connect();
  }, [me]);

  useEffect(() => {
    setPref("td.adminTab", tab);
    history.replaceState(null, "", `#${tab}`);
  }, [tab]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
      const i = TABS.findIndex((x) => x.key === e.key);
      const alt = e.altKey && /^[1-3]$/.test(e.key) ? Number(e.key) - 1 : -1;
      const j = i >= 0 ? i : alt;
      if (j >= 0) {
        e.preventDefault();
        setTab(TABS[j].id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!me) return <main className="login-page"><div className="panel login-card"><p className="small muted">{err || "Loading…"}</p></div></main>;
  if (!me.admin) {
    return (
      <main className="login-page">
        <div className="panel login-card">
          <h1 className="login-title">Admins only</h1>
          <p className="small muted">You are signed in as <b>{me.user}</b> ({me.role}). The admin page is for admin accounts — ask an
            admin to change your role.</p>
          <div className="login-actions">
            <a className="btn primary" href="/">Open the terminal</a>
            <button type="button" className="btn ghost" onClick={() => void logout()}>Log out</button>
          </div>
        </div>
      </main>
    );
  }
  return (
    <div className="app admin-app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">TD</span>
          <span>TRADEDESK<span className="brand-sub"> · ADMIN</span></span>
        </div>
        <span className="admin-badge">ADMIN</span>
        <div className="top-right" style={{ marginLeft: "auto" }}>
          <span className="user-chip" title={`Signed in as ${me.user}`}>{me.user}<b> · admin</b></span>
          <a className="btn" href="/" title="Back to the trading terminal">Terminal</a>
          <button type="button" className="btn ghost" onClick={() => void logout()}>Log out</button>
        </div>
      </header>
      <nav className="tabs" role="tablist" aria-label="Admin sections">
        {TABS.map((t) => (
          <button key={t.id} role="tab" type="button" className="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)} title={t.sub}>
            <span className="fkey">{t.key}</span>{t.label}
          </button>
        ))}
      </nav>
      <main style={{ minHeight: 0, display: "grid" }} role="tabpanel" aria-label={`${tab} section`}>
        <Suspense fallback={<Empty>Loading…</Empty>}>
          {tab === "users" && <Users />}
          {tab === "history" && <History />}
          {tab === "system" && <System />}
        </Suspense>
      </main>
      <footer className="statusbar" aria-label="Status">
        <span>ADMIN <b>{me.user}</b></span><span className="sep">│</span>
        <span>{TABS.find((t) => t.id === tab)?.sub}</span>
        <span style={{ marginLeft: "auto" }}>Changes here apply to everyone using this TradeDesk</span>
      </footer>
    </div>
  );
}

