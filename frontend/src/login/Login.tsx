import { useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";
import { getMe, logout, safeNext, type Me } from "../lib/auth";
import { setPref } from "../lib/storage";

/* /login — sign in with a user account; create the first admin (on the host itself, while there are no
   accounts); or use a legacy access key someone shared. */

const next = safeNext(new URLSearchParams(location.search).get("next"));

function Brand() {
  return (
    <div className="login-brand">
      <span className="brand-mark">TD</span>
      <span>TRADEDESK<span className="brand-sub"> · NSE TERMINAL</span></span>
    </div>
  );
}

function Field(props: { id: string; label: string; type?: string; value: string; onChange: (v: string) => void; auto?: string; focus?: boolean }) {
  return (
    <div className="field">
      <label htmlFor={props.id}>{props.label}</label>
      <input id={props.id} type={props.type ?? "text"} value={props.value} autoFocus={props.focus} required
        autoComplete={props.auto} spellCheck={false} onChange={(e) => props.onChange(e.target.value)} />
    </div>
  );
}

export default function Login() {
  const [me, setMe] = useState<Me | null>(null);
  const [mode, setMode] = useState<"login" | "key">("login");
  const [user, setUser] = useState("");
  const [pw, setPw] = useState("");
  const [pw2, setPw2] = useState("");
  const [key, setKey] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getMe().then(setMe).catch(() => setErr("Cannot reach the gateway — is TradeDesk running?"));
  }, []);

  const go = () => location.replace(next);
  const submit = (path: string, body: object) => {
    setBusy(true);
    setErr("");
    api(path, { method: "POST", body: JSON.stringify(body) })
      .then(getMe)
      .then((m) => {
        // the login only counts if the browser kept the session cookie (blocked cookies, http vs https)
        if (m.via === "session") go();
        else setErr("Logged in, but this browser did not keep the session cookie — allow cookies for this site (and use the same http/https address the host gave you).");
      })
      .catch((e: ApiError) => setErr(e.status === 401 ? "Wrong user name or password." : e.message))
      .finally(() => setBusy(false));
  };

  if (!me) {
    return <main className="login-page"><div className="panel login-card"><Brand /><p className="muted small">{err || "Loading…"}</p></div></main>;
  }

  const setup = me.setup && me.can_setup;
  let body;
  if (me.via === "session") {
    body = (
      <>
        <h1 className="login-title">Signed in</h1>
        <p className="small muted">You are signed in as <b>{me.user}</b> ({me.role}).</p>
        <div className="login-actions">
          <button type="button" className="btn primary" onClick={go}>Continue</button>
          {me.admin && next !== "/admin" && <a className="btn" href="/admin">Admin page</a>}
          <button type="button" className="btn ghost" onClick={() => void logout()}>Log out</button>
        </div>
      </>
    );
  } else if (setup && mode === "login") {
    const mismatch = pw2 !== "" && pw !== pw2;
    body = (
      <form onSubmit={(e) => { e.preventDefault(); if (!mismatch) submit("/api/auth/setup", { username: user, password: pw }); }}>
        <h1 className="login-title">Create the admin account</h1>
        <p className="small muted">No accounts exist yet. This first account is an admin: it can add users and open the admin
          page. Once it exists, everyone logs in to use the terminal.</p>
        <Field id="u" label="User name" value={user} onChange={setUser} auto="username" focus />
        <Field id="p" label="Password (8+ characters)" type="password" value={pw} onChange={setPw} auto="new-password" />
        <Field id="p2" label="Repeat password" type="password" value={pw2} onChange={setPw2} auto="new-password" />
        {mismatch && <p className="err small">The passwords differ.</p>}
        <div className="login-actions">
          <button type="submit" className="btn primary" disabled={busy || mismatch}>Create admin &amp; sign in</button>
          {!me.login_required && <a className="btn ghost" href="/">Skip — use the terminal without accounts</a>}
          {me.login_required && <button type="button" className="btn ghost" onClick={() => setMode("key")}>I have an access key</button>}
        </div>
      </form>
    );
  } else if (mode === "key") {
    body = (
      <form onSubmit={(e) => {
        e.preventDefault();
        const k = key.trim();
        if (!k) return;
        setErr("");
        api<Me>("/api/auth/me", { headers: { "X-Access-Key": k } }).then((m) => {
          if (m.via !== "key") { setErr("That access key was not accepted."); return; }
          setPref("td.key", k);
          location.replace(next === "/admin" ? "/" : next);
        }).catch((er: ApiError) => setErr(er.message));
      }}>
        <h1 className="login-title">Use an access key</h1>
        <p className="small muted">For share links (<code>?key=…</code>) from the host. Access keys open the terminal, not the admin page.</p>
        <Field id="k" label="Access key" value={key} onChange={setKey} auto="off" focus />
        <div className="login-actions">
          <button type="submit" className="btn primary">Continue</button>
          <button type="button" className="btn ghost" onClick={() => setMode("login")}>Log in with an account</button>
        </div>
      </form>
    );
  } else {
    body = (
      <form onSubmit={(e) => { e.preventDefault(); submit("/api/auth/login", { username: user, password: pw }); }}>
        <h1 className="login-title">Log in</h1>
        {me.setup && (
          <p className="banner warn small">No accounts exist yet. Set <code>ADMIN_USER</code> and <code>ADMIN_PASSWORD</code> in
            the host's <code>.env</code> and restart, or open <code>http://localhost:8000/login</code> on the host itself to
            create the first admin.</p>
        )}
        <Field id="u" label="User name" value={user} onChange={setUser} auto="username" focus />
        <Field id="p" label="Password" type="password" value={pw} onChange={setPw} auto="current-password" />
        <div className="login-actions">
          <button type="submit" className="btn primary" disabled={busy}>Log in</button>
          <button type="button" className="btn ghost" onClick={() => setMode("key")}>I have an access key</button>
          {!me.login_required && <a className="btn ghost" href="/">Back to the terminal</a>}
        </div>
      </form>
    );
  }
  return (
    <main className="login-page">
      <div className="panel login-card">
        <Brand />
        {body}
        {err && <p className="err small" role="alert">{err}</p>}
        <p className="tiny muted login-foot">Research tool · not investment advice</p>
      </div>
    </main>
  );
}
