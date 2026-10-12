import { useCallback, useEffect, useState } from "react";
import { Empty, Panel, Seg } from "../components/ui";
import { api, ApiError } from "../lib/api";
import { dateIST, timeIST } from "../lib/format";

/* Admin page · USERS: accounts (admin = this page and history pulls, viewer = the terminal), your own
   password, and the legacy access keys from .env. */

type Role = "admin" | "viewer";
interface UserRow { name: string; role: Role; disabled: boolean; created: number | null; by: string | null; last_login: number | null }
interface UsersResp { users: UserRow[]; me: string; online: string[]; access_keys: string[] }

const when = (ts: number | null) => (ts ? `${dateIST(ts)} ${timeIST(ts)}` : "never");

function PasswordInput(props: { id: string; label: string; value: string; onChange: (v: string) => void; auto: string }) {
  return (
    <div className="field">
      <label htmlFor={props.id}>{props.label}</label>
      <input id={props.id} type="password" required minLength={8} maxLength={128} autoComplete={props.auto}
        value={props.value} onChange={(e) => props.onChange(e.target.value)} />
    </div>
  );
}

type Act = (p: Promise<unknown>, ok: string, onOk?: () => void) => void;

function UserRowView({ u, me, online, act }: { u: UserRow; me: string; online: boolean; act: Act }) {
  const self = u.name === me;
  const [pw, setPw] = useState<string | null>(null);
  const [confirmDel, setConfirmDel] = useState(false);
  useEffect(() => {
    if (!confirmDel) return;
    const id = window.setTimeout(() => setConfirmDel(false), 4000);
    return () => window.clearTimeout(id);
  }, [confirmDel]);
  const patch = (body: object, ok: string, onOk?: () => void) =>
    act(api(`/api/admin/users/${encodeURIComponent(u.name)}`, { method: "PATCH", body: JSON.stringify(body) }), ok, onOk);
  return (
    <>
      <tr style={{ cursor: "default" }} className={u.disabled ? "dim" : ""}>
        <td className="sym">
          <span className={`dot ${online ? "on" : "off"}`} title={online ? "online now" : "offline"} aria-label={online ? "online" : "offline"} />{" "}
          {u.name}{self && <small>you</small>}
        </td>
        <td className="l">
          <select className="cell-select" value={u.role} disabled={self} aria-label={`Role of ${u.name}`}
            title={self ? "Ask another admin to change your role" : undefined}
            onChange={(e) => patch({ role: e.target.value }, `${u.name} is now ${e.target.value}.`)}>
            <option value="viewer">viewer</option>
            <option value="admin">admin</option>
          </select>
        </td>
        <td className={`l ${u.disabled ? "down" : "up"}`}>{u.disabled ? "disabled" : "active"}</td>
        <td className="l muted">{when(u.last_login)}</td>
        <td className="l muted">{when(u.created)}{u.by ? ` · by ${u.by}` : ""}</td>
        <td className="l">
          <div className="row-actions">
            {!self && <button type="button" className="btn" onClick={() => setPw(pw === null ? "" : null)}>Password</button>}
            {!self && (
              <button type="button" className="btn" onClick={() => patch({ disabled: !u.disabled }, `${u.name} ${u.disabled ? "enabled" : "disabled — signed out everywhere"}.`)}>
                {u.disabled ? "Enable" : "Disable"}
              </button>
            )}
            {!self && (
              <button type="button" className={`btn danger ${confirmDel ? "armed" : ""}`}
                onClick={() => (confirmDel ? act(api(`/api/admin/users/${encodeURIComponent(u.name)}`, { method: "DELETE" }), `${u.name} deleted.`) : setConfirmDel(true))}>
                {confirmDel ? "Confirm delete" : "Delete"}
              </button>
            )}
          </div>
        </td>
      </tr>
      {pw !== null && (
        <tr className="sub-row">
          <td colSpan={6}>
            <form className="inline-form" onSubmit={(e) => { e.preventDefault(); patch({ password: pw }, `New password set for ${u.name} — their other sessions ended.`, () => setPw(null)); }}>
              <label htmlFor={`pw-${u.name}`}>New password for {u.name}</label>
              <input id={`pw-${u.name}`} type="password" required minLength={8} maxLength={128} autoComplete="new-password" autoFocus
                value={pw} onChange={(e) => setPw(e.target.value)} />
              <button type="submit" className="btn primary">Set password</button>
              <button type="button" className="btn ghost" onClick={() => setPw(null)}>Cancel</button>
            </form>
          </td>
        </tr>
      )}
    </>
  );
}

export default function UsersTab() {
  const [data, setData] = useState<UsersResp | null>(null);
  const [err, setErr] = useState("");
  const [msg, setMsg] = useState("");
  const [name, setName] = useState("");
  const [pw, setPw] = useState("");
  const [role, setRole] = useState<Role>("viewer");
  const [old, setOld] = useState("");
  const [n1, setN1] = useState("");
  const [n2, setN2] = useState("");
  const [mine, setMine] = useState("");
  const [addMsg, setAddMsg] = useState("");
  const [adding, setAdding] = useState(false);

  const load = useCallback(() => {
    api<UsersResp>("/api/admin/users").then((d) => { setData(d); setErr(""); }).catch((e: ApiError) => setErr(e.message));
  }, []);
  useEffect(() => {
    load();
    const id = window.setInterval(load, 10000);
    return () => window.clearInterval(id);
  }, [load]);
  const act: Act = (p, ok, onOk) => {
    setMsg("");
    p.then(() => { setMsg(ok); onOk?.(); load(); }).catch((e: ApiError) => setMsg(e.message));
  };

  if (err && !data) return <div className="desk"><Empty>{err}</Empty></div>;
  const lost = /admin login required|login required/i.test(err);
  const online = new Set(data?.online ?? []);
  const admins = (data?.users ?? []).filter((u) => u.role === "admin" && !u.disabled).length;
  return (
    <div className="desk desk-users">
      {lost && <div className="banner warn" role="alert" style={{ gridColumn: "1 / -1" }}>{err} — the list below is out of date.</div>}
      <Panel code="USERS" title="Accounts" area="list" className="cap-m"
        sub={data ? `${data.users.length} users · ${admins} admin${admins === 1 ? "" : "s"} · admins open this page, viewers use the terminal` : ""}
        foot={msg ? <span role="status">{msg}</span> : "Disabling a user or changing their password signs them out everywhere."}>
        {!data ? <Empty>Loading…</Empty> : (
          <table className="tbl users-tbl">
            <thead><tr><th className="l">User</th><th className="l">Role</th><th className="l">Status</th><th className="l">Last login</th><th className="l">Created</th><th className="l">Actions</th></tr></thead>
            <tbody>
              {data.users.map((u) => <UserRowView key={u.name} u={u} me={data.me} online={online.has(u.name)} act={act} />)}
            </tbody>
          </table>
        )}
      </Panel>
      <Panel code="ADD" title="Add a user" area="add">
        <form className="form" onSubmit={(e) => {
          e.preventDefault();
          setAdding(true);
          setAddMsg("");
          api("/api/admin/users", { method: "POST", body: JSON.stringify({ username: name, password: pw, role }) })
            .then(() => { setAddMsg(`${name.toLowerCase()} added as ${role}.`); setName(""); setPw(""); load(); })
            .catch((er: ApiError) => setAddMsg(er.message))
            .finally(() => setAdding(false));
        }}>
          <div className="field">
            <label htmlFor="nu">User name</label>
            <input id="nu" required pattern="[A-Za-z0-9][A-Za-z0-9._\-]{1,31}" title="2-32 letters, digits, . _ -" autoComplete="off"
              spellCheck={false} value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <PasswordInput id="np" label="Password (8+ characters)" value={pw} onChange={setPw} auto="new-password" />
          <div className="field">
            <span className="field-label">Role</span>
            <Seg label="Role" value={role} onChange={setRole}
              options={[{ v: "viewer", label: "Viewer", title: "Uses the terminal" }, { v: "admin", label: "Admin", title: "Also this page: users, history pulls, system" }]} />
          </div>
          <button type="submit" className="btn primary" style={{ height: 30 }} disabled={adding}>Add user</button>
          {addMsg && <span className="small muted" role="status" style={{ gridColumn: "1 / -1" }}>{addMsg}</span>}
        </form>
      </Panel>
      <Panel code="YOU" title="Your password" area="mine">
        <form className="form" onSubmit={(e) => {
          e.preventDefault();
          if (n1 !== n2) { setMine("The new passwords differ."); return; }
          api("/api/auth/password", { method: "POST", body: JSON.stringify({ old, new: n1 }) })
            .then(() => { setMine("Password changed — your other sessions ended."); setOld(""); setN1(""); setN2(""); })
            .catch((er: ApiError) => setMine(er.message));
        }}>
          <PasswordInput id="op" label="Current password" value={old} onChange={setOld} auto="current-password" />
          <PasswordInput id="n1" label="New password" value={n1} onChange={setN1} auto="new-password" />
          <PasswordInput id="n2" label="Repeat new password" value={n2} onChange={setN2} auto="new-password" />
          <button type="submit" className="btn primary" style={{ height: 30 }}>Change password</button>
          {mine && <span className="small muted" role="status" style={{ gridColumn: "1 / -1" }}>{mine}</span>}
        </form>
        <div className="banner">
          Access keys from <code>ACCESS_KEYS</code> in <code>.env</code> still open the terminal (share links with{" "}
          <code>?key=…</code>), never this page{data?.access_keys.length ? `: ${data.access_keys.join(", ")}` : " — none set"}.
        </div>
      </Panel>
    </div>
  );
}
