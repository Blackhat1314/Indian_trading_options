/** REST helper. The gateway origin defaults to the page origin (served by the gateway or the Vite proxy).
 *  Friends can connect to a shared gateway with ?key=<access key> (stored locally afterwards). */
import { getPref, setPref } from "./storage";

const params = new URLSearchParams(location.search);
const urlKey = params.get("key");
if (urlKey) setPref("td.key", urlKey);

export const ACCESS_KEY = urlKey ?? getPref("td.key") ?? "";   // legacy share links (ACCESS_KEYS)
export const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export function wsUrl(): string {
  const base = API_BASE || location.origin;
  const u = new URL("/ws", base);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  if (ACCESS_KEY) u.searchParams.set("key", ACCESS_KEY);
  return u.toString();
}

export class ApiError extends Error {
  constructor(public status: number, msg: string) {
    super(msg);
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  // X-TD-Request: the gateway requires it on state changes (a cross-site page cannot send it with the cookie)
  const headers: Record<string, string> = { "Content-Type": "application/json", "X-TD-Request": "1" };
  if (ACCESS_KEY) headers["X-Access-Key"] = ACCESS_KEY;
  const r = await fetch(API_BASE + path, { credentials: "same-origin", ...init, headers: { ...headers, ...(init?.headers as object) } });
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const j = await r.json();
      msg = j.detail ?? msg;
    } catch {
      /* ignore */
    }
    throw new ApiError(r.status, msg);
  }
  return r.json() as Promise<T>;
}
