/** Login session helpers shared by the terminal, /login and /admin. The session itself is an HttpOnly
 *  cookie set by the gateway — scripts never see it. */
import { api } from "./api";
import { setPref } from "./storage";

export interface Me {
  user: string | null;
  role: "admin" | "viewer" | null;
  via: "session" | "key" | "local" | null;   // login session · legacy access key · no accounts configured
  admin: boolean;
  login_required: boolean;
  setup: boolean;        // no accounts exist yet
  can_setup: boolean;    // …and this browser is on the host itself, so it may create the first admin
}

export const getMe = () => api<Me>("/api/auth/me");

/** A path on this site only (no open redirect): parsed like the browser will, so tricks such as "/\t/evil.com"
 *  (tabs and newlines are dropped by URL parsing) or "/\\evil.com" cannot leave the origin. Never carries ?key=. */
export function safeNext(raw: string | null | undefined, fallback = "/"): string {
  if (!raw || !raw.startsWith("/")) return fallback;
  try {
    const u = new URL(raw, location.origin);
    if (u.origin !== location.origin) return fallback;
    u.searchParams.delete("key");
    return u.pathname + u.search + u.hash;
  } catch {
    return fallback;
  }
}

export function toLogin(next: string = location.pathname + location.search + location.hash): void {
  location.replace(`/login?next=${encodeURIComponent(safeNext(next))}`);
}

export async function logout(): Promise<void> {
  try {
    await api("/api/auth/logout", { method: "POST" });     // only the gateway can clear the HttpOnly cookie
  } catch {
    window.alert("Could not reach the gateway — you are still signed in. Try again in a moment.");
    return;
  }
  setPref("td.key", "");
  location.replace("/login");
}
