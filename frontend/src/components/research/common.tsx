/* Small building blocks shared by the research desks: the advice disclaimer (honesty rule: every desk shows it),
 * stat tiles, warning lists and API-error notes, so the four desks read and fail the same way. */
import type { CSSProperties, ReactNode } from "react";
import { ApiError } from "../../lib/api";
import "./research.css";

/** Readable message for anything a research.* call can throw (ApiError carries the gateway's `detail`). */
export function errorText(e: unknown): string {
  if (e instanceof ApiError) {
    const m = typeof e.message === "string" && e.message && e.message !== "[object Object]" ? e.message : "";
    if (e.status === 401) return "Signed out — log in again.";
    if (e.status === 404) return m || "Not found.";
    if (e.status === 422) return m || "Request rejected — check the inputs.";
    if (e.status === 429) return m || "Too many requests — try again in a minute.";
    return m || `Gateway error (HTTP ${e.status}).`;
  }
  if (e instanceof TypeError && /fetch|network|load failed/i.test(e.message)) return "Gateway unreachable — is TradeDesk running?";
  if (e instanceof Error) return e.message;
  return String(e);
}

/** "Research tool, not investment advice" note; children add a desk-specific sentence. */
export function Disclaimer({ children, style }: { children?: ReactNode; style?: CSSProperties }) {
  return (
    <p className="rs-disclaimer" role="note" style={style}>
      <b>Research tool, not investment advice.</b> Forecasts are probability ranges estimated from past data, not
      promises — prices can and do move outside every band. {children}
    </p>
  );
}

/** One figure: label (sentence case), value, optional sub line. tone: "up" | "down" | "amber" | "cyan" | "muted". */
export function Stat({ label, value, sub, tone, title }: { label: string; value: ReactNode; sub?: ReactNode; tone?: string; title?: string }) {
  return (
    <div className="rs-stat" title={title}>
      <div className="rs-stat-label">{label}</div>
      <div className={`rs-stat-val ${tone ?? ""}`}>{value}</div>
      {sub !== undefined && sub !== null && <div className="rs-stat-sub">{sub}</div>}
    </div>
  );
}

/** Auto-fit grid of <Stat>s; `min` = minimum tile width in px (default 120). */
export function Stats({ children, min = 120 }: { children: ReactNode; min?: number }) {
  return <div className="rs-stats" style={{ "--rs-stat-min": `${min}px` } as CSSProperties}>{children}</div>;
}

/** Amber list of model/data warnings; renders nothing when empty. */
export function Warnings({ items, label = "Warnings" }: { items?: (string | null | undefined)[] | null; label?: string }) {
  const list = (items ?? []).filter((s): s is string => !!s && !!s.trim());
  if (!list.length) return null;
  return (
    <ul className="rs-warnings" aria-label={label}>
      {list.map((w, i) => <li key={i}>{w}</li>)}
    </ul>
  );
}

/** Error state for a failed research call, with an optional retry. */
export function ErrorNote({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  return (
    <div className="rs-error" role="alert">
      <span>{errorText(error)}</span>
      {onRetry && <button type="button" className="btn" onClick={onRetry}>Retry</button>}
    </div>
  );
}
