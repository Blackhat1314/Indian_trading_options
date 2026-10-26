/* Small building blocks of Basic mode: cards, stock badges, price changes, an ⓘ tip, a loader hook. */
import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useLive } from "../lib/store";

export const ok = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const INR = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2, minimumFractionDigits: 2 });
const INR0 = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
export const rupees = (v: number | null | undefined, whole = false) => (ok(v) ? `₹${(whole ? INR0 : INR).format(v)}` : "—");
export const num = (v: number | null | undefined) => (ok(v) ? INR0.format(v) : "—");

export function Card({ title, sub, action, children, className, id }: {
  title?: ReactNode; sub?: ReactNode; action?: ReactNode; children: ReactNode; className?: string; id?: string;
}) {
  return (
    <section className={`b-card ${className ?? ""}`} aria-labelledby={id && title ? `${id}-t` : undefined}>
      {(title || action) && (
        <header className="b-card-head">
          <div>
            {title && <h2 className="b-card-title" id={id ? `${id}-t` : undefined}>{title}</h2>}
            {sub && <p className="b-card-sub">{sub}</p>}
          </div>
          {action && <div className="b-card-action">{action}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

const PALETTE = ["#2563eb", "#0e9384", "#7a5af8", "#dd2590", "#e04f16", "#15803d", "#0891b2", "#9333ea"];

export function Badge({ symbol, size = 34 }: { symbol: string; size?: number }) {
  let h = 0;
  for (const ch of symbol) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  const c = PALETTE[h % PALETTE.length];
  return (
    <span className="b-badge" aria-hidden="true" style={{ width: size, height: size, color: c, background: `${c}14`, borderColor: `${c}33` }}>
      {symbol.replace(/[^A-Z0-9]/gi, "").slice(0, 2).toUpperCase()}
    </span>
  );
}

/** ▲ 1.24% in green / ▼ 0.80% in red: the sign is in the arrow too, never colour alone. */
export function Change({ v, d = 2, suffix = "%" }: { v: number | null | undefined; d?: number; suffix?: string }) {
  if (!ok(v)) return <span className="b-flat">—</span>;
  const tone = v > 0.004 ? "b-up" : v < -0.004 ? "b-down" : "b-flat";
  return <span className={tone}>{v > 0.004 ? "▲" : v < -0.004 ? "▼" : "•"} {Math.abs(v).toFixed(d)}{suffix}</span>;
}

export function Tip({ text }: { text: string }) {
  return <span className="b-tip" tabIndex={0} role="note" aria-label={text} data-tip={text}>i</span>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="b-empty">{children}</div>;
}

export function StockLink({ symbol, children }: { symbol: string; children: ReactNode }) {
  const openStock = useLive((s) => s.openStock);
  return <button type="button" className="b-link-row" onClick={() => openStock(symbol)}>{children}</button>;
}

/** Fetch once per `deps`, keep the last good value while reloading. */
export function useLoad<T>(load: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const ref = useRef(load);
  ref.current = load;
  const [n, setN] = useState(0);
  const reload = useCallback(() => setN((x) => x + 1), []);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    ref.current().then((r) => { if (alive) { setData(r); setError(null); } })
      .catch((e) => { if (alive) setError(e); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, n]);
  return { data, error, loading, reload };
}

export function errText(e: unknown): string {
  if (!e) return "";
  const m = (e as { message?: string }).message ?? String(e);
  return m.length > 160 ? `${m.slice(0, 160)}…` : m;
}
