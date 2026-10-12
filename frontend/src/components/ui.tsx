import type { ReactNode } from "react";
import { cls, pct } from "../lib/format";

export function Panel(props: {
  code?: string; title: string; sub?: ReactNode; actions?: ReactNode; children: ReactNode;
  area?: string; pad?: boolean; foot?: ReactNode; className?: string; id?: string; bar?: ReactNode;
}) {
  return (
    <section className={`panel ${props.className ?? ""}`} style={props.area ? { gridArea: props.area } : undefined}
      aria-labelledby={props.id ? `${props.id}-t` : undefined}>
      <header className="panel-head">
        {props.code && <span className="panel-code">{props.code}</span>}
        <h2 className="panel-title" id={props.id ? `${props.id}-t` : undefined} style={{ margin: 0 }}>{props.title}</h2>
        {props.sub && <span className="panel-sub">{props.sub}</span>}
        {props.actions && <div className="panel-actions">{props.actions}</div>}
      </header>
      {props.bar && <div className="panel-bar">{props.bar}</div>}
      <div className={`panel-body ${props.pad ? "pad" : ""}`}>{props.children}</div>
      {props.foot && <footer className="panel-foot">{props.foot}</footer>}
    </section>
  );
}

export function Seg<T extends string>(props: { value: T; options: { v: T; label: ReactNode; title?: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="seg" role="group" aria-label={props.label}>
      {props.options.map((o) => (
        <button key={o.v} type="button" aria-pressed={props.value === o.v} title={o.title} onClick={() => props.onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Chg({ v, d = 2, arrow = true }: { v: number | null | undefined; d?: number; arrow?: boolean }) {
  return <span className={cls(v)}>{pct(v, d, arrow)}</span>;
}

/** Centered bar for z-scores / signed values in [-max, +max]. */
export function ZBar({ v, max = 3, title }: { v: number; max?: number; title?: string }) {
  const w = Math.min(50, (Math.abs(v) / max) * 50);
  return (
    <div className="bar center" title={title ?? v.toFixed(2)} role="img" aria-label={`${title ?? "value"} ${v.toFixed(2)}`}>
      <i className={v >= 0 ? "pos" : "neg"} style={{ width: `${w}%` }} />
    </div>
  );
}

export function Spark({ data, width = 84, height = 24, color }: { data: number[]; width?: number; height?: number; color?: string }) {
  if (!data || data.length < 2) return <svg width={width} height={height} aria-hidden="true" />;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const rng = max - min || 1;
  const step = width / (data.length - 1);
  const pts = data.map((v, i) => `${(i * step).toFixed(1)},${(height - 2 - ((v - min) / rng) * (height - 4)).toFixed(1)}`).join(" ");
  const up = data[data.length - 1] >= data[0];
  return (
    <svg width={width} height={height} aria-hidden="true">
      <polyline points={pts} fill="none" stroke={color ?? (up ? "var(--up)" : "var(--down)")} strokeWidth="1.4" />
    </svg>
  );
}

export function Kpi({ label, value, sub, tone, title }: { label: string; value: ReactNode; sub?: ReactNode; tone?: string; title?: string }) {
  const tip = title ?? (typeof sub === "string" ? `${label}: ${sub}` : label);
  return (
    <div className="kpi" title={tip}>
      <div className="kpi-label">{label}</div>
      <div className={`kpi-val ${tone ?? ""}`}>{value}</div>
      {sub !== undefined && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function Icon({ name, size = 14 }: { name: "search" | "bell" | "x" | "help" | "refresh" | "trash" | "plus" | "layers"; size?: number }) {
  const p: Record<string, ReactNode> = {
    search: <><circle cx="11" cy="11" r="7" /><path d="M20 20l-3.5-3.5" /></>,
    bell: <><path d="M6 8a6 6 0 1112 0c0 7 3 9 3 9H3s3-2 3-9" /><path d="M10.3 21a1.94 1.94 0 003.4 0" /></>,
    x: <path d="M18 6L6 18M6 6l12 12" />,
    help: <><circle cx="12" cy="12" r="10" /><path d="M9.1 9a3 3 0 015.8 1c0 2-3 3-3 3M12 17h.01" /></>,
    refresh: <><path d="M21 12a9 9 0 11-3-6.7L21 8" /><path d="M21 3v5h-5" /></>,
    trash: <><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" /></>,
    plus: <path d="M12 5v14M5 12h14" />,
    layers: <><path d="M12 2l10 5-10 5L2 7l10-5z" /><path d="M2 17l10 5 10-5M2 12l10 5 10-5" /></>,
  };
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{p[name]}</svg>
  );
}
