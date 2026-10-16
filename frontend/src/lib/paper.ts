/** PAPER TRADE desk — the paper-trading service's REST contract (gateway /api/paper/*) and the "paper" ws topic.
 *  Money in ₹; percentages are fractions (0.0123 = 1.23 %) unless the name ends in _pct_points; timestamps are epoch
 *  seconds; dates are yyyymmdd ints. Simulated fills only — nothing here places a real order. */
import { api } from "./api";

export type Strategy = "intraday" | "swing";
export type Side = "LONG" | "SHORT";

export interface Reason { feature: string; label: string; contrib_pct?: number; z?: number; weight?: number }
export interface Regime { lights: number; risk_scale: number; text: string }

export interface TradeReasons {
  setup: string;                 // "Overextended fade", "Opening-range breakout", "Momentum leader"…
  summary: string;               // one plain-English sentence on why the trade was taken
  drivers: Reason[];             // intraday: model drivers (contrib_pct, signed); swing: factor z-scores
  facts: Record<string, number | string | null>;
  pred?: number; edge?: number; cost?: number;   // intraday: predicted return, edge after costs, estimated cost
  score?: number;                // swing factor score
  sizing: string;
  regime?: Regime;
}

export interface Position {
  id: string; strategy: Strategy; symbol: string; side: Side; qty: number;
  entry: number; entry_ts: number; ltp: number | null; stop: number; initial_stop: number;
  pnl: number; pnl_pct: number; r_mult: number | null; decision: string;
  days_held?: number; reasons: TradeReasons;
}

export interface Pending { symbol: string; strategy: "swing"; planned_at: number; reasons: TradeReasons }
export interface StratSettings { enabled: boolean; capital: number; leverage?: number }

export interface Stats {
  trades: number; win_rate: number | null; profit_factor: number | null; avg_pnl: number | null;
  avg_r: number | null; best: number | null; worst: number | null; max_dd: number | null; charges: number;
}

export interface PaperService {
  running: boolean; clock: number | null; phase: string | null; mode: "live" | "sim";
  models: { source: "local" | "shipped" | "none"; trained_to: number | null };
  note?: string;
}

export interface IntradayAccount {
  capital: number; leverage: number; equity: number; pnl_total: number; pnl_today: number; realized_today: number;
  unrealized: number; gross_exposure: number; buying_power: number; trades_today: number;
  loss_limit_hit: boolean; positions: Position[]; stats: Stats;
}

export interface SwingAccount {
  capital: number; equity: number; cash: number; invested: number; pnl_total: number; unrealized: number;
  positions: Position[]; pending: Pending[]; stats: Stats;
  regime: (Regime & { asof: number | null }) | null;
}

export interface PaperState {
  user: string;
  service: PaperService;
  settings: { intraday: StratSettings; swing: StratSettings };
  intraday: IntradayAccount;
  swing: SwingAccount;
  equity_curve: { date: number; intraday: number; swing: number }[];
}

export interface Trade {
  id: string; strategy: Strategy; symbol: string; side: Side; qty: number;
  entry: number; exit: number; entry_ts: number; exit_ts: number; why_exit: string;
  exit_note: string;
  pnl: number; pnl_gross: number; pnl_pct: number; r_mult: number | null; charges: Record<string, number>; slippage: number;
  holding: string;
  decision: string; reasons: TradeReasons;
}

export interface Proposal {
  symbol: string; side: Side; price: number; stop: number; setup: string; pred: number; edge: number; cost: number;
  drivers: Reason[]; facts: Record<string, number | null>;
}

export interface Decision {
  time: "09:30" | "10:00"; at: number | null; status: "pending" | "done" | "skipped"; note: string; considered: number;
  proposals: Proposal[];
}

export interface SwingCandidate { symbol: string; score: number; setup: string; drivers: Reason[]; price: number; stop: number }

export interface PaperSignals {
  date: string | null;
  intraday: { decisions: Decision[] };
  swing: { asof: number | null; regime: Regime | null; candidates: SwingCandidate[] };
}

export interface PStats {
  start: number; end: number; days: number; return: number; cagr: number; sharpe: number; max_dd: number; trades: number;
  win_rate?: number; profit_factor?: number; avg_ret?: number; charges?: number;
}

export interface Report {
  kind: Strategy; generated_at: number; capital: number; leverage?: number; method: string;
  all: PStats; before_holdout?: PStats; holdout?: PStats; holdout_from?: number;
  benchmark?: PStats & { name: string }; holdout_benchmark?: PStats;
  equity: [number, number][]; benchmark_equity?: [number, number][];
  monthly?: { month: number; pnl: number }[];
  by_setup?: { setup: string; n: number; win: number; pnl: number }[];
  by_side?: Record<string, { n: number; pnl: number }>;
  source?: "local" | "shipped";                  // run on this server, or the report shipped with the app
}

export interface Evidence { intraday: Report | null; swing: Report | null }

export interface SettingsPatch {
  intraday?: { enabled?: boolean; capital?: number; leverage?: number };
  swing?: { enabled?: boolean; capital?: number };
}

export type BookScope = Strategy | "all";

export const CAPITAL_MIN = 10_000;
export const CAPITAL_MAX = 100_000_000;     // ₹10 crore
export const LEV_MIN = 1;
export const LEV_MAX = 5;

const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const paperApi = {
  state: (signal?: AbortSignal) => api<PaperState>("/api/paper/state", { signal }),
  settings: (patch: SettingsPatch) => api<PaperState>("/api/paper/settings", post(patch)),
  reset: (strategy: BookScope) => api<PaperState>("/api/paper/reset", post({ strategy })),
  book: (strategy: BookScope, limit = 200) => api<{ trades: Trade[] }>(`/api/paper/book?strategy=${strategy}&limit=${limit}`),
  signals: () => api<PaperSignals>("/api/paper/signals"),
  evidence: () => api<Evidence>("/api/paper/evidence"),
};
