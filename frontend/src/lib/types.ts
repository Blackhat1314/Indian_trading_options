export type Phase = "OPEN" | "PRE_OPEN" | "PRE_OPEN_MATCH" | "POST_CLOSE" | "CLOSED";

export interface Instrument {
  sid: number;          // Dhan security id (unique only within a segment)
  key: number;          // unique instrument key used by quotes, bars and depth
  seg: string;
  symbol: string;
  kind: "EQ" | "FUT" | "IDX" | "MACRO";
  underlying: string;
  sector: string;
  lot: number;
}

export interface Clock { ts: number; phase: Phase; date: string; time: string; source: string; minute: number }

export interface IndexRow { name: string; sid: number; ltp: number; pc: number; chg: number; o: number; h: number; l: number }

export interface Breadth {
  n: number; adv: number; dec: number; unch: number; ad_ratio: number; above_vwap: number; above_vwap_pct: number;
  near_high: number; near_low: number; above_pdh: number; below_pdl: number; up_turnover: number;
  down_turnover: number; median_chg: number; hist: number[]; hist_bins: string[];
}

export interface SectorRow {
  sector: string; name: string; n: number; chg: number; median: number; adv: number; dec: number; turnover: number;
  best: { symbol: string; chg: number }; worst: { symbol: string; chg: number };
  members: { symbol: string; chg: number; sid: number; turnover: number }[];
}

export interface MoverRow { sid: number; symbol: string; ltp: number; chg: number; rvol: number; turnover: number; sector: string }

export interface Market {
  ts: number; phase: Phase; indices: IndexRow[]; breadth: Breadth; sectors: SectorRow[];
  movers: { gainers: MoverRow[]; losers: MoverRow[]; active: MoverRow[]; rvol: MoverRow[] };
}

export interface Factors { rvol: number; resid: number; vwap: number; orb: number; oi: number; news: number; sector: number; obi: number }

export interface RankRow {
  sid: number; symbol: string; sector: string; ltp: number; chg: number; gap: number; rvol: number; resid: number;
  vwap_atr: number; or_pos: number; orb: number; oi_tag: string; oi_chg: number; news: number; news_type: string | null;
  headline: string | null; obi: number; turnover: number; score: number; score_s: number; rank: number;
  eligible: boolean; f: Factors; atr_pct: number; vwap: number; orh: number; orl: number; high: number; low: number;
  open: number; pc: number;
}

export interface RankSnap { ts: number; minute: number; rows: RankRow[]; weights: Record<string, number> }

export interface Pick {
  sid: number; symbol: string; sector: string; entry: number; score: number; chips: string[]; f: Factors;
  rvol: number; gap: number; chg_at_lock: number; headline: string | null; invalidation: number; target: number;
  oi_tag: string; ltp?: number; chg?: number; pnl?: number; mfe?: number; mae?: number; status?: string;
  hit_at?: string | null; spark?: number[];
  p_up?: number | null; p_beat?: number | null;     // the 09:30 neural model's picks
}

export interface LockStats {
  universe_avg: number; long_avg: number; short_avg: number; long_hits_top5: number; long_hits_top20: number;
  short_hits_bottom5: number; leaders: { symbol: string; ret: number }[]; laggards: { symbol: string; ret: number }[];
}

/** The 09:30 neural model's picks, made at the same lock as the TOP5 (td.neural.live). */
export interface NeuralTop {
  status: "ok" | "pending" | "missing" | "no_edge" | "error" | "off";
  note?: string; verdict?: string; edge?: boolean; edge_note?: string | null; trained_at?: number;
  computed_at?: number; late_s?: number; n?: number; skipped?: number; uses?: string | null;
  long?: Pick[]; short?: Pick[]; since_lock?: LockStats | null;
}

export interface TopSnap {
  locked: boolean; provisional?: boolean; missed?: boolean; closed?: string; date?: string; lock_ts?: number; lock_time: string;
  seconds_to_lock?: number; long: Pick[]; short: Pick[];
  abstained?: { long: number; short: number };
  universe_since_lock?: null | LockStats;
  neural?: NeuralTop;
}

export interface FnoRow {
  sid: number; eq_sid: number | null; symbol: string; sector: string; ltp: number; chg: number; oi: number;
  oi_chg: number; oi_chg_abs: number; tag: string; basis: number; oi_value_cr: number; vol: number; lot: number;
}
export interface FnoSnap { groups: Record<"LB" | "SB" | "SC" | "LU", FnoRow[]>; counts: Record<string, number>; spurts: FnoRow[]; all: FnoRow[] }

export interface ChainRow {
  strike: number;
  ce_ltp: number; ce_oi: number; ce_oi_chg: number; ce_vol: number; ce_iv: number; ce_delta: number; ce_gamma: number;
  pe_ltp: number; pe_oi: number; pe_oi_chg: number; pe_vol: number; pe_iv: number; pe_delta: number; pe_gamma: number;
}
export interface ChainSnap {
  name: string; spot: number; atm: number; pcr_oi: number | null; pcr_vol: number | null; max_pain: number | null;
  atm_iv: number | null; straddle: number; expected_move_pct: number | null; skew_25d: number | null;
  total_gex: number; gamma_flip: number | null; call_wall: number; put_wall: number; ce_oi_total: number;
  pe_oi_total: number; ce_oi_chg_total: number; pe_oi_chg_total: number; rows: ChainRow[];
  gex: { strike: number; gex: number }[]; expiry?: string; expiries?: string[]; ts?: number;
  pcr_hist?: [number, number | null][];
}

export interface SetupStat { n: number; hit_rate: number; stop_rate: number; avg_fwd: number; median_fwd: number; target_pct: number; stop_pct: number; horizon: number }
export interface Zone { ts: number; top: number; bottom: number; dir: number; filled?: boolean; fill_pct?: number; mitigated?: boolean }
export interface SwingRow {
  symbol: string; sid: number | null; sector: string; close: number; chg: number; setup: string; trend: number; rsi: number;
  atr_pct: number; vol_ratio: number | null; rs63: number | null; rs_rank: number; dist_hh55: number | null;
  contraction: number | null; e20: number | null; e50: number | null; e200: number | null; atr: number | null;
  smc_trend: number; smc_event: string | null; smc_event_dir: number | null; smc_age: number | null;
  bull_fvg: Zone | null; bear_fvg: Zone | null; score: number; stop?: number; target?: number;
  stats: SetupStat | null; live: boolean;
}
export interface SwingSnap { ts: number; rows: SwingRow[]; stats: Record<string, SetupStat>; rules: { target: number; stop: number; horizon: number } }

export interface NewsItem {
  id: string; ts: number; title: string; text: string; source: string; category: string; type: string; sign: number;
  weight: number; novel: boolean; symbols: string[]; impacts: Record<string, number>; sentiment: string;
}

export interface Alert { id: string; user: string; symbol: string; type: string; value: number | string | null; note: string; repeat: boolean; active: boolean; created: number; fired: number }
export interface AlertFire { alert: Alert; ts: number; symbol: string; ltp: number; chg: number; msg: string }

export interface SmcResult {
  trend: number;
  swings: { idx: number; ts: number; price: number; kind: "H" | "L" }[];
  events: { ts: number; price: number; dir: number; kind: string; from_ts: number }[];
  fvgs: (Zone & { idx: number })[];
  obs: (Zone & { idx: number })[];
  sweeps: { ts: number; price: number; dir: number }[];
}
export interface ChartData {
  symbol: string; sid: number; tf: string; bars: [number, number, number, number, number, number][];
  vwap: [number, number][]; levels: Record<string, number | null>; smc: SmcResult;
}

export interface Depth { sid: number; ts?: number; bid?: [number, number, number][]; ask?: [number, number, number][]; pending?: boolean }
