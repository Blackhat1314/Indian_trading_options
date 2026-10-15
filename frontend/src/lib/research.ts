/** Research desks (AI ANALYST, SCREENER, MONTE CARLO, QUANT): response types of /api/research/... and fetch helpers.
 *  Single source of truth for the shapes the gateway returns (backend: td/services/research_api.py).
 *  Conventions: ts = epoch seconds of the bar's IST midnight (same as the lake); returns and probabilities
 *  are fractions (0.12 = 12 %) unless a field name ends in _pct; prices in rupees. */
import { api } from "./api";

// ------------------------------------------------------------------ common
export interface SearchHit { symbol: string; name: string; sector: string; in_universe: boolean; has_history: boolean }

export interface HistoryInfo {
  symbol: string; name: string; sector: string;
  bars: number; first: string | null; last: string | null; years: number;     // ISO dates of the daily bars held
  spot: number | null; spot_source: "live" | "close" | null;
  fetched: boolean;              // true when this call pulled missing history from Dhan
  warning: string | null;        // e.g. "only 1.3 years of daily history" or "Dhan token expired — using the lake"
  closes: [number, number][];    // [ts, close], daily, up to the last 5 years
}

// ------------------------------------------------------------------ Monte Carlo
export type McModel = "ensemble" | "garch" | "bootstrap" | "jump" | "regime" | "gbm";
export type McDrift = "capm" | "historical" | "zero" | "quant";

export interface McRequest {
  horizon_days: number;          // trading days, 5…756 (756 ≈ 3 years)
  paths: number;                 // 500…50 000 (paths × horizon is capped server-side)
  model: McModel;
  drift: McDrift;
  lookback_years: number;        // calibration window, 1…10
  targets: number[];             // up to 6 price levels → probability of touching / ending beyond
  seed?: number;
}

export interface McBand { t: number; ts: number; p5: number; p10: number; p25: number; p50: number; p75: number; p90: number; p95: number; mean: number }

export interface McHorizon {
  label: string;                 // "1M" "3M" "6M" "1Y" "2Y" "3Y", or "<n>D" for a horizon that is not a checkpoint
  days: number; ts: number;
  p5: number; p25: number; p50: number; p75: number; p95: number; mean: number;
  prob_up: number;               // P(price at horizon > spot)
  exp_ret: number;               // mean simple return
  median_ret: number;
  ann_median_ret: number;        // median return annualised (CAGR)
}

export interface McTarget { price: number; direction: "up" | "down"; prob_touch: number; prob_end_beyond: number }

export interface McGarch { omega: number; alpha: number; beta: number; nu: number; persistence: number; long_run_vol: number; current_vol: number }

export interface McResult {
  symbol: string; asof: string; spot: number; spot_source: "live" | "close";
  horizon_days: number; paths: number; model: McModel; drift: McDrift; seed: number; elapsed_ms: number;
  history: [number, number][];   // [ts, close], last ~1 year, for the chart's left side
  fan: McBand[];                 // ≤ 200 points; t = trading days ahead (t = 0 is spot)
  sample_paths: number[][];      // 30 simulated paths, values aligned with fan[i].t
  terminal: {
    edges: number[]; counts: number[];          // histogram of the final price: edges.length = counts.length + 1
    mean: number; median: number; std: number; skew: number;
    p1: number; p5: number; p25: number; p75: number; p95: number; p99: number;
    prob_up: number; exp_ret: number; median_ret: number;
  };
  risk: {                        // fractions of spot; losses positive
    var95: number; var99: number; cvar95: number; cvar99: number;
    prob_loss_10: number; prob_loss_20: number; prob_gain_20: number;
    max_dd_median: number; max_dd_p90: number; prob_dd_20: number;
  };
  horizons: McHorizon[];         // checkpoints inside the horizon (21, 63, 126, 252, 504, 756) + the horizon
  targets: McTarget[];
  inputs: {
    bars: number; first: string; last: string; lookback_years: number;
    mu_annual: number; sigma_annual: number;     // historical log-return mean / vol over the lookback
    drift_annual: number; drift_note: string;    // the drift actually simulated (arithmetic, annual) and why
    beta: number | null; rf: number; erp: number;
    garch: McGarch | null;
    jump: { lambda: number; mu_j: number; sigma_j: number } | null;          // jumps/year, mean & sd of log jump
    regime: { labels: string[]; mu_annual: number[]; vol_annual: number[]; p_stay: number[]; current: number; probs: number[] } | null;
    bootstrap_block: number | null;              // mean block length (days)
    components: { model: McModel; weight: number }[] | null;   // ensemble members
  };
  warnings: string[];
}

export interface McBacktestRequest { horizon_days: number; model: McModel; drift: McDrift; lookback_years: number; origins: number; paths: number }

export interface McBacktest {
  symbol: string; model: McModel; drift: McDrift; horizon_days: number; origins: number; paths: number; elapsed_ms: number;
  coverage: { level: number; hit_rate: number; expected: number }[];   // central intervals 0.5 / 0.8 / 0.9
  pit_hist: number[];            // 10 bins of the probability-integral transform, fractions summing to 1 (flat 0.1 = calibrated)
  crps: number; crps_naive: number; skill: number;   // CRPS in % of spot; skill = 1 − crps / crps_naive
  direction_hit: number;         // share of origins where the simulated median got the direction right
  effective_n: number;           // non-overlapping windows (honest sample size)
  points: { ts: number; target_ts: number; spot: number; realized: number; p5: number; p50: number; p95: number }[];   // ts = origin, target_ts = date of the realised price
  verdict: string;               // plain-language summary
  warnings: string[];
}

// ------------------------------------------------------------------ Quant model
export type Signal = "UP" | "DOWN" | "NEUTRAL";
export type Conf = "low" | "medium" | "high";

export interface QuantHorizon {
  h: number; label: string;      // 1 "Next day", 5 "1 week", 21 "1 month"
  p_up: number;                  // calibrated probability the close h days ahead is above today's
  exp_ret: number; ret_sd: number;
  range68: [number, number]; range95: [number, number];   // price ranges
  price_median: number;
  signal: Signal; confidence: Conf;
  oos_auc: number | null; oos_acc: number | null; base_rate: number | null;   // walk-forward, whole universe
  edge: boolean;                 // the walk-forward verdict (AUC ≥ 0.52, 2 s.e. above 0.5, every fold ≥ 0.5, calibrated spread)
  edge_note: string | null;      // why there is no / only a weak edge
  flat: boolean;                 // calibration flattened this horizon: p_up = the base rate for every stock, exp_ret 0, median = spot
}

export interface QuantComponent {
  key: string; label: string;    // e.g. "trend" "Kalman trend"
  value: number; display: string;
  z: number | null;              // standardised reading
  reading: string;               // one plain sentence
  tone: "up" | "down" | "neutral";
}

export interface QuantContribution { feature: string; label: string; value: number; contribution: number }   // calibrated log-odds (b·coef·z), 1-month model

export interface QuantMetric {
  h: number; auc: number; acc: number; brier: number; brier_base: number;
  logloss: number; logloss_base: number; base_rate: number; ic: number; n_test: number; folds: number;
  fold_auc: (number | null)[]; auc_sd: number | null; auc_se: number | null; auc_pooled: number | null;   // auc = mean of folds
  calib_b: number;               // calibration spread learnt walk-forward (0 = probabilities flattened to the base rate)
  edge: boolean; edge_note: string | null;
}

export interface QuantModelInfo {
  status: "ready" | "training" | "missing" | "error";
  progress: number; message: string;
  trained_at: number | null; symbols: number; rows: number; features: string[];
  metrics: QuantMetric[];
}

export interface QuantResult {
  symbol: string; asof: string; spot: number;
  status: QuantModelInfo["status"];
  horizons: QuantHorizon[];
  components: QuantComponent[];
  contributions: QuantContribution[];
  contributions_flat: boolean;   // the 1-month P(up) is flattened to the base rate, so it depends on no input (contributions = [])
  regime: {
    state: "bull" | "bear" | "sideways"; labels: string[]; probs: number[]; mu_annual: number[]; vol_annual: number[]; expected_duration: number[];
    // displayed regime = the Monte Carlo desk's HMM fit on the latest window (dates of its first / last bar, years)
    fit_start: string; fit_end: string; fit_years: number;
    model_fit_end: string | null;   // end of the fixed HMM fit behind the model's own (causal) regime feature
  } | null;
  vol: { garch_now: number; garch_21d: number; long_run: number; ewma: number; realized_21: number; percentile: number } | null;   // annualised; percentile 0…1
  chart: {
    ts: number[]; close: number[]; trend: number[]; upper: number[]; lower: number[]; bull_prob: number[];   // last ~1 year;
    // upper/lower = trend channel (Kalman level ± 2 sd of price around it); bull_prob may be [] without an HMM
    forecast: { t: number; ts: number; p50: number; lo68: number; hi68: number; lo95: number; hi95: number }[];   // next 21 trading days
  } | null;
  stock_metrics: { h: number; auc: number | null; acc: number | null; n: number }[];   // walk-forward on this stock only
  model: QuantModelInfo;
  warnings: string[];
}

export interface QuantLeader { symbol: string; sector: string; close: number; p_up: number; exp_ret: number; signal: Signal }
export interface QuantLeaderboard {
  h: number; asof: string;
  up: QuantLeader[]; down: QuantLeader[];   // highest / lowest P(up); both empty when `flat`
  model: QuantModelInfo; edge: boolean; edge_note: string | null;
  flat: boolean;                 // every stock gets the same P(up) (the base rate) at this horizon: no ranking
  base_p: number | null;         // that base rate (the model's training up-share at this horizon)
}

// ------------------------------------------------------------------ Screener fundamentals
// Screener numbers keep Screener's units: percentages in percent (12.5 = 12.5 %), money in ₹ crore.
export interface FinRow { label: string; values: (number | null)[]; is_pct: boolean }
export interface FinTable { title: string; columns: string[]; rows: FinRow[]; unit: string }   // unit e.g. "Rs. Crores"

export interface ScoreCheck { label: string; pass: boolean | null; value: string; rule: string }
export interface ScorePart { key: "quality" | "growth" | "valuation" | "health" | "ownership"; label: string; score: number; max: number; checks: ScoreCheck[] }

export interface Fundamentals {
  symbol: string; name: string; url: string; view: "consolidated" | "standalone";
  fetched_at: number; cached: boolean; stale: boolean;   // stale = Screener unreachable, older copy served
  about: string; sector: string | null; industry: string | null;
  bse_code: string | null; nse_code: string | null; website: string | null;
  ratios: { key: string; label: string; value: number | null; text: string; unit: string }[];   // the top ratio box
  pros: string[]; cons: string[];
  growth: { title: string; rows: { period: string; value: number | null }[] }[];   // compounded growth boxes (%)
  quarters: FinTable | null; profit_loss: FinTable | null; balance_sheet: FinTable | null;
  cash_flow: FinTable | null; ratios_table: FinTable | null;
  shareholding: { quarterly: FinTable | null; yearly: FinTable | null };
  peers: { columns: string[]; rows: { name: string; symbol: string | null; values: (number | null)[] }[] } | null;
  derived: { key: string; label: string; value: number | null; unit: string; note: string }[];
  scorecard: { total: number; max: number; grade: string; parts: ScorePart[] };
  warnings: string[];
}

// ------------------------------------------------------------------ AI analyst
export type Stance = "bullish" | "moderately_bullish" | "neutral" | "moderately_bearish" | "bearish";

export interface AiScenario { name: "bull" | "base" | "bear"; probability: number; target_price: number; return_pct: number; thesis: string }

export interface AiHorizonView {
  horizon: "6M" | "1Y" | "2Y" | "3Y";
  stance: Stance; confidence: Conf;
  expected_return_low_pct: number; expected_return_high_pct: number;
  scenarios: AiScenario[];       // bull / base / bear, probabilities sum to 1
  key_drivers: string[]; key_risks: string[]; rationale: string;
  mc_check: { p5: number; p50: number; p95: number; base_in_band: boolean; note: string } | null;   // vs Monte Carlo
}

export interface AiFact { label: string; value: string }
export interface AiDossier { sections: { key: string; title: string; facts: AiFact[] }[] }

export interface AiReport {
  symbol: string; name: string; asof: string; generated_at: number;
  provider: string; model: string;
  ai: boolean;                   // false = rule-based view (no AI key configured or the model call failed)
  summary: string; overall_stance: Stance; overall_confidence: Conf;
  horizons: AiHorizonView[];     // 6M, 1Y, 2Y, 3Y
  technical_view: string; fundamental_view: string; market_view: string; valuation_view: string;
  catalysts: string[]; risks: string[]; what_would_change_view: string[]; data_gaps: string[];
  score: { technical: number; fundamental: number; valuation: number; momentum: number; risk: number };   // 0…10 (risk: 10 = riskiest)
  dossier: AiDossier;
  usage: { input_tokens: number; output_tokens: number; cache_read_input_tokens: number | null } | null;
  warnings: string[];
  disclaimer: string;
}

export interface AiStep { key: string; label: string; status: "pending" | "running" | "done" | "skipped" | "error"; detail: string }

export interface AiJob {
  id: string; symbol: string;
  status: "queued" | "running" | "done" | "error";
  steps: AiStep[];
  result: AiReport | null; error: string | null;
  cached: boolean;               // today's earlier report returned without a new model call
  started: number; finished: number | null;
}

export interface AiConfig {
  enabled: boolean; provider: string; model: string; reason: string;   // reason: why disabled / how to enable
  daily_limit: number; used_today: number; user_limit: number; user_used_today: number;
}

// ------------------------------------------------------------------ fetch helpers
const enc = encodeURIComponent;
const post = <T>(path: string, body: unknown) => api<T>(path, { method: "POST", body: JSON.stringify(body) });

export const research = {
  search: (q: string, limit = 12) => api<SearchHit[]>(`/api/research/search?q=${enc(q)}&limit=${limit}`),
  history: (symbol: string, fetch = true) => api<HistoryInfo>(`/api/research/history/${enc(symbol)}?fetch=${fetch ? 1 : 0}`),
  montecarlo: (symbol: string, req: McRequest) => post<McResult>(`/api/research/montecarlo/${enc(symbol)}`, req),
  backtest: (symbol: string, req: McBacktestRequest) => post<McBacktest>(`/api/research/montecarlo/${enc(symbol)}/backtest`, req),
  quant: (symbol: string) => api<QuantResult>(`/api/research/quant/${enc(symbol)}`),
  quantModel: () => api<QuantModelInfo>("/api/research/quant-model"),
  quantTrain: () => post<QuantModelInfo>("/api/research/quant-model/train", {}),
  leaderboard: (h = 21, n = 15) => api<QuantLeaderboard>(`/api/research/quant-leaderboard?h=${h}&n=${n}`),
  screener: (symbol: string, refresh = false, view: "auto" | "consolidated" | "standalone" = "auto") =>
    api<Fundamentals>(`/api/research/screener/${enc(symbol)}?refresh=${refresh ? 1 : 0}&view=${view}`),
  aiConfig: () => api<AiConfig>("/api/research/ai-config"),
  aiStart: (symbol: string, refresh = false) => post<AiJob>(`/api/research/ai/${enc(symbol)}`, { refresh }),
  aiJob: (id: string) => api<AiJob>(`/api/research/ai-job/${enc(id)}`),
  aiLatest: (symbol: string) => api<AiReport>(`/api/research/ai/${enc(symbol)}/latest`),
};
