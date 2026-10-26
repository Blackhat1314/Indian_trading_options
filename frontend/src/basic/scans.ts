/* Basic mode's ready-made ideas: simple, named filters over the live data the terminal already has.
 * Each says in one line what it finds; none of them is a prediction — the Ideas page says so. */
import type { FnoRow, RankRow, SwingRow, SwingSnap } from "../lib/types";
import { SETUP, chipText } from "./words";

export interface IdeaRow { symbol: string; sector: string; price: number; chg: number; why: string; extra?: string }

export interface Scan {
  id: string;
  name: string;
  tags: string[];
  desc: string;
  when: "today" | "swing";
  rows: (d: ScanData) => IdeaRow[];
  note?: (d: ScanData) => string | null;
}

export interface ScanData { rank: RankRow[]; fno: FnoRow[]; swing: SwingRow[]; swingStats: SwingSnap["stats"] }

const ok = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const liquid = (r: RankRow) => ok(r.turnover) && r.turnover >= 5;          // ₹5 crore traded today or more
const live = (r: RankRow): IdeaRow => ({ symbol: r.symbol, sector: r.sector, price: r.ltp, chg: r.chg, why: "" });

function fromRank(filter: (r: RankRow) => boolean, sort: (a: RankRow, b: RankRow) => number,
  why: (r: RankRow) => string, limit = 25) {
  return (d: ScanData) => d.rank.filter((r) => ok(r.ltp) && r.ltp > 0 && liquid(r) && filter(r)).sort(sort).slice(0, limit)
    .map((r) => ({ ...live(r), why: why(r) }));
}

function swingSetup(setup: string) {
  return (d: ScanData) => d.swing.filter((r) => r.setup === setup).sort((a, b) => b.score - a.score).slice(0, 25)
    .map((r) => ({
      symbol: r.symbol, sector: r.sector, price: r.close, chg: r.chg, why: SETUP[setup]?.text ?? setup,
      extra: ok(r.stop) && ok(r.target) ? `stop ₹${r.stop!.toFixed(1)} · target ₹${r.target!.toFixed(1)}` : undefined,
    }));
}

function setupNote(setup: string) {
  return (d: ScanData) => {
    const s = d.swingStats?.[setup];
    if (!s || !s.n) return null;
    return `Across the history in the data, this setup reached +${s.target_pct.toFixed(0)}% before −${s.stop_pct.toFixed(0)}% `
      + `in ${s.hit_rate.toFixed(0)}% of ${s.n.toLocaleString("en-IN")} cases (within ${s.horizon} trading days). `
      + `The rest hit the stop first or ran out of time.`;
  };
}

export const SCANS: Scan[] = [
  {
    id: "breakout", name: "Breaking out today", tags: ["First 15 min", "Volume", "Today"], when: "today",
    desc: "Rose above their 9:15–9:30 high with more volume than usual.",
    rows: fromRank((r) => r.orb >= 1 && r.chg > 0 && r.rvol >= 1.2, (a, b) => b.rvol - a.rvol,
      (r) => `Above its first-15-minute high · volume ${r.rvol.toFixed(1)}× usual`),
  },
  {
    id: "volume", name: "Volume rocketing", tags: ["Volume", "Today"], when: "today",
    desc: "Trading at least twice their usual volume for this time of day — someone big is active.",
    rows: fromRank((r) => r.rvol >= 2, (a, b) => b.rvol - a.rvol,
      (r) => `Volume ${r.rvol.toFixed(1)}× usual · ${r.chg >= 0 ? "up" : "down"} ${Math.abs(r.chg).toFixed(1)}%`),
  },
  {
    id: "strong", name: "Beating the market", tags: ["Relative strength", "Today"], when: "today",
    desc: "Up much more than the market's move would explain.",
    rows: fromRank((r) => r.resid > 1, (a, b) => b.resid - a.resid,
      (r) => `${r.resid.toFixed(1)}% stronger than the market`),
  },
  {
    id: "weak", name: "Lagging the market", tags: ["Relative weakness", "Today"], when: "today",
    desc: "Down much more than the market's move would explain — to avoid, or to watch for a turn.",
    rows: fromRank((r) => r.resid < -1, (a, b) => a.resid - b.resid,
      (r) => `${Math.abs(r.resid).toFixed(1)}% weaker than the market`),
  },
  {
    id: "gaphold", name: "Gap up holding", tags: ["Gap", "Today"], when: "today",
    desc: "Opened at least 1% higher and are still above their opening price.",
    rows: fromRank((r) => r.gap >= 1 && r.ltp > r.open, (a, b) => b.gap - a.gap,
      (r) => `Opened ${r.gap.toFixed(1)}% higher and holding above the open`),
  },
  {
    id: "vwap", name: "Buyers in control", tags: ["Average price", "Today"], when: "today",
    desc: "Rising and trading well above today's average price.",
    rows: fromRank((r) => r.vwap_atr > 0.3 && r.chg > 0, (a, b) => b.vwap_atr - a.vwap_atr,
      () => "Well above today's average price and rising"),
  },
  {
    id: "news", name: "In the news", tags: ["News", "Today"], when: "today",
    desc: "Stocks with a fresh headline today.",
    rows: fromRank((r) => !!r.headline, (a, b) => Math.abs(b.news) - Math.abs(a.news), (r) => r.headline ?? ""),
  },
  {
    id: "lb", name: "Futures: bets on a rise", tags: ["F&O", "Open interest", "Today"], when: "today",
    desc: "Price up and futures open interest up — traders are adding positions that gain if it rises.",
    rows: (d) => d.fno.filter((r) => r.tag === "LB").sort((a, b) => b.oi_chg - a.oi_chg).slice(0, 25)
      .map((r) => ({ symbol: r.symbol, sector: r.sector, price: r.ltp, chg: r.chg, why: `${chipText("LB")} · OI +${r.oi_chg.toFixed(1)}%` })),
  },
  {
    id: "sb", name: "Futures: bets on a fall", tags: ["F&O", "Open interest", "Today"], when: "today",
    desc: "Price down and futures open interest up — traders are adding positions that gain if it falls.",
    rows: (d) => d.fno.filter((r) => r.tag === "SB").sort((a, b) => b.oi_chg - a.oi_chg).slice(0, 25)
      .map((r) => ({ symbol: r.symbol, sector: r.sector, price: r.ltp, chg: r.chg, why: `${chipText("SB")} · OI +${r.oi_chg.toFixed(1)}%` })),
  },
  ...(["BREAKOUT", "PULLBACK", "VCP", "MOMENTUM"] as const).map((k) => ({
    id: `swing-${k.toLowerCase()}`, name: SETUP[k].name, tags: ["Swing", "2 weeks", "Daily chart"], when: "swing" as const,
    desc: `${SETUP[k].text}. A setup for holding a few days to two weeks.`,
    rows: swingSetup(k), note: setupNote(k),
  })),
];
