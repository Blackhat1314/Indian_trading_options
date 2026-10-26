/* Basic mode speaks plain English: every code the terminal shows (RVOL, ORB, LB, VWAP…) has a sentence here.
 * One place, so the same thing is always called the same thing. */
import type { RankRow } from "../lib/types";

const ok = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const n1 = (v: number) => Math.abs(v).toFixed(1);

/** A pick's factor chip ("RVOL 2.1x", "ORB", "LB" …) as a short sentence. */
export function chipText(chip: string): string {
  const m = chip.match(/^RVOL ([\d.]+)x$/);
  if (m) return `Volume ${m[1]}× its usual`;
  const g = chip.match(/^GAP ([+-][\d.]+)%$/);
  if (g) return `Opened ${n1(parseFloat(g[1]))}% ${parseFloat(g[1]) >= 0 ? "higher" : "lower"}`;
  return ({
    ORB: "Broke above its first-15-minute high",
    ORBD: "Broke below its first-15-minute low",
    "VWAP+": "Trading above today's average price",
    "VWAP-": "Trading below today's average price",
    LB: "Futures traders are adding bets on a rise",
    SB: "Futures traders are adding bets on a fall",
    SC: "Traders betting on a fall are exiting",
    LU: "Traders betting on a rise are exiting",
    RS: "Stronger than the market today",
    RW: "Weaker than the market today",
  } as Record<string, string>)[chip] ?? "In the news today";
}

export function statusText(status: string | undefined, hitAt?: string | null): { text: string; tone: "up" | "down" | "flat" } {
  if (status === "TARGET") return { text: `Target reached${hitAt ? ` at ${hitAt}` : ""}`, tone: "up" };
  if (status === "INVALID") return { text: `Stop hit${hitAt ? ` at ${hitAt}` : ""}`, tone: "down" };
  if (status === "PROV") return { text: "Not final until 9:30", tone: "flat" };
  return { text: "Still open", tone: "flat" };
}

export const SETUP: Record<string, { name: string; text: string }> = {
  BREAKOUT: { name: "Breakout", text: "Breaking out to a new high after a quiet spell" },
  PULLBACK: { name: "Dip in an uptrend", text: "Pulled back to its average inside a rising trend" },
  VCP: { name: "Tightening", text: "Swings getting smaller — often comes before a breakout" },
  MOMENTUM: { name: "Steady climber", text: "Rising steadily and stronger than most stocks" },
  BREAKDOWN: { name: "Breakdown", text: "Breaking down to a new low" },
};

/** Today's action of a live stock, as short sentences. */
export function todaySentences(r: RankRow): string[] {
  const out: string[] = [];
  if (ok(r.rvol) && r.rvol >= 1.5) out.push(`Volume is ${r.rvol.toFixed(1)}× its usual for this time of day`);
  else if (ok(r.rvol) && r.rvol > 0 && r.rvol < 0.6) out.push("Quieter than usual today");
  if (ok(r.gap) && Math.abs(r.gap) >= 0.8) out.push(`Opened ${n1(r.gap)}% ${r.gap > 0 ? "above" : "below"} yesterday's close`);
  if (ok(r.orb) && r.orb >= 1) out.push("Broke above its first-15-minute high");
  else if (ok(r.orb) && r.orb <= -1) out.push("Broke below its first-15-minute low");
  if (ok(r.vwap_atr) && Math.abs(r.vwap_atr) > 0.15) out.push(`Trading ${r.vwap_atr > 0 ? "above" : "below"} today's average price`);
  if (ok(r.resid) && Math.abs(r.resid) > 1) out.push(`${r.resid > 0 ? "Stronger" : "Weaker"} than the market by ${n1(r.resid)}%`);
  const oi = r.oi_tag ? chipText(r.oi_tag) : "";
  if (oi && r.oi_tag) out.push(oi);
  if (r.headline) out.push(`In the news: “${r.headline}”`);
  return out;
}

export interface Term { term: string; text: string; tags?: string[] }

export const GLOSSARY: Term[] = [
  { term: "Stock / share", text: "A small piece of ownership in a company. Its price moves as buyers and sellers change their minds about the company's future." },
  { term: "Index (NIFTY 50, BANK NIFTY)", text: "A basket of big stocks tracked as one number. NIFTY 50 = India's 50 largest companies; when people say “the market is up”, they usually mean this." },
  { term: "Previous close", text: "Yesterday's last price. Today's “+1.2%” is measured from it." },
  { term: "Gap up / gap down", text: "The stock opened above / below yesterday's close, usually because of news overnight." },
  { term: "Volume", text: "How many shares changed hands. “Volume 2× its usual” means twice the normal amount for this time of day — big players may be active." },
  { term: "Average price (VWAP)", text: "Today's average traded price, weighted by volume. Above it, buyers have been in control so far today; below it, sellers." },
  { term: "First-15-minute range", text: "The high and low of 9:15–9:30. Breaking above it is a common sign the day's move is up." },
  { term: "Stronger / weaker than the market", text: "How much the stock moved beyond what the market's move would explain (taking its usual sensitivity to NIFTY into account)." },
  { term: "Trend", text: "The direction prices have been moving for weeks or months. Most experienced traders trade with the trend, not against it." },
  { term: "Moving average", text: "The average closing price of the last N days (50-day, 200-day). Price above its 200-day average = a long-term uptrend." },
  { term: "Breakout", text: "Price rising above a level it failed to cross for a while. Many breakouts fail — always use a stop." },
  { term: "Stop-loss", text: "The price at which you accept you were wrong and exit. Decide it before you buy, never after." },
  { term: "Target", text: "The price where you plan to take profit. Here: about one day's typical move (intraday) or +4% (swing)." },
  { term: "Position size", text: "How many shares to buy so that, if the stop is hit, you lose only what you planned (for example 1% of your capital)." },
  { term: "Risk per trade", text: "The most you are willing to lose on one idea. Beginners: 0.5–1% of capital. Ten losses in a row then cost 5–10%, not everything." },
  { term: "Long / short", text: "Long = buy hoping it rises. Short = sell first hoping to buy back lower (intraday or with futures) — riskier, not for beginners." },
  { term: "F&O (futures & options)", text: "Contracts that move with a stock or index, with borrowed money (leverage). SEBI's 2024 study: about 9 in 10 individual F&O traders lost money." },
  { term: "Open interest (OI)", text: "How many futures/options contracts are still open. Rising price + rising OI = traders adding bets on a rise." },
  { term: "Market cap", text: "The company's total value on the market: share price × number of shares. Large caps are steadier; small caps move more." },
  { term: "P/E (price to earnings)", text: "Price divided by one year's profit per share — how many years of today's profit you pay for. High P/E = the market expects growth." },
  { term: "ROE / ROCE", text: "Profit as a % of shareholders' money (ROE) or of all capital used (ROCE). Above ~15% is generally a sign of a good business." },
  { term: "Debt to equity", text: "Borrowings compared with shareholders' money. Lower is safer; banks and lenders are an exception (borrowing is their business)." },
  { term: "Swing trade", text: "Holding for days to a few weeks to catch one move. Slower and calmer than intraday trading — a better start for most beginners." },
  { term: "Edge", text: "A reason to expect better than coin-flip results, proven on data the idea was not built on. Without a measured edge, a signal is just a guess with a story." },
  { term: "Track record", text: "How past picks actually did. A few good days prove nothing; judge over 60+ days." },
  { term: "Paper trading", text: "Practising with virtual money on real prices. Do this for weeks before risking real money." },
];

export const RULES: { title: string; text: string }[] = [
  { title: "Decide your stop before you buy", text: "Write down the price where you'll exit if wrong. Never move it further away." },
  { title: "Risk 1% or less per idea", text: "With ₹1,00,000, lose at most ₹1,000 if the stop hits. The calculator on every stock page does the maths." },
  { title: "Practise first", text: "Paper trade for a few weeks. If you can't follow your own rules with virtual money, real money won't help." },
  { title: "Stay away from F&O at the start", text: "Leverage turns small mistakes into big losses. Learn on cash stocks." },
  { title: "Judge ideas by their record", text: "Every idea here shows how similar ideas did before. Prefer ones with a long, honest record over exciting stories or tips." },
];
