import { create } from "zustand";
import type { Me } from "./auth";
import { getPref, setPref } from "./storage";
import type {
  AlertFire, ChainSnap, Clock, FnoSnap, Instrument, Market, NewsItem, RankSnap, SwingSnap, TopSnap,
} from "./types";

export type DeskId = "intraday" | "market" | "fno" | "swing" | "news" | "alerts" | "geo";
const DESK_IDS: DeskId[] = ["intraday", "market", "fno", "swing", "news", "alerts", "geo"];

export interface Toast { id: number; kind: "alert" | "event"; title: string; body: string; ts: number }

interface Quotes { ts: number; q: Record<string, [number, number, number, number, number]> }

export interface LiveState {
  conn: "connecting" | "open" | "closed" | "denied";
  connError: string;
  latencyMs: number | null;
  lastMsg: number;
  user: string;
  source: string;
  desk: DeskId;
  selected: string;              // selected symbol (equity)
  chainName: string;
  universe: Instrument[];
  bySymbol: Record<string, Instrument>;
  clock: Clock | null;
  market: Market | null;
  rank: RankSnap | null;
  top: TopSnap | null;
  fno: FnoSnap | null;
  chains: Record<string, ChainSnap>;
  swing: SwingSnap | null;
  news: NewsItem[];
  quotes: Quotes | null;
  geo: Record<string, unknown>;    // geo:* snapshots keyed by topic
  toasts: Toast[];
  unreadAlerts: number;
  helpOpen: boolean;
  me: Me | null;
  setDesk: (d: DeskId) => void;
  setMe: (m: Me) => void;
  select: (s: string) => void;
  setChain: (n: string) => void;
  setUniverse: (u: Instrument[]) => void;
  ingest: (topic: string, data: unknown, event?: boolean) => void;
  dismiss: (id: number) => void;
  setConn: (c: LiveState["conn"], err?: string) => void;
  setHelp: (v: boolean) => void;
  clearUnread: () => void;
}

let toastId = 1;

export const useLive = create<LiveState>((set, get) => ({
  conn: "connecting",
  connError: "",
  latencyMs: null,
  lastMsg: 0,
  user: "",
  source: "",
  desk: DESK_IDS.includes(getPref("td.desk") as DeskId) ? (getPref("td.desk") as DeskId) : "intraday",  // SYSTEM moved to /admin
  selected: getPref("td.sel") || "",
  chainName: getPref("td.chain") || "NIFTY",
  universe: [],
  bySymbol: {},
  clock: null,
  market: null,
  rank: null,
  top: null,
  fno: null,
  chains: {},
  swing: null,
  news: [],
  quotes: null,
  geo: {},
  toasts: [],
  unreadAlerts: 0,
  helpOpen: false,
  me: null,
  setMe: (m) => set({ me: m }),
  setDesk: (d) => {
    setPref("td.desk", d);
    set({ desk: d, ...(d === "alerts" ? { unreadAlerts: 0 } : {}) });
  },
  select: (s) => {
    setPref("td.sel", s);
    set({ selected: s });
  },
  setChain: (n) => {
    setPref("td.chain", n);
    set({ chainName: n });
  },
  setUniverse: (u) => {
    const bySymbol: Record<string, Instrument> = {};
    for (const i of u) bySymbol[i.symbol] = i;
    set({ universe: u, bySymbol });
  },
  setConn: (c, err = "") => set({ conn: c, connError: err }),
  setHelp: (v) => set({ helpOpen: v }),
  clearUnread: () => set({ unreadAlerts: 0 }),
  dismiss: (id) => set({ toasts: get().toasts.filter((t) => t.id !== id) }),
  ingest: (topic, data, event) => {
    const now = Date.now() / 1000;
    const patch: Partial<LiveState> = { lastMsg: now };
    if (topic === "hello") {
      const d = data as { user: string; source: string };
      patch.user = d.user;
      patch.source = d.source;
    } else if (topic === "clock") patch.clock = data as Clock;
    else if (topic === "market") patch.market = data as Market;
    else if (topic === "rank") patch.rank = data as RankSnap;
    else if (topic === "top") patch.top = data as TopSnap;
    else if (topic === "fno") patch.fno = data as FnoSnap;
    else if (topic === "swing") patch.swing = data as SwingSnap;
    else if (topic === "news") patch.news = (data as { items: NewsItem[] }).items;
    else if (topic === "quotes") patch.quotes = data as Quotes;
    else if (topic.startsWith("geo:")) patch.geo = { ...get().geo, [topic]: data };
    else if (topic.startsWith("chain:")) patch.chains = { ...get().chains, [topic.slice(6)]: data as ChainSnap };
    else if (topic === "pong") {
      const t = (data as { t: number }).t;
      patch.latencyMs = Math.max(0, Math.round((now - t) * 1000));
    } else if (event && (topic === "alert" || topic === "event")) {
      const isAlert = topic === "alert";
      const d = data as AlertFire & { kind?: string; msg?: string };
      const t: Toast = {
        id: toastId++, kind: isAlert ? "alert" : "event",
        title: isAlert ? `ALERT · ${d.symbol}` : (d.kind ?? "EVENT"),
        body: d.msg ?? "", ts: now,
      };
      patch.toasts = [...get().toasts, t].slice(-5);
      if (isAlert && get().desk !== "alerts") patch.unreadAlerts = get().unreadAlerts + 1;
      setTimeout(() => get().dismiss(t.id), isAlert ? 12000 : 7000);
    }
    set(patch);
  },
}));
