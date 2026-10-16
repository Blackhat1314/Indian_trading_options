"""Paper trading service: runs the intraday and swing strategies for every user who switched them on.

    python -m td.services.paper          (run.py and docker compose start it with the other services)

It reads what the analytics service already publishes — the market clock (td:snap:clock), last prices
(td:snap:quotes) and today's minute bars per instrument (td:bars:*) — so live, simulated and replayed markets
all work the same way, and a restart mid-session loses nothing.

Timeline of a session (market time):
  09:15+    swing entries planned at yesterday's close fill at the first prices
  09:30     intraday decision #1 (features from 09:15-09:29 bars); entries at the current price
  10:00     intraday decision #2 (stricter edge; skipped for an account past its daily loss limit)
  any time  stops are checked against every new price (intraday and swing)
  15:15     every intraday position is squared off
  15:30+    swing end of day: trailing stops and time stops on the close, then tomorrow's candidates
Users' settings/reset requests arrive from the gateway on td:paper:cmd; each account's state is published to
td:paper:state:<user> for the gateway, and today's public signals to the "paper" topic.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .. import bus
from ..auth import UserStore
from ..config import settings
from ..dhan.instruments import Universe
from ..lake import Lake
from ..paper import intraday as I
from ..paper import swing as SW
from ..paper.broker import Account, accounts_dir
from ..paper.daydata import SESSION, load_days
from ..paper.features import daily_context, intraday_features, swing_features
from ..timeutil import from_epoch

log = logging.getLogger("td.paper")

K_CMD = "td:paper:cmd"
K_STATE = "td:paper:state:"
K_SETTINGS = "td:paper:settings:"
HIST_DAYS = 20                 # prior sessions of minute bars kept for relative volume
DECISION_DELAY = 0.15          # minutes after the decision minute starts (bars of the previous minute pushed)
DECISION_GRACE = 5.0           # a decision is taken at most this many minutes late, else skipped
SQUARE_OFF = I.EXIT_MINUTE     # 15:15
SWING_EOD_MINUTE = SESSION + 1 # 15:31
INDEX = "NIFTY"


def state_key(user: str) -> str:
    return K_STATE + user


def _bars_to_day(rows: list) -> np.ndarray:
    """[[minute, o, h, l, c, v], ...] -> [375, 5] with empty minutes carried forward (as daydata does)."""
    arr = np.full((SESSION, 5), np.nan, np.float32)
    for m, o, h, l, c, v in rows:
        if 0 <= int(m) < SESSION:
            arr[int(m)] = (o, h, l, c, v)
    c = pd.Series(arr[:, 3]).ffill().bfill().to_numpy()
    empty = np.isnan(arr[:, 3])
    for k in range(4):
        arr[:, k] = np.where(empty, c, arr[:, k])
    arr[:, 4] = np.where(empty, 0.0, arr[:, 4])
    return arr


def _ymd(d: date) -> int:
    return int(d.strftime("%Y%m%d"))


class PaperService:
    def __init__(self, r):
        self.r = r
        self.lake = Lake()
        self.users = UserStore()
        self.accounts: dict[str, Account] = {}
        self.models = I.load_models()
        self.universe: Universe | None = None
        self.eq_key: dict[str, int] = {}          # symbol -> instrument key (td:bars / quotes id)
        self.key_sym: dict[int, str] = {}
        self.idx_key: int | None = None
        self.day: date | None = None
        self.ctx: dict[str, pd.DataFrame] = {}
        self.hist: dict[str, tuple] = {}
        self.idx_ctx: pd.DataFrame | None = None
        self.idx_hist: tuple | None = None
        self.ltp: dict[str, float] = {}
        self.clock = 0.0
        self.phase = "CLOSED"
        self.minute = -1.0
        self.signals: dict = {"date": None, "intraday": {"decisions": []}, "swing": {"asof": None, "regime": None,
                                                                                     "candidates": []}}
        self.decided: dict[int, str] = {}         # minute -> "done" / "skipped" for today
        self.swing_filled_day: int | None = None
        self.swing_eod_day: int | None = None
        self.regime: dict | None = None
        self.loading = False
        self.monitors = self._load_monitors()
        self.shadow_day: int | None = None
        self.last_publish = 0.0
        self.last_discover = 0.0
        self.train_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ setup
    async def setup(self) -> None:
        for _ in range(180):
            raw = await self.r.get(bus.KEY_UNIVERSE)
            if raw:
                break
            log.info("waiting for the ingestor to publish the universe ...")
            await asyncio.sleep(1)
        else:
            raise RuntimeError("universe not available — is the ingestor running?")
        self.universe = Universe.from_json(bus.loads(raw))
        for i in self.universe.instruments:
            if i.kind == "EQ":
                self.eq_key[i.symbol] = i.key
                self.key_sym[i.key] = i.symbol
            elif i.kind == "IDX" and i.symbol == INDEX:
                self.idx_key = i.key
                self.key_sym[i.key] = i.symbol
        for p in accounts_dir().glob("*.json"):
            if p.name.count(".") == 1:
                self.accounts[p.stem] = Account(p.stem)
        log.info("paper: %d stocks, %d accounts, models: %s", len(self.eq_key), len(self.accounts),
                 {k: v.meta.get("source") for k, v in self.models.items()} or "none")

    @staticmethod
    def _monitor_path() -> Path:
        name = "monitor_sim.json" if settings.data_source == "sim" else "monitor.json"
        return Path(settings.data_dir) / "paper" / name

    def _load_monitors(self) -> dict[int, I.EdgeMonitor]:
        """Each decision time's shadow record: this server's own, else (live) the seed shipped with the models —
        the walk-forward backtest's latest real-market signals — else empty (trust the backtest until a record
        builds up). The simulator keeps its own record: synthetic markets say nothing about real ones."""
        paths = [self._monitor_path()]
        if settings.data_source != "sim":
            paths.append(I.PRETRAINED / "monitor_seed.json")
        for path in paths:
            try:
                d = json.loads(path.read_text())
                return {m: I.EdgeMonitor(I.MONITOR_N, d.get(str(m), [])) for m in I.DECISIONS}
            except (OSError, ValueError):
                continue
        return {m: I.EdgeMonitor(I.MONITOR_N) for m in I.DECISIONS}

    def _save_monitors(self) -> None:
        path = self._monitor_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({str(m): mon.hist for m, mon in self.monitors.items()}))
        tmp.replace(path)

    def account(self, user: str) -> Account:
        if user not in self.accounts:
            self.accounts[user] = Account(user)
        return self.accounts[user]

    # ------------------------------------------------------------------ history for today
    def _load_context(self, day: date) -> None:
        """Daily context (with a row for today) and the last HIST_DAYS sessions of minute bars, per symbol."""
        today = _ymd(day)
        nd = self.lake.read("D", INDEX)
        iclose = None
        if nd is not None and len(nd):
            nd = nd.sort_values("ts")
            nd = nd[pd.to_datetime(nd["ts"] + 19800, unit="s").dt.strftime("%Y%m%d").astype(int) < today]
            iclose = pd.Series(nd["close"].to_numpy(float),
                               index=pd.to_datetime(nd["ts"] + 19800, unit="s").dt.strftime("%Y%m%d").astype(int).to_numpy())
            self.idx_ctx = daily_context(nd, next_date=today)
        try:
            d, b, p = load_days(self.lake, INDEX)
            keep = d < today
            self.idx_hist = (d[keep][-HIST_DAYS:], b[keep][-HIST_DAYS:], p[keep][-HIST_DAYS:])
        except Exception:
            self.idx_hist = None
        ctx, hist = {}, {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for sym in self.eq_key:
                D = self.lake.read("D", sym)
                if D is None or len(D) < 30:
                    continue
                D = D[pd.to_datetime(D["ts"] + 19800, unit="s").dt.strftime("%Y%m%d").astype(int) < today]
                if len(D) < 30:
                    continue
                ctx[sym] = daily_context(D, iclose, next_date=today)
                try:
                    dd, bb, pp = load_days(self.lake, sym)
                    keep = dd < today
                    hist[sym] = (dd[keep][-HIST_DAYS:], bb[keep][-HIST_DAYS:], pp[keep][-HIST_DAYS:])
                except Exception:
                    pass
        self.ctx, self.hist = ctx, hist
        log.info("context for %s: %d stocks with daily history, %d with minute history", day, len(ctx), len(hist))

    async def new_day(self, day: date) -> None:
        self.day = day
        self.decided = {}
        self.signals = {"date": day.isoformat(),
                        "intraday": {"decisions": [{"time": t, "at": None, "status": "pending", "note": "",
                                                    "considered": 0, "proposals": []} for t in I.DECISIONS.values()]},
                        "swing": self.signals.get("swing", {"asof": None, "regime": None, "candidates": []})}
        for a in self.accounts.values():
            for pos in list(a.intraday["positions"]):          # the service was down at 15:15 that day
                px = self.ltp.get(pos["symbol"]) or pos["entry"]
                a.close_position(pos, px, self.clock, "square-off (late)",
                                 "The paper service was not running at 15:15, so this intraday position was squared "
                                 "off at the last known price when it restarted.", daily_vol=pos["atr"] / px)
            a.new_day(_ymd(day))
        self.ltp = {}
        self.loading = True
        task = asyncio.create_task(asyncio.to_thread(self._load_context, day))
        try:
            while not task.done():
                await self.heartbeat()
                await asyncio.wait({task}, timeout=2)
            task.result()
        finally:
            self.loading = False

    # ------------------------------------------------------------------ market data
    async def read_market(self) -> bool:
        raw_clock, raw_q, raw_an = await self.r.mget([bus.SNAP_PREFIX + "clock", bus.SNAP_PREFIX + "quotes",
                                                      bus.k_status("analytics")])
        an = bus.loads(raw_an) if raw_an else None
        if not an or time.time() - float(an.get("updated", 0)) > 15:
            return False                     # analytics not running: the clock / quotes in Redis may be stale
        clk = (bus.loads(raw_clock) or {}).get("data") if raw_clock else None
        if not clk:
            return False
        self.clock = float(clk["ts"])
        self.phase = clk.get("phase", "CLOSED")
        dt = from_epoch(self.clock)
        self.minute = (dt.hour * 60 + dt.minute - 555) + dt.second / 60.0
        q = ((bus.loads(raw_q) or {}).get("data") or {}).get("q", {}) if raw_q else {}
        for k, v in q.items():
            sym = self.key_sym.get(int(k))
            if sym and v and v[0]:
                self.ltp[sym] = float(v[0])
        return True

    async def today_bars(self, symbols: list[str]) -> dict[str, list]:
        keys = [self.eq_key[s] if s != INDEX else self.idx_key for s in symbols]
        pipe = self.r.pipeline(transaction=False)
        for k in keys:
            pipe.get(bus.k_bars(k))
        out = {}
        today = self.day.isoformat() if self.day else ""
        for s, raw in zip(symbols, await pipe.execute()):
            d = bus.loads(raw) if raw else None
            if d and d.get("day") == today and d.get("bars"):
                out[s] = d["bars"]
        return out

    # ------------------------------------------------------------------ intraday decision
    def _features(self, m: int, bars: dict[str, list]) -> pd.DataFrame:
        today = _ymd(self.day)
        ib = None
        if self.idx_hist is not None and INDEX in bars:
            d, b, p = self.idx_hist
            ib = (np.r_[d, today].astype(np.int32), np.concatenate([b, _bars_to_day(bars[INDEX])[None]]))
        rows = []
        with np.errstate(divide="ignore", invalid="ignore"):
            for sym, rb in bars.items():
                if sym == INDEX or sym not in self.ctx or sym not in self.hist:
                    continue
                if not any(int(x[0]) < m for x in rb):
                    continue
                d, b, p = self.hist[sym]
                dates = np.r_[d, today].astype(np.int32)
                tensor = np.concatenate([b, _bars_to_day(rb)[None]])
                printed = np.r_[p, SESSION].astype(np.int16)
                f = intraday_features(tensor, m, self.ctx[sym], dates, printed,
                                      ib[1] if ib else None, self.idx_ctx, ib[0] if ib else None)
                row = f.iloc[-1].copy()
                row["symbol"] = sym
                row["minute_volume"] = float(tensor[-1, m - 1, 4])
                rows.append(row)
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).reset_index(drop=True)
        frame["date"], frame["m"] = today, m
        return I.prepare(frame)

    async def decide_intraday(self, m: int) -> None:
        label = I.DECISIONS[m]
        slot = next(x for x in self.signals["intraday"]["decisions"] if x["time"] == label)
        slot["at"] = self.clock
        late = self.minute - m
        if late > DECISION_GRACE:
            slot.update(status="skipped", note=f"service was not running at {label} (now {late:.0f} min late)")
            self.decided[m] = "skipped"
            return
        if not self.models:
            slot.update(status="skipped", note="no trained models (run python -m td.paper.train)")
            self.decided[m] = "skipped"
            return
        symbols = list(self.ctx) + [INDEX]
        bars = await self.today_bars(symbols)
        frame = await asyncio.to_thread(self._features, m, bars)
        if frame.empty:
            slot.update(status="skipped", note="no minute bars for today yet")
            self.decided[m] = "skipped"
            return
        # the price we would actually trade at: the latest quote, else the decision bar's close
        frame["price"] = [self.ltp.get(s, p) for s, p in zip(frame["symbol"], frame["price"])]
        taken = {x["symbol"] for d in self.signals["intraday"]["decisions"] for x in d["proposals"]}
        props = await asyncio.to_thread(I.decide, frame, m, self.models, 1_000_000.0, taken)
        slot.update(status="done", considered=int(len(frame)), proposals=[self._proposal_view(p) for p in props],
                    note=(f"{len(props)} trade(s) cleared the edge-over-cost bar" if props
                          else "no stock cleared the edge-over-cost bar — no trade is a position too"))
        self.decided[m] = "done"
        mon = self.monitors[m]
        slot["monitor"] = mon.state()
        if props and settings.paper_edge_monitor and not mon.allows():
            slot["note"] += (f" · edge monitor: the last {len(mon.hist)} {label} signals averaged "
                             f"{mon.state()['avg_net']:+.2%} after costs, so paper accounts sit this one out "
                             "(signals are still tracked; trading resumes when the record turns positive)")
            return
        mvol = dict(zip(frame["symbol"], frame["minute_volume"]))
        for user, acct in self.accounts.items():
            st = acct.settings["intraday"]
            if not st["enabled"] or not props:
                continue
            self._fill_intraday(acct, props, m, mvol)
        log.info("decision %s: %d candidates, %d proposals", label, len(frame), len(props))

    def _proposal_view(self, p: dict) -> dict:
        return {"symbol": p["symbol"], "side": "LONG" if p["side"] > 0 else "SHORT", "price": round(p["price"], 2),
                "stop": round(p["price"] - p["side"] * p["stop_dist"], 2), "setup": p["setup"],
                "pred": p["pred"], "edge": p["edge"], "cost": p["cost"], "drivers": p["drivers"], "facts": p["facts"]}

    def _fill_intraday(self, acct: Account, props: list[dict], m: int, mvol: dict) -> None:
        st = acct.settings["intraday"]
        cap, lev = st["capital"], st["leverage"]
        eq = acct.intraday_equity(self.ltp)
        day_pnl = acct.intraday["realized_today"] + sum(acct._open_pnl(p, self.ltp.get(p["symbol"]))
                                                        for p in acct.intraday["positions"])
        if day_pnl <= -I.DAILY_LOSS_LIMIT * cap:
            acct.intraday["loss_limit_hit"] = True
            acct.dirty = True
            return
        held = {p["symbol"] for p in acct.intraday["positions"]} | set(acct.intraday.get("traded_today", []))
        sized = []
        for p in props:
            if p["symbol"] in held:
                continue
            price = self.ltp.get(p["symbol"], p["price"])
            notional = min(I.RISK_PER_TRADE * eq / p["risk_pct"], I.MAX_POSITION_FRAC * lev * eq)
            cost = I.cost_estimate(notional)
            if p["pred"] - cost <= I.MIN_EDGE[m]:
                continue                                   # too small an order for its fixed costs
            sized.append((p, price, notional, cost))
        budget = lev * eq - acct.intraday_gross(self.ltp)
        want = sum(x[2] for x in sized)
        scale = min(1.0, budget / want) if want > 0 else 0.0
        for p, price, notional, cost in sized:
            qty = int(notional * scale / price)
            if qty < 1:
                continue
            stop = price - p["side"] * p["stop_dist"]
            risk_amt = qty * p["stop_dist"]
            reasons = {
                "setup": p["setup"],
                "summary": self._intraday_summary(p),
                "drivers": p["drivers"], "facts": p["facts"], "pred": p["pred"], "edge": p["pred"] - cost, "cost": cost,
                "sizing": (f"risk {I.RISK_PER_TRADE:.1%} of ₹{eq:,.0f} at a {p['risk_pct']:.2%} stop"
                           + (f", scaled ×{scale:.2f} to stay within {lev:g}x buying power" if scale < 1 else "")
                           + f" → {qty} shares (₹{qty * price:,.0f}, {qty * price / eq:.2f}x of equity), "
                             f"₹{risk_amt:,.0f} at risk"),
            }
            acct.open_position("intraday", p["symbol"], p["side"], qty, price, self.clock, stop, p["atr"],
                               I.DECISIONS[m], reasons, minute_volume=mvol.get(p["symbol"], 0.0),
                               daily_vol=p["atr"] / max(price, 1e-9))
            acct.intraday.setdefault("traded_today", []).append(p["symbol"])

    @staticmethod
    def _intraday_summary(p: dict) -> str:
        side = "Long" if p["side"] > 0 else "Short"
        top = ", ".join(d["label"] for d in p["drivers"][:2])
        return (f"{side} at {p['decision']}: '{p['setup']}'. The models expect {p['pred']:+.2%} to 15:15 against an "
                f"estimated {p['cost']:.2%} cost (edge {p['edge']:+.2%}); strongest inputs: {top}.")

    # ------------------------------------------------------------------ exits
    def manage_exits(self) -> None:
        for acct in self.accounts.values():
            for pos in list(acct.intraday["positions"]):
                px = self.ltp.get(pos["symbol"])
                if not px:
                    continue
                if pos["side"] * (px - pos["stop"]) <= 0:
                    acct.close_position(pos, px, self.clock, "stop",
                                        f"Price {px:,.2f} reached the stop {pos['stop']:,.2f} (beyond the "
                                        f"{'opening range' if pos['decision'] == '09:30' else 'morning extreme'}).",
                                        stop_order=True, daily_vol=pos["atr"] / px)
                elif self.minute >= SQUARE_OFF:
                    acct.close_position(pos, px, self.clock, "time exit 15:15",
                                        "Intraday positions are squared off at 15:15, before the broker's auto square-off.",
                                        daily_vol=pos["atr"] / px)
            for pos in list(acct.swing["positions"]):
                px = self.ltp.get(pos["symbol"])
                if px and self.phase == "OPEN" and px <= pos["stop"]:
                    trailed = pos["stop"] > pos["initial_stop"] + 1e-9
                    acct.close_position(pos, px, self.clock, "trailing stop" if trailed else "stop",
                                        (f"Price {px:,.2f} fell to the trailing stop {pos['stop']:,.2f} "
                                         f"({SW.CONFIG.trail_atr:g} ATR below the best close)") if trailed else
                                        (f"Price {px:,.2f} fell to the initial stop {pos['stop']:,.2f} "
                                         f"({SW.CONFIG.stop_atr:g} ATR below the entry)."),
                                        stop_order=True, daily_vol=pos["atr"] / px)

    async def record_shadows(self) -> None:
        """After 15:15: what every one of today's signals made (taken or not), for the edge monitor."""
        from ..paper.sim import ExitRules, simulate
        decisions = self.signals["intraday"]["decisions"]
        syms = sorted({p["symbol"] for d in decisions for p in d["proposals"]})
        if not syms:
            return
        bars = await self.today_bars(syms)
        rules = ExitRules(exit_minute=SQUARE_OFF)
        for d in decisions:
            m = next((k for k, v in I.DECISIONS.items() if v == d["time"]), None)
            if m is None or not d["proposals"]:
                continue
            out = []
            for p in d["proposals"]:
                if p["symbol"] not in bars:
                    continue
                day = _bars_to_day(bars[p["symbol"]])[None]
                side = 1 if p["side"] == "LONG" else -1
                res = simulate(day, np.array([m]), np.array([side]), np.array([p["stop"]]), np.array([1.0]), rules)
                if np.isfinite(res["r"][0]):
                    out.append(float(res["r"][0] * res["risk"][0] / res["entry"][0]) - I.cost_estimate(100_000.0))
            self.monitors[m].extend(out)
            d["monitor"] = self.monitors[m].state()
        await asyncio.to_thread(self._save_monitors)
        log.info("edge monitor: %s", {I.DECISIONS[m]: mon.state() for m, mon in self.monitors.items()})

    # ------------------------------------------------------------------ swing
    def fill_swing_pending(self) -> None:
        C = SW.CONFIG
        for acct in self.accounts.values():
            pend, keep = acct.swing["pending"], []
            if not pend:
                continue
            if not acct.settings["swing"]["enabled"]:
                acct.swing["pending"] = []
                acct.dirty = True
                continue
            for p in pend:
                px = self.ltp.get(p["symbol"])
                if not px:
                    keep.append(p)
                    continue
                if len(acct.swing["positions"]) >= C.max_positions or \
                        any(q["symbol"] == p["symbol"] for q in acct.swing["positions"]):
                    continue
                eq, cash = acct.swing_equity(self.ltp), acct.swing_cash()
                stop = px - C.stop_atr * p["atr"]
                rs = p["reasons"].get("regime", {}).get("risk_scale", 1.0)
                qty = int(min(rs * C.risk_per_trade * eq / max(px - stop, 1e-9), C.max_weight * eq / px, cash * 0.995 / px))
                if qty < 1:
                    continue
                reasons = dict(p["reasons"])
                reasons["sizing"] = (f"risk {rs * C.risk_per_trade:.2%} of ₹{eq:,.0f} (regime scale {rs:.0%}) at a "
                                     f"{C.stop_atr:g} ATR stop ({(px - stop) / px:.2%}) → {qty} shares "
                                     f"(₹{qty * px:,.0f}, {qty * px / eq:.1%} of equity; cap {C.max_weight:.0%})")
                acct.open_position("swing", p["symbol"], 1, qty, px, self.clock, stop, p["atr"], p["decision"], reasons,
                                   daily_vol=p["atr"] / px, extra={"entry_date": _ymd(self.day)})
            acct.swing["pending"] = keep
            acct.dirty = True

    def _swing_scan(self, today_bars: dict[str, list]) -> tuple[pd.DataFrame, dict | None]:
        """Score every stock on today's close (lake history + today's bar)."""
        today = _ymd(self.day)
        ts0 = int(pd.Timestamp(self.day).timestamp()) - 19800 + 12 * 3600
        nd = self.lake.read("D", INDEX)
        if nd is None or nd.empty:
            return pd.DataFrame(), None
        def with_today(D: pd.DataFrame, rows: list | None) -> pd.DataFrame:
            D = D.sort_values("ts")
            D = D[pd.to_datetime(D["ts"] + 19800, unit="s").dt.strftime("%Y%m%d").astype(int) < today]
            if rows:
                a = np.asarray([r[1:6] for r in rows], float)
                bar = {"ts": ts0, "open": a[0, 0], "high": a[:, 1].max(), "low": a[:, 2].min(), "close": a[-1, 3],
                       "volume": a[:, 4].sum()}
                D = pd.concat([D, pd.DataFrame([bar])], ignore_index=True)
            return D
        nd = with_today(nd, today_bars.get(INDEX))
        iclose = pd.Series(nd["close"].to_numpy(float),
                           index=pd.to_datetime(nd["ts"] + 19800, unit="s").dt.strftime("%Y%m%d").astype(int).to_numpy())
        parts = []
        with np.errstate(divide="ignore", invalid="ignore"):
            for sym in self.eq_key:
                D = self.lake.read("D", sym)
                if D is None or len(D) < 260:
                    continue
                f = swing_features(with_today(D, today_bars.get(sym)), iclose)
                f["symbol"] = sym
                parts.append(f.tail(1))
            if not parts:
                return pd.DataFrame(), None
            panel = pd.concat(parts, ignore_index=True)
            panel = panel[panel["date"] == today]
            vix = self.lake.read("D", "INDIAVIX")
            reg = SW.regime_table(nd, panel, vix if vix is not None and len(vix) else None)
        sc = SW.score(panel)
        r = reg.loc[today] if today in reg.index else None
        regime = None
        if r is not None:
            lights = int(r["lights"])
            bits = []
            if r["m_e200"] < 0:
                bits.append("NIFTY below its 200-day EMA")
            if r["m_e50"] < 0:
                bits.append("NIFTY below its 50-day EMA")
            if r["breadth50"] < 0.40:
                bits.append(f"only {r['breadth50']:.0%} of stocks above their 50-day EMA")
            regime = {"lights": lights, "risk_scale": float(r["risk_scale"]), "asof": today,
                      "text": ("All clear: full risk on new positions." if not bits else
                               f"{lights} warning light(s): " + "; ".join(bits) + f" → new positions at "
                               f"{float(r['risk_scale']):.0%} risk" + (" (no new entries)." if lights >= 3 else "."))}
        return sc, regime

    async def swing_end_of_day(self) -> None:
        today = _ymd(self.day)
        C = SW.CONFIG
        bars = await self.today_bars(list(self.eq_key) + [INDEX])
        if len(bars) < 10:
            log.info("swing EOD %s skipped: no bars for today (holiday or no feed)", today)
            return
        sc, regime = await asyncio.to_thread(self._swing_scan, bars)
        self.regime = regime
        close = {s: float(b[-1][4]) for s, b in bars.items() if b}
        # manage open positions on the close
        for acct in self.accounts.values():
            for pos in list(acct.swing["positions"]):
                c = close.get(pos["symbol"]) or self.ltp.get(pos["symbol"])
                if not c:
                    continue
                if pos.get("last_eod") == today:
                    continue
                pos["last_eod"] = today
                pos["sessions"] = pos.get("sessions", 0) + 1
                if pos["sessions"] >= C.max_days:
                    acct.close_position(pos, c, self.clock, f"time stop {C.max_days} sessions",
                                        f"Held {pos['sessions']} sessions without the trailing stop being hit: the time "
                                        "stop frees the slot for a fresher leader.", daily_vol=pos["atr"] / c)
                    continue
                pos["best"] = max(pos.get("best", pos["entry"]), c)
                new_stop = pos["best"] - C.trail_atr * pos["atr"]
                if new_stop > pos["stop"]:
                    pos["stop"] = round(new_stop, 4)
                acct.dirty = True
        # tomorrow's candidates
        cands = []
        if not sc.empty:
            top = sc.sort_values("score", ascending=False).head(25)
            for _, row in top.iterrows():
                drivers = SW.reasons(row)
                cands.append({"symbol": row["symbol"], "score": round(float(row["score"]), 3),
                              "setup": "Momentum leader" if row.get("hh252_pct", -1) > -0.05 else "Trend strength",
                              "drivers": drivers, "price": round(float(row["close"]), 2),
                              "stop": round(float(row["close"] - C.stop_atr * row["atr"]), 2),
                              "atr": float(row["atr"])})
        self.signals["swing"] = {"asof": today, "regime": regime, "candidates": cands}
        scale = regime["risk_scale"] if regime else 1.0
        for acct in self.accounts.values():
            if not acct.settings["swing"]["enabled"] or scale <= 0:
                continue
            held = {p["symbol"] for p in acct.swing["positions"]} | {p["symbol"] for p in acct.swing["pending"]}
            free = C.max_positions - len(acct.swing["positions"]) - len(acct.swing["pending"])
            for c in cands:
                if free <= 0:
                    break
                if c["symbol"] in held:
                    continue
                top2 = ", ".join(f"{d['label']} ({d['z']:+.1f}σ)" for d in c["drivers"][:3])
                acct.swing["pending"].append({
                    "symbol": c["symbol"], "strategy": "swing", "planned_at": self.clock, "atr": c["atr"],
                    "decision": f"EOD {self.day.isoformat()}",
                    "reasons": {"setup": c["setup"], "score": c["score"], "drivers": c["drivers"],
                                "facts": {"close": c["price"], "planned stop": c["stop"], "ATR": round(c["atr"], 2)},
                                "summary": (f"Ranked among the strongest liquid F&O stocks at the close (score "
                                            f"{c['score']:+.2f}): {top2}. Buy at the next open; stop "
                                            f"{C.stop_atr:g} ATR below, then trail {C.trail_atr:g} ATR under the best close."),
                                "sizing": "sized at the fill (next open)", "regime": regime or {}}})
                free -= 1
            acct.dirty = True
        for acct in self.accounts.values():
            acct.mark_day(today, {**self.ltp, **close})
        log.info("swing EOD %s: %d scored, %d candidates, regime %s", today, len(sc), len(cands),
                 regime["lights"] if regime else "?")

    # ------------------------------------------------------------------ model upkeep
    RETRAIN_DAYS = 30
    MIN_TRAIN_DAYS = 250

    def _local_model_age_days(self) -> float:
        files = list(I.model_dir().glob("intraday_m*_*.json"))
        if len(files) < 2 * len(I.DECISIONS):
            return float("inf")
        return (time.time() - min(f.stat().st_mtime for f in files)) / 86400

    def _enough_history(self) -> bool:
        deep = 0
        for sym in list(self.eq_key)[:120]:
            try:
                d, _, _ = load_days(self.lake, sym)
            except Exception:
                continue
            deep += len(d) >= self.MIN_TRAIN_DAYS
        return deep >= 50

    def _retrain(self) -> str:
        from ..paper.train import build_dataset, train
        if not self._enough_history():
            return "skipped: under 250 days of minute history for 50 stocks (pull history on the admin page)"
        ds = build_dataset(self.lake, workers=2)
        I.save_models(train(ds))
        return f"trained on {len(ds)} rows"

    async def maybe_retrain(self) -> None:
        """Monthly, after the close, in live mode: refit the intraday models on this server's own history."""
        if settings.data_source == "sim" or (self.train_task and not self.train_task.done()):
            return
        if self._local_model_age_days() < self.RETRAIN_DAYS:
            return
        async def job():
            try:
                msg = await asyncio.to_thread(self._retrain)
                self.models = I.load_models()
                log.info("intraday models: %s (now using %s models)", msg,
                         {v.meta.get("source") for v in self.models.values()})
            except Exception:
                log.exception("intraday model retraining failed")
        self.train_task = asyncio.create_task(job())

    # ------------------------------------------------------------------ commands from the gateway
    async def commands(self) -> None:
        for _ in range(50):
            raw = await self.r.lpop(K_CMD)
            if not raw:
                return
            try:
                c = json.loads(raw)
                acct = self.account(c["user"])
                if c["kind"] == "settings":
                    acct.update_settings(c["patch"])
                elif c["kind"] == "reset":
                    acct.reset(c["strategy"])
                if self.day:
                    acct.new_day(_ymd(self.day))
                acct.save()
                await self.publish_account(c["user"], acct)
            except Exception:
                log.exception("bad paper command %r", raw[:200])

    # ------------------------------------------------------------------ publishing
    def service_info(self) -> dict:
        src = {v.meta.get("source") for v in self.models.values()}
        trained = max((v.meta.get("train_to") or 0 for v in self.models.values()), default=0) or None
        return {"running": True, "clock": self.clock or None, "phase": self.phase, "mode": settings.data_source
                if settings.data_source == "sim" else "live",
                "models": {"source": ("local" if "local" in src else "shipped" if "shipped" in src else "none"),
                           "trained_to": trained},
                "note": "loading history…" if self.loading else self._monitor_note(),
                "edge_monitor": {"enabled": settings.paper_edge_monitor,
                                 **{I.DECISIONS[m]: mon.state() for m, mon in self.monitors.items()}}}

    def _monitor_note(self) -> str:
        if not settings.paper_edge_monitor:
            return ""
        off = [I.DECISIONS[m] for m, mon in self.monitors.items() if not mon.allows()]
        if not off:
            return ""
        avg = ", ".join(f"{I.DECISIONS[m]} {self.monitors[m].state()['avg_net']:+.2%}" for m in self.monitors
                        if not self.monitors[m].allows())
        return (f"Intraday is sitting out at {' and '.join(off)}: the last {I.MONITOR_N} signals averaged {avg} after "
                "costs (edge monitor). Signals are still tracked; trading resumes by itself when they turn profitable.")

    async def publish_account(self, user: str, acct: Account) -> None:
        view = acct.view(self.ltp, self.service_info(), self.regime)
        await self.r.set(state_key(user), json.dumps(view, default=float), ex=600)

    async def publish(self) -> None:
        for user, acct in self.accounts.items():
            if acct.dirty:
                acct.save()
            await self.publish_account(user, acct)
        await bus.publish_snapshot(self.r, "paper", self.signals)

    # ------------------------------------------------------------------ main loop
    def discover_accounts(self) -> None:
        """Accounts the gateway created while this service was down or starting."""
        for p in accounts_dir().glob("*.json"):
            if p.name.count(".") == 1 and p.stem not in self.accounts:
                self.accounts[p.stem] = Account(p.stem)
                if self.day:
                    self.accounts[p.stem].new_day(_ymd(self.day))
                log.info("paper account %s picked up", p.stem)

    async def heartbeat(self) -> None:
        await self.r.set(bus.k_status("paper"), bus.dumps({"pid": os.getpid(), "clock": self.clock, "phase": self.phase,
                                                           "accounts": len(self.accounts), "loading": self.loading,
                                                           "updated": time.time()}), ex=30)

    async def step(self) -> None:
        if time.monotonic() - self.last_discover > 10:
            self.last_discover = time.monotonic()
            self.discover_accounts()
        await self.commands()
        await self.heartbeat()
        if not await self.read_market():
            return
        day = from_epoch(self.clock).date()
        if self.day != day:
            await self.new_day(day)
        if self.loading:
            return
        today = _ymd(day)
        trading_day = self.phase in ("OPEN", "POST_CLOSE") or (self.phase == "CLOSED" and self.minute >= SESSION)
        if self.phase == "OPEN" and self.swing_filled_day != today and self.minute >= 1.0:
            self.fill_swing_pending()
            self.swing_filled_day = today
        for m in I.DECISIONS:
            if m not in self.decided and self.phase == "OPEN" and self.minute >= m + DECISION_DELAY:
                await self.decide_intraday(m)
        if trading_day:
            self.manage_exits()
        if self.shadow_day != today and self.minute >= SQUARE_OFF + 1 and trading_day:
            self.shadow_day = today
            await self.record_shadows()
        if self.swing_eod_day != today and self.minute >= SWING_EOD_MINUTE and self.phase in ("POST_CLOSE", "CLOSED") \
                and any(self.ltp.values()):
            self.swing_eod_day = today
            await self.swing_end_of_day()
            await self.maybe_retrain()
        if time.monotonic() - self.last_publish >= 2.0:
            self.last_publish = time.monotonic()
            await self.publish()

    async def run(self) -> None:
        await self.setup()
        while True:
            t0 = time.monotonic()
            try:
                await self.step()
            except Exception:
                log.exception("paper step failed")
            await asyncio.sleep(max(0.2, 1.0 - (time.monotonic() - t0)))


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    r = bus.connect()
    await PaperService(r).run()


if __name__ == "__main__":
    asyncio.run(main())
