"""Paper accounts and the trade book.

One account per user, with two independent books:
  intraday  a margin account: capital x leverage of buying power, P&L realised at every exit, every position
            squared off the same day. Equity = capital + realised P&L to date + open P&L.
  swing     a cash (delivery) account, long only: buying spends cash (price + charges), selling returns it.
            Equity = cash + market value of the holdings.
Fills: market orders fill at the given price moved against us by the slippage model; charges are the exact
statutory + brokerage schedule (costs.py) on each order's value. Every closed trade is appended to the user's
book (one JSON object per line) with why it was opened, why it was closed and what it cost.

Files (single writer: the paper service): data/paper/accounts/<user>.json and <user>.book.jsonl.
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path

from ..config import settings
from ..timeutil import from_epoch
from .costs import charges, fill_price, slippage_bps

MIN_CAPITAL, MAX_CAPITAL = 10_000.0, 100_000_000.0
MAX_LEVERAGE = 5.0
CURVE_DAYS = 800


def accounts_dir() -> Path:
    return Path(settings.data_dir) / "paper" / "accounts"


def _atomic_write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, p)


def default_state(user: str) -> dict:
    return {"user": user, "version": 1, "created": time.time(),
            "settings": {"intraday": {"enabled": False, "capital": 1_000_000.0, "leverage": 5.0},
                         "swing": {"enabled": False, "capital": 1_000_000.0}},
            "intraday": {"pnl_total": 0.0, "charges_total": 0.0, "day": 0, "realized_today": 0.0, "trades_today": 0,
                         "loss_limit_hit": False, "positions": [], "decided": {}},
            "swing": {"pnl_total": 0.0, "charges_total": 0.0, "positions": [], "pending": []},
            "equity_curve": []}


def validate_settings(cur: dict, patch: dict) -> dict:
    """Merge a settings patch into the current settings, raising ValueError with a readable message."""
    out = json.loads(json.dumps(cur))
    for strat in ("intraday", "swing"):
        p = patch.get(strat)
        if p is None:
            continue
        if not isinstance(p, dict):
            raise ValueError(f"{strat} settings must be an object")
        if "enabled" in p:
            out[strat]["enabled"] = bool(p["enabled"])
        if "capital" in p and p["capital"] is not None:
            c = float(p["capital"])
            if not math.isfinite(c) or c < MIN_CAPITAL or c > MAX_CAPITAL:
                raise ValueError(f"{strat} capital must be between ₹{MIN_CAPITAL:,.0f} and ₹{MAX_CAPITAL:,.0f}")
            out[strat]["capital"] = round(c, 2)
        if strat == "intraday" and "leverage" in p and p["leverage"] is not None:
            lev = float(p["leverage"])
            if not math.isfinite(lev) or lev < 1 or lev > MAX_LEVERAGE:
                raise ValueError("leverage must be between 1x and 5x")
            out[strat]["leverage"] = round(lev, 2)
        if strat == "swing" and "leverage" in p:
            raise ValueError("swing trades are delivery (cash) only: no leverage")
    return out


class Account:
    def __init__(self, user: str, root: Path | None = None):
        self.user = user
        self.root = root or accounts_dir()
        self.path = self.root / f"{user}.json"
        self.book_path = self.root / f"{user}.book.jsonl"
        self.s = default_state(user)
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                base = default_state(user)
                for k, v in loaded.items():
                    if isinstance(v, dict) and isinstance(base.get(k), dict):
                        base[k].update(v)
                    else:
                        base[k] = v
                self.s = base
            except Exception:
                bad = self.path.with_suffix(f".corrupt.{int(time.time())}")
                self.path.rename(bad)
        self.dirty = False

    # ------------------------------------------------------------------ persistence
    def save(self) -> None:
        _atomic_write(self.path, json.dumps(self.s, separators=(",", ":")))
        self.dirty = False

    def _append_book(self, rec: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.book_path, "a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def book(self, strategy: str = "all", limit: int = 200) -> list[dict]:
        if not self.book_path.exists():
            return []
        out = []
        with open(self.book_path) as f:
            for line in f:
                try:
                    t = json.loads(line)
                except ValueError:
                    continue
                if strategy in ("all", t.get("strategy")):
                    out.append(t)
        return out[-limit:][::-1]

    # ------------------------------------------------------------------ settings
    @property
    def settings(self) -> dict:
        return self.s["settings"]

    def update_settings(self, patch: dict) -> None:
        new = validate_settings(self.settings, patch)
        old_cap = self.settings["swing"]["capital"]
        self.s["settings"] = new
        self.s.setdefault("events", []).append({"ts": time.time(), "kind": "settings", "settings": new,
                                                "swing_capital_change": new["swing"]["capital"] - old_cap})
        self.s["events"] = self.s["events"][-100:]
        self.dirty = True

    def reset(self, strategy: str) -> None:
        base = default_state(self.user)
        for strat in (("intraday", "swing") if strategy == "all" else (strategy,)):
            self.s[strat] = base[strat]
        if strategy == "all":
            self.s["equity_curve"] = []
        if self.book_path.exists():
            keep = [] if strategy == "all" else [t for t in self.book("all", 10**9)[::-1] if t.get("strategy") != strategy]
            archive = self.book_path.with_suffix(f".{int(time.time())}.reset.jsonl")
            self.book_path.rename(archive)
            for t in keep:
                self._append_book(t)
        self.dirty = True

    # ------------------------------------------------------------------ intraday
    @property
    def intraday(self) -> dict:
        return self.s["intraday"]

    @property
    def swing(self) -> dict:
        return self.s["swing"]

    def new_day(self, day: int) -> None:
        d = self.intraday
        if d["day"] != day:
            d.update(day=day, realized_today=0.0, trades_today=0, loss_limit_hit=False, decided={}, traded_today=[])
            self.dirty = True

    def intraday_equity(self, ltp: dict[str, float]) -> float:
        cap = self.settings["intraday"]["capital"]
        return cap + self.intraday["pnl_total"] + sum(self._open_pnl(p, ltp.get(p["symbol"])) for p in self.intraday["positions"])

    def intraday_gross(self, ltp: dict[str, float]) -> float:
        return sum(p["qty"] * (ltp.get(p["symbol"]) or p["entry"]) for p in self.intraday["positions"])

    def open_position(self, strategy: str, symbol: str, side: int, qty: int, price: float, ts: float, stop: float,
                      atr: float, decision: str, reasons: dict, minute_volume: float = 0.0, daily_vol: float = 0.02,
                      extra: dict | None = None) -> dict:
        bps = slippage_bps(qty, price, minute_volume or qty * 20, daily_vol)
        px = fill_price(side, price, bps)
        pos = {"id": uuid.uuid4().hex[:12], "strategy": strategy, "symbol": symbol, "side": side, "qty": int(qty),
               "entry": round(px, 4), "ref_entry": price, "entry_ts": ts, "stop": round(stop, 4),
               "initial_stop": round(stop, 4), "atr": atr, "best": px, "decision": decision, "reasons": reasons,
               "entry_slip": round(abs(px - price) * qty, 2), "sessions": 0, **(extra or {})}
        if strategy == "swing":
            fee = charges(qty * px, 0, "delivery")["total"]
            pos["buy_fee"] = fee
            self.swing["positions"].append(pos)
        else:
            self.intraday["positions"].append(pos)
            self.intraday["trades_today"] += 1
        self.dirty = True
        return pos

    def close_position(self, pos: dict, price: float, ts: float, why: str, note: str, stop_order: bool = False,
                       minute_volume: float = 0.0, daily_vol: float = 0.02) -> dict:
        side, qty = pos["side"], pos["qty"]
        bps = slippage_bps(qty, price, minute_volume or qty * 20, daily_vol, stop=stop_order)
        px = fill_price(-side, price, bps)
        exit_slip = abs(px - price) * qty
        if pos["strategy"] == "swing":
            ch = charges(0, qty * px, "delivery")
            buy_ch = charges(qty * pos["entry"], 0, "delivery")
            ch = {k: round(ch.get(k, 0) + buy_ch.get(k, 0), 2) for k in set(ch) | set(buy_ch)}
            book = self.swing
        else:
            buy, sell = (qty * pos["entry"], qty * px) if side > 0 else (qty * px, qty * pos["entry"])
            ch = charges(buy, sell, "intraday")
            book = self.intraday
        gross = round(side * (px - pos["entry"]) * qty, 2)
        pnl = round(gross - ch["total"], 2)                    # the number the book shows is the number booked
        risk = abs(pos["entry"] - pos["initial_stop"]) * qty
        held = ts - pos["entry_ts"]
        holding = f"{pos.get('sessions', 0)} sessions" if pos["strategy"] == "swing" else \
            f"{int(held // 3600)}h {int(held % 3600 // 60)}m"
        rec = {"id": pos["id"], "strategy": pos["strategy"], "symbol": pos["symbol"],
               "side": "LONG" if side > 0 else "SHORT", "qty": qty, "entry": pos["entry"], "exit": round(px, 4),
               "entry_ts": pos["entry_ts"], "exit_ts": ts, "why_exit": why, "exit_note": note,
               "pnl": round(pnl, 2), "pnl_gross": round(gross, 2), "pnl_pct": pnl / max(qty * pos["entry"], 1e-9),
               "r_mult": round(pnl / risk, 3) if risk > 0 else None, "charges": ch,
               "slippage": round(pos.get("entry_slip", 0) + exit_slip, 2), "holding": holding,
               "decision": pos["decision"], "reasons": pos["reasons"], "initial_stop": pos["initial_stop"],
               "date": int(from_epoch(ts).strftime("%Y%m%d"))}
        book["pnl_total"] += pnl
        book["charges_total"] = book.get("charges_total", 0.0) + ch["total"]
        if pos["strategy"] == "intraday":
            self.intraday["realized_today"] += pnl
        book["positions"] = [p for p in book["positions"] if p["id"] != pos["id"]]
        self._append_book(rec)
        self.dirty = True
        return rec

    # ------------------------------------------------------------------ swing
    def swing_cash(self) -> float:
        """Capital + realised P&L - what the open holdings cost (price + buy charges). Realised P&L already nets
        both legs' charges, so at each exit cash grows by exactly the sale proceeds minus the sell charges."""
        spent = sum(p["qty"] * p["entry"] + p.get("buy_fee", 0.0) for p in self.swing["positions"])
        return self.settings["swing"]["capital"] + self.swing["pnl_total"] - spent

    def swing_equity(self, ltp: dict[str, float]) -> float:
        return self.swing_cash() + sum(p["qty"] * (ltp.get(p["symbol"]) or p["entry"]) for p in self.swing["positions"])

    # ------------------------------------------------------------------ marks
    @staticmethod
    def _open_pnl(p: dict, ltp: float | None) -> float:
        if not ltp:
            return 0.0
        return p["side"] * (ltp - p["entry"]) * p["qty"]

    def mark_day(self, day: int, ltp: dict[str, float]) -> None:
        """End-of-day equity point (one per date, overwritten while the day is open)."""
        pt = {"date": day, "intraday": round(self.intraday_equity(ltp), 2), "swing": round(self.swing_equity(ltp), 2)}
        curve = self.s["equity_curve"]
        if curve and curve[-1]["date"] == day:
            curve[-1] = pt
        else:
            curve.append(pt)
        self.s["equity_curve"] = curve[-CURVE_DAYS:]
        self.dirty = True

    # ------------------------------------------------------------------ views
    def _pos_view(self, p: dict, ltp: dict[str, float]) -> dict:
        last = ltp.get(p["symbol"])
        pnl = self._open_pnl(p, last)
        risk = abs(p["entry"] - p["initial_stop"]) * p["qty"]
        return {"id": p["id"], "strategy": p["strategy"], "symbol": p["symbol"],
                "side": "LONG" if p["side"] > 0 else "SHORT", "qty": p["qty"], "entry": p["entry"],
                "entry_ts": p["entry_ts"], "ltp": last, "stop": p["stop"], "initial_stop": p["initial_stop"],
                "pnl": round(pnl, 2), "pnl_pct": pnl / max(p["qty"] * p["entry"], 1e-9),
                "r_mult": round(pnl / risk, 2) if risk > 0 and last else None, "decision": p["decision"],
                "days_held": p.get("sessions"), "reasons": p["reasons"]}

    def stats(self, strategy: str) -> dict:
        trades = self.book(strategy, 10**9)
        out = {"trades": len(trades), "win_rate": None, "profit_factor": None, "avg_pnl": None, "avg_r": None,
               "best": None, "worst": None, "max_dd": None,
               "charges": round(self.s[strategy].get("charges_total", 0.0), 2)}
        if trades:
            pnl = [t["pnl"] for t in trades]
            wins, losses = sum(x for x in pnl if x > 0), -sum(x for x in pnl if x < 0)
            rs = [t["r_mult"] for t in trades if t.get("r_mult") is not None]
            out.update(win_rate=sum(1 for x in pnl if x > 0) / len(pnl),
                       profit_factor=(wins / losses) if losses > 0 else None, avg_pnl=sum(pnl) / len(pnl),
                       avg_r=(sum(rs) / len(rs)) if rs else None, best=max(pnl), worst=min(pnl))
        curve = [pt[strategy] for pt in self.s["equity_curve"]]
        if len(curve) > 1:
            peak, dd = curve[0], 0.0
            for v in curve:
                peak = max(peak, v)
                dd = min(dd, v / peak - 1 if peak > 0 else 0.0)
            out["max_dd"] = dd
        return out

    def view(self, ltp: dict[str, float], service: dict, regime: dict | None) -> dict:
        st = self.settings
        ipos = [self._pos_view(p, ltp) for p in self.intraday["positions"]]
        spos = [self._pos_view(p, ltp) for p in self.swing["positions"]]
        i_eq = self.intraday_equity(ltp)
        unreal_i = sum(p["pnl"] for p in ipos)
        cap_i, lev = st["intraday"]["capital"], st["intraday"]["leverage"]
        gross = self.intraday_gross(ltp)
        s_cash = self.swing_cash()
        invested = sum(p["qty"] * (ltp.get(p["symbol"]) or p["entry"]) for p in self.swing["positions"])
        return {
            "user": self.user, "service": service, "settings": st,
            "intraday": {"capital": cap_i, "leverage": lev, "equity": round(i_eq, 2),
                         "pnl_total": round(self.intraday["pnl_total"] + unreal_i, 2),
                         "pnl_today": round(self.intraday["realized_today"] + unreal_i, 2),
                         "realized_today": round(self.intraday["realized_today"], 2), "unrealized": round(unreal_i, 2),
                         "gross_exposure": round(gross, 2), "buying_power": round(max(0.0, lev * i_eq - gross), 2),
                         "trades_today": self.intraday["trades_today"], "loss_limit_hit": self.intraday["loss_limit_hit"],
                         "positions": ipos, "stats": self.stats("intraday")},
            "swing": {"capital": st["swing"]["capital"], "equity": round(s_cash + invested, 2), "cash": round(s_cash, 2),
                      "invested": round(invested, 2),
                      "pnl_total": round(self.swing["pnl_total"] + sum(p["pnl"] for p in spos), 2),
                      "unrealized": round(sum(p["pnl"] for p in spos), 2), "positions": spos,
                      "pending": self.swing["pending"], "stats": self.stats("swing"), "regime": regime},
            "equity_curve": self.s["equity_curve"],
        }
