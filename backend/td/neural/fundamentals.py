"""Point-in-time fundamentals from a parsed Screener.in page (td.research.screener).

A quarter's results are not known on the last day of the quarter: listed companies publish them within 45 days
(60 for the March quarter, which carries the audited full year). Each reported number therefore becomes usable
only `lag` days after its period ends — a model trained without that lag "knows" results weeks before the market
did and looks brilliant in a backtest it could never repeat.

Screener shows the latest restated figures, about 12 annual columns and the last ~13 quarters. Days before a
quarter's data exists get NaN (the network sees a mask); restatements are not modelled.

`daily_values(f, ts, close)` returns one row per daily bar with the FUND_KEYS below, as of the end of that day.
Price-based ratios (earnings and book yield) use that day's close, so they move daily; the rest step on results.
"""
from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np

from ..research.screener import _equity, _points, is_financial
from ..timeutil import IST

FUND_KEYS: tuple[str, ...] = (
    "earn_yield",       # trailing-4-quarter EPS / close
    "book_yield",       # book value per share / close
    "q_sales_yoy",      # latest quarter's revenue vs the same quarter a year earlier (log ratio)
    "q_np_yoy",         # latest quarter's net profit growth vs a year earlier (signed, capped)
    "q_opm",            # latest quarter's operating (or financing) margin
    "q_opm_yoy",        # its change vs a year earlier
    "a_roe",            # last full year's net profit / equity
    "a_de",             # borrowings / equity (not for lenders, whose borrowings are their business)
    "a_sales_cagr3",    # 3-year revenue CAGR
    "a_payout",         # dividend payout ratio
    "days_since_q",     # trading days since the latest quarter became public (post-results drift)
)
FUND_LABEL = {
    "earn_yield": "Earnings yield (TTM EPS / price)", "book_yield": "Book yield (book value / price)",
    "q_sales_yoy": "Quarterly revenue growth (YoY)", "q_np_yoy": "Quarterly profit growth (YoY)",
    "q_opm": "Operating margin (latest quarter)", "q_opm_yoy": "Margin change (YoY)",
    "a_roe": "Return on equity (last year)", "a_de": "Debt / equity", "a_sales_cagr3": "3-year revenue CAGR",
    "a_payout": "Dividend payout", "days_since_q": "Days since the last results",
}
QUARTER_LAG_DAYS = 45
YEAR_END_LAG_DAYS = 60          # March quarter / full year: audited results, 60 days


def _month_end(ym: int) -> date:
    y, m = divmod(ym, 12)
    return date(y, m + 1, calendar.monthrange(y, m + 1)[1])


def _avail_ts(ym: int, lag: int) -> int:
    """Epoch (IST midnight) of the first day a period ending in month `ym` can be used."""
    d = _month_end(ym) + timedelta(days=lag)
    return int(datetime(d.year, d.month, d.day, tzinfo=IST).timestamp())


def _growth(new: float, old: float, cap: float = 3.0) -> float:
    if old is None or new is None or old == 0:
        return math.nan
    return float(np.clip((new - old) / abs(old), -cap, cap))


def _logratio(new: float, old: float) -> float:
    if not new or not old or new <= 0 or old <= 0:
        return math.nan
    return float(math.log(new / old))


@dataclass
class Event:
    ts: int                     # first day the values are public
    values: dict[str, float]    # FUND_KEYS subset (price-independent) and the per-share inputs eps_ttm / bvps


def events(f: dict | None) -> list[Event]:
    """Every result announcement as (public from, values known from then on), oldest first."""
    if not f:
        return []
    fin = is_financial(f)
    out: list[Event] = []
    q = f.get("quarters")
    sales = {p.ym: p.v for p in _points(q, "Sales", "Revenue")}
    np_ = {p.ym: p.v for p in _points(q, "Net Profit")}
    eps = {p.ym: p.v for p in _points(q, "EPS in Rs")}
    opm = {p.ym: p.v / 100.0 for p in _points(q, "OPM %", "Financing Margin %")}
    for ym in sorted(set(sales) | set(np_) | set(eps)):
        lag = YEAR_END_LAG_DAYS if ym % 12 == 2 else QUARTER_LAG_DAYS
        v: dict[str, float] = {}
        prev = ym - 12
        if ym in sales and prev in sales:
            v["q_sales_yoy"] = _logratio(sales[ym], sales[prev])
        if ym in np_ and prev in np_:
            v["q_np_yoy"] = _growth(np_[ym], np_[prev])
        if ym in opm:                           # OPM % for companies, Financing Margin % for lenders
            v["q_opm"] = opm[ym]
            if prev in opm:
                v["q_opm_yoy"] = opm[ym] - opm[prev]
        last4 = [eps.get(ym - 3 * k) for k in range(4)]
        if all(x is not None for x in last4):
            v["eps_ttm"] = float(sum(last4))
        v["q_ym"] = float(ym)
        out.append(Event(_avail_ts(ym, lag), v))

    pl, bs = f.get("profit_loss"), f.get("balance_sheet")
    a_sales = {p.ym: p.v for p in _points(pl, "Sales", "Revenue") if p.months == 12}
    a_np = {p.ym: p.v for p in _points(pl, "Net Profit") if p.months == 12}
    a_eps = {p.ym: p.v for p in _points(pl, "EPS in Rs") if p.months == 12}
    a_pay = {p.ym: p.v / 100.0 for p in _points(pl, "Dividend Payout %") if p.months == 12}
    equity = {p.ym: p.v for p in _equity(bs)}
    debt = {p.ym: p.v for p in _points(bs, "Borrowings", "Borrowing")}
    for ym in sorted(set(a_np) | set(equity)):
        v = {}
        eq = equity.get(ym)
        if eq and eq > 0 and ym in a_np:
            v["a_roe"] = float(np.clip(a_np[ym] / eq, -2, 2))
        if eq and eq > 0 and ym in debt and not fin:
            v["a_de"] = float(min(debt[ym] / eq, 20.0))
        if ym in a_sales and ym - 36 in a_sales and a_sales[ym] > 0 and a_sales[ym - 36] > 0:
            v["a_sales_cagr3"] = float((a_sales[ym] / a_sales[ym - 36]) ** (1 / 3) - 1)
        if ym in a_pay:
            v["a_payout"] = float(np.clip(a_pay[ym], 0, 3))
        # book value per share: equity (₹ cr) / shares (cr), shares = net profit / EPS of the same year
        if eq and ym in a_np and a_eps.get(ym) and abs(a_eps[ym]) > 1e-6 and a_np[ym] / a_eps[ym] > 0:
            v["bvps"] = float(eq / (a_np[ym] / a_eps[ym]))
        if v:
            out.append(Event(_avail_ts(ym, YEAR_END_LAG_DAYS), v))
    out.sort(key=lambda e: e.ts)
    return out


def daily_values(f: dict | None, ts: np.ndarray, close: np.ndarray) -> np.ndarray:
    """(n, len(FUND_KEYS)) as of each bar (ts = IST-midnight epochs of daily bars, close = that day's close)."""
    n = len(ts)
    out = np.full((n, len(FUND_KEYS)), np.nan)
    evs = events(f)
    if not evs or n == 0:
        return out
    keys = list(FUND_KEYS) + ["eps_ttm", "bvps", "q_ym"]
    col = {k: i for i, k in enumerate(keys)}
    state = np.full((len(evs), len(keys)), np.nan)
    cur = np.full(len(keys), np.nan)
    for i, e in enumerate(evs):                 # forward-fill: a value stays known until the next one replaces it
        for k, v in e.values.items():
            if k in col and v is not None and math.isfinite(v):
                cur[col[k]] = v
        state[i] = cur
    ev_ts = np.array([e.ts for e in evs], dtype=np.int64)
    j = np.searchsorted(ev_ts, np.asarray(ts, dtype=np.int64), side="right") - 1
    known = j >= 0
    S = np.where(known[:, None], state[np.clip(j, 0, None)], np.nan)
    for k in FUND_KEYS:
        if k in col and k not in ("earn_yield", "book_yield", "days_since_q"):
            out[:, FUND_KEYS.index(k)] = S[:, col[k]]
    c = np.asarray(close, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, FUND_KEYS.index("earn_yield")] = np.clip(S[:, col["eps_ttm"]] / c, -1, 1)
        out[:, FUND_KEYS.index("book_yield")] = np.clip(S[:, col["bvps"]] / c, -5, 5)
    # trading days since the latest quarter went public: the bar index distance to that quarter's first bar
    q_pub = np.full(n, -1, dtype=np.int64)
    q_events = np.array([e.ts for e in evs if "q_ym" in e.values], dtype=np.int64)
    if q_events.size:
        jq = np.searchsorted(q_events, np.asarray(ts, dtype=np.int64), side="right") - 1
        ok = jq >= 0
        first_bar = np.searchsorted(np.asarray(ts, dtype=np.int64), q_events[np.clip(jq, 0, None)], side="left")
        q_pub = np.where(ok, first_bar, -1)
    days = np.arange(n) - q_pub
    out[:, FUND_KEYS.index("days_since_q")] = np.where(q_pub >= 0, np.log1p(np.clip(days, 0, 400)), np.nan)
    return out
