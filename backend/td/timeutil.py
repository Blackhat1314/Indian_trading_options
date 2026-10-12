"""IST clock helpers and NSE session phases.

All analytics use *market time* (the timestamp carried by ticks), never wall clock,
so the simulator can run accelerated and backtests can replay history.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

PRE_OPEN_START = time(9, 0)
PRE_OPEN_END = time(9, 8)
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
CLOSING_SESSION_END = time(16, 0)


def now_ist() -> datetime:
    return datetime.now(tz=IST)


def from_epoch(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=IST)


def to_epoch(dt: datetime) -> float:
    return dt.timestamp()


def parse_hms(s: str) -> time:
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.append(0)
    return time(parts[0], parts[1], parts[2])


def at(d: date, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=IST)


def phase(dt: datetime) -> str:
    """Return the NSE session phase for an IST datetime."""
    if dt.weekday() >= 5:
        return "CLOSED"
    t = dt.timetz().replace(tzinfo=None)
    if t < PRE_OPEN_START:
        return "CLOSED"
    if t < PRE_OPEN_END:
        return "PRE_OPEN"
    if t < MARKET_OPEN:
        return "PRE_OPEN_MATCH"
    if t < MARKET_CLOSE:
        return "OPEN"
    if t < CLOSING_SESSION_END:
        return "POST_CLOSE"
    return "CLOSED"


def minute_of_session(dt: datetime) -> int:
    """Minutes elapsed since 09:15 (0 = first minute). Negative before open."""
    open_dt = at(dt.date(), MARKET_OPEN)
    return int((dt - open_dt).total_seconds() // 60)


def session_minutes() -> int:
    return (MARKET_CLOSE.hour * 60 + MARKET_CLOSE.minute) - (MARKET_OPEN.hour * 60 + MARKET_OPEN.minute)


def previous_weekday(d: date) -> date:
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def trading_days_back(end: date, n: int) -> list[date]:
    """n weekdays ending at `end` (inclusive), oldest first. Holidays ignored (simulator use)."""
    out: list[date] = []
    d = end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return list(reversed(out))
