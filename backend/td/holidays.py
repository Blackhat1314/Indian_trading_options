"""NSE trading holidays (equity and F&O segments).

The live app uses them so a holiday is a closed day: no 09:30 lock on yesterday's frozen prices, no scorecard,
and the history backfill runs at full speed. NSE publishes next year's list in December — add those dates (or
any special closure) with NSE_HOLIDAYS=2027-01-26,2027-03-22 in .env until this list is updated.

The simulator ignores holidays: it makes up a session for every weekday.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache

# NSE circular for 2026 (weekday closures only; weekend holidays change nothing)
NSE_HOLIDAYS: dict[date, str] = {
    date(2026, 1, 15): "Municipal Corporation elections (Maharashtra)",
    date(2026, 1, 26): "Republic Day",
    date(2026, 3, 3): "Holi",
    date(2026, 3, 26): "Shri Ram Navami",
    date(2026, 3, 31): "Shri Mahavir Jayanti",
    date(2026, 4, 3): "Good Friday",
    date(2026, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
    date(2026, 5, 1): "Maharashtra Day",
    date(2026, 5, 28): "Bakri Id",
    date(2026, 6, 26): "Muharram",
    date(2026, 9, 14): "Ganesh Chaturthi",
    date(2026, 10, 2): "Mahatma Gandhi Jayanti",
    date(2026, 10, 20): "Dussehra",
    date(2026, 11, 10): "Diwali Balipratipada",
    date(2026, 11, 24): "Prakash Gurpurb Sri Guru Nanak Dev",
    date(2026, 12, 25): "Christmas",
}


@lru_cache(maxsize=8)
def _extra(spec: str) -> frozenset[date]:
    out = set()
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                out.add(date.fromisoformat(part))
            except ValueError:
                pass                                  # a typo must not stop the app
    return frozenset(out)


def holiday_name(d: date) -> str | None:
    """Why NSE is shut on `d` (a weekday holiday), else None. Always None in simulator mode."""
    from .config import settings
    if settings.data_source == "sim":
        return None
    if d in NSE_HOLIDAYS:
        return NSE_HOLIDAYS[d]
    if d in _extra(settings.nse_holidays):
        return "NSE holiday (NSE_HOLIDAYS)"
    return None


def is_holiday(d: date) -> bool:
    return holiday_name(d) is not None


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and not is_holiday(d)
