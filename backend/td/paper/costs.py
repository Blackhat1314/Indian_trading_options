"""What a trade really costs in India: broker + statutory charges and slippage.

Charges follow the usual NSE equity schedule (approximate, configurable in .env as they change):
  brokerage     ₹20 or 0.03% per executed order, whichever is lower (discount-broker style; Dhan intraday)
  STT           intraday 0.025% on the sell side; delivery 0.1% on buy and sell
  exchange      NSE transaction charge 0.00297% of turnover
  SEBI fee      ₹10 per crore (0.0001%)
  stamp duty    intraday 0.003%, delivery 0.015%, buy side only
  GST           18% on brokerage + exchange + SEBI fees
  DP charge     delivery sells: a flat fee per scrip per day (₹15.34 incl. GST)

Slippage is modelled, not assumed away: half the bid-ask spread plus a square-root market-impact term
(impact ∝ σ · sqrt(order size / volume), the standard empirical law), and stop orders fill worse than market
orders because they trigger into a moving price.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ChargeSchedule:
    brokerage_flat: float = 20.0
    brokerage_pct: float = 0.0003
    stt_intraday_sell: float = 0.00025
    stt_delivery: float = 0.001
    exchange: float = 0.0000297
    sebi: float = 0.000001
    stamp_intraday: float = 0.00003
    stamp_delivery: float = 0.00015
    gst: float = 0.18
    dp_per_sell: float = 15.34
    delivery_brokerage: bool = True        # some brokers charge ₹0 on delivery; set False for those


DEFAULT = ChargeSchedule()


def _brokerage(value: float, s: ChargeSchedule) -> float:
    return min(s.brokerage_flat, s.brokerage_pct * value) if value > 0 else 0.0


def charges(buy_value: float, sell_value: float, product: str = "intraday",
            s: ChargeSchedule = DEFAULT) -> dict:
    """Charges for one round trip (or one leg: pass 0 for the other side). product: intraday | delivery."""
    delivery = product == "delivery"
    brk = 0.0
    if not delivery or s.delivery_brokerage:
        brk = _brokerage(buy_value, s) + _brokerage(sell_value, s)
    stt = (s.stt_delivery * (buy_value + sell_value)) if delivery else s.stt_intraday_sell * sell_value
    exch = s.exchange * (buy_value + sell_value)
    sebi = s.sebi * (buy_value + sell_value)
    stamp = (s.stamp_delivery if delivery else s.stamp_intraday) * buy_value
    gst = s.gst * (brk + exch + sebi)
    dp = s.dp_per_sell if delivery and sell_value > 0 else 0.0
    out = {"brokerage": brk, "stt": stt, "exchange": exch, "sebi": sebi, "stamp": stamp, "gst": gst, "dp": dp}
    out = {k: round(v, 2) for k, v in out.items()}
    out["total"] = round(sum(out.values()), 2)
    return out


@dataclass(frozen=True)
class SlippageModel:
    half_spread_bps: float = 2.0          # liquid F&O stocks in the first hour: 1-3 bps half spread
    impact_coef: float = 0.6              # Almgren-style: impact = coef * sigma_minute * sqrt(qty / minute volume)
    stop_extra_bps: float = 4.0           # stops trigger into momentum: extra adverse fill
    floor_bps: float = 2.0
    cap_bps: float = 60.0


SLIPPAGE = SlippageModel()


def slippage_bps(qty: float, price: float, minute_volume: float, daily_vol: float, stop: bool = False,
                 m: SlippageModel = SLIPPAGE) -> float:
    """Adverse fill in basis points for an order of `qty` shares into a minute that traded `minute_volume`.

    daily_vol: the stock's daily return volatility (e.g. ATR / price); the per-minute sigma is daily / sqrt(375).
    """
    sigma_min = max(daily_vol, 0.005) / math.sqrt(375)
    part = qty / max(minute_volume, 1.0)
    impact = m.impact_coef * sigma_min * math.sqrt(min(part, 4.0)) * 1e4
    bps = m.half_spread_bps + impact + (m.stop_extra_bps if stop else 0.0)
    return float(min(m.cap_bps, max(m.floor_bps, bps)))


def fill_price(side: int, price: float, bps: float) -> float:
    """side +1 buys (pays up), -1 sells (gets less)."""
    return price * (1 + side * bps / 1e4)


def round_trip_cost_pct(notional: float, product: str = "intraday", slip_bps_each: float = 4.0) -> float:
    """Approximate total cost of a round trip as a fraction of notional (charges + two slippages)."""
    ch = charges(notional, notional, product)["total"]
    return ch / notional + 2 * slip_bps_each / 1e4
