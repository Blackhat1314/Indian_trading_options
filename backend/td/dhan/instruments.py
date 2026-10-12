"""Universe construction.

Dhan mode: downloads Dhan's detailed scrip master (cached daily) and resolves
  - every F&O stock's NSE_EQ security id
  - its near-month stock future (for OI build-up)
  - the main indices (IDX_I)
Sim mode: builds a synthetic universe with the same shape (fake security ids).
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import httpx

from ..config import settings
from ..timeutil import now_ist
from ..universe_data import FNO_STOCKS, INDICES, SECTOR_OF

log = logging.getLogger("td.instruments")


SEG_CODE = {"IDX_I": 0, "NSE_EQ": 1, "NSE_FNO": 2, "NSE_CURRENCY": 3, "BSE_EQ": 4, "MCX_COMM": 5,
            "BSE_CURRENCY": 7, "BSE_FNO": 8}


def ikey(seg: str | int, sid: int) -> int:
    """Unique instrument key across segments. Dhan security ids are only unique *within* a segment
    (NIFTY index = IDX_I 13 and ABB = NSE_EQ 13), so everything inside TradeDesk is keyed by this.
    NSE equities keep their Dhan id; other segments are offset by (segment code + 1) x 10^7."""
    code = SEG_CODE.get(seg, 1) if isinstance(seg, str) else int(seg)
    return int(sid) if code == 1 else (code + 1) * 10_000_000 + int(sid)


@dataclass
class Instrument:
    sid: int
    seg: str            # IDX_I | NSE_EQ | NSE_FNO
    symbol: str         # RELIANCE, NIFTY, RELIANCE-FUT
    kind: str           # IDX | EQ | FUT
    underlying: str     # RELIANCE (for futures), same as symbol for EQ
    sector: str = "OTHER"
    lot: int = 1
    tick: float = 0.05
    expiry: str = ""
    isin: str = ""
    name: str = ""
    key: int = 0          # unique across segments (see ikey); `sid` stays the Dhan security id for API calls

    def __post_init__(self):
        self.key = ikey(self.seg, self.sid)


@dataclass
class Universe:
    instruments: list[Instrument] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[Instrument]:
        return [i for i in self.instruments if i.kind == kind]

    def equities(self) -> list[Instrument]:
        return self.by_kind("EQ")

    def futures(self) -> list[Instrument]:
        return self.by_kind("FUT")

    def indices(self) -> list[Instrument]:
        return self.by_kind("IDX")

    def to_json(self) -> list[dict]:
        return [asdict(i) for i in self.instruments]

    @classmethod
    def from_json(cls, rows: list[dict]) -> "Universe":
        return cls([Instrument(**r) for r in rows])


# ----------------------------------------------------------------------- sim
def sim_universe(n: int | None = None) -> Universe:
    n = n or settings.sim_symbols
    out: list[Instrument] = []
    for name, (sid, _, _) in INDICES.items():
        out.append(Instrument(sid=sid, seg="IDX_I", symbol=name, kind="IDX", underlying=name, sector="INDEX"))
    for i, (sym, sector, _, _) in enumerate(FNO_STOCKS[:n]):
        # Dhan-like ids: equities 1..n overlap the index ids (13, 21, 25, 27) exactly as on Dhan,
        # so the simulator exercises the same cross-segment id clash as live data
        eq_sid = 1 + i
        fut_sid = 35000 + i
        out.append(Instrument(sid=eq_sid, seg="NSE_EQ", symbol=sym, kind="EQ", underlying=sym, sector=sector,
                              name=sym))
        out.append(Instrument(sid=fut_sid, seg="NSE_FNO", symbol=f"{sym}-FUT", kind="FUT", underlying=sym,
                              sector=sector, lot=max(1, int(500000 / max(1.0, _ref_px(sym))))))
    return Universe(out)


def _ref_px(sym: str) -> float:
    for s, _, px, _ in FNO_STOCKS:
        if s == sym:
            return px
    return 1000.0


# ---------------------------------------------------------------------- dhan
def _cache_path() -> Path:
    p = Path(settings.data_dir) / "cache" / f"scrip-master-{now_ist().date().isoformat()}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def download_scrip_master() -> str:
    path = _cache_path()
    if path.exists() and path.stat().st_size > 1_000_000:
        return path.read_text(encoding="utf-8", errors="ignore")
    log.info("downloading scrip master %s", settings.dhan_scrip_master_url)
    with httpx.Client(timeout=120, follow_redirects=True) as c:
        r = c.get(settings.dhan_scrip_master_url)
        r.raise_for_status()
        text = r.text
    path.write_text(text, encoding="utf-8")
    return text


def _parse_expiry(s: str) -> date | None:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s[:19] if " " in s else s, fmt).date()
        except ValueError:
            continue
    return None


def dhan_universe(csv_text: str | None = None) -> Universe:
    """Resolve the F&O universe from Dhan's detailed scrip master."""
    text = csv_text or download_scrip_master()
    rows = list(csv.DictReader(io.StringIO(text)))
    today = now_ist().date()

    def g(r, *keys):
        for k in keys:
            if k in r and r[k] not in (None, ""):
                return r[k].strip()
        return ""

    # near-month stock futures (NSE, FUTSTK, earliest expiry >= today)
    futs: dict[str, tuple[date, dict]] = {}
    for r in rows:
        if g(r, "EXCH_ID", "SEM_EXM_EXCH_ID") != "NSE":
            continue
        if g(r, "INSTRUMENT", "SEM_INSTRUMENT_NAME") != "FUTSTK":
            continue
        und = g(r, "UNDERLYING_SYMBOL", "SM_SYMBOL_NAME")
        exp = _parse_expiry(g(r, "SM_EXPIRY_DATE", "SEM_EXPIRY_DATE"))
        if not und or not exp or exp < today:
            continue
        if und not in futs or exp < futs[und][0]:
            futs[und] = (exp, r)

    wanted = set(futs)
    if settings.universe == "custom" and settings.symbols:
        wanted = {s.strip().upper() for s in settings.symbols.split(",") if s.strip()}

    out: list[Instrument] = []
    # indices
    idx_lookup = {"NIFTY": "NIFTY", "BANKNIFTY": "BANKNIFTY", "FINNIFTY": "FINNIFTY",
                  "MIDCPNIFTY": "MIDCPNIFTY", "INDIAVIX": "INDIA VIX"}
    found_idx: dict[str, int] = {}
    for r in rows:
        if g(r, "INSTRUMENT", "SEM_INSTRUMENT_NAME") != "INDEX" or g(r, "EXCH_ID", "SEM_EXM_EXCH_ID") != "NSE":
            continue
        sym = g(r, "SYMBOL_NAME", "SM_SYMBOL_NAME", "SEM_TRADING_SYMBOL").upper()
        disp = g(r, "DISPLAY_NAME", "SEM_CUSTOM_SYMBOL").upper()
        for key, needle in idx_lookup.items():
            if key not in found_idx and needle in (sym, disp, sym.replace(" ", ""), disp.replace(" ", "")):
                found_idx[key] = int(g(r, "SECURITY_ID", "SEM_SMST_SECURITY_ID"))
    for name, (fallback_sid, _, _) in INDICES.items():
        out.append(Instrument(sid=found_idx.get(name, fallback_sid), seg="IDX_I", symbol=name, kind="IDX",
                              underlying=name, sector="INDEX"))

    # cash equities (series EQ) for the wanted underlyings
    for r in rows:
        if g(r, "EXCH_ID", "SEM_EXM_EXCH_ID") != "NSE" or g(r, "INSTRUMENT", "SEM_INSTRUMENT_NAME") != "EQUITY":
            continue
        if g(r, "SERIES", "SEM_SERIES") not in ("EQ", ""):
            continue
        sym = g(r, "UNDERLYING_SYMBOL", "SYMBOL_NAME", "SEM_TRADING_SYMBOL").upper()
        if sym not in wanted:
            continue
        out.append(Instrument(sid=int(g(r, "SECURITY_ID", "SEM_SMST_SECURITY_ID")), seg="NSE_EQ", symbol=sym,
                              kind="EQ", underlying=sym, sector=SECTOR_OF.get(sym, "OTHER"),
                              tick=float(g(r, "TICK_SIZE", "SEM_TICK_SIZE") or 0.05),
                              isin=g(r, "ISIN"), name=g(r, "DISPLAY_NAME", "SEM_CUSTOM_SYMBOL")))
    eq_syms = {i.symbol for i in out if i.kind == "EQ"}
    for und, (exp, r) in futs.items():
        if und not in eq_syms:
            continue
        out.append(Instrument(sid=int(g(r, "SECURITY_ID", "SEM_SMST_SECURITY_ID")), seg="NSE_FNO",
                              symbol=f"{und}-FUT", kind="FUT", underlying=und, sector=SECTOR_OF.get(und, "OTHER"),
                              lot=int(float(g(r, "LOT_SIZE", "SEM_LOT_UNITS") or 1)), expiry=exp.isoformat()))
    # GEO desk: near-month MCX commodity and NSE currency futures (crude, gas, gold, silver, USDINR)
    macro_wanted = {s.strip().upper() for s in settings.geo_macro_symbols.split(",") if s.strip()}
    macro: dict[str, tuple[date, dict, str]] = {}
    for r in rows:
        ex, inst = g(r, "EXCH_ID", "SEM_EXM_EXCH_ID"), g(r, "INSTRUMENT", "SEM_INSTRUMENT_NAME")
        if not ((ex == "MCX" and inst == "FUTCOM") or (ex == "NSE" and inst == "FUTCUR")):
            continue
        und = g(r, "UNDERLYING_SYMBOL", "SM_SYMBOL_NAME").upper()
        if und not in macro_wanted:
            continue
        exp = _parse_expiry(g(r, "SM_EXPIRY_DATE", "SEM_EXPIRY_DATE"))
        if not exp or exp < today:
            continue
        if und not in macro or exp < macro[und][0]:
            macro[und] = (exp, r, "MCX_COMM" if ex == "MCX" else "NSE_CURRENCY")
    for und, (exp, r, seg) in macro.items():
        out.append(Instrument(sid=int(g(r, "SECURITY_ID", "SEM_SMST_SECURITY_ID")), seg=seg, symbol=f"{und}-FUT",
                              kind="MACRO", underlying=und, sector="MACRO",
                              lot=int(float(g(r, "LOT_SIZE", "SEM_LOT_UNITS") or 1)), expiry=exp.isoformat()))
    log.info("universe: %d equities, %d futures, %d indices", len(eq_syms),
             sum(1 for i in out if i.kind == "FUT"), sum(1 for i in out if i.kind == "IDX"))
    return Universe(out)


def build_universe() -> Universe:
    if settings.data_source == "sim":
        return sim_universe()
    return dhan_universe()
