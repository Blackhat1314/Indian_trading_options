"""Screener.in fundamentals: a polite page client, a pure HTML parser, derived metrics and a scorecard.

The SCREENER desk shows any listed company's statements and the AI analyst reasons over the same numbers, so
both go through `ScreenerClient.fundamentals` — one fetch path, one parser, one on-disk cache.

Screener.in has no public API, so the client behaves like one careful reader: it fetches only the company page
and its peers fragment (never search / sort / paging URLs, which robots.txt disallows), one request at a time
with a minimum spacing, and caches the parsed result on disk for hours. When Screener is unreachable the last
copy is served and marked stale. Nothing is ever filled in by guessing: a number that is not on the page is null,
and every derived metric says how it was computed or why it is missing.

Units: statement values stay as Screener prints them — ₹ crore for money, percentages in percent (12.5 = 12.5 %)
— and are labelled by FinTable.unit / FinRow.is_pct and the `unit` field of ratios and derived metrics.
Banks and NBFCs have different statements (Revenue / Financing Profit, no OPM, deposits); `is_financial` detects
them so metrics that make no sense for a lender are reported as not applicable instead of failing a check.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from pathlib import Path
from statistics import median
from typing import Any, Literal
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from ..config import settings

log = logging.getLogger("td.research.screener")

BASE_URL = "https://www.screener.in"
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/140.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
           "Accept-Language": "en-IN,en;q=0.9"}
SYMBOL_RE = re.compile(r"^(?=.*[A-Z0-9])[A-Z0-9&._-]{1,40}$")
View = Literal["auto", "consolidated", "standalone"]
VIEWS = ("auto", "consolidated", "standalone")
CACHE_VERSION = 1          # bump when the Fundamentals shape changes so old cache files are ignored


# ------------------------------------------------------------------ text and number helpers
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")
_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun",
                                        "jul", "aug", "sep", "oct", "nov", "dec"))}
_COL_RE = re.compile(r"^([A-Za-z]{3})[a-z]*\.?\s+(\d{4})(?:\s+(\d{1,2})\s*m)?$")


def parse_number(text: str | None) -> float | None:
    """Screener cell text → float. '1,23,456' → 123456, '-5' → -5, '12%' → 12, '₹ 1,234 Cr.' → 1234.

    Blank, '-', '--' or anything holding no number (or two numbers, like '1,612 / 1,152') → None.
    """
    if not text:
        return None
    s = (text.replace("\xa0", " ").replace("−", "-").replace("–", "-")
         .replace(",", "").replace("₹", " ").replace("Rs.", " ").strip())
    s = re.sub(r"-\s+(?=\d)", "-", s)
    nums = _NUM_RE.findall(s)
    if len(nums) != 1:
        return None
    try:
        v = float(nums[0])
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _text(node: Any) -> str:
    if node is None:
        return ""
    return " ".join(node.get_text(" ", strip=True).replace("\xa0", " ").split())


def _label(node: Any) -> str:
    """Row label without Screener's '+' schedule expander ('Sales +' → 'Sales')."""
    return re.sub(r"\s*\+$", "", _text(node)).strip()


def _slug(s: str) -> str:
    s = s.lower().replace("p/e", "pe").replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def col_period(label: str) -> tuple[int, int] | None:
    """Column label → (months since year 0, months in the period). 'Mar 2024' → (24291, 12),
    'Mar 2024 15m' → (24291, 15) (a changed financial year), 'TTM' → None."""
    m = _COL_RE.match(label.strip())
    if not m or m.group(1).lower() not in _MONTHS:
        return None
    return int(m.group(2)) * 12 + _MONTHS[m.group(1).lower()], int(m.group(3) or 12)


# ------------------------------------------------------------------ page parser
def _fin_table(table: Tag | None, title: str, unit: str) -> dict | None:
    if table is None:
        return None
    head = table.find("thead")
    header = head.find("tr") if head else None
    if header is None:
        header = next((tr for tr in table.find_all("tr") if tr.find("th")), None)
    if header is None:
        return None
    columns = [_text(th) for th in header.find_all("th")[1:]]
    if not columns:
        return None
    rows = []
    for tr in (table.find("tbody") or table).find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        label = _label(cells[0])
        if not label or label.lower() == "raw pdf":
            continue
        texts = [_text(td) for td in cells[1:1 + len(columns)]]
        values = [parse_number(t) for t in texts] + [None] * (len(columns) - len(texts))
        is_pct = label.endswith("%") or any(t.endswith("%") for t in texts if t)
        rows.append({"label": label, "values": values, "is_pct": is_pct})
    if not rows:
        return None
    return {"title": title, "columns": columns, "rows": rows, "unit": unit}


def _section(soup: BeautifulSoup, sid: str) -> tuple[Tag | None, str, str, str | None]:
    """(section, title, unit, view shown) for a statement section such as #profit-loss."""
    sec = soup.find("section", id=sid)
    if sec is None:
        return None, "", "", None
    title = _text(sec.find("h2"))
    sub = _text(sec.find("p", class_="sub"))
    m = re.search(r"Figures in (.+?)(?:\s*/|$)", sub)
    unit = m.group(1).strip() if m else ("%" if "percentage" in sub.lower() else "")
    shown = sub.split(" ", 1)[0].lower() if sub else ""
    return sec, title, unit, shown if shown in ("consolidated", "standalone") else None


def _statement(soup: BeautifulSoup, sid: str) -> tuple[dict | None, str | None]:
    sec, title, unit, shown = _section(soup, sid)
    if sec is None:
        return None, None
    table = sec.select_one("[data-result-table] table.data-table") or sec.select_one("table.data-table")
    return _fin_table(table, title, unit), shown


def _top_ratios(soup: BeautifulSoup) -> list[dict]:
    out = []
    for li in soup.select("#top-ratios li"):
        name = _text(li.select_one(".name"))
        vnode = li.select_one(".value")
        if not name or vnode is None:
            continue
        text = _text(vnode)
        num_nodes = vnode.select(".number")
        nums = [_text(n) for n in num_nodes]
        unit = " ".join(" ".join(s for s in vnode.find_all(string=True)
                                 if not s.find_parent(class_="number")).split())
        if len(nums) == 2 and "/" in name:            # "High / Low  ₹ 1,612 / 1,152" — the 52-week range
            cur = unit.replace("/", "").strip()
            for key, label, raw in (("high", "52W High", nums[0]), ("low", "52W Low", nums[1])):
                out.append({"key": key, "label": label, "value": parse_number(raw),
                            "text": f"{cur} {raw}".strip() if raw else "", "unit": cur})
            continue
        value = parse_number(nums[0]) if len(nums) == 1 else (parse_number(text) if not num_nodes else None)
        out.append({"key": _slug(name), "label": name, "value": value,
                    "text": text if value is not None else "", "unit": unit})
    return out


def _growth_boxes(soup: BeautifulSoup) -> list[dict]:
    out = []
    for t in soup.select("table.ranges-table"):
        title = _text(t.find("th"))
        rows = []
        for tr in t.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 2:
                rows.append({"period": _text(tds[0]).rstrip(":").strip(), "value": parse_number(_text(tds[1]))})
        if title and rows:
            out.append({"title": title, "rows": rows})
    return out


def _links(soup: BeautifulSoup) -> dict:
    top = soup.find(id="top") or soup
    website = bse = nse = None
    for a in top.find_all("a", href=True):
        href = a["href"]
        if website is None and a.find("i", class_="icon-link"):
            website = href
        if bse is None and "bseindia.com/stock-share-price" in href:
            m = re.search(r"/(\d{6})/?$", href)
            bse = m.group(1) if m else None
        if nse is None and "nseindia.com/get-quotes/equity" in href:
            q = parse_qs(urlparse(href).query).get("symbol")
            nse = q[0].strip().upper() if q else None
    return {"website": website, "bse_code": bse, "nse_code": nse}


def _shareholding(soup: BeautifulSoup) -> dict:
    sec, title, unit, _ = _section(soup, "shareholding")
    out: dict[str, dict | None] = {"quarterly": None, "yearly": None}
    if sec is None:
        return out
    title = title or "Shareholding Pattern"
    unit = unit or "%"
    for key, div_id in (("quarterly", "quarterly-shp"), ("yearly", "yearly-shp")):
        div = sec.find("div", id=div_id)
        if div is not None:
            out[key] = _fin_table(div.select_one("table.data-table"), f"{title} ({key})", unit)
    if out["quarterly"] is None and out["yearly"] is None:      # older layout without the tabs
        out["quarterly"] = _fin_table(sec.select_one("table.data-table"), f"{title} (quarterly)", unit)
    return out


def _parse_soup(soup: BeautifulSoup, symbol: str, url: str, view: str) -> tuple[dict, dict]:
    """(Fundamentals, page meta: warehouse_id for the peers call, other_view_hint when Screener highlights the
    'View Standalone/Consolidated' link — it does so when the other view has the longer history)."""
    info = soup.find(id="company-info")
    meta = {"warehouse_id": info.get("data-warehouse-id") if info is not None else None,
            "other_view_hint": any(_text(a).startswith("View ") for a in soup.select("section p.sub a.strong"))}
    top = soup.find(id="top")
    name = _text(top.find("h1")) if top is not None else ""
    if not name:
        name = _text(soup.find("title")).split(" share price")[0].strip()
    peers_sec = soup.find("section", id="peers")

    def by_title(*titles: str) -> str | None:
        for t in titles:
            a = peers_sec.find("a", title=t) if peers_sec is not None else None
            if a is not None and _text(a):
                return _text(a)
        return None

    tables, shown_views = {}, []
    for key, sid in (("quarters", "quarters"), ("profit_loss", "profit-loss"), ("balance_sheet", "balance-sheet"),
                     ("cash_flow", "cash-flow"), ("ratios_table", "ratios")):
        tables[key], shown = _statement(soup, sid)
        if shown:
            shown_views.append(shown)
    shown_view = max(set(shown_views), key=shown_views.count) if shown_views else None

    f: dict[str, Any] = {
        "symbol": symbol, "name": name or symbol, "url": url,
        "view": shown_view or (view if view in ("consolidated", "standalone") else "consolidated"),
        "fetched_at": 0.0, "cached": False, "stale": False,
        "about": _text(soup.select_one(".company-profile .about")),
        "sector": by_title("Sector", "Broad Sector"), "industry": by_title("Industry", "Broad Industry"),
        **_links(soup),
        "ratios": _top_ratios(soup),
        "pros": [_text(li) for li in soup.select("#analysis .pros li") if _text(li)],
        "cons": [_text(li) for li in soup.select("#analysis .cons li") if _text(li)],
        "growth": _growth_boxes(soup),
        **tables,
        "shareholding": _shareholding(soup),
        "peers": None, "derived": [], "scorecard": {}, "warnings": [],
    }
    if not has_statements(f):
        f["warnings"].append(f"Screener.in shows no {f['view']} financial statements for {symbol}.")
    else:
        for key, what in (("profit_loss", "annual profit & loss"), ("balance_sheet", "balance sheet"),
                          ("cash_flow", "cash flow statement"), ("quarters", "quarterly results")):
            if f[key] is None:
                f["warnings"].append(f"No {what} on the Screener.in page.")
    finalize(f)
    return f, meta


def _parse_html(html: str, symbol: str, url: str, view: str) -> tuple[dict, dict] | None:
    soup = BeautifulSoup(html, "lxml")
    if soup.find(id="top") is None and soup.find("section", id="profit-loss") is None:
        return None
    return _parse_soup(soup, symbol, url, view)


def parse_company_page(html: str, symbol: str, url: str, view: str) -> dict:
    """Parse a Screener.in company page into a Fundamentals dict (peers null; the client adds them)."""
    return _parse_soup(BeautifulSoup(html, "lxml"), symbol, url, view)[0]


def parse_peers(html: str) -> dict | None:
    """Peers fragment (/api/company/{warehouse_id}/peers/) → {columns, rows[{name, symbol, values}]}.
    The footer median row is kept as the last row (symbol null, name 'Median: N Co.')."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        return None
    header = next((tr for tr in table.find_all("tr") if tr.find("th") and not tr.find("td")), None)
    if header is None:
        return None
    names = [_text(th) for th in header.find_all("th")]
    name_idx = next((i for i, n in enumerate(names) if n.lower() == "company"), 1)
    columns = names[name_idx + 1:]
    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) <= name_idx:
            continue
        name = _text(tds[name_idx])
        if not name:
            continue
        symbol = None
        a = tds[name_idx].find("a", href=True)
        if a is not None:
            m = re.match(r"^/company/([^/]+)/", a["href"])
            code = unquote(m.group(1)).upper() if m else ""
            symbol = code if code and not code.isdigit() else None     # BSE-only listings carry a numeric code
        values = [parse_number(_text(td)) for td in tds[name_idx + 1:name_idx + 1 + len(columns)]]
        rows.append({"name": name, "symbol": symbol, "values": values + [None] * (len(columns) - len(values))})
    return {"columns": columns, "rows": rows} if columns and rows else None


# ------------------------------------------------------------------ table access
_FIN_LABELS = {"financing profit", "financing margin %", "deposits", "gross npa %", "net npa %"}
_FIN_WORDS = re.compile(r"bank|nbfc|non banking|financ|insurance|lending", re.I)


def _labels(t: dict | None) -> set[str]:
    return {r["label"].lower() for r in t["rows"]} if t else set()


def is_financial(f: dict) -> bool:
    """Banks / NBFCs / insurers: Screener shows Revenue, Financing Profit / Margin and NPAs instead of OPM."""
    labels = _labels(f.get("profit_loss")) | _labels(f.get("quarters")) | _labels(f.get("balance_sheet"))
    if labels & _FIN_LABELS:
        return True
    if labels and "opm %" not in labels:
        return "revenue" in labels or bool(_FIN_WORDS.search(f"{f.get('industry') or ''} {f.get('sector') or ''}"))
    return False


def has_statements(f: dict) -> bool:
    for key in ("profit_loss", "quarters"):
        t = f.get(key)
        if t and any(v is not None for r in t["rows"] for v in r["values"]):
            return True
    return False


def _row(t: dict | None, *names: str) -> dict | None:
    if not t:
        return None
    want = [n.lower() for n in names]
    for n in want:
        for r in t["rows"]:
            if r["label"].lower() == n:
                return r
    return None


class _Pt:
    __slots__ = ("col", "ym", "months", "v")

    def __init__(self, col: str, ym: int, months: int, v: float):
        self.col, self.ym, self.months, self.v = col, ym, months, v


def _points(t: dict | None, *names: str) -> list[_Pt]:
    """Dated, non-null values of a row (TTM and undated columns dropped), oldest first."""
    r = _row(t, *names)
    if r is None:
        return []
    out = []
    for col, v in zip(t["columns"], r["values"], strict=False):
        per = col_period(col)
        if per is not None and v is not None:
            out.append(_Pt(col, per[0], per[1], v))
    return sorted(out, key=lambda p: p.ym)


def _ttm(t: dict | None, *names: str) -> float | None:
    r = _row(t, *names)
    if r is None:
        return None
    for col, v in zip(t["columns"], r["values"], strict=False):
        if col.strip().upper() == "TTM":
            return v
    return None


def _at(points: list[_Pt], ym: int, tol: int = 0) -> _Pt | None:
    best = min(points, key=lambda p: abs(p.ym - ym), default=None)
    return best if best is not None and abs(best.ym - ym) <= tol else None


def _annual_span(f: dict) -> float:
    """Years covered by full-year P&L columns that hold sales or profit."""
    pts = [p for p in (_points(f.get("profit_loss"), "Sales", "Revenue")
                       or _points(f.get("profit_loss"), "Net Profit")) if p.months == 12]
    return (pts[-1].ym - pts[0].ym) / 12 if len(pts) >= 2 else 0.0


def _cagr(points: list[_Pt], years: int) -> tuple[float | None, str]:
    full = [p for p in points if p.months == 12]          # a 15-month year would overstate growth
    if not full:
        return None, "no annual figures"
    end = full[-1]
    start = _at(full, end.ym - 12 * years, tol=3)
    if start is None or start.ym >= end.ym:
        have = (end.ym - full[0].ym) / 12
        return None, f"needs {years} years of annual figures (Screener shows {have:.0f})"
    if start.v <= 0 or end.v <= 0:
        return None, f"{start.col} {start.v:,.0f} → {end.col} {end.v:,.0f}: not positive, CAGR undefined"
    yrs = (end.ym - start.ym) / 12
    return ((end.v / start.v) ** (1 / yrs) - 1) * 100, f"{start.col} → {end.col}"


def _ratio(f: dict, key: str) -> float | None:
    for r in f.get("ratios") or []:
        if r.get("key") == key:
            return r.get("value")
    return None


def _peer_median_pe(f: dict) -> tuple[float | None, str]:
    peers = f.get("peers")
    if not peers or not peers.get("rows"):
        return None, "peer table unavailable"
    idx = next((i for i, c in enumerate(peers["columns"]) if c.lower().startswith("p/e")), None)
    if idx is None:
        return None, "peer table has no P/E column"
    for r in peers["rows"]:
        if r["symbol"] is None and r["name"].lower().startswith("median"):
            v = r["values"][idx] if idx < len(r["values"]) else None
            if v is not None:
                return v, f"Screener peer median ({r['name'].split(':', 1)[-1].strip()})"
    me = (f.get("symbol") or "").upper()
    vals = [r["values"][idx] for r in peers["rows"]
            if r["symbol"] and r["symbol"] != me and idx < len(r["values"]) and (r["values"][idx] or 0) > 0]
    if len(vals) < 2:
        return None, "too few peers with a P/E"
    return median(vals), f"median of {len(vals)} peers"


# ------------------------------------------------------------------ derived metrics
def _metrics(f: dict) -> tuple[dict[str, dict], dict[str, Any]]:
    """All derived metrics keyed by name, plus raw facts the scorecard needs (not part of the output)."""
    fin = is_financial(f)
    pl, q, bs, cf, rt = (f.get(k) for k in ("profit_loss", "quarters", "balance_sheet", "cash_flow", "ratios_table"))
    shq = (f.get("shareholding") or {}).get("quarterly")
    out: dict[str, dict] = {}
    facts: dict[str, Any] = {"financial": fin}
    lender = "not meaningful for banks / NBFCs"

    def add(key: str, label: str, value: float | None, unit: str, note: str = "") -> None:
        ok = value is not None and math.isfinite(value)
        out[key] = {"key": key, "label": label, "value": round(float(value), 2) if ok else None,
                    "unit": unit, "note": note}

    sales_row = _row(pl, "Sales", "Revenue")
    sales_name = sales_row["label"] if sales_row else "Sales"
    sales = _points(pl, "Sales", "Revenue")
    profit = _points(pl, "Net Profit")
    eps = _points(pl, "EPS in Rs", "EPS")
    no_pl = "no profit & loss table on Screener.in"

    for key, label, pts in (("sales_cagr_3y", f"{sales_name} CAGR 3y", sales),
                            ("sales_cagr_5y", f"{sales_name} CAGR 5y", sales),
                            ("profit_cagr_3y", "Net profit CAGR 3y", profit),
                            ("profit_cagr_5y", "Net profit CAGR 5y", profit),
                            ("eps_cagr_3y", "EPS CAGR 3y", eps)):
        v, note = _cagr(pts, int(key[-2])) if pts else (None, no_pl if pl is None else "row not on the page")
        add(key, label, v, "%", note)

    for key, label, names in (("qtr_sales_yoy", f"Latest quarter {sales_name.lower()} YoY", ("Sales", "Revenue")),
                              ("qtr_profit_yoy", "Latest quarter net profit YoY", ("Net Profit",))):
        pts = _points(q, *names)
        cur = pts[-1] if pts else None
        prev = _at(pts, cur.ym - 12, tol=1) if cur else None
        facts[key] = (cur.v, prev.v) if cur and prev else None
        if cur is None or prev is None:
            add(key, label, None, "%", "no quarterly results" if q is None else "needs the same quarter last year")
        elif prev.v <= 0:
            add(key, label, None, "%", f"{cur.col} {cur.v:,.0f} vs {prev.col} {prev.v:,.0f} Cr (base not positive)")
        else:
            add(key, label, (cur.v / prev.v - 1) * 100, "%", f"{cur.col} vs {prev.col}")

    # margins
    if fin:
        fm = _ttm(pl, "Financing Margin %")
        fm_pts = _points(pl, "Financing Margin %")
        fm, note = (fm, "TTM") if fm is not None else ((fm_pts[-1].v, fm_pts[-1].col) if fm_pts else (None, ""))
        add("opm_latest", "Financing margin", fm, "%",
            f"lender: Screener's Financing Margin ({note}) replaces OPM" if fm is not None
            else "lender: no OPM; Financing Margin not on the page")
        add("opm_trend_3y", "OPM change 3y", None, "pp", lender)
    else:
        opm_pts = _points(pl, "OPM %")
        opm_ttm = _ttm(pl, "OPM %")
        if opm_ttm is not None:
            add("opm_latest", "Operating margin", opm_ttm, "%", "TTM")
        elif opm_pts:
            add("opm_latest", "Operating margin", opm_pts[-1].v, "%", opm_pts[-1].col)
        else:
            add("opm_latest", "Operating margin", None, "%", no_pl if pl is None else "no OPM row on the page")
        full = [p for p in opm_pts if p.months == 12]
        start = _at(full, full[-1].ym - 36, tol=3) if full else None
        if full and start is not None and start.ym < full[-1].ym:
            add("opm_trend_3y", "OPM change 3y", full[-1].v - start.v, "pp", f"{start.col} → {full[-1].col}")
        else:
            add("opm_trend_3y", "OPM change 3y", None, "pp", "needs 3 years of OPM")

    # returns
    roe, note = _ratio(f, "roe"), "Screener top ratio"
    if roe is None:
        roe_pts = _points(rt, "ROE %")
        roe, note = (roe_pts[-1].v, f"ratios table, {roe_pts[-1].col}") if roe_pts else (None, "")
    equity_pts = _equity(bs)
    fy_profit = [p for p in profit if p.months == 12]
    if roe is None and fy_profit:
        roe, note = _over_average(fy_profit[-1], equity_pts, "equity")
    add("roe", "Return on equity", roe, "%", note if roe is not None else "not on the page")

    roce, note = _ratio(f, "roce"), "Screener top ratio"
    if roce is None:
        roce_pts = _points(rt, "ROCE %")
        roce, note = (roce_pts[-1].v, f"ratios table, {roce_pts[-1].col}") if roce_pts else (None, "not on the page")
    add("roce", "Return on capital employed", roce, "%", f"{note}; {lender}" if fin and roce is not None else note)

    roa, note = _over_average(fy_profit[-1], _points(bs, "Total Assets"), "total assets") if fy_profit else (None, "")
    add("roa", "Return on assets", roa, "%", note if roa is not None else "needs a balance sheet and net profit")

    # balance sheet health
    borrow = _points(bs, "Borrowings", "Borrowing")
    if fin:
        add("debt_to_equity", "Debt / equity", None, "x", lender + " (deposits and borrowings fund the loan book)")
    elif borrow and equity_pts:
        eq = _at(equity_pts, borrow[-1].ym)
        if eq is not None and eq.v > 0:
            add("debt_to_equity", "Debt / equity", borrow[-1].v / eq.v, "x",
                f"borrowings ÷ (equity capital + reserves), {borrow[-1].col}")
        else:
            add("debt_to_equity", "Debt / equity", None, "x", "equity not positive")
    else:
        add("debt_to_equity", "Debt / equity", None, "x", "balance sheet not on the page")

    pbt_ttm, int_ttm = _ttm(pl, "Profit before tax"), _ttm(pl, "Interest")
    pbt_pts, int_pts = _points(pl, "Profit before tax"), _points(pl, "Interest")
    if pbt_ttm is not None and int_ttm is not None:
        pbt, interest, when = pbt_ttm, int_ttm, "TTM"
    elif pbt_pts and int_pts:
        pbt, interest, when = pbt_pts[-1].v, int_pts[-1].v, pbt_pts[-1].col
    else:
        pbt = interest = None
        when = ""
    facts["interest"] = interest
    if fin:
        add("interest_coverage", "Interest coverage", None, "x", lender + " (interest is their cost of goods)")
    elif interest is None:
        add("interest_coverage", "Interest coverage", None, "x", "profit before tax / interest not on the page")
    elif interest <= 0:
        add("interest_coverage", "Interest coverage", None, "x",
            f"no interest expense ({when}) — effectively debt free")
    else:
        add("interest_coverage", "Interest coverage", (pbt + interest) / interest, "x",
            f"(profit before tax + interest) ÷ interest, {when}")

    # cash flow
    cfo = _points(cf, "Cash from Operating Activity")
    fcf_pts = _points(cf, "Free Cash Flow")
    if fin:
        add("fcf_latest", "Free cash flow", None, "₹ Cr", lender + " (operating cash flow includes lending)")
        add("cash_conversion_3y", "Cash conversion 3y", None, "%", lender)
    else:
        if fcf_pts:
            add("fcf_latest", "Free cash flow", fcf_pts[-1].v, "₹ Cr", f"Screener's free cash flow, {fcf_pts[-1].col}")
        else:
            cfi = _points(cf, "Cash from Investing Activity")
            last = cfo[-1] if cfo else None
            inv = _at(cfi, last.ym) if last else None
            if last and inv:
                add("fcf_latest", "Free cash flow", last.v + inv.v, "₹ Cr",
                    f"proxy: operating + investing cash flow, {last.col} (capex not broken out)")
            else:
                add("fcf_latest", "Free cash flow", None, "₹ Cr", "cash flow statement not on the page")
        pairs = [(c, _at(profit, c.ym)) for c in cfo[-3:]]
        pairs = [(c, p) for c, p in pairs if p is not None]
        np_sum = sum(p.v for _, p in pairs)
        if len(pairs) == 3 and np_sum > 0:
            add("cash_conversion_3y", "Cash conversion 3y", sum(c.v for c, _ in pairs) / np_sum * 100, "%",
                f"operating cash flow ÷ net profit, {pairs[0][0].col} – {pairs[-1][0].col}")
        else:
            add("cash_conversion_3y", "Cash conversion 3y", None, "%",
                "needs 3 years of operating cash flow and positive total profit")
    facts["profit_last3"] = [p.v for p in profit if p.months == 12][-3:]

    payout = _points(pl, "Dividend Payout %")
    add("dividend_payout", "Dividend payout", payout[-1].v if payout else None, "%",
        payout[-1].col if payout else "not on the page")

    # ownership
    prom = _points(shq, "Promoters")
    if not prom:
        add("promoter_holding", "Promoter holding", None, "%",
            "no promoter group listed (widely held)" if shq else "shareholding not on the page")
        add("promoter_change_4q", "Promoter holding change 4Q", None, "pp", out["promoter_holding"]["note"])
    else:
        add("promoter_holding", "Promoter holding", prom[-1].v, "%",
            prom[-1].col + (" — widely held, no promoter stake" if prom[-1].v == 0 else ""))
        before = _at(prom, prom[-1].ym - 12, tol=1)
        add("promoter_change_4q", "Promoter holding change 4Q",
            prom[-1].v - before.v if before else None, "pp",
            f"{before.col} → {prom[-1].col}" if before else "needs 4 quarters of history")
    facts["promoter_now"] = prom[-1].v if prom else None
    fii, dii = _points(shq, "FIIs"), _points(shq, "DIIs")
    inst = {}
    for pts in (fii, dii):
        for p in pts:
            inst.setdefault(p.ym, []).append(p)
    inst_pts = [_Pt(v[0].col, ym, 3, sum(p.v for p in v)) for ym, v in sorted(inst.items()) if len(v) == 2]
    before = _at(inst_pts, inst_pts[-1].ym - 12, tol=1) if inst_pts else None
    add("fii_dii_change_4q", "FII + DII holding change 4Q", inst_pts[-1].v - before.v if before else None, "pp",
        f"{before.col} → {inst_pts[-1].col}" if before else "needs 4 quarters of FII and DII holding")

    # valuation
    pe, price, book = _ratio(f, "stock_pe"), _ratio(f, "current_price"), _ratio(f, "book_value")
    facts["pe"] = pe
    eps_ttm = _ttm(pl, "EPS in Rs", "EPS")
    if pe is not None and pe > 0:
        add("earnings_yield", "Earnings yield", 100 / pe, "%", f"1 ÷ P/E {pe:g}")
    elif eps_ttm is not None and price:
        add("earnings_yield", "Earnings yield", eps_ttm / price * 100, "%", "TTM EPS ÷ price")
    else:
        add("earnings_yield", "Earnings yield", None, "%", "no P/E or TTM EPS")
    p3 = out["profit_cagr_3y"]["value"]
    if pe is not None and pe > 0 and p3 is not None and p3 > 0:
        add("peg", "PEG", pe / p3, "x", f"P/E {pe:g} ÷ 3y profit CAGR {p3:.1f} %")
    else:
        add("peg", "PEG", None, "x", "needs a positive P/E and positive 3y profit CAGR")
    if price and book and book > 0:
        add("price_to_book", "Price / book", price / book, "x", f"price ₹{price:,.2f} ÷ book value ₹{book:,.2f}")
    else:
        add("price_to_book", "Price / book", None, "x", "price or book value not on the page")
    med, note = _peer_median_pe(f)
    facts["peer_median_pe"] = med
    if pe is not None and pe > 0 and med is not None and med > 0:
        add("pe_vs_peers", "P/E vs peer median", pe / med, "x", f"P/E {pe:g} vs {note} {med:g}")
    else:
        add("pe_vs_peers", "P/E vs peer median", None, "x", note if med is None else "no positive P/E")

    if fin:
        for key, label, row in (("gross_npa", "Gross NPA", "Gross NPA %"), ("net_npa", "Net NPA", "Net NPA %")):
            pts = _points(q, row)
            add(key, label, pts[-1].v if pts else None, "%",
                pts[-1].col if pts else "not reported in this view (try standalone)")
    return out, facts


def _over_average(flow: _Pt, stock: list[_Pt], what: str) -> tuple[float | None, str]:
    """flow ÷ average of the balance-sheet item at the start and end of the same fiscal year, in %.
    Matched by date, so a half-year balance sheet (e.g. 'Sep 2026') never pairs with full-year profit."""
    end, start = _at(stock, flow.ym, tol=1), _at(stock, flow.ym - 12, tol=1)
    if end is None or end.v <= 0:
        return None, ""
    if start is None or start.v <= 0:
        return flow.v / end.v * 100, f"net profit {flow.col} ÷ year-end {what}"
    return flow.v / ((start.v + end.v) / 2) * 100, f"net profit {flow.col} ÷ average {what}"


def _equity(bs: dict | None) -> list[_Pt]:
    cap = {p.ym: p for p in _points(bs, "Equity Capital")}
    res = {p.ym: p for p in _points(bs, "Reserves")}
    return [_Pt(cap[ym].col, ym, 12, cap[ym].v + res[ym].v) for ym in sorted(cap) if ym in res]


def derived_metrics(f: dict) -> list[dict]:
    """Growth, margins, returns, leverage, cash and ownership metrics computed from the parsed page.
    Each {key, label, value, unit, note}; value null with a note when an input is missing or meaningless."""
    return list(_metrics(f)[0].values())


# ------------------------------------------------------------------ scorecard
def _fmt(v: float | None, unit: str) -> str:
    if v is None:
        return "—"
    if unit == "%":
        return f"{v:.1f} %"
    if unit == "pp":
        return f"{v:+.2f} pp"
    if unit == "x":
        return f"{v:.2f}×"
    if unit == "₹ Cr":
        return f"₹ {v:,.0f} Cr"
    return f"{v:g}"


def scorecard(f: dict) -> dict:
    """Five transparent parts (quality, growth, valuation, health, ownership) of simple pass/fail checks.

    A check whose input is missing or does not apply (e.g. debt/equity for a bank) has pass=null and does not
    count toward the maximum, so lenders and data gaps are never penalised. Grade A…E from the pass ratio,
    "N/A" when fewer than 6 checks could be evaluated.
    """
    m, facts = _metrics(f)
    fin = facts["financial"]
    rf = settings.risk_free_rate * 100

    def v(key: str) -> float | None:
        return m[key]["value"] if key in m else None

    def check(label: str, rule: str, key: str | None, test, na: str | None = None) -> dict:
        if na is not None:
            return {"label": label, "pass": None, "value": na, "rule": rule}
        val = v(key) if key else None
        if val is None:
            return {"label": label, "pass": None, "value": m[key]["note"] if key in m else "—", "rule": rule}
        return {"label": label, "pass": bool(test(val)), "value": _fmt(val, m[key]["unit"]), "rule": rule}

    na_fin = "n/a for banks / NBFCs" if fin else None
    quality = [
        check("ROCE above 15 %", "ROCE > 15 %", "roce", lambda x: x > 15, na_fin),
        check("ROE above 15 %", "ROE > 15 %", "roe", lambda x: x > 15),
        check("Operating margin stable or rising", "OPM change over 3 years ≥ −2 pp", "opm_trend_3y",
              lambda x: x >= -2, na_fin),
        check("Profits backed by cash", "Operating cash flow ÷ net profit over 3 years ≥ 80 %",
              "cash_conversion_3y", lambda x: x >= 80, na_fin),
    ]
    if fin:
        quality.append(check("ROA above 1 %", "Return on assets > 1 % (lenders)", "roa", lambda x: x > 1))

    growth = [
        check(f"{'Revenue' if fin else 'Sales'} growth", f"3-year {'revenue' if fin else 'sales'} CAGR > 10 %",
              "sales_cagr_3y", lambda x: x > 10),
        check("Profit growth", "3-year net profit CAGR > 10 %", "profit_cagr_3y", lambda x: x > 10),
        check("EPS growth", "3-year EPS CAGR > 10 %", "eps_cagr_3y", lambda x: x > 10),
    ]
    top_line = "revenue" if fin else "sales"
    for key, label, rule in (("qtr_sales_yoy", f"Latest quarter {top_line} up",
                              f"{top_line.capitalize()} above the same quarter last year"),
                             ("qtr_profit_yoy", "Latest quarter profit up",
                              "Net profit above the same quarter last year")):
        pair = facts.get(key)
        if pair is None:
            growth.append(check(label, rule, key, lambda x: x > 0))
        else:
            cur, prev = pair
            growth.append({"label": label, "pass": cur > prev, "rule": rule,
                           "value": _fmt(v(key), "%") if v(key) is not None
                           else f"₹ {cur:,.0f} Cr vs ₹ {prev:,.0f} Cr"})

    pe, p3 = facts.get("pe"), v("profit_cagr_3y")
    if pe is not None and pe > 0 and p3 is not None and p3 <= 0:
        peg_check = {"label": "Growth justifies the P/E", "pass": False, "rule": "PEG (P/E ÷ 3y profit CAGR) < 1.5",
                     "value": f"P/E {pe:g} with 3y profit CAGR {p3:.1f} %"}
    else:
        peg_check = check("Growth justifies the P/E", "PEG (P/E ÷ 3y profit CAGR) < 1.5", "peg", lambda x: x < 1.5)
    med = facts.get("peer_median_pe")
    valuation = [
        check("P/E below sector median", "P/E < median P/E of Screener peers", "pe_vs_peers", lambda x: x < 1,
              None if med is not None else "peer table unavailable"),
        peg_check,
        check("Earnings yield beats the risk-free rate", f"Earnings yield > {rf:.1f} % (India 10y G-sec)",
              "earnings_yield", lambda x: x > rf),
        check("Price / book below 3", "P/B < 3", "price_to_book", lambda x: x < 3),
    ]
    if valuation[0]["pass"] is not None:
        valuation[0]["value"] = f"{_fmt(v('pe_vs_peers'), 'x')} the peer median ({med:g})"

    interest = facts.get("interest")
    if not fin and interest is not None and interest <= 0:
        coverage = {"label": "Interest comfortably covered", "pass": True, "rule": "EBIT ÷ interest > 3×",
                    "value": "no interest expense"}
    else:
        coverage = check("Interest comfortably covered", "EBIT ÷ interest > 3×", "interest_coverage",
                         lambda x: x > 3, na_fin)
    last3 = facts.get("profit_last3") or []
    profitable = ({"label": "Consistently profitable", "pass": all(x > 0 for x in last3),
                   "rule": "Net profit > 0 in each of the last 3 years",
                   "value": ", ".join(f"₹ {x:,.0f} Cr" for x in last3)}
                  if len(last3) == 3 else
                  {"label": "Consistently profitable", "pass": None,
                   "rule": "Net profit > 0 in each of the last 3 years", "value": "needs 3 years of profit"})
    health = [
        check("Low debt", "Debt / equity < 0.5", "debt_to_equity", lambda x: x < 0.5, na_fin),
        coverage,
        check("Positive free cash flow", "Free cash flow > 0 in the latest year", "fcf_latest",
              lambda x: x > 0, na_fin),
        profitable,
    ]
    if fin:
        health += [check("Gross NPA below 3 %", "Gross NPA < 3 % (latest quarter)", "gross_npa", lambda x: x < 3),
                   check("Net NPA below 1 %", "Net NPA < 1 % (latest quarter)", "net_npa", lambda x: x < 1)]

    promoter_na = None
    if facts.get("promoter_now") in (None, 0):
        promoter_na = "no promoter group (widely held)" if (f.get("shareholding") or {}).get("quarterly") else None
    ownership = [
        check("Promoter holding stable or rising", "Promoter holding change over 4 quarters ≥ −1 pp",
              "promoter_change_4q", lambda x: x >= -1, promoter_na),
        check("Institutions buying", "FII + DII holding change over 4 quarters > 0 pp", "fii_dii_change_4q",
              lambda x: x > 0),
        check("Pays a dividend", "Dividend payout > 0 % (latest year)", "dividend_payout", lambda x: x > 0),
    ]

    parts = []
    for key, label, checks in (("quality", "Quality", quality), ("growth", "Growth", growth),
                               ("valuation", "Valuation", valuation), ("health", "Financial health", health),
                               ("ownership", "Ownership", ownership)):
        parts.append({"key": key, "label": label, "score": sum(c["pass"] is True for c in checks),
                      "max": sum(c["pass"] is not None for c in checks), "checks": checks})
    total, mx = sum(p["score"] for p in parts), sum(p["max"] for p in parts)
    grade = "N/A"
    if mx >= 6:
        r = total / mx
        grade = "A" if r >= 0.8 else "B" if r >= 0.65 else "C" if r >= 0.5 else "D" if r >= 0.35 else "E"
    return {"total": total, "max": mx, "grade": grade, "parts": parts}


def finalize(f: dict) -> dict:
    """(Re)compute derived metrics and the scorecard — call after peers are attached."""
    f["derived"] = derived_metrics(f)
    f["scorecard"] = scorecard(f)
    return f


# ------------------------------------------------------------------ client
class ScreenerError(RuntimeError):
    """Screener.in could not be reached or returned an unusable page, and no cached copy exists."""


class ScreenerClient:
    """Fetches, parses and caches Screener.in company pages, one polite request at a time."""

    def __init__(self, cache_dir: Path | str | None = None, ttl_hours: float | None = None,
                 min_interval: float | None = None, transport: httpx.AsyncBaseTransport | None = None,
                 retry_pause: float = 5.0, min_refresh_age: float = 600.0, timeout: float = 20.0):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(settings.data_dir) / "cache" / "screener"
        self.ttl = (settings.screener_ttl_hours if ttl_hours is None else ttl_hours) * 3600
        self.min_interval = settings.screener_min_interval if min_interval is None else min_interval
        self.retry_pause = retry_pause
        self.min_refresh_age = min_refresh_age      # a refresh younger than this is served from cache
        self.timeout = timeout
        self._transport = transport
        self._lock = asyncio.Lock()                 # one request in flight, spaced by min_interval
        self._last = -math.inf                      # monotonic time the last request finished
        self._sym_locks: dict[str, asyncio.Lock] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self.requests = 0

    # ---- public
    async def fundamentals(self, symbol: str, view: View = "auto", refresh: bool = False) -> dict:
        sym = symbol.strip().upper()
        if not SYMBOL_RE.match(sym):
            raise KeyError(f"{symbol!r} is not a valid NSE/BSE symbol")
        if view not in VIEWS:
            raise ValueError(f"view must be one of {', '.join(VIEWS)}")
        loop = asyncio.get_running_loop()
        if loop is not self._loop:                   # asyncio locks belong to one event loop
            self._loop, self._lock, self._sym_locks = loop, asyncio.Lock(), {}
        lock = self._sym_locks.setdefault(f"{sym}:{view}", asyncio.Lock())
        async with lock:                             # concurrent callers for one symbol share a single fetch
            cached = self._read_cache(sym, view)
            age = time.time() - cached["fetched_at"] if cached else math.inf
            if cached and age < self.ttl and (not refresh or age < self.min_refresh_age):
                note = (f"Fetched from Screener.in {age / 60:.0f} min ago; refresh is limited to once per "
                        f"{self.min_refresh_age / 60:.0f} min." if refresh else None)
                return self._served(cached, stale=False, note=note)
            try:
                f = await self._fetch(sym, view)
            except ScreenerError as e:
                if cached is None:
                    raise
                log.warning("screener %s: %s — serving the cached copy", sym, e)
                when = time.strftime("%d %b %Y %H:%M", time.localtime(cached["fetched_at"]))
                return self._served(cached, stale=age >= self.ttl,
                                    note=f"Screener.in unavailable ({e}); showing the copy fetched {when}.")
            self._write_cache(sym, view, f)
            return f

    async def aclose(self) -> None:
        """Nothing to release: each request uses its own short-lived HTTP client (they are ≥ seconds apart)."""

    # ---- fetching
    async def _get(self, path: str) -> httpx.Response:
        """GET one Screener path; one request at a time, ≥ min_interval apart, one retry on 429/5xx."""
        async with self._lock:
            resp: httpx.Response | None = None
            for attempt in range(2):
                wait = self._last + self.min_interval - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self.requests += 1
                try:
                    async with httpx.AsyncClient(transport=self._transport, headers=HEADERS, timeout=self.timeout,
                                                 follow_redirects=True) as http:
                        resp = await http.get(BASE_URL + path)
                except httpx.HTTPError as e:
                    raise ScreenerError(f"network error: {type(e).__name__}") from e
                finally:
                    self._last = time.monotonic()
                if resp.status_code != 429 and resp.status_code < 500:
                    return resp
                if attempt == 0:
                    pause = self.retry_pause
                    retry_after = resp.headers.get("retry-after", "")
                    if retry_after.isdigit():
                        pause = max(pause, min(float(retry_after), 60.0))
                    log.info("screener %s → HTTP %s, retrying in %.0f s", path, resp.status_code, pause)
                    await asyncio.sleep(pause)
            assert resp is not None
            return resp

    async def _page(self, sym: str, view: str) -> tuple[dict, dict]:
        path = f"/company/{quote(sym, safe='')}/" + ("consolidated/" if view == "consolidated" else "")
        resp = await self._get(path)
        if resp.status_code == 404:
            raise KeyError(f"{sym} is not on Screener.in")
        if resp.status_code != 200:
            raise ScreenerError(f"HTTP {resp.status_code}")
        parsed = await asyncio.to_thread(_parse_html, resp.text, sym, str(resp.url), view)
        if parsed is None:
            raise ScreenerError("page layout not recognised")
        f, meta = parsed
        landed = re.match(r"^/company/([^/]+)/", urlparse(str(resp.url)).path)
        if landed and unquote(landed.group(1)).upper() != sym:
            f["warnings"].append(f"Screener.in redirected {sym} to {unquote(landed.group(1)).upper()}.")
        return f, meta

    async def _fetch(self, sym: str, view: str) -> dict:
        notes: list[str] = []
        if view == "standalone":
            f, meta = await self._page(sym, "standalone")
        else:
            f, meta = await self._page(sym, "consolidated")
            if view == "auto" and not has_statements(f):
                f, meta = await self._page(sym, "standalone")
                notes.append("Screener.in has no consolidated statements for this company — showing standalone.")
            elif view == "auto" and _annual_span(f) < 3 and meta["other_view_hint"]:
                # consolidation began recently (e.g. a new subsidiary) and Screener points at the standalone view
                try:
                    alt, alt_meta = await self._page(sym, "standalone")
                except (KeyError, ScreenerError) as e:
                    log.info("screener %s standalone: %s", sym, e)
                else:
                    if _annual_span(alt) > _annual_span(f):
                        notes.append(f"Consolidated figures cover only {_annual_span(f):.1f} years — showing "
                                     f"standalone ({_annual_span(alt):.1f} years).")
                        f, meta = alt, alt_meta
        wid = meta["warehouse_id"]
        if wid:
            try:
                resp = await self._get(f"/api/company/{quote(wid, safe='')}/peers/")
                if resp.status_code == 200:
                    f["peers"] = await asyncio.to_thread(parse_peers, resp.text)
                if f["peers"] is None:
                    notes.append("Peer comparison unavailable from Screener.in.")
            except ScreenerError as e:
                notes.append(f"Peer comparison unavailable ({e}).")
        f["warnings"] = notes + f["warnings"]
        f["fetched_at"] = round(time.time(), 3)
        return finalize(f)

    # ---- cache
    def _cache_path(self, sym: str, view: str) -> Path:
        return self.cache_dir / f"{quote(sym, safe='')}.{view}.json"

    def _read_cache(self, sym: str, view: str) -> dict | None:
        try:
            data = json.loads(self._cache_path(sym, view).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
            return None
        f = data.get("fundamentals")
        return f if isinstance(f, dict) and isinstance(f.get("fetched_at"), (int, float)) else None

    def _write_cache(self, sym: str, view: str, f: dict) -> None:
        path = self._cache_path(sym, view)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": CACHE_VERSION, "fundamentals": f}, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(path)
        except OSError as e:
            log.warning("screener cache write failed for %s: %s", sym, e)

    @staticmethod
    def _served(cached: dict, stale: bool, note: str | None) -> dict:
        f = dict(cached)
        f["cached"], f["stale"] = True, stale
        f["warnings"] = ([note] if note else []) + list(cached.get("warnings") or [])
        return f
