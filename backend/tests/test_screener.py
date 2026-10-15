"""Screener.in fundamentals (td.research.screener): parser, number parsing, derived metrics, scorecard and the
polite caching client (httpx.MockTransport — no network)."""
import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from td.research import screener as sc

FIXTURE = Path(__file__).parent / "fixtures" / "screener_sample.html"
HTML = FIXTURE.read_text(encoding="utf-8")
URL = "https://www.screener.in/company/ACME/consolidated/"

PEERS_HTML = """
<div class="responsive-holder fill-card-width" data-page-results>
  <table class="data-table text-nowrap striped mark-visited">
    <tbody>
      <tr>
        <th class="text">S.No.</th>
        <th class="text">Company</th>
        <th data-tooltip="Current Price">CMP <span>Rs.</span></th>
        <th data-tooltip="Price to Earning">P/E <span></span></th>
        <th data-tooltip="Market Capitalization">Mar Cap <span>Rs.Cr.</span></th>
        <th data-tooltip="Return on capital employed">ROCE <span>%</span></th>
      </tr>
      <tr data-row-company-id="9001">
        <td class="text">1.</td>
        <td class="text"><a href="/company/ACME/consolidated/" target="_blank">Acme Widgets</a></td>
        <td>1234.00</td><td>59.90</td><td>12340.00</td><td>24.00</td>
      </tr>
      <tr data-row-company-id="9002">
        <td class="text">2.</td>
        <td class="text"><a href="/company/BETAGEAR/" target="_blank">Beta Gears</a></td>
        <td>500.00</td><td>30.00</td><td>5000.00</td><td>18.00</td>
      </tr>
      <tr data-row-company-id="9003">
        <td class="text">3.</td>
        <td class="text"><a href="/company/M%26N/consolidated/" target="_blank">M&amp;N Tools</a></td>
        <td>80.00</td><td>25.00</td><td>900.00</td><td></td>
      </tr>
      <tr data-row-company-id="9004">
        <td class="text">4.</td>
        <td class="text"><a href="/company/543210/" target="_blank">Gamma Castings</a></td>
        <td>100.00</td><td>20.00</td><td>450.00</td><td>12.00</td>
      </tr>
    </tbody>
    <tfoot>
      <tr><td></td><td class="text">Median: 4 Co.</td>
        <td>290.00</td><td>27.50</td><td>2950.00</td><td>18.00</td></tr>
    </tfoot>
  </table>
</div>
"""


def run(coro):
    return asyncio.run(coro)


def parsed() -> dict:
    return sc.parse_company_page(HTML, "ACME", URL, "consolidated")


def metric(f: dict, key: str) -> dict:
    return next(d for d in f["derived"] if d["key"] == key)


def checks(f: dict) -> dict[str, dict]:
    return {c["label"]: c for p in f["scorecard"]["parts"] for c in p["checks"]}


# ------------------------------------------------------------------ synthetic page builder (bank / edge cases)
def _table(cols: list[str], rows: list[tuple[str, list[str]]]) -> str:
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(f'<tr><td class="text">{label}</td>' + "".join(f"<td>{v}</td>" for v in vals) + "</tr>"
                   for label, vals in rows)
    return (f'<table class="data-table responsive-text-nowrap"><thead><tr><th class="text"></th>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table>")


def build_page(name: str, statements: dict, shp=None, ratios=(), view="Consolidated",
               industry="Private Sector Bank", hint=False) -> str:
    """Screener-shaped page: statements = {section id: (title, columns, rows)}, shp = (columns, rows);
    hint = Screener highlights the 'View Standalone' link (its nudge that the other view has more history)."""
    lis = "".join(f'<li><span class="name">{n}</span><span class="nowrap value">{pre} <span class="number">{v}</span>'
                  f" {post}</span></li>" for n, pre, v, post in ratios)
    secs = ""
    for sid, (title, cols, rows) in statements.items():
        link = f"<a href='#' class='{'strong' if hint else ''}'>View Standalone</a>"
        sub = f"{view} Figures in Rs. Crores / {link}" if sid != "ratios" else f"{view} / {link}"
        secs += (f'<section id="{sid}" class="card"><h2>{title}</h2><p class="sub">{sub}</p>'
                 f"<div data-result-table>{_table(cols, rows)}</div></section>")
    if shp:
        secs += ('<section id="shareholding"><h2>Shareholding Pattern</h2><p class="sub">Numbers in percentages</p>'
                 f'<div id="quarterly-shp">{_table(*shp)}</div></section>')
    return (f'<html><body><div id="company-info" data-warehouse-id="555"></div>'
            f'<div class="card" id="top"><h1>{name}</h1><ul id="top-ratios">{lis}</ul></div>'
            f'<section id="peers"><a title="Sector">Financial Services</a><a title="Industry">{industry}</a></section>'
            f"{secs}</body></html>")


YEARS = ["Mar 2021", "Mar 2022", "Mar 2023", "Mar 2024", "Mar 2025", "Mar 2026"]
QTRS = ["Jun 2025", "Sep 2025", "Dec 2025", "Mar 2026", "Jun 2026"]


def bank_page() -> str:
    pl = ("Profit & Loss", YEARS + ["TTM"], [
        ("Revenue +", ["1,000", "1,100", "1,250", "1,450", "1,650", "1,900", "1,950"]),
        ("Interest", ["500", "550", "620", "720", "820", "940", "960"]),
        ("Expenses +", ["300", "320", "350", "390", "430", "480", "490"]),
        ("Financing Profit", ["200", "230", "280", "340", "400", "480", "500"]),
        ("Financing Margin %", ["20%", "21%", "22%", "23%", "24%", "25%", "26%"]),
        ("Other Income +", ["50", "60", "70", "80", "90", "100", "100"]),
        ("Profit before tax", ["200", "227", "267", "320", "373", "427", "440"]),
        ("Tax %", ["25%", "25%", "25%", "25%", "25%", "25%", ""]),
        ("Net Profit +", ["150", "170", "200", "240", "280", "320", "330"]),
        ("EPS in Rs", ["15.00", "17.00", "20.00", "24.00", "28.00", "32.00", "33.00"]),
        ("Dividend Payout %", ["20%", "20%", "20%", "22%", "22%", "24%", ""]),
    ])
    q = ("Quarterly Results", QTRS, [
        ("Revenue +", ["450", "460", "480", "510", "500"]),
        ("Financing Profit", ["110", "115", "120", "135", "130"]),
        ("Net Profit +", ["75", "78", "80", "87", "85"]),
        ("Gross NPA %", ["1.20%", "1.15%", "1.10%", "1.05%", "1.00%"]),
        ("Net NPA %", ["0.40%", "0.38%", "0.35%", "0.32%", "0.30%"]),
    ])
    bs = ("Balance Sheet", YEARS, [
        ("Equity Capital", ["50"] * 6),
        ("Reserves", ["1,000", "1,150", "1,330", "1,550", "1,800", "2,080"]),
        ("Deposits", ["7,000", "8,000", "9,200", "10,500", "11,800", "13,500"]),
        ("Borrowing", ["1,000", "1,100", "1,200", "1,300", "1,350", "1,370"]),
        ("Total Liabilities", ["9,050", "10,300", "11,780", "13,400", "15,000", "17,000"]),
        ("Total Assets", ["9,050", "10,300", "11,780", "13,400", "15,000", "17,000"]),
    ])
    cf = ("Cash Flows", YEARS, [
        ("Cash from Operating Activity +", ["-300", "-350", "-400", "-420", "-500", "-600"]),
        ("Cash from Investing Activity +", ["-20", "-25", "-30", "-30", "-35", "-40"]),
        ("Free Cash Flow", ["-330", "-380", "-440", "-460", "-540", "-650"]),
    ])
    rt = ("Ratios", YEARS, [("ROE %", ["13%", "14%", "15%", "16%", "17%", "17%"])])
    shp = (QTRS, [("FIIs +", ["30.00%", "30.50%", "31.00%", "31.20%", "31.50%"]),
                  ("DIIs +", ["40.00%", "40.20%", "40.40%", "40.60%", "41.00%"]),
                  ("Public +", ["30.00%", "29.30%", "28.60%", "28.20%", "27.50%"])])
    ratios = [("Market Cap", "₹", "40,000", "Cr."), ("Current Price", "₹", "380", ""), ("Stock P/E", "", "12.0", ""),
              ("Book Value", "₹", "200", ""), ("ROCE", "", "7.50", "%"), ("ROE", "", "17.0", "%")]
    return build_page("Example Bank Ltd", {"quarters": q, "profit-loss": pl, "balance-sheet": bs, "cash-flow": cf,
                                           "ratios": rt}, shp=shp, ratios=ratios)


def empty_page() -> str:
    """What Screener serves on /consolidated/ for a company that only reports standalone figures."""
    rows = [("Sales +", []), ("Net Profit +", [])]
    statements = {sid: (title, [], rows) for sid, title in
                  (("quarters", "Quarterly Results"), ("profit-loss", "Profit & Loss"))}
    return build_page("Solo Ltd", statements, ratios=[("Market Cap", "₹", "", "Cr.")], view="", industry="Textiles")


# ------------------------------------------------------------------ number parsing
@pytest.mark.parametrize("text,expected", [
    ("1,23,456", 123456.0), ("207,559", 207559.0), ("-5", -5.0), ("12%", 12.0), ("0.51 %", 0.51),
    ("₹ 16,49,075 Cr.", 1649075.0), ("−3.2", -3.2), ("- 4", -4.0), ("+7%", 7.0), (".5", 0.5), ("50.27%", 50.27),
    ("Rs. 1,234", 1234.0), ("Rs.5", 5.0),
    ("", None), (None, None), ("-", None), ("--", None), ("%", None), ("abc", None), ("1,612 / 1,152", None),
])
def test_parse_number(text, expected):
    assert sc.parse_number(text) == expected


def test_col_period():
    assert sc.col_period("Mar 2024") == (2024 * 12 + 2, 12)
    assert sc.col_period("Mar 2024 15m") == (2024 * 12 + 2, 15)
    assert sc.col_period("Sep 2026") == (2026 * 12 + 8, 12)
    assert sc.col_period("TTM") is None
    assert sc.col_period("") is None


# ------------------------------------------------------------------ parser
def test_identity_links_and_profile():
    f = parsed()
    assert f["symbol"] == "ACME" and f["name"] == "Acme Widgets Ltd" and f["url"] == URL
    assert f["view"] == "consolidated"
    assert f["about"] == ("Acme Widgets makes precision widgets and gear assemblies for the automotive and "
                          "appliance industries.")
    assert f["sector"] == "Industrial Products" and f["industry"] == "Gears & Widgets"
    assert (f["bse_code"], f["nse_code"], f["website"]) == ("599001", "ACME", "https://www.acme-widgets.example")
    assert f["pros"] == ["Company is almost debt free.",
                         "Company has a good return on equity (ROE) track record: 3 Years ROE 20.0%"]
    assert f["cons"] == ["Stock is trading at 11.8 times its book value"]
    assert f["warnings"] == [] and f["peers"] is None and f["stale"] is False


def test_shape_matches_the_frontend_contract():
    """Keys of Fundamentals / FinTable / derived / scorecard exactly as frontend/src/lib/research.ts declares."""
    f = parsed()
    assert set(f) == {"symbol", "name", "url", "view", "fetched_at", "cached", "stale", "about", "sector", "industry",
                      "bse_code", "nse_code", "website", "ratios", "pros", "cons", "growth", "quarters",
                      "profit_loss", "balance_sheet", "cash_flow", "ratios_table", "shareholding", "peers",
                      "derived", "scorecard", "warnings"}
    assert set(f["quarters"]) == {"title", "columns", "rows", "unit"}
    assert set(f["quarters"]["rows"][0]) == {"label", "values", "is_pct"}
    assert set(f["shareholding"]) == {"quarterly", "yearly"}
    assert all(set(r) == {"key", "label", "value", "text", "unit"} for r in f["ratios"])
    assert all(set(d) == {"key", "label", "value", "unit", "note"} for d in f["derived"])
    assert set(f["scorecard"]) == {"total", "max", "grade", "parts"}
    for p in f["scorecard"]["parts"]:
        assert set(p) == {"key", "label", "score", "max", "checks"}
        assert all(set(c) == {"label", "pass", "value", "rule"} and isinstance(c["value"], str) for c in p["checks"])
    assert [p["key"] for p in f["scorecard"]["parts"]] == ["quality", "growth", "valuation", "health", "ownership"]
    f["peers"] = sc.parse_peers(PEERS_HTML)
    assert all(set(r) == {"name", "symbol", "values"} for r in f["peers"]["rows"])


def test_top_ratios():
    r = {x["key"]: x for x in parsed()["ratios"]}
    assert list(r) == ["market_cap", "current_price", "high", "low", "stock_pe", "book_value", "dividend_yield",
                       "roce", "roe", "face_value"]
    assert r["market_cap"] == {"key": "market_cap", "label": "Market Cap", "value": 12340.0,
                               "text": "₹ 12,340 Cr.", "unit": "₹ Cr."}
    assert (r["high"]["value"], r["low"]["value"], r["high"]["unit"], r["low"]["text"]) == (1500.0, 900.0, "₹", "₹ 900")
    assert r["high"]["label"] == "52W High"
    assert (r["stock_pe"]["value"], r["stock_pe"]["unit"]) == (59.9, "")
    assert (r["dividend_yield"]["value"], r["dividend_yield"]["unit"]) == (0.49, "%")
    assert r["roe"]["value"] == 20.4 and r["face_value"]["value"] == 10.0


def test_growth_boxes():
    g = {b["title"]: b["rows"] for b in parsed()["growth"]}
    assert list(g) == ["Compounded Sales Growth", "Compounded Profit Growth", "Stock Price CAGR", "Return on Equity"]
    assert g["Compounded Sales Growth"] == [{"period": "10 Years", "value": None}, {"period": "5 Years", "value": 14.0},
                                           {"period": "3 Years", "value": 10.0}, {"period": "TTM", "value": 9.0}]
    assert g["Stock Price CAGR"][-1] == {"period": "1 Year", "value": -4.0}


def test_statement_tables():
    f = parsed()
    q = f["quarters"]
    assert q["title"] == "Quarterly Results" and q["unit"] == "Rs. Crores"
    assert q["columns"] == QTRS
    labels = [r["label"] for r in q["rows"]]
    assert "Raw PDF" not in labels and labels[0] == "Sales" and "Net Profit" in labels
    assert q["rows"][0] == {"label": "Sales", "values": [310.0, 320.0, 335.0, 366.0, 341.0], "is_pct": False}
    opm = next(r for r in q["rows"] if r["label"] == "OPM %")
    assert opm["is_pct"] and opm["values"][-1] == 22.0

    pl = f["profit_loss"]
    assert pl["title"] == "Profit & Loss" and pl["columns"] == YEARS + ["TTM"]
    payout = next(r for r in pl["rows"] if r["label"] == "Dividend Payout %")
    assert payout["values"] == [20.0, 20.0, 25.0, 25.0, 30.0, 30.0, None]       # blank TTM cell → null
    assert next(r for r in pl["rows"] if r["label"] == "Sales")["values"][2] == 1000.0

    assert f["balance_sheet"]["rows"][2]["label"] == "Borrowings"
    cf = f["cash_flow"]
    assert next(r for r in cf["rows"] if r["label"] == "Cash from Investing Activity")["values"][0] == -60.0
    assert f["ratios_table"]["unit"] == "" and f["ratios_table"]["rows"][-1]["label"] == "ROCE %"


def test_shareholding_tabs():
    sh = parsed()["shareholding"]
    qt, yr = sh["quarterly"], sh["yearly"]
    assert qt["columns"] == QTRS and yr["columns"] == ["Mar 2024", "Mar 2025", "Mar 2026", "Jun 2026"]
    assert qt["unit"] == "%" and qt["title"] == "Shareholding Pattern (quarterly)"
    assert qt["rows"][0] == {"label": "Promoters", "values": [55.0, 54.8, 54.6, 54.5, 54.2], "is_pct": True}
    holders = qt["rows"][-1]
    assert holders["label"] == "No. of Shareholders" and not holders["is_pct"] and holders["values"][-1] == 152000.0
    assert yr["rows"][0]["values"][0] == 56.0


def test_parse_peers():
    p = sc.parse_peers(PEERS_HTML)
    assert p["columns"] == ["CMP Rs.", "P/E", "Mar Cap Rs.Cr.", "ROCE %"]
    names = [(r["name"], r["symbol"]) for r in p["rows"]]
    assert names == [("Acme Widgets", "ACME"), ("Beta Gears", "BETAGEAR"), ("M&N Tools", "M&N"),
                     ("Gamma Castings", None), ("Median: 4 Co.", None)]           # BSE-only code → no symbol
    assert p["rows"][2]["values"] == [80.0, 25.0, 900.0, None]
    assert sc.parse_peers("<div>Loading peers table ...</div>") is None


def test_empty_statements_and_json_safety():
    f = sc.parse_company_page(empty_page(), "SOLO", "u", "consolidated")
    assert not sc.has_statements(f) and f["profit_loss"] is None
    assert f["warnings"] == ["Screener.in shows no consolidated financial statements for SOLO."]
    assert f["ratios"][0]["value"] is None and f["scorecard"]["grade"] == "N/A"
    json.dumps(f, allow_nan=False)
    json.dumps(parsed(), allow_nan=False)


# ------------------------------------------------------------------ derived metrics
def test_derived_metrics_exact():
    f = parsed()
    assert [d["key"] for d in f["derived"]] == [
        "sales_cagr_3y", "sales_cagr_5y", "profit_cagr_3y", "profit_cagr_5y", "eps_cagr_3y", "qtr_sales_yoy",
        "qtr_profit_yoy", "opm_latest", "opm_trend_3y", "roe", "roce", "roa", "debt_to_equity",
        "interest_coverage", "fcf_latest", "cash_conversion_3y", "dividend_payout", "promoter_holding",
        "promoter_change_4q", "fii_dii_change_4q", "earnings_yield", "peg", "price_to_book", "pe_vs_peers"]
    expect = {
        "sales_cagr_3y": ((1350 / 1000) ** (1 / 3) - 1) * 100,
        "sales_cagr_5y": ((1350 / 700) ** (1 / 5) - 1) * 100,
        "profit_cagr_3y": ((200 / 124) ** (1 / 3) - 1) * 100,
        "profit_cagr_5y": ((200 / 75) ** (1 / 5) - 1) * 100,
        "eps_cagr_3y": ((20 / 12.4) ** (1 / 3) - 1) * 100,
        "qtr_sales_yoy": (341 / 310 - 1) * 100,
        "qtr_profit_yoy": (54 / 48 - 1) * 100,
        "opm_latest": 22.0,                      # TTM
        "opm_trend_3y": 2.0,                     # 22 % (Mar 2026) − 20 % (Mar 2023)
        "roe": 20.4, "roce": 24.0,
        "roa": 200 / ((1250 + 1400) / 2) * 100,
        "debt_to_equity": 105 / (100 + 950),
        "interest_coverage": (274 + 10) / 10,    # TTM
        "fcf_latest": 95.0,
        "cash_conversion_3y": (140 + 160 + 190) / (149 + 173 + 200) * 100,
        "dividend_payout": 30.0,
        "promoter_holding": 54.2,
        "promoter_change_4q": 54.2 - 55.0,
        "fii_dii_change_4q": (16.5 + 11.0) - (15.0 + 10.0),
        "earnings_yield": 100 / 59.9,
        "peg": 59.9 / round(((20 / 12.4) ** (1 / 3) - 1) * 100, 2),        # per-share (EPS) growth
        "price_to_book": 1234 / 105,
    }
    for key, want in expect.items():
        assert metric(f, key)["value"] == pytest.approx(want, abs=0.006), key
    assert metric(f, "sales_cagr_3y")["note"] == "Mar 2023 → Mar 2026"
    assert metric(f, "qtr_sales_yoy")["note"] == "Jun 2026 vs Jun 2025"
    assert metric(f, "fcf_latest")["unit"] == "₹ Cr"
    assert metric(f, "pe_vs_peers")["value"] is None and "peer" in metric(f, "pe_vs_peers")["note"]
    f["peers"] = sc.parse_peers(PEERS_HTML)
    sc.finalize(f)
    assert metric(f, "pe_vs_peers")["value"] == pytest.approx(59.9 / 27.5, abs=0.006)


def test_missing_inputs_are_null_with_a_note():
    f = parsed()
    f["cash_flow"] = None
    f["profit_loss"]["rows"] = [r for r in f["profit_loss"]["rows"] if r["label"] != "OPM %"]
    sc.finalize(f)
    assert metric(f, "fcf_latest")["value"] is None and "cash flow" in metric(f, "fcf_latest")["note"]
    assert metric(f, "cash_conversion_3y")["value"] is None
    assert metric(f, "opm_latest")["value"] is None and "OPM" in metric(f, "opm_latest")["note"]
    c = checks(f)
    assert c["Positive free cash flow"]["pass"] is None and c["Profits backed by cash"]["pass"] is None
    assert c["Operating margin stable or rising"]["pass"] is None


def test_half_year_balance_sheet_and_roe_fallback():
    f = parsed()
    bs = f["balance_sheet"]
    bs["columns"].append("Sep 2026")                     # Screener adds a half-year balance sheet mid-year
    for r in bs["rows"]:
        r["values"].append({"Total Assets": 1600.0, "Reserves": 1000.0, "Borrowings": 400.0}.get(r["label"],
                                                                                              r["values"][-1]))
    f["ratios"] = [r for r in f["ratios"] if r["key"] != "roe"]
    f["ratios_table"] = None
    sc.finalize(f)
    roa = metric(f, "roa")
    assert roa["value"] == pytest.approx(200 / ((1250 + 1400) / 2) * 100, abs=0.006)     # FY profit, FY assets
    roe = metric(f, "roe")
    assert roe["value"] == pytest.approx(200 / ((900 + 1050) / 2) * 100, abs=0.006)
    assert roe["note"] == "net profit Mar 2026 ÷ average equity"
    de = metric(f, "debt_to_equity")                     # leverage is a point-in-time stock: latest column
    assert de["value"] == pytest.approx(400 / 1100, abs=0.006) and de["note"].endswith("Sep 2026")


def test_cagr_skips_a_15_month_year_and_uses_real_elapsed_time():
    cols = ["Dec 2021", "Dec 2022", "Mar 2024 15m", "Mar 2025", "Mar 2026"]
    page = build_page("Shift Ltd", {"profit-loss": ("Profit & Loss", cols, [
        ("Sales +", ["100", "120", "180", "160", "176"]), ("OPM %", ["10%"] * 5),
        ("Net Profit +", ["10", "12", "18", "16", "-2"])])}, industry="Foods")
    f = sc.parse_company_page(page, "SHIFT", "u", "consolidated")
    m = metric(f, "sales_cagr_3y")
    # target Mar 2023 → nearest full year is Dec 2022 (3 months away); the 15-month period is never an endpoint
    assert m["value"] == pytest.approx(((176 / 120) ** (1 / 3.25) - 1) * 100, abs=0.006)
    assert m["note"] == "Dec 2022 → Mar 2026"
    assert metric(f, "sales_cagr_5y")["value"] is None and "needs 5 years" in metric(f, "sales_cagr_5y")["note"]
    assert metric(f, "profit_cagr_3y")["value"] is None and "not positive" in metric(f, "profit_cagr_3y")["note"]
    assert sc._annual_span(f) == pytest.approx(4.25)


# ------------------------------------------------------------------ scorecard
def test_scorecard_on_fixture():
    f = parsed()
    f["peers"] = sc.parse_peers(PEERS_HTML)
    sc.finalize(f)
    card = f["scorecard"]
    parts = {p["key"]: (p["score"], p["max"]) for p in card["parts"]}
    assert parts == {"quality": (4, 4), "growth": (5, 5), "valuation": (0, 4), "health": (4, 4), "ownership": (3, 3)}
    assert (card["total"], card["max"], card["grade"]) == (16, 20, "A")
    c = checks(f)
    assert c["P/E below sector median"]["pass"] is False and "27.5" in c["P/E below sector median"]["value"]
    assert c["Low debt"] == {"label": "Low debt", "pass": True, "value": "0.10×", "rule": "Debt / equity < 0.5"}
    assert c["Earnings yield beats the risk-free rate"]["rule"].startswith("Earnings yield > 6.5 %")
    for p in card["parts"]:
        assert 0 <= p["score"] <= p["max"] == sum(x["pass"] is not None for x in p["checks"])


def test_scorecard_without_peers_does_not_penalise():
    card = parsed()["scorecard"]
    val = next(p for p in card["parts"] if p["key"] == "valuation")
    assert val["max"] == 3 and val["checks"][0]["pass"] is None
    assert (card["total"], card["max"]) == (16, 19)


def test_bank_is_scored_without_industrial_metrics():
    f = sc.parse_company_page(bank_page(), "EXBANK", "u", "consolidated")
    assert sc.is_financial(f)
    assert sc.is_financial(parsed()) is False
    d = {x["key"]: x for x in f["derived"]}
    assert d["opm_latest"]["label"] == "Financing margin" and d["opm_latest"]["value"] == 26.0
    for key in ("debt_to_equity", "interest_coverage", "fcf_latest", "cash_conversion_3y", "opm_trend_3y"):
        assert d[key]["value"] is None and "banks / NBFCs" in d[key]["note"], key
    assert d["roa"]["value"] == pytest.approx(320 / 16000 * 100)
    assert (d["gross_npa"]["value"], d["net_npa"]["value"]) == (1.0, 0.3)
    assert d["promoter_holding"]["value"] is None and "widely held" in d["promoter_holding"]["note"]
    assert d["sales_cagr_3y"]["label"] == "Revenue CAGR 3y"
    assert d["sales_cagr_3y"]["value"] == pytest.approx(((1900 / 1250) ** (1 / 3) - 1) * 100, abs=0.006)

    c = checks(f)
    for label in ("ROCE above 15 %", "Operating margin stable or rising", "Profits backed by cash", "Low debt",
                  "Interest comfortably covered", "Positive free cash flow", "Promoter holding stable or rising"):
        assert c[label]["pass"] is None, label
    assert c["ROA above 1 %"]["pass"] is True
    assert c["Gross NPA below 3 %"]["pass"] is True and c["Net NPA below 1 %"]["pass"] is True
    assert c["Revenue growth"]["pass"] is True and c["Latest quarter revenue up"]["pass"] is True
    card = f["scorecard"]
    assert card["max"] == sum(x["pass"] is not None for x in c.values())
    assert card["grade"] in "ABCDE" and card["total"] / card["max"] >= 0.65      # a healthy lender grades well


def test_scorecard_loss_maker_and_debt_free():
    f = parsed()
    pl = f["profit_loss"]
    for r in pl["rows"]:
        if r["label"] == "Interest":
            r["values"] = [0.0] * len(r["values"])
        if r["label"] == "Net Profit":
            r["values"][-2] = -5.0                       # Mar 2026 loss
    f["ratios"] = [r for r in f["ratios"] if r["key"] != "stock_pe"]
    sc.finalize(f)
    c = checks(f)
    assert c["Interest comfortably covered"] == {"label": "Interest comfortably covered", "pass": True,
                                                 "rule": "EBIT ÷ interest > 3×", "value": "no interest expense"}
    assert c["Consistently profitable"]["pass"] is False
    assert c["Growth justifies the P/E"]["pass"] is None          # no P/E for a loss maker → not scored
    assert metric(f, "earnings_yield")["note"] == "TTM EPS ÷ price"


def _with_pe_and_eps(pe: float, eps: list | None) -> dict:
    """The fixture with another Stock P/E and EPS row (None drops the row). Its net profit grows 124 → 200 over
    Mar 2023 → Mar 2026, a 3y CAGR of 17.3 %."""
    f = parsed()
    for r in f["ratios"]:
        if r["key"] == "stock_pe":
            r["value"] = pe
    pl = f["profit_loss"]
    if eps is None:
        pl["rows"] = [r for r in pl["rows"] if r["label"] != "EPS in Rs"]
    else:
        next(r for r in pl["rows"] if r["label"] == "EPS in Rs")["values"] = eps
    return sc.finalize(f)


def test_peg_uses_per_share_growth():
    profit3 = ((200 / 124) ** (1 / 3) - 1) * 100
    # EPS grew with profit: PEG = 14.2 ÷ 17.3 % passes, and the rule names the growth it used
    f = _with_pe_and_eps(14.2, [5.0, 7.5, 12.4, 14.9, 17.3, 20.0, 20.5])
    peg, c = metric(f, "peg"), checks(f)["Growth justifies the P/E"]
    assert peg["value"] == pytest.approx(14.2 / round(profit3, 2), abs=0.006)
    assert peg["note"] == f"P/E 14.2 ÷ 3y EPS CAGR {profit3:.1f} %"
    assert c["pass"] is True and c["rule"] == "PEG (P/E ÷ 3y EPS CAGR) < 1.5"

    # a merger issued many new shares: net profit CAGR 17.3 % but EPS CAGR 5 % → PEG 2.85 fails (was 0.82)
    eps3 = 12.4 * 1.05 ** 3
    f = _with_pe_and_eps(14.2, [5.0, 7.5, 12.4, 12.9, 13.6, round(eps3, 2), 14.5])
    peg, c = metric(f, "peg"), checks(f)["Growth justifies the P/E"]
    e3 = metric(f, "eps_cagr_3y")["value"]
    assert e3 == pytest.approx(5.0, abs=0.01)
    assert peg["value"] == pytest.approx(14.2 / e3, abs=0.006)
    assert "3y EPS CAGR 5.0 %" in peg["note"] and "net profit CAGR 17.3 % differs" in peg["note"]
    assert c["pass"] is False and c["value"] == "2.85×" and "EPS" in c["rule"]

    # EPS shrank while net profit grew: no PEG, and the check fails on per-share growth (not a profit-based pass)
    f = _with_pe_and_eps(14.2, [5.0, 7.5, 12.4, 12.0, 11.5, 11.0, 11.0])
    peg, c = metric(f, "peg"), checks(f)["Growth justifies the P/E"]
    assert peg["value"] is None and peg["note"].startswith("3y EPS CAGR -3.9 % is not positive")
    assert c["pass"] is False and c["value"] == "P/E 14.2 with 3y EPS CAGR -3.9 %"


def test_peg_falls_back_to_profit_growth_and_says_so():
    profit3 = ((200 / 124) ** (1 / 3) - 1) * 100
    f = _with_pe_and_eps(14.2, None)                     # no EPS row on the page
    assert metric(f, "eps_cagr_3y")["value"] is None
    peg, c = metric(f, "peg"), checks(f)["Growth justifies the P/E"]
    assert peg["value"] == pytest.approx(14.2 / round(profit3, 2), abs=0.006)
    assert peg["note"] == (f"P/E 14.2 ÷ 3y net profit CAGR {profit3:.1f} %; no 3y EPS CAGR (row not on the page), "
                           "so net profit growth stands in")
    assert c["pass"] is True and c["rule"] == "PEG (P/E ÷ 3y net profit CAGR; EPS CAGR unavailable) < 1.5"


# ------------------------------------------------------------------ client (httpx.MockTransport)
class Site:
    """Fake www.screener.in: path → response or list of responses (consumed in order, last one repeats).
    Every request is checked against robots.txt rules and logged with its time."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.log: list[tuple[float, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode()
        assert request.url.host == "www.screener.in"
        assert "?" not in path and not path.startswith("/user/") and "/search" not in path, path
        assert "Mozilla" in request.headers["user-agent"]
        self.log.append((time.monotonic(), path))
        r = self.routes.get(path, (404, "<html><title>Error 404</title></html>"))
        if isinstance(r, list):
            r = r.pop(0) if len(r) > 1 else r[0]
        if isinstance(r, Exception):
            raise r
        status, body, *headers = r
        return httpx.Response(status, text=body, headers=headers[0] if headers else None)

    @property
    def paths(self) -> list[str]:
        return [p for _, p in self.log]


def client(tmp_path, site: Site, **kw) -> sc.ScreenerClient:
    kw = {"ttl_hours": 12, "min_interval": 0, "retry_pause": 0, **kw}
    return sc.ScreenerClient(cache_dir=tmp_path, transport=httpx.MockTransport(site), **kw)


def acme_routes(**extra) -> dict:
    return {"/company/ACME/consolidated/": (200, HTML), "/api/company/7700001/peers/": (200, PEERS_HTML), **extra}


def test_fetch_parse_and_cache(tmp_path):
    site = Site(acme_routes())
    c = client(tmp_path, site)

    async def go():
        a = await c.fundamentals("acme")
        b = await c.fundamentals("ACME")
        await c.aclose()
        return a, b

    a, b = run(go())
    assert site.paths == ["/company/ACME/consolidated/", "/api/company/7700001/peers/"]
    assert a["cached"] is False and a["stale"] is False and a["fetched_at"] > 0
    assert a["url"] == URL and a["peers"]["rows"][1]["symbol"] == "BETAGEAR"
    assert checks(a)["P/E below sector median"]["pass"] is False                # scorecard recomputed with peers
    assert b["cached"] is True and b["stale"] is False and b["name"] == "Acme Widgets Ltd"
    disk = json.loads((tmp_path / "ACME.auto.json").read_text(encoding="utf-8"))
    assert disk["version"] == sc.CACHE_VERSION and disk["fundamentals"]["scorecard"] == a["scorecard"]


def test_cache_from_another_version_is_ignored(tmp_path):
    (tmp_path / "ACME.auto.json").write_text(json.dumps({"version": -1, "fundamentals": {"fetched_at": time.time()}}))
    site = Site(acme_routes())
    f = run(client(tmp_path, site).fundamentals("ACME"))
    assert f["cached"] is False and site.paths[0] == "/company/ACME/consolidated/"


def test_ttl_expiry_refetches(tmp_path):
    site = Site(acme_routes())
    c = client(tmp_path, site, ttl_hours=0)

    async def go():
        await c.fundamentals("ACME")
        return await c.fundamentals("ACME")

    assert run(go())["cached"] is False
    assert site.paths.count("/company/ACME/consolidated/") == 2


def test_refresh_is_rate_limited(tmp_path):
    site = Site(acme_routes())

    async def go(min_age):
        c = client(tmp_path, site, min_refresh_age=min_age)
        return await c.fundamentals("ACME", refresh=True)

    first = run(go(600))                                  # nothing cached yet → fetch
    again = run(go(600))                                  # refreshed seconds ago → served from cache
    assert first["cached"] is False and again["cached"] is True
    assert "refresh is limited" in again["warnings"][0]
    assert run(go(0))["cached"] is False                  # old enough → really refreshed
    assert site.paths.count("/company/ACME/consolidated/") == 2


def test_stale_cache_when_screener_is_down(tmp_path):
    site = Site(acme_routes())
    run(client(tmp_path, site).fundamentals("ACME"))
    site.routes["/company/ACME/consolidated/"] = (503, "busy")
    site.log.clear()
    f = run(client(tmp_path, site, ttl_hours=0).fundamentals("ACME"))
    assert f["stale"] is True and f["cached"] is True
    assert f["warnings"][0].startswith("Screener.in unavailable (HTTP 503)")
    assert site.paths == ["/company/ACME/consolidated/"] * 2              # retried once, then gave up


def test_down_without_cache_raises(tmp_path):
    site = Site({"/company/ACME/consolidated/": (502, "bad gateway")})
    with pytest.raises(sc.ScreenerError, match="HTTP 502"):
        run(client(tmp_path, site).fundamentals("ACME"))
    site2 = Site({"/company/ACME/consolidated/": httpx.ConnectError("refused")})
    with pytest.raises(sc.ScreenerError, match="network error"):
        run(client(tmp_path, site2).fundamentals("ACME"))
    site3 = Site({"/company/ACME/consolidated/": (200, "<html><body><form>Login</form></body></html>")})
    with pytest.raises(sc.ScreenerError, match="layout"):
        run(client(tmp_path, site3).fundamentals("ACME"))


def test_404_is_key_error_and_bad_input(tmp_path):
    site = Site({})
    with pytest.raises(KeyError, match="not on Screener.in"):
        run(client(tmp_path, site).fundamentals("NOSUCHCO"))
    assert site.paths == ["/company/NOSUCHCO/consolidated/"]
    with pytest.raises(KeyError):
        run(client(tmp_path, site).fundamentals("bad sym!"))
    with pytest.raises(ValueError):
        run(client(tmp_path, site).fundamentals("ACME", view="annual"))


def test_404_is_remembered_without_new_requests(tmp_path):
    site = Site({})
    c = client(tmp_path, site)

    async def ask(n, sym="NOSUCHCO", **kw):
        for _ in range(n):
            with pytest.raises(KeyError, match=f"{sym} is not on Screener.in"):
                await c.fundamentals(sym, **kw)

    async def go():
        await ask(3)
        await ask(2, refresh=True)                       # refresh does not bypass a remembered 404
        await asyncio.gather(ask(1, "JUNK"), ask(1, "JUNK"))       # the waiter sees the first caller's 404
        assert c._sym_locks == {}                        # no lock entry left behind per unknown symbol
        c.missing_ttl = 0
        c._missing.clear()
        await ask(2)                                     # expired at once → each call asks Screener.in again

    run(go())
    assert site.paths == ["/company/NOSUCHCO/consolidated/", "/company/JUNK/consolidated/"] + \
        ["/company/NOSUCHCO/consolidated/"] * 2


def test_remembered_404s_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(sc.ScreenerClient, "MISSING_MAX", 3)
    c = client(tmp_path, Site({}))
    for sym in ("A1", "A2", "A3", "A4"):
        with pytest.raises(KeyError):
            run(c.fundamentals(sym))
    assert list(c._missing) == ["A2:auto", "A3:auto", "A4:auto"]          # the oldest was forgotten


def test_full_queue_turns_requests_away(tmp_path):
    routes = {}
    for sym in ("CO1", "CO2", "CO3", "CO4"):
        routes[f"/company/{sym}/consolidated/"] = (200, HTML.replace("Acme Widgets Ltd", f"{sym} Ltd"))
    routes["/api/company/7700001/peers/"] = (200, PEERS_HTML)
    site = Site(routes)
    c = client(tmp_path, site, min_interval=0.05, max_queued=2)

    async def go():
        return await asyncio.gather(*(c.fundamentals(s) for s in ("CO1", "CO2", "CO3", "CO4")),
                                    return_exceptions=True)

    res = run(go())
    assert [r["name"] for r in res[:2]] == ["CO1 Ltd", "CO2 Ltd"]          # admitted fetches finish, with peers
    assert all(r["peers"] is not None for r in res[:2])
    for e in res[2:]:
        assert type(e) is sc.ScreenerError and e.busy and str(e).startswith("busy")   # router maps it to 502
    assert not any("CO3" in p or "CO4" in p for p in site.paths)            # turned away without a request
    assert c._queued == 0 and c._sym_locks == {}
    assert run(c.fundamentals("CO3"))["name"] == "CO3 Ltd"                # the queue drains, later calls work


def test_full_queue_serves_the_cached_copy(tmp_path):
    site = Site(acme_routes())
    run(client(tmp_path, site).fundamentals("ACME"))
    site.log.clear()
    c = client(tmp_path, site, ttl_hours=0, max_queued=0)
    f = run(c.fundamentals("ACME"))
    assert f["cached"] is True and f["stale"] is True and site.paths == []
    assert f["warnings"][0].startswith("Not refetched: the gateway's Screener.in queue is full")


def test_cached_copy_follows_the_current_rules(tmp_path):
    site = Site(acme_routes())
    first = run(client(tmp_path, site).fundamentals("ACME"))
    path = tmp_path / "ACME.auto.json"
    disk = json.loads(path.read_text(encoding="utf-8"))
    disk["fundamentals"]["derived"] = [{"key": "peg", "label": "PEG", "value": 0.1, "unit": "x", "note": "old rule"}]
    disk["fundamentals"]["scorecard"] = {"total": 0, "max": 0, "grade": "N/A", "parts": []}
    path.write_text(json.dumps(disk), encoding="utf-8")
    again = run(client(tmp_path, site).fundamentals("ACME"))
    assert again["cached"] is True and site.paths.count("/company/ACME/consolidated/") == 1
    assert again["derived"] == first["derived"] and again["scorecard"] == first["scorecard"]


def test_429_is_retried_once(tmp_path):
    site = Site(acme_routes(**{"/company/ACME/consolidated/": [(429, "slow down"), (200, HTML)]}))
    f = run(client(tmp_path, site).fundamentals("ACME"))
    assert f["name"] == "Acme Widgets Ltd"
    assert site.paths == ["/company/ACME/consolidated/"] * 2 + ["/api/company/7700001/peers/"]


def test_requests_are_spaced_globally(tmp_path):
    beta = HTML.replace("Acme Widgets Ltd", "Beta Gears Ltd").replace('data-warehouse-id="7700001"',
                                                                       'data-warehouse-id="7700002"')
    site = Site(acme_routes(**{"/company/BETA/consolidated/": (200, beta),
                               "/api/company/7700002/peers/": (200, PEERS_HTML)}))
    c = client(tmp_path, site, min_interval=0.12)

    async def go():
        return await asyncio.gather(c.fundamentals("ACME"), c.fundamentals("BETA"))

    a, b = run(go())
    assert (a["name"], b["name"]) == ("Acme Widgets Ltd", "Beta Gears Ltd")
    times = [t for t, _ in site.log]
    assert len(times) == 4
    assert min(t2 - t1 for t1, t2 in zip(times, times[1:], strict=False)) >= 0.11


def test_concurrent_callers_share_one_fetch(tmp_path):
    site = Site(acme_routes())
    c = client(tmp_path, site)

    async def go():
        return await asyncio.gather(*(c.fundamentals("ACME") for _ in range(3)))

    res = run(go())
    assert site.paths.count("/company/ACME/consolidated/") == 1
    assert sorted(r["cached"] for r in res) == [False, True, True]


def test_one_client_survives_several_event_loops(tmp_path):
    site = Site(acme_routes())
    c = client(tmp_path, site, ttl_hours=0)

    async def burst():
        return await asyncio.gather(*(c.fundamentals("ACME") for _ in range(2)))

    for _ in range(2):                                   # contended locks in two different loops
        assert [f["name"] for f in run(burst())] == ["Acme Widgets Ltd"] * 2


def test_ampersand_symbol_is_url_encoded(tmp_path):
    page = HTML.replace("Acme Widgets Ltd", "M&amp;N Tools Ltd")
    site = Site({"/company/M%26N/consolidated/": (200, page), "/api/company/7700001/peers/": (200, PEERS_HTML)})
    f = run(client(tmp_path, site).fundamentals("m&n"))
    assert f["symbol"] == "M&N" and f["name"] == "M&N Tools Ltd"
    assert site.paths[0] == "/company/M%26N/consolidated/"
    assert (tmp_path / "M%26N.auto.json").exists()
    assert not any("redirected" in w for w in f["warnings"])


def test_auto_falls_back_to_standalone(tmp_path):
    standalone = HTML.replace("Consolidated\n", "Standalone\n")
    site = Site({"/company/ACME/consolidated/": (200, empty_page()), "/company/ACME/": (200, standalone),
                 "/api/company/7700001/peers/": (200, PEERS_HTML)})
    f = run(client(tmp_path, site).fundamentals("ACME"))
    assert f["view"] == "standalone" and f["url"] == "https://www.screener.in/company/ACME/"
    assert "no consolidated statements" in f["warnings"][0]
    assert site.paths == ["/company/ACME/consolidated/", "/company/ACME/", "/api/company/7700001/peers/"]

    site.log.clear()
    g = run(client(tmp_path, site).fundamentals("ACME", view="standalone"))
    assert g["view"] == "standalone" and site.paths == ["/company/ACME/", "/api/company/7700001/peers/"]


def test_short_consolidated_history_prefers_longer_standalone(tmp_path):
    def young(hint):
        return build_page("Young Group", {"profit-loss": ("Profit & Loss", ["Mar 2025", "Mar 2026"], [
            ("Sales +", ["900", "1,000"]), ("OPM %", ["20%", "21%"]), ("Net Profit +", ["90", "100"])])},
            industry="Foods", hint=hint)

    standalone = HTML.replace("Consolidated\n", "Standalone\n")
    site = Site({"/company/YNG/consolidated/": (200, young(True)), "/company/YNG/": (200, standalone),
                 "/api/company/7700001/peers/": (200, PEERS_HTML)})
    f = run(client(tmp_path, site).fundamentals("YNG"))
    assert f["warnings"][0].startswith("Consolidated figures cover only 1.0 years — showing standalone (5.0 years)")
    assert metric(f, "sales_cagr_5y")["value"] is not None and f["view"] == "standalone"
    assert site.paths[:2] == ["/company/YNG/consolidated/", "/company/YNG/"]

    # no nudge from Screener → keep the consolidated view, no extra request
    site = Site({"/company/YNG/consolidated/": (200, young(False)), "/company/YNG/": (200, HTML)})
    g = run(client(tmp_path / "b", site).fundamentals("YNG"))
    assert "/company/YNG/" not in site.paths and g["name"] == "Young Group"
    assert metric(g, "sales_cagr_3y")["value"] is None


def test_peer_failure_is_soft(tmp_path):
    site = Site(acme_routes(**{"/api/company/7700001/peers/": (500, "oops")}))
    f = run(client(tmp_path, site).fundamentals("ACME"))
    assert f["peers"] is None and f["warnings"] == ["Peer comparison unavailable from Screener.in."]
    assert checks(f)["P/E below sector median"]["pass"] is None
    assert site.paths.count("/api/company/7700001/peers/") == 2


def test_redirect_to_new_symbol_is_reported(tmp_path):
    site = Site(acme_routes(**{
        "/company/OLDCO/consolidated/": (301, "", {"location": "https://www.screener.in/company/ACME/consolidated/"})}))
    f = run(client(tmp_path, site).fundamentals("OLDCO"))
    assert f["symbol"] == "OLDCO" and f["url"] == URL
    assert "Screener.in redirected OLDCO to ACME." in f["warnings"]
