"""NeuralEdge: point-in-time fundamentals, news timing, the panel, and (with PyTorch) train → ONNX → serve."""
import asyncio
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from td.config import settings
from td.lake import Lake
from td.neural import dataset as ds
from td.neural import fundamentals as fu
from td.neural import news as nw
from td.research.data import clean_daily
from td.research.screener import parse_company_page
from td.timeutil import IST
from td.universe_data import FNO_STOCKS

FIXTURE = Path(__file__).parent / "fixtures" / "screener_sample.html"


def bar_ts(days: list[date]) -> np.ndarray:
    return np.array([int(datetime(d.year, d.month, d.day, tzinfo=IST).timestamp()) for d in days])


def weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def panel(n_stocks=6, n_days=1100, seed=0):
    rng = np.random.default_rng(seed)
    days = weekdays(date(2022, 6, 1), n_days)
    ts = bar_ts(days)
    mkt = rng.normal(0.0003, 0.01, n_days)
    frames = {}
    for s, *_ in FNO_STOCKS[:n_stocks]:
        r = 0.9 * mkt + rng.normal(0, 0.015, n_days)
        c = 100 * np.exp(np.cumsum(r))
        frames[s] = pd.DataFrame({"ts": ts, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                                  "volume": rng.lognormal(13, 0.3, n_days).astype(np.int64)})
    n = 20000 * np.exp(np.cumsum(mkt))
    idx = pd.DataFrame({"ts": ts, "open": n, "high": n, "low": n, "close": n, "volume": 0})
    return frames, idx


# ------------------------------------------------------------------ fundamentals
def test_fundamentals_only_after_they_were_published():
    f = parse_company_page(FIXTURE.read_text(), "SAMPLE", "https://www.screener.in/company/SAMPLE/", "consolidated")
    evs = fu.events(f)
    assert evs and all(a.ts <= b.ts for a, b in zip(evs, evs[1:]))
    # the June 2025 quarter is public 45 days after 30 June, the March 2026 one (full year) 60 days after 31 March
    jun = next(e for e in evs if e.values.get("q_ym") == 2025 * 12 + 5)
    assert datetime.fromtimestamp(jun.ts, tz=IST).date() == date(2025, 8, 14)
    mar = next(e for e in evs if e.values.get("q_ym") == 2026 * 12 + 2)
    assert datetime.fromtimestamp(mar.ts, tz=IST).date() == date(2026, 5, 30)
    days = [date(2026, 5, 29), date(2026, 6, 1)]
    v = fu.daily_values(f, bar_ts(days), np.array([1000.0, 1000.0]))
    roe = fu.FUND_KEYS.index("a_roe")
    assert np.isnan(v[0, roe]) or v[0, roe] != v[1, roe]          # the FY26 numbers arrive on 30 May, not before
    assert np.isfinite(v[1, roe]) and np.isfinite(v[1, fu.FUND_KEYS.index("book_yield")])


def test_no_fundamentals_is_all_nan():
    v = fu.daily_values(None, bar_ts(weekdays(date(2026, 1, 1), 5)), np.ones(5))
    assert v.shape == (5, len(fu.FUND_KEYS)) and np.isnan(v).all()


# ------------------------------------------------------------------ news
def write_news(root: Path, day: str, items: list[tuple[float, dict]]) -> None:
    d = root / day
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"rt": [t for t, _ in items], "json": [orjson.dumps(i).decode() for _, i in items]}),
                   d / "news-101500.parquet")


def test_news_after_the_close_counts_for_the_next_day(tmp_path):
    days = [date(2026, 10, 19), date(2026, 10, 21)]                # 20 Oct is a holiday: no bar
    ts = bar_ts(days)
    at = lambda d, h, m: datetime(d.year, d.month, d.day, h, m, tzinfo=IST).timestamp()  # noqa: E731
    write_news(tmp_path, "2026-10-19", [
        (at(days[0], 11, 0), {"id": "a", "title": "Acme wins large order", "sentiment": "positive", "symbols": ["ACME"]}),
        (at(days[0], 18, 0), {"id": "b", "title": "Acme CFO resigns", "sentiment": "negative", "symbols": ["ACME"]}),
        (at(days[0], 18, 0), {"id": "b", "title": "Acme CFO resigns", "sentiment": "negative", "symbols": ["ACME"]}),
        (at(days[0], 12, 0), {"id": "c", "title": "Sim headline", "symbols": ["ACME"], "source": "SIM"}),
    ])
    df = nw.load_archive(tmp_path)
    assert len(df) == 2                                             # duplicate and simulator headline dropped
    f = nw.daily_features(df, "ACME", ts, nw.archive_start(df))
    n, sent, on = (nw.NEWS_KEYS.index(k) for k in ("news_n", "news_sent", "news_on"))
    assert f[0, n] == pytest.approx(np.log1p(1.0)) and f[0, sent] > 0      # 19 Oct: only the 11:00 headline
    assert f[1, sent] < f[0, sent]                                  # the evening resignation lands on the 21st
    assert f[:, on].tolist() == [1.0, 1.0]
    before = nw.daily_features(df, "ACME", bar_ts([date(2026, 10, 1)]), nw.archive_start(df))
    assert before[0, on] == 0.0                                     # before the archive: "no archive", not "no news"


# ------------------------------------------------------------------ panel
def test_panel_rows_sequences_and_serving_inputs_line_up():
    frames, idx = panel()
    P = ds.build(frames, idx)
    assert P.rows > 1000 and len(P.folds) >= 4 and P.static.shape[1] == len(P.static_keys)
    seq = P.seq(np.array([0, P.rows - 1]))
    assert seq.shape == (2, ds.SEQ_LEN, len(ds.TECH_KEYS)) and np.isfinite(seq).all()
    # the row's own features are the last step of its sequence
    assert np.allclose(seq[1, -1], P.zrow[P.rows - 1])
    L = ds.build(frames, idx, hmm=P.hmm, sectors=P.sectors, last_only=True)
    assert L.rows == len(frames) and L.static_keys == P.static_keys
    last_day = P.row_cal == P.row_cal.max()
    for i in range(L.rows):                                         # serving == the training row of the same day
        s = L.row_sym[i]
        j = np.flatnonzero(last_day & (P.row_sym == P.syms.index(L.syms[s])))[0]
        assert np.allclose(L.static[i], P.static[j]) and np.allclose(L.seq(np.array([i]))[0], P.seq(np.array([j]))[0])


def test_panel_needs_nifty():
    frames, _ = panel(3, 900)
    with pytest.raises(ValueError, match="NIFTY"):
        ds.build(frames, None)


# ------------------------------------------------------------------ server without a model
def test_runtime_reports_a_missing_model(tmp_path, monkeypatch):
    from td.neural.runtime import NeuralService
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    info = asyncio.run(NeuralService(data=None).info())
    assert info["status"] == "missing" and info["folder"].endswith("neural")


# ------------------------------------------------------------------ train → ONNX → serve (needs PyTorch)
def test_train_export_and_serve(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from td.neural import train as tr
    from td.neural.runtime import NeuralService
    frames, idx = panel(6, 1100, seed=2)
    P = ds.build(frames, idx)
    ens, meta = tr.train(P, tr.TrainConfig(seeds=1, epochs=1, batch=512, gbm=False, device="cpu"))
    assert [h["h"] for h in meta["horizons"]] == list(ds.HORIZONS)
    assert all("neural" in h["metrics"] and "quantedge" in h["metrics"] for h in meta["horizons"])
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "data_source", "dhan")
    onnx_p, meta_p = tr.export(ens, P, meta, tmp_path / "neural")
    assert json.loads(meta_p.read_text())["onnx_check_max_diff"] < 1e-4
    lake = Lake(tmp_path / "lake" / "dhan")
    for s, df in frames.items():
        lake.write("D", s, df)
    lake.write("D", "NIFTY", idx)

    class Data:
        def __init__(self):
            self.lake = lake

        async def daily(self, s, fetch=False):
            return clean_daily(lake.read("D", s)), {}

        async def index_daily(self, name):
            return clean_daily(lake.read("D", name))

    async def go():
        svc = NeuralService(Data())
        assert (await svc.info())["status"] == "computing"
        await svc._predictions(wait=True)
        sym = sorted(frames)[0]
        r = await svc.result(sym)
        assert r["status"] == "ready" and len(r["horizons"]) == 3
        assert all(0 < h["p_up"] < 1 for h in r["horizons"])
        lb = await svc.leaderboard(5, 3)
        assert lb["n"] == len(frames) and (lb["ranked"] or (lb["top"] == [] and lb["edge_note"]))
    asyncio.run(go())
