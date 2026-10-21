"""The 09:30 model: 1-minute day tables, the panel, honest walk-forward vs the TOP5 ranker, ONNX, live scoring."""
import json
import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from td.analytics import intraday as rk
from td.analytics.context import load_context
from td.lake import Lake
from td.neural import intraday as it
from td.neural import minute as mn
from td.timeutil import IST
from td.universe_data import FNO_STOCKS, SECTOR_OF


def weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def midnight(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=IST).timestamp())


# ------------------------------------------------------------------ the minute table
def minute_frame(days: list[date], seed=0, skip=()):
    """Full 1-minute sessions (375 bars) for `days`; minutes in `skip` (day index, minute) are left out."""
    rng = np.random.default_rng(seed)
    rows = []
    px = 100.0
    for di, d in enumerate(days):
        t0 = midnight(d) + 9 * 3600 + 15 * 60
        for k in range(mn.SESSION):
            o = px
            px = px * math.exp(rng.normal(0, 0.001))
            if (di, k) in skip:
                continue
            rows.append((t0 + 60 * k, o, max(o, px) * 1.0005, min(o, px) * 0.9995, px, 1000 + 10 * k + di))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


def test_day_table_lock_price_close_and_usual_volume():
    days = weekdays(date(2026, 9, 1), 17)
    m = minute_frame(days, skip={(16, 14), (16, 3)})
    t = mn.day_table(m)
    assert list(t.columns) == mn.COLUMNS and len(t) == len(days)
    last = t.iloc[-1]
    day_bars = m[m["ts"] >= midnight(days[-1])].reset_index(drop=True)
    # 09:29 is missing on the last day: the 09:30 price is the 09:28 close
    assert last["px_lock"] == pytest.approx(day_bars["close"].iat[12])
    assert last["close"] == pytest.approx(day_bars["close"].iat[-1])
    assert last["pre"] == 13 and np.isnan(last["c14"]) and np.isnan(last["o3"])
    # usual volume: positional mean over the previous 14 full sessions — what load_context builds for the live ranker
    lake = Lake(None)
    exp = np.mean([[1000 + 10 * k + di for k in range(15)] for di in range(2, 16)], axis=0)
    assert np.allclose(last[mn.BASE_COLS].to_numpy(dtype=float), exp)
    assert np.isnan(t.iloc[0][mn.BASE_COLS].to_numpy(dtype=float)).all()      # nothing before the first day
    assert np.isfinite(last[list(mn.DAY_KEYS)].to_numpy(dtype=float)).all()
    assert last["m_late_vol"] == pytest.approx(day_bars["volume"].iloc[-30:].sum() / day_bars["volume"].sum(), rel=1e-5)
    del lake


def test_usual_volume_matches_the_live_context(tmp_path):
    days = weekdays(date(2026, 8, 3), 20)
    m = minute_frame(days, seed=3)
    lake = Lake(tmp_path)
    lake.write("1m", "ACME", m)
    lake.write("D", "ACME", pd.DataFrame({"ts": [midnight(d) for d in days], "open": 100.0, "high": 101.0,
                                          "low": 99.0, "close": 100.0, "volume": 375_000}))
    ctx, _ = load_context(lake, ["ACME"], days[-1])
    live = it.base_from_curve(ctx["ACME"].base_cum, ctx["ACME"].adv)
    t = mn.day_table(m)
    assert np.allclose(t.iloc[-1][mn.BASE_COLS].to_numpy(dtype=float), live)


# ------------------------------------------------------------------ a synthetic market with 1-minute tables
def market(n_stocks=30, n_days=900, n_intraday=520, seed=0, signal=0.0):
    """Daily bars for every day and day tables for the last `n_intraday` days, built from one path per stock-day:
    gap, the first 15 one-minute bars, the rest of the day. With `signal`, the stock's move after 09:30 continues its
    first-15-minute move (relative to the market) when its 1-month trend is up and reverses it when down — a pattern
    that needs the daily context, which the TOP5 ranker never looks at."""
    rng = np.random.default_rng(seed)
    days = weekdays(date(2022, 1, 3), n_days)
    syms = [s for s, *_ in FNO_STOCKS[:n_stocks]]
    first = n_days - n_intraday
    mk_gap = rng.normal(0, 0.004, n_days)
    mk_f = rng.normal(0, 0.003, (n_days, 15))
    mk_rest = rng.normal(0.0002, 0.006, n_days)
    frames, tables = {}, {}

    def day_rows(name, px0, gap, fpath, rest, vol0, k):
        """One day of one instrument → (daily bar, table row)."""
        o = px0 * math.exp(gap)
        steps = np.r_[0.0, np.cumsum(fpath)]
        c = o * np.exp(steps[1:])
        op = o * np.exp(steps[:-1])
        hi = np.maximum(op, c) * 1.0007
        lo = np.minimum(op, c) * 0.9993
        v = vol0 * np.exp(rng.normal(0, 0.3, 15)) * np.linspace(2.0, 1.0, 15)
        close = c[-1] * math.exp(rest)
        ph = max(c[-1], close) * 1.003
        pl = min(c[-1], close) * 0.997
        row = {"day": mn.ist_day(midnight(days[k])), "n": 375, "pre": 15, "post": 360}
        for i in range(15):
            row[f"o{i}"], row[f"h{i}"], row[f"l{i}"], row[f"c{i}"], row[f"v{i}"] = op[i], hi[i], lo[i], c[i], v[i]
        row.update({"open": o, "px_lock": c[-1], "close": close, "post_high": ph, "post_low": pl,
                    "vol_day": v.sum() * 8})
        for key in mn.DAY_KEYS:
            row[key] = rng.normal()
        bar = (midnight(days[k]), o, max(hi.max(), ph), min(lo.min(), pl), close, int(v.sum() * 8))
        return bar, row

    for s in syms + ["NIFTY"]:
        is_idx = s == "NIFTY"
        beta = 1.0 if is_idx else rng.uniform(0.7, 1.3)
        px = 20000.0 if is_idx else rng.uniform(200, 2000)
        vol0 = 0 if is_idx else rng.uniform(2e3, 2e4)
        bars, rows = [], []
        closes = []
        for k in range(n_days):
            g = beta * mk_gap[k] + (0 if is_idx else rng.normal(0, 0.006))
            fp = beta * mk_f[k] + (0 if is_idx else rng.normal(0, 0.0015, 15))
            rest = beta * mk_rest[k] + (0 if is_idx else rng.normal(0, 0.008))
            if signal and not is_idx and k > 30:
                trend = math.log(closes[-1] / closes[-22])
                rel = float(fp.sum() - beta * mk_f[k].sum())
                rest += signal * (1 if trend > 0 else -1) * rel
            bar, row = day_rows(s, px, g, fp, rest, vol0, k)
            px = bar[4]
            closes.append(px)
            bars.append(bar)
            if k >= first:
                rows.append(row)
        frames[s] = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
        t = pd.DataFrame(rows)
        base = t[[f"v{i}" for i in range(15)]].rolling(14, min_periods=1).mean().shift(1)
        for i in range(15):
            t[f"b{i}"] = base[f"v{i}"] if not is_idx else 0.0
        tables[s] = t[mn.COLUMNS].copy()
    idx = frames.pop("NIFTY")
    return frames, idx, tables


@pytest.fixture(scope="module")
def noise_panel():
    frames, idx, tables = market(seed=1)
    return it.build(frames, idx, tables)


def test_panel_rows_line_up_with_the_ranker_and_the_daily_context(noise_panel):
    IP = noise_panel
    assert IP.rows > 5000 and len(IP.folds) >= 4
    assert IP.bars.shape == (IP.rows, 15, len(it.BAR_KEYS)) and IP.intra.shape == (IP.rows, len(it.INTRA_KEYS))
    assert np.isfinite(IP.bars).all() and np.isfinite(IP.intra).all() and np.isfinite(IP.fwd).all()
    # each row's daily context is the PREVIOUS trading day
    prev_day = mn.ist_day(IP.daily.row_ts[IP.drow])
    assert (prev_day < IP.row_day).all() and (IP.row_day - prev_day <= 4).all()
    # beat = return minus the day's median: half the rows per day beat it
    beat = (IP.fwd[:, 1] > 0).astype(float)
    assert 0.45 < beat.mean() < 0.55
    # the HMM was fitted before the first test day
    assert IP.daily.hmm_end_ts < mn.day_of_epoch(int(IP.cal_days[IP.folds[0][0]]))
    # ranker picks come from the live ranker code: at most TOP_N each side, never both sides for a stock
    for c in np.unique(IP.row_cal)[:50]:
        r = IP.row_cal == c
        assert IP.ranker[r, 3].sum() <= rk.RankConfig().top_n and IP.ranker[r, 4].sum() <= rk.RankConfig().top_n
        assert not ((IP.ranker[r, 3] > 0) & (IP.ranker[r, 4] > 0)).any()


def test_day_features_match_a_live_ranker_run():
    """day_features' ranker numbers are those of td.analytics.intraday.rank on the same 09:30 states."""
    frames, idx, tables = market(n_stocks=25, n_days=400, n_intraday=40, seed=4)
    k = -1
    syms = sorted(frames)
    t = {s: tables[s].iloc[k] for s in syms}
    ctx = {s: it.daily_context(frames[s].iloc[:k], idx.iloc[:k]) for s in syms}
    nctx = it.daily_context(idx.iloc[:k], None)
    bars = np.stack([mn.first_bars(tables[s].iloc[[k]])[0] for s in syms])
    di = it.DayInputs(syms=syms, bars=bars, base=np.stack([t[s][mn.BASE_COLS].to_numpy(dtype=float) for s in syms]),
                      pdc=np.array([ctx[s]["close"][-1] for s in syms]), atr=np.array([ctx[s]["atr"][-1] for s in syms]),
                      adv=np.array([ctx[s]["adv"][-1] for s in syms]), beta=np.array([ctx[s]["beta"][-1] for s in syms]),
                      nifty=mn.first_bars(tables["NIFTY"].iloc[[k]])[0], nifty_pdc=float(nctx["close"][-1]),
                      nifty_atr=float(nctx["atr"][-1]), news=np.zeros((len(syms), 3)))
    res = it.day_features(di, rk.RankConfig())
    assert res["ok"].all() and res["intra"].shape == (len(syms), len(it.INTRA_KEYS))
    states, _ = it._states(di)
    rows = rk.rank(states, {}, (float(tables["NIFTY"]["px_lock"].iat[k]) / di.nifty_pdc - 1) * 100, 15.0, {}, rk.RankConfig(), 15)
    by = {r["symbol"]: r for r in rows}
    z_resid = it.INTRA_KEYS.index("z_resid")
    for i, s in enumerate(res["sym"]):
        assert res["ranker"][i, 0] == pytest.approx(by[s]["score"])
        assert res["intra"][i, z_resid] == pytest.approx(by[s]["f"]["resid"] / 3, abs=1e-6)
    # the bars are in ATR units of the previous close: the 09:29 close equals chg_atr
    c = it.BAR_KEYS.index("c")
    assert np.allclose(res["bars"][:, -1, c], res["intra"][:, it.INTRA_KEYS.index("chg_atr")], atol=1e-5)


def test_top_bottom_respects_liquidity_and_sector_caps():
    score = np.arange(12, dtype=float)[::-1]
    liquid = np.ones(12, bool)
    liquid[1] = False
    sector = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"])
    lo, sh = it.top_bottom(score, liquid, sector, n=3, per_sector=2)
    assert lo.tolist() == [0, 2, 3]                     # 1 is illiquid; A capped at two
    assert sh.tolist() == [11, 10, 8] and not set(lo) & set(sh)


# ------------------------------------------------------------------ training (PyTorch)
def _cfg(**kw):
    from td.neural import train as tr
    base = dict(seeds=1, epochs=6, patience=2, batch=512, gbm=False, device="cpu")
    return tr.TrainConfig(**{**base, **kw})


def test_noise_has_no_edge_and_no_picks(noise_panel, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from td.neural import train_intraday as ti
    from td.neural.live import LiveIntraday
    ens, meta = ti.train(noise_panel, _cfg(gbm=True))
    assert meta["edge"] is False and "no neural picks" in meta["verdict"]
    assert {h["head"] for h in meta["heads"]} == set(it.HEADS)
    assert set(meta["trading"]) >= {"neural", "ranker", "linear", "gbm", "blend", "selected"}
    dep = meta["deploy"]
    assert dep["model"] in ti.DEPLOYABLE and dep["judged_on"] == "selected"
    assert [t["fold"] for t in dep["selection"]] == list(range(2, len(noise_panel.folds) + 1))
    assert (dep["gbm_file"] is not None) == (dep["model"] in ("gbm", "blend"))
    files = ti.export(ens, noise_panel, meta, tmp_path)
    assert len(files) == (3 if dep["gbm_file"] else 2)
    saved = json.loads((tmp_path / "intraday.json").read_text())
    assert "_gbm" not in saved and saved["deploy"]["model"] == dep["model"]
    nn = LiveIntraday(folder=tmp_path)
    nn.load()
    assert nn.error is None and nn.state == "no_edge" and (nn.gbm is not None) == bool(dep["gbm_file"])


def test_tree_export_matches_sklearn():
    from sklearn.ensemble import HistGradientBoostingClassifier

    from td.neural import trees
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 30)).astype(np.float32)
    X[::13, 4] = np.nan                                         # missing values follow the learnt direction
    y = (np.nan_to_num(X[:, 0]) + 0.6 * X[:, 1] * X[:, 2] + rng.normal(size=4000) > 0).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
                                         early_stopping=False, random_state=0).fit(X, y)
    m = json.loads(json.dumps(trees.export_hgb(clf)))           # through JSON, as on the server
    assert np.abs(trees.predict_proba(m, X) - clf.predict_proba(X)[:, 1]).max() < 1e-9
    with pytest.raises(ValueError, match="inputs"):
        trees.predict_proba(m, X[:, :10])


def test_best_model_is_chosen_on_earlier_folds_only(noise_panel):
    """The choosing rule is walked forward: a model that only shone in fold 1 is chosen for fold 2 (and its bad
    fold 2 counts against the rule), then the steadily better one takes over."""
    from td.neural import train_intraday as ti
    IP = noise_panel
    rng = np.random.default_rng(1)
    preds = {"neural": [], "gbm": [], "blend": []}
    for k, (s, e) in enumerate(IP.folds):
        te = np.flatnonzero((IP.row_cal >= s) & (IP.row_cal < e))
        r = IP.ret_pct[te]
        good, bad, mid = r, -r, r + rng.normal(0, 3 * r.std(), r.size)
        sig = lambda v: 1 / (1 + np.exp(-v))                     # noqa: E731
        base = np.array([0.5, 0.5])
        preds["neural"].append((k, te, np.column_stack([sig(good if k == 0 else bad)] * 2), base))
        preds["gbm"].append((k, te, np.column_stack([sig(mid if k == 0 else good)] * 2), base))
        preds["blend"].append((k, te, np.column_stack([sig(rng.normal(size=r.size))] * 2), base))
    sel, trail, final, ls = ti.select(IP, preds)
    assert [t["chosen"] for t in trail] == ["neural"] + ["gbm"] * (len(IP.folds) - 2)
    assert final == "gbm" and ls["neural"][1] < 0 < ls["gbm"][1]
    assert len(sel["tests"]) == len(IP.folds) - 1 and sel["parts"]["beat"][0][2] == 1   # fold 2 = neural's bad one
    single, trail1, final1, _ = ti.select(IP, {"neural": preds["neural"]})
    assert trail1 == [] and final1 == "neural" and not single["tests"]


def test_live_blend_averages_the_network_and_the_trees(noise_panel):
    from sklearn.ensemble import HistGradientBoostingClassifier

    from td.neural import trees
    from td.neural.live import LiveIntraday
    from td.neural.train_intraday import IntradayTask, flat
    IP = noise_panel
    rows = np.arange(0, IP.rows, 7)
    X = flat(IP, rows)
    gbm = {h: trees.export_hgb(HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(X, (IP.fwd[rows, i] > 0)))
           for i, h in enumerate(it.HEADS)}
    nn = LiveIntraday()
    nn.sess, nn.gbm = _Sess(), gbm
    nn.meta = {"deploy": {"model": "blend"}}
    x = IntradayTask(IP).inputs(rows[:50])
    want = (0.5 + np.column_stack([trees.predict_proba(gbm[h], X[:50]) for h in it.HEADS])) / 2
    assert np.allclose(nn.raw(*x), want)
    nn.meta = {"deploy": {"model": "gbm"}}
    assert np.allclose(nn.raw(*x), want * 2 - 0.5)


def test_planted_pattern_is_found_and_beats_the_ranker(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from td.neural import train_intraday as ti
    frames, idx, tables = market(seed=5, signal=1.5)
    IP = it.build(frames, idx, tables)
    ens, meta = ti.train(IP, _cfg(epochs=6, search=2))
    beat = next(h for h in meta["heads"] if h["head"] == "beat")["metrics"]
    assert beat["neural"]["auc"] > 0.58 and beat["neural"]["auc"] > beat["ranker"]["auc"] + 0.03
    nn, rkr = meta["trading"]["neural"], meta["trading"]["ranker"]
    assert nn["ls_avg"] > 0 and nn["ic"] > rkr["ic"]
    assert meta["edge"] is True and meta["verdict"].startswith("beats")
    assert len(meta["search"]["chosen_per_fold"]) == len(IP.folds) and meta["search"]["candidates"] == 2
    files = ti.export(ens, IP, meta, tmp_path)
    assert json.loads(files[1].read_text())["onnx_check_max_diff"] < 1e-4


# ------------------------------------------------------------------ live 09:30 scoring == training rows
class _Sess:
    """Stands in for onnxruntime: records the inputs, answers 0.5."""
    def __init__(self):
        self.inputs = None

    def run(self, names, feeds):
        self.inputs = feeds
        return [np.full((len(feeds["seq"]), 2), 0.5, dtype=np.float32)]


def test_live_scoring_feeds_the_network_exactly_the_training_inputs(noise_panel):
    from td.analytics.state import HistoryContext, InstState
    from td.neural.live import LiveIntraday
    from td.neural.train_intraday import IntradayTask
    frames, idx, tables = market(seed=1)                    # the market noise_panel was built from
    IP = noise_panel
    c = int(IP.row_cal.max())
    rows = np.flatnonzero(IP.row_cal == c)
    dday = int(IP.cal_days[c])
    day = date(1970, 1, 1) + timedelta(days=dday)

    def state(sym, t, ctxd, j, sector):
        st = InstState(hash(sym) % 10_000, "NSE_EQ", sym, "EQ", sym, sector)
        row = t[t["day"] == dday].iloc[0]
        b = mn.first_bars(t[t["day"] == dday])[0]
        st.bars = {m: list(b[m]) for m in range(15) if np.isfinite(b[m, 3])}
        h = HistoryContext()
        h.pdc, h.atr, h.adv, h.beta = (float(ctxd[k][j]) for k in ("close", "atr", "adv", "beta"))
        cum = np.cumsum(np.nan_to_num(row[mn.BASE_COLS].to_numpy(dtype=float)))
        h.base_cum = list(cum) + [cum[-1]]
        st.ctx, st.pc, st.ltp = h, h.pdc, float(row["px_lock"])
        st.vol = int(np.nansum(b[:, 4]))
        return st

    eqs, live_rows = [], {}
    for r in rows:
        sym = IP.syms[IP.row_sym[r]]
        ctxd = it.daily_context(frames[sym], idx)
        j = int(np.searchsorted(ctxd["day"], dday)) - 1
        eqs.append(state(sym, tables[sym], ctxd, j, SECTOR_OF.get(sym, "OTHER")))
        d = IP.drow[r]
        live_rows[sym] = (IP.daily.seq(np.array([d]))[0], IP.daily.static[d], int(mn.ist_day(IP.daily.row_ts[d])))
    nctx = it.daily_context(idx, None)
    nifty = state("NIFTY", tables["NIFTY"], nctx, int(np.searchsorted(nctx["day"], dday)) - 1, "INDEX")
    nn = LiveIntraday()
    nn.meta = {"heads": [{"calibration": {"base": 0.5, "b": 1.0}}] * 2, "ranker_weights": IP.weights, "edge": True,
               "top_n": 5, "per_sector": 2, "trained_at": 0}
    nn.sess = _Sess()
    nn.ctx = {"day": day.isoformat(), "rows": live_rows}
    res = nn.score(day, eqs, nifty, np.zeros((len(eqs), 3)), 5.0)
    assert res["status"] == "ok" and 1 <= len(res["long"]) <= 5 and 1 <= len(res["short"]) <= 5
    got = nn.sess.inputs
    want = IntradayTask(IP).inputs(rows)
    order = [list(res["p"]).index(IP.syms[IP.row_sym[r]]) for r in rows]
    for name, w in zip(IntradayTask.input_names, want):
        assert np.allclose(got[name][order], w, atol=1e-5), name


# ------------------------------------------------------------------ the analytics worker: lock → track → scorecard
class _FakeNN:
    state = "ready"

    def __init__(self, longs, shorts):
        self.longs, self.shorts = longs, shorts
        self.ctx = {"rows": {}}

    def load(self):
        pass

    def ready_for(self, day):
        return True

    async def prepare(self, day):
        pass

    def summary(self):
        return {"status": self.state, "verdict": "beats the TOP5 ranker", "edge": True, "trained_at": 1.0}

    def score(self, day, eqs, nifty, news, min_turnover_cr):
        p = {s.symbol: [0.5, 0.9 if s.symbol in self.longs else 0.1 if s.symbol in self.shorts else 0.5] for s in eqs}
        return {**self.summary(), "status": "ok", "long": self.longs, "short": self.shorts, "p": p, "n": len(eqs),
                "skipped": 0}


def test_engine_locks_tracks_and_scores_the_neural_picks(monkeypatch, tmp_path):
    import asyncio

    import fakeredis.aioredis

    from td import bus
    from td.analytics.state import InstState
    from td.config import settings
    from td.services import analytics as an
    monkeypatch.setattr(settings, "data_source", "dhan")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "nse_holidays", "")
    e = an.Engine(fakeredis.aioredis.FakeRedis())
    for i in range(30):
        s = InstState(i + 1, "NSE_EQ", f"S{i}", "EQ", f"S{i}", f"SEC{i % 6}")
        s.pc, s.open, s.ltp, s.vol, s.atp = 100.0, 100.0, 100.5 + 0.05 * i, 10_000_000, 100.2
        s.high, s.low, s.orh, s.orl = s.ltp + 0.2, 99.8, s.ltp + 0.2, 99.8
        e.eqs.append(s)
        e.eq_by_symbol[s.symbol] = s
        e.states[s.sid] = s
    e.rows = rk.rank(e.eqs, {}, 0.0, 15.0, {}, e.cfg)
    e.nn = _FakeNN(["S29", "S28"], ["S0"])
    day = date(2026, 10, 21)
    e.clock = datetime(2026, 10, 21, 9, 30, 5, tzinfo=IST).timestamp()
    asyncio.run(e.maybe_lock())
    nn = e.top["neural"]
    assert e.top["locked"] and nn["status"] == "ok" and [p["symbol"] for p in nn["long"]] == ["S29", "S28"]
    assert nn["long"][0]["p_beat"] == 0.9 and nn["long"][0]["entry"] == e.eq_by_symbol["S29"].ltp
    assert nn["short"][0]["target"] < nn["short"][0]["entry"] < nn["short"][0]["invalidation"]
    saved = bus.loads(asyncio.run(e.r.get(bus.k_top(day.isoformat()))))
    assert saved["neural"]["long"][0]["symbol"] == "S29"                      # restored after a restart
    for i, s in enumerate(e.eqs):                                             # the afternoon: S29 +2 %, S0 −2 %
        s.ltp = s.lock_px * (1.02 if s.symbol == "S29" else 0.98 if s.symbol == "S0" else 1 + 0.0001 * i)
    v = e.top_view()["neural"]
    assert "p" not in v and v["long"][0]["pnl"] == pytest.approx(2.0) and v["short"][0]["pnl"] == pytest.approx(2.0)
    assert v["since_lock"]["long_avg"] == pytest.approx((2.0 + 0.28) / 2, abs=0.01)
    e.clock = datetime(2026, 10, 21, 15, 31, tzinfo=IST).timestamp()
    asyncio.run(e.maybe_scorecard())
    card = bus.loads(asyncio.run(e.r.lindex(bus.KEY_SCORECARD, 0)))
    assert card["neural"]["picks"] == ["S29", "S28"] and card["neural"]["short_avg"] == pytest.approx(2.0)
    assert card["neural"]["ic"] > 0
    from td.services.gateway import _neural_summary
    ns = _neural_summary([card])
    assert ns["days"] == 1 and ns["neural"]["avg_long"] == pytest.approx(1.14, abs=0.01) and "top5" in ns


def test_engine_without_a_model_or_edge_makes_no_neural_picks(monkeypatch, tmp_path):
    import asyncio

    import fakeredis.aioredis

    from td.config import settings
    from td.services import analytics as an
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    e = an.Engine(fakeredis.aioredis.FakeRedis())
    assert asyncio.run(e._neural_picks(date(2026, 10, 21)))["status"] == "missing"
    nn = _FakeNN([], [])
    nn.state = "no_edge"
    e.nn = nn
    out = asyncio.run(e._neural_picks(date(2026, 10, 21)))
    assert out["status"] == "no_edge" and "long" not in out
