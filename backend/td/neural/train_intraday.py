"""Training the 09:30 → close model: walk-forward over trading days against the live TOP5 ranker, settings search,
calibration, final fit, ONNX export (intraday.onnx + intraday.json).

Same honesty rules as the daily model (td.neural.train):
- Test folds are blocks of trading days; a fold is predicted by networks trained only on days that end PURGE days
  before it. The daily context's HMM was fitted before the first test day (td.neural.intraday.build).
- Early stopping and the settings search use a validation slice from the end of each training window only.
- Four models are scored on identical test rows:
    neural   this network (mean of `seeds` trained copies)
    ranker   the live TOP5 ranker (td.analytics.intraday) — its score for the AUC/IC, its actual picks (eligibility,
             direction filters, sector caps, abstaining) for the trading numbers
    linear   an L2 logistic regression on the same inputs, flattened
    gbm      gradient-boosted trees on the same flattened inputs (optional)
- Two heads: "up" (09:30 → close > 0) and "beat" (beats the day's median stock). The picks use "beat": the TOP5 is
  a cross-sectional bet, and the market's own direction is a coin toss the network should not be paid for.
- Trading numbers per test day: the 5 best / 5 worst liquid stocks (≤ 2 per sector), returns from the 09:30 price
  to the close net of COST_PCT round trip, the long/short average and its t-statistic over days, and the daily rank
  IC. The model counts as having an edge only if its "beat" head passes the rank-IC test (td.neural.train.
  rank_metrics: IC ≥ 0.02, 2 block-bootstrap SEs above zero, positive in every fold) AND its long/short return is
  positive with t ≥ 2 — two different ways of being right, both required.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..research import quant as qm
from . import dataset as ds
from . import intraday as it
from . import minute as mn
from . import train as tr

log = logging.getLogger("td.neural.train_intraday")

MODEL_NAME = "NeuralEdge-0930"
MODEL_VERSION = 1
MODELS = ("neural", "ranker", "linear", "gbm")
INTRA_DEFAULTS = {"bar_hidden": 32}
INTRA_SPACE = {"bar_hidden": (16, 32, 48)}
EDGE_T = 2.0


class IntradayTask:
    input_names = ("seq", "static", "bars", "intra")

    def __init__(self, IP: it.IPanel):
        self.IP = IP
        self.row_cal = IP.row_cal
        self.purge = it.PURGE
        self.n_heads = len(it.HEADS)

    def inputs(self, idx: np.ndarray) -> list[np.ndarray]:
        d = self.IP.drow[idx]
        return [self.IP.daily.seq(d), self.IP.daily.static[d], self.IP.bars[idx], self.IP.intra[idx]]

    def net(self, hp: dict):
        from .model import IntradayNet
        P = self.IP.daily
        return IntradayNet(P.Z.shape[1], P.static.shape[1], len(it.BAR_KEYS), len(it.INTRA_KEYS), self.n_heads, **hp)


def flat(IP: it.IPanel, idx: np.ndarray) -> np.ndarray:
    """The inputs as one row of numbers, for the linear and tree baselines."""
    d = IP.drow[idx]
    b = IP.bars[idx]
    return np.hstack([IP.daily.zrow[d], IP.daily.static[d], IP.intra[idx], b[:, -1, :], b[:, :, 3], b[:, :, 4]]
                     ).astype(np.float32)


def _linear(X: np.ndarray, tr_: np.ndarray, te: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xt = X[tr_].astype(np.float64)
    mu, sd = Xt.mean(axis=0), Xt.std(axis=0)
    sd[sd < 1e-9] = 1.0
    b0, w, _ = qm.logistic_fit((Xt - mu) / sd, y[tr_], qm.ALPHA0)
    return qm._sigmoid(b0 + ((X[te] - mu) / sd) @ w)


# --------------------------------------------------------------------------- trading numbers
def trading(IP: it.IPanel, rows: np.ndarray, score: np.ndarray | None, ranker_picks: bool = False) -> dict:
    """Day-by-day picks on test rows and what they returned (net of costs). `score` aligned with `rows`."""
    cal = IP.row_cal[rows]
    order = np.argsort(cal, kind="stable")
    rows, cal = rows[order], cal[order]
    sc = score[order] if score is not None else None
    cuts = np.flatnonzero(np.r_[True, cal[1:] != cal[:-1], True])
    longs, shorts, lsr, ics, green_l, green_s, n_l, n_s, uni, top20 = [], [], [], [], 0, 0, 0, 0, [], 0
    for a, b in zip(cuts[:-1], cuts[1:]):
        r = rows[a:b]
        ret = IP.ret_pct[r]
        uni.append(float(ret.mean()))
        if ranker_picks:
            li = np.flatnonzero(IP.ranker[r, 3] > 0)
            si = np.flatnonzero(IP.ranker[r, 4] > 0)
            s_day = IP.ranker[r, 0]
        else:
            s_day = sc[a:b]
            li, si = it.top_bottom(s_day, IP.liquid[r], IP.sector[r])
        if b - a >= 10:
            v = qm._spearman(np.asarray(s_day, dtype=float), ret)
            if math.isfinite(v):
                ics.append(v)
        lr = float(ret[li].mean()) - it.COST_PCT if li.size else None
        sr = float(-ret[si].mean()) - it.COST_PCT if si.size else None
        if lr is not None:
            longs.append(lr)
            green_l += int((ret[li] > it.COST_PCT).sum())
            n_l += li.size
            top = set(np.argsort(-ret)[:20].tolist())
            top20 += sum(1 for i in li if i in top)
        if sr is not None:
            shorts.append(sr)
            green_s += int((-ret[si] > it.COST_PCT).sum())
            n_s += si.size
        if lr is not None and sr is not None:
            lsr.append((lr + sr) / 2)
    days = len(cuts) - 1

    def mt(x):
        a = np.asarray(x, dtype=float)
        if a.size < 3 or a.std(ddof=1) == 0:
            return None
        return round(float(a.mean() / a.std(ddof=1) * math.sqrt(a.size)), 2)
    f = lambda x, nd=3: round(float(np.mean(x)), nd) if len(x) else None  # noqa: E731
    return {"days": days, "long_days": len(longs), "short_days": len(shorts),
            "long_avg_net": f(longs), "short_avg_net": f(shorts), "ls_avg": f(lsr), "ls_t": mt(lsr),
            "long_green_pct": round(100 * green_l / n_l, 1) if n_l else None,
            "short_green_pct": round(100 * green_s / n_s, 1) if n_s else None,
            "long_top20_pct": round(100 * top20 / n_l, 1) if n_l else None,
            "ic": f(ics, 4), "ic_t": mt(ics), "ic_pos_pct": round(100 * float(np.mean(np.array(ics) > 0)), 1) if ics else None,
            "universe_avg": f(uni), "cost_pct": it.COST_PCT}


def has_edge(beat: dict | None, trade: dict | None) -> tuple[bool, str | None]:
    if not beat or not beat.get("edge"):
        return False, (beat or {}).get("edge_note") or "no reliable edge in the walk-forward"
    t = (trade or {}).get("ls_t")
    avg = (trade or {}).get("ls_avg")
    if t is None or avg is None or avg <= 0 or t < EDGE_T:
        return False, (f"ranks stocks better than chance (IC {beat.get('ic') or 0:+.3f}) but its picks did not make money "
                       f"reliably after costs (long/short {avg if avg is not None else float('nan'):+.3f}% a day, "
                       f"t = {t if t is not None else float('nan'):.1f}; needs t ≥ {EDGE_T:g})")
    return True, None


def verdict(trade: dict[str, dict], edge: bool) -> str:
    nn, rk = trade.get("neural") or {}, trade.get("ranker") or {}
    if not edge:
        return "no reliable edge — no neural picks are made; the TOP5 ranker stays in charge"
    a, b = nn.get("ls_avg"), rk.get("ls_avg")
    if a is None or b is None:
        return "has an edge (the ranker made no complete long/short days to compare)"
    if a is not None and a > b and (nn.get("ic") or 0) > (rk.get("ic") or 0):
        return f"beats the TOP5 ranker (long/short {a:+.3f}% vs {b:+.3f}% a day, IC {nn['ic']:+.3f} vs {rk['ic']:+.3f})"
    return f"has an edge but does not beat the TOP5 ranker (long/short {a:+.3f}% vs {b:+.3f}% a day)"


# --------------------------------------------------------------------------- the run
def walk_forward(IP: it.IPanel, cfg: tr.TrainConfig, dev, say: tr.Say):
    task = IntradayTask(IP)
    cands = tr.candidates(cfg, INTRA_DEFAULTS, INTRA_SPACE)
    y = (IP.fwd > 0).astype(np.float32)
    m = np.ones_like(y)
    out = {name: {h: [] for h in it.HEADS} for name in MODELS}
    tests: dict[str, list] = {name: [] for name in MODELS}       # (rows, beat score) per fold for trading
    epochs, chosen = [], []
    nf = len(IP.folds)
    X = None
    for k, (s, e) in enumerate(IP.folds):
        te = np.flatnonzero((IP.row_cal >= s) & (IP.row_cal < e) & (IP.row_cal > IP.row_hmm_cal))
        cut = s - it.PURGE - 1
        trn = np.flatnonzero((IP.row_cal <= cut) & (IP.row_hmm_cal <= cut))
        if trn.size < 2000 or te.size < 200:
            log.info("fold %d skipped (%d train / %d test rows)", k + 1, trn.size, te.size)
            continue
        say(0.05 + 0.75 * k / nf, f"fold {k + 1}/{nf}: training on {trn.size:,} rows ({IP.date(s)} → {IP.date(e - 1)})")
        fit_i, val_i = tr.split_val(task, trn, cfg.val_frac)
        nets, hp, eps, trials = tr.fit_ensemble(
            task, fit_i, val_i, cfg, cands, 1000 * k, dev, y, m,
            say=lambda msg: say(0.05 + 0.75 * k / nf, f"fold {k + 1}/{nf}: {msg}"))
        epochs += eps
        chosen.append({"fold": k + 1, "hp": hp, "trials": trials})
        p = tr.predict_nets(nets, task, te, dev)
        base = y[trn].mean(axis=0)
        rs = qm._sigmoid(IP.ranker[te, 0])
        for hi, h in enumerate(it.HEADS):
            out["neural"][h].append((te, p[:, hi], k, float(base[hi])))
            out["ranker"][h].append((te, rs, k, float(base[hi])))
        tests["neural"].append((te, p[:, 1]))
        tests["ranker"].append((te, IP.ranker[te, 0]))
        say(0.05 + 0.75 * (k + 0.6) / nf, f"fold {k + 1}/{nf}: baselines")
        if X is None:
            X = flat(IP, np.arange(IP.rows))
        for hi, h in enumerate(it.HEADS):
            pl = _linear(X, trn, te, y[:, hi])
            out["linear"][h].append((te, pl, k, float(base[hi])))
            if hi == 1:
                tests["linear"].append((te, pl))
            if cfg.gbm:
                pg = tr.gbm_fit_predict(X, trn, te, y[:, hi], seed=k)
                out["gbm"][h].append((te, pg, k, float(base[hi])))
                if hi == 1:
                    tests["gbm"].append((te, pg))
    return out, tests, epochs, chosen


def train(IP: it.IPanel, cfg: tr.TrainConfig | None = None, say: tr.Say | None = None):
    torch = tr._torch()
    cfg = cfg or tr.TrainConfig()
    say = say or (lambda f, m: None)
    t0 = time.perf_counter()
    dev = tr.device_of(cfg)
    log.info("09:30 model on %s: %d stocks, %d rows, %d days, %d test folds", tr.device_name(dev), len(IP.syms),
             IP.rows, IP.cal_days.size, len(IP.folds))
    wf, tests, epochs, chosen = walk_forward(IP, cfg, dev, say)
    heads = []
    for hi, h in enumerate(it.HEADS):
        res = {name: tr.score(wf[name][h], IP, hi, 1, rank=(h == "beat")) for name in MODELS
               if len(wf[name][h]) >= 2}
        if "neural" not in res:
            raise ValueError("the walk-forward produced too few test folds — more 1-minute history is needed")
        heads.append({"head": h, "label": it.HEAD_LABEL[h], "metrics": res})
    trade = {}
    for name in MODELS:
        if not tests[name]:
            continue
        rows = np.concatenate([a for a, _ in tests[name]])
        sc = np.concatenate([b for _, b in tests[name]])
        trade[name] = trading(IP, rows, None if name == "ranker" else sc, ranker_picks=(name == "ranker"))
    beat = heads[1]["metrics"]["neural"]
    edge, edge_note = has_edge(beat, trade.get("neural"))

    say(0.85, f"final fit on all {IP.rows:,} rows")
    from .model import Ensemble, n_params
    task = IntradayTask(IP)
    cands = tr.candidates(cfg, INTRA_DEFAULTS, INTRA_SPACE)
    y = (IP.fwd > 0).astype(np.float32)
    m = np.ones_like(y)
    rows = np.arange(IP.rows)
    fit_i, val_i = tr.split_val(task, rows, cfg.val_frac)
    n_ep = int(np.median(epochs)) if epochs else cfg.epochs
    nets, hp, _, trials = tr.fit_ensemble(task, fit_i, val_i, cfg, cands, 99_000, dev, y, m,
                                          epochs=cfg.epochs if val_i.size else n_ep,
                                          say=lambda msg: say(0.85, f"final fit: {msg}"))
    ens = Ensemble([n.cpu() for n in nets]).eval()
    for hi, hz in enumerate(heads):
        hz["calibration"] = tr.calibration_of(wf["neural"][hz["head"]], IP, hi, float(y[:, hi].mean()))
    P = IP.daily
    meta = {
        "model": MODEL_NAME, "version": MODEL_VERSION, "trained_at": time.time(),
        "device": tr.device_name(dev), "torch": torch.__version__,
        "seq_len": ds.SEQ_LEN, "tech_keys": list(ds.TECH_KEYS), "static_keys": P.static_keys,
        "bar_keys": list(it.BAR_KEYS), "intra_keys": list(it.INTRA_KEYS), "heads": heads,
        "sectors": P.sectors, "symbols": IP.syms, "rows": IP.rows, "days": int(IP.cal_days.size),
        "params": sum(n_params(n) for n in nets) // max(len(nets), 1), "seeds": cfg.seeds, "config": asdict(cfg),
        "epochs_median": n_ep, "hmm": P.hmm, "hmm_end_ts": P.hmm_end_ts, "with_minute": "m_on" in P.static_keys,
        "ranker_weights": IP.weights, "cost_pct": it.COST_PCT, "top_n": it.TOP_N, "per_sector": it.MAX_PER_SECTOR,
        "trading": trade, "edge": edge, "edge_note": edge_note, "verdict": verdict(trade, edge),
        "coverage": IP.coverage, "notes": IP.notes,
        "search": tr.search_summary(cfg, cands, [{"fold": c["fold"], "hp": c["hp"]} for c in chosen],
                                    {"hp": hp, "trials": trials}),
        "evaluation": {"test_start": IP.date(IP.folds[0][0]), "test_end": IP.date(IP.folds[-1][1] - 1),
                       "folds": [[IP.date(a), IP.date(b - 1)] for a, b in IP.folds], "purge_days": it.PURGE,
                       "hmm_fit_end": ds.date_of(P.hmm_end_ts) if P.hmm_end_ts else None,
                       "baselines": [n for n in MODELS if n != "neural" and (cfg.gbm or n != "gbm")]},
        "minute_keys": list(mn.DAY_KEYS),
        "timing": {"total_s": round(time.perf_counter() - t0, 1)},
    }
    return ens, meta


def export(ens, IP: it.IPanel, meta: dict, out_dir: Path | str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path, meta_path = out / "intraday.onnx", out / "intraday.json"
    sample = np.linspace(0, IP.rows - 1, min(IP.rows, 256)).astype(int)
    diff = tr.export_onnx(ens, IntradayTask(IP).inputs(sample), IntradayTask.input_names, "p", onnx_path)
    tr.write_meta(meta, onnx_path, meta_path, diff)
    return onnx_path, meta_path
