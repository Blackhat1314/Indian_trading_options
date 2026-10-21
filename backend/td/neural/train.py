"""NeuralEdge training: walk-forward against baselines, a settings search inside each training window, calibration,
final fit, ONNX export. The fitting machinery here is shared with the 09:30 model (td.neural.train_intraday).

Honesty rules (QuantEdge's, so the comparison is fair):
- Same rows, same test folds, same HMM cutoff. Every test fold is predicted by networks trained only on rows that
  end 2·21 trading days before it (one network serves all three horizons, so it takes the longest purge).
- Early stopping AND the settings search (`--search N`: N network sizes / dropout / learning-rate settings) look
  only at a validation slice cut from the END of each fold's own training window (again purged), never at the
  test fold. The setting chosen in each fold is recorded, so a lucky pick can't hide.
- Each horizon is scored for three models on identical test rows:
    neural     this network (mean of `seeds` trained copies);
    quantedge  QuantEdge's L2 logistic regression on the same 26 technical features (what the QUANT desk runs);
    gbm        gradient-boosted trees on technical + static inputs (sklearn, optional) — a strong tabular baseline.
  Metrics: mean per-fold AUC with a block-bootstrap SE, daily cross-sectional rank IC, Brier / log loss vs the base
  rate, all with probabilities calibrated on earlier folds only; `edge_status` decides whether a horizon has an
  edge at all. The verdict says whether the network beats QuantEdge, not just whether it is above 0.5.
- The exported model is refit on every labelled row; its calibration spread comes from the walk-forward.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..research import quant as qm
from . import dataset as ds
from . import fundamentals as fu
from . import minute as mn
from . import news as nw

log = logging.getLogger("td.neural.train")

MODEL_NAME = "NeuralEdge"
MODEL_VERSION = 1
MODELS = ("neural", "quantedge", "gbm")
BEAT_MARGIN = 0.003          # the network "beats" QuantEdge at a horizon by at least this much mean fold AUC
IC_BEAT_MARGIN = 0.005       # … and at ranking stocks by at least this much mean daily IC
MIN_IC = 0.02                # a ranking edge needs at least this mean daily rank IC
IC_BOOT_REPS = 500

NET_DEFAULTS = {"hidden": 48, "static_hidden": 64, "dropout": 0.25, "input_dropout": 0.1}
SPACE: dict[str, tuple] = {
    "hidden": (32, 48, 64, 96), "static_hidden": (32, 64, 128), "dropout": (0.1, 0.25, 0.4),
    "input_dropout": (0.0, 0.1, 0.2), "lr": (5e-4, 1e-3, 2e-3, 4e-3), "weight_decay": (1e-4, 1e-3, 1e-2),
}


@dataclass
class TrainConfig:
    seeds: int = 5
    epochs: int = 15
    patience: int = 3
    batch: int = 2048
    lr: float = 2e-3
    weight_decay: float = 1e-3
    val_frac: float = 0.15
    gbm: bool = True
    device: str = "auto"            # auto | cuda | cpu
    threads: int = 0                # CPU threads for torch (0 = torch's default)
    search: int = 0                 # network settings tried per training window (0 or 1 = the defaults only)
    search_seed: int = 7


Say = Callable[[float, str], None]


# --------------------------------------------------------------------------- torch plumbing
def _torch():
    try:
        import torch
    except ImportError as e:                          # the server never needs it
        raise RuntimeError("PyTorch is not installed — see 'NeuralEdge: train on your PC' in the README") from e
    return torch


def device_of(cfg: TrainConfig):
    torch = _torch()
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    if cfg.device == "cpu" or (cfg.device == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("--device cuda asked for, but PyTorch sees no CUDA GPU (install the CUDA build of torch)")
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


def device_name(dev) -> str:
    torch = _torch()
    return torch.cuda.get_device_name(dev) if dev.type == "cuda" else f"CPU ({torch.get_num_threads()} threads)"


# --------------------------------------------------------------------------- tasks: what a network reads
class DailyTask:
    """The daily panel as network inputs."""
    input_names = ("seq", "static")

    def __init__(self, P: ds.Panel):
        self.P = P
        self.row_cal = P.row_cal
        self.purge = ds.PURGE_DAYS
        self.n_heads = len(ds.HEAD_KEYS)

    def inputs(self, idx: np.ndarray) -> list[np.ndarray]:
        return [self.P.seq(idx), self.P.static[idx]]

    def net(self, hp: dict):
        from .model import NeuralEdgeNet
        return NeuralEdgeNet(self.P.Z.shape[1], self.P.static.shape[1], self.n_heads, **hp)


def candidates(cfg: TrainConfig, defaults: dict | None = None, space: dict | None = None) -> list[dict]:
    """The settings to try: the defaults first, then distinct random draws from the space (same list for every
    training window, so folds differ only in which one their own validation slice picked)."""
    base = {**NET_DEFAULTS, **(defaults or {}), "lr": cfg.lr, "weight_decay": cfg.weight_decay}
    sp = {**SPACE, **(space or {})}
    out, seen = [base], {tuple(sorted(base.items()))}
    rng = np.random.default_rng(cfg.search_seed)
    for _ in range(2000):
        if len(out) >= max(cfg.search, 1):
            break
        hp = {k: v[int(rng.integers(len(v)))] for k, v in sp.items()}
        hp = {k: (float(x) if isinstance(x, float) else int(x)) for k, x in hp.items()}
        key = tuple(sorted(hp.items()))
        if key not in seen:
            seen.add(key)
            out.append(hp)
    return out


def split_val(task, rows: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Train rows → (fit rows, validation rows): the last `frac` of the window's dates validate, with a purge."""
    cal = task.row_cal
    days = np.unique(cal[rows])
    if days.size < 200 or frac <= 0:
        return rows, rows[:0]
    edge = days[int(days.size * (1 - frac))]
    va = rows[cal[rows] >= edge]
    fit = rows[cal[rows] < edge - task.purge]
    if fit.size < 1000 or va.size < 200:
        return rows, rows[:0]
    return fit, va


def fit_net(task, fit_idx: np.ndarray, val_idx: np.ndarray, cfg: TrainConfig, hp: dict, seed: int, dev,
            y: np.ndarray, m: np.ndarray, epochs: int | None = None):
    """One network. With validation rows: early stopping on validation log loss (best epoch kept); without:
    exactly `epochs` epochs. Returns (net, epochs run to the best state, best validation loss or None)."""
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    arch = {k: v for k, v in hp.items() if k not in ("lr", "weight_decay")}
    net = task.net(arch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=float(hp.get("lr", cfg.lr)),
                            weight_decay=float(hp.get("weight_decay", cfg.weight_decay)))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    n_ep = epochs or cfg.epochs

    def tensors(b):
        xs = [torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True) for x in task.inputs(b)]
        return xs, torch.from_numpy(y[b]).to(dev), torch.from_numpy(m[b]).to(dev)

    def val_loss() -> float:
        net.eval()
        tot, cnt = 0.0, 0.0
        with torch.no_grad():
            for a in range(0, val_idx.size, cfg.batch * 4):
                xs, yy, mm = tensors(val_idx[a:a + cfg.batch * 4])
                tot += float((bce(net(*xs), yy) * mm).sum())
                cnt += float(mm.sum())
        return tot / max(cnt, 1.0)

    best, best_state, best_ep, bad = math.inf, None, 0, 0
    for ep in range(1, n_ep + 1):
        net.train()
        order = rng.permutation(fit_idx)
        for a in range(0, order.size, cfg.batch):
            xs, yy, mm = tensors(order[a:a + cfg.batch])
            loss = (bce(net(*xs), yy) * mm).sum() / mm.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if val_idx.size:
            v = val_loss()
            if v < best - 1e-5:
                best, best_ep, bad = v, ep, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    else:
        best_ep = n_ep
    net.eval()
    return net, best_ep, (best if math.isfinite(best) else None)


def fit_ensemble(task, fit_idx: np.ndarray, val_idx: np.ndarray, cfg: TrainConfig, cands: list[dict], seed0: int,
                 dev, y: np.ndarray, m: np.ndarray, epochs: int | None = None, say: Callable[[str], None] | None = None):
    """Settings search on the validation slice (when there is one and more than one candidate), then `seeds`
    networks with the chosen settings (the search's own network of that setting is the first seed).
    Returns (nets, chosen settings, epochs of each net, the search trials)."""
    trials: list[dict] = []
    best_net, best_hp, best_ep = None, cands[0], None
    if len(cands) > 1 and val_idx.size:
        best_loss = math.inf
        for j, hp in enumerate(cands):
            if say:
                say(f"settings {j + 1}/{len(cands)}")
            net, ep, loss = fit_net(task, fit_idx, val_idx, cfg, hp, seed0, dev, y, m, epochs)
            trials.append({"hp": hp, "val_loss": None if loss is None else round(loss, 6), "epochs": ep})
            if loss is not None and loss < best_loss:
                best_loss, best_net, best_hp, best_ep = loss, net, hp, ep
            else:
                del net
    nets, eps = [], []
    if best_net is not None:
        nets.append(best_net)
        eps.append(best_ep)
    for sd in range(len(nets), cfg.seeds):
        net, ep, _ = fit_net(task, fit_idx, val_idx, cfg, best_hp, seed0 + sd, dev, y, m, epochs)
        nets.append(net)
        eps.append(ep)
    return nets, best_hp, eps, trials


def predict_nets(nets: list, task, idx: np.ndarray, dev, batch: int = 8192) -> np.ndarray:
    torch = _torch()
    out = np.empty((idx.size, task.n_heads), dtype=np.float64)
    with torch.no_grad():
        for a in range(0, idx.size, batch):
            b = idx[a:a + batch]
            xs = [torch.from_numpy(np.ascontiguousarray(x)).to(dev) for x in task.inputs(b)]
            p = torch.stack([torch.sigmoid(n(*xs)) for n in nets]).mean(dim=0)
            out[a:a + b.size] = p.float().cpu().numpy()
    return out


def _labels(P: ds.Panel) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(P.fwd)
    return np.where(m, (P.fwd > 0).astype(np.float32), 0.0).astype(np.float32), m.astype(np.float32)


# --------------------------------------------------------------------------- baselines
def _quantedge(P: ds.Panel, tr: np.ndarray, te: np.ndarray, h: int, y: np.ndarray) -> np.ndarray:
    X = P.zrow.astype(np.float64)
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
    sd[sd < 1e-9] = 1.0
    b0, w, _ = qm.logistic_fit((X[tr] - mu) / sd, y[tr], qm.ALPHA0 * h)
    return qm._sigmoid(b0 + ((X[te] - mu) / sd) @ w)


def gbm_fit_predict(X: np.ndarray, tr: np.ndarray, te: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=300, l2_regularization=1.0, early_stopping=False,
                                         random_state=seed)
    clf.fit(X[tr], y[tr])
    return clf.predict_proba(X[te])[:, 1]


def _gbm(P: ds.Panel, tr: np.ndarray, te: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    return gbm_fit_predict(np.hstack([P.zrow, P.static]).astype(np.float32), tr, te, y, seed)


# --------------------------------------------------------------------------- scoring (QuantEdge's metrics)
def score(parts: list[tuple[np.ndarray, np.ndarray, int, float]], P, hi: int, h: int, rank: bool = False) -> dict:
    """Metrics of one model at one head from its walk-forward predictions [(rows, p_raw, fold, base_train)].
    P needs .fwd (labels: > 0 is a 1) and .row_cal (the date of each row). `rank`: a cross-sectional head, whose
    edge is judged by its daily rank IC (rank_metrics) instead of its AUC."""
    y_all = (P.fwd[:, hi] > 0).astype(float)
    p_cal_parts = []
    for j, (idx, pr, _, base) in enumerate(parts):
        if j == 0:
            p_cal_parts.append(pr)
        else:
            b = qm.calib_fit(np.concatenate([o[1] for o in parts[:j]]),
                             np.concatenate([np.full(o[0].size, o[3]) for o in parts[:j]]),
                             np.concatenate([y_all[o[0]] for o in parts[:j]]))
            p_cal_parts.append(qm.calib_apply(pr, base, b))
    idx_all = np.concatenate([o[0] for o in parts])
    p_raw = np.concatenate([o[1] for o in parts])
    p_cal = np.concatenate(p_cal_parts)
    base_train = np.concatenate([np.full(o[0].size, o[3]) for o in parts])
    y = y_all[idx_all]
    fold_auc = [qm.auc_score(o[1], y_all[o[0]]) for o in parts]
    fold_id = np.concatenate([np.full(o[0].size, o[2]) for o in parts])
    se = qm.auc_block_se(p_raw, y, P.row_cal[idx_all], fold_id, block=max(2 * h, 10))
    good = [a for a in fold_auc if a is not None]
    if se is not None and len(good) > 1:
        se = max(se, float(np.std(good, ddof=1)) / math.sqrt(len(good)))
    met = qm._metrics(h, len(parts), p_raw, p_cal, y, P.fwd[idx_all, hi], P.row_cal[idx_all], base_train, good, se)
    met["calib_b"] = round(qm.calib_fit(p_raw, base_train, y), 4)
    met["edge"], met["edge_note"] = qm.edge_status(met)
    if rank:
        met.update(rank_metrics(p_raw, P.fwd[idx_all, hi], P.row_cal[idx_all], fold_id, h))
    return met


def rank_metrics(p: np.ndarray, fwd: np.ndarray, cal: np.ndarray, fold: np.ndarray, h: int) -> dict:
    """How well a score orders stocks against each other: the mean daily Spearman IC with a standard error from a
    block bootstrap over test DATES (blocks of 2·h days inside each fold — h-day labels overlap, so neighbouring
    days' ICs are nearly the same number), floored at the spread of the fold means; the top-minus-bottom 10 %
    h-day return on non-overlapping dates (every h-th test day). Edge: IC ≥ MIN_IC, IC − 2·SE > 0 and a positive
    mean IC in every fold."""
    order = np.lexsort((cal,))
    p, fwd, cal, fold = p[order], fwd[order], cal[order], fold[order]
    cuts = np.flatnonzero(np.r_[True, cal[1:] != cal[:-1], True])
    ics, days, folds, spreads = [], [], [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 10:
            continue
        v = qm._spearman(p[a:b], fwd[a:b])
        if not math.isfinite(v):
            continue
        ics.append(v)
        days.append(int(cal[a]))
        folds.append(int(fold[a]))
    if len(ics) < 20:
        return {"ic_se": None, "ic_t": None, "fold_ic": [], "spread": None, "spread_t": None,
                "edge": False, "edge_note": "no reliable ranking edge (too few test days)"}
    ics, days, folds = np.array(ics), np.array(days), np.array(folds)
    fold_ic = [float(ics[folds == f].mean()) for f in np.unique(folds)]
    rng = np.random.default_rng(0)
    blocks = []
    for f in np.unique(folds):
        d = days[folds == f]
        blk = (d - d.min()) // max(2 * h, 1)
        for bk in np.unique(blk):
            blocks.append(ics[folds == f][blk == bk])
    means = []
    for _ in range(IC_BOOT_REPS):
        pick = rng.integers(0, len(blocks), len(blocks))
        means.append(float(np.concatenate([blocks[i] for i in pick]).mean()))
    se = float(np.std(means, ddof=1))
    if len(fold_ic) > 1:
        se = max(se, float(np.std(fold_ic, ddof=1)) / math.sqrt(len(fold_ic)))
    ic = float(ics.mean())
    # top-minus-bottom decile on every h-th test day (non-overlapping holding periods)
    first = days.min()
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 20 or (int(cal[a]) - first) % h:
            continue
        o = np.argsort(p[a:b])
        q = max(1, (b - a) // 10)
        r = np.expm1(fwd[a:b]) * 100
        spreads.append(float(r[o[-q:]].mean() - r[o[:q]].mean()))
    sp = np.array(spreads)
    sp_t = float(sp.mean() / sp.std(ddof=1) * math.sqrt(sp.size)) if sp.size > 2 and sp.std(ddof=1) > 0 else None
    head = "no reliable ranking edge"
    if ic < MIN_IC:
        edge, note = False, f"{head} (daily rank IC {ic:+.3f} < {MIN_IC})"
    elif ic - 2 * se <= 0:
        edge, note = False, (f"{head} (IC {ic:+.3f} ± {se:.3f}: not clearly above zero once the overlap of "
                             f"{h}-day returns is accounted for)")
    elif min(fold_ic) <= 0:
        edge, note = False, f"{head} (IC {ic:+.3f} on average but {min(fold_ic):+.3f} in its worst fold)"
    else:
        edge, note = True, None
    return {"ic": qm._f(ic), "ic_se": qm._f(se), "ic_t": qm._f(ic / se, 2) if se > 0 else None, "fold_ic": [qm._f(x) for x in fold_ic],
            "ic_days": int(ics.size), "spread": qm._f(sp.mean(), 3) if sp.size else None,
            "spread_t": qm._f(sp_t, 2) if sp_t is not None else None, "spread_periods": int(sp.size),
            "edge": edge, "edge_note": note}


def verdict(neural: dict, quantedge: dict | None) -> str:
    if not neural.get("edge"):
        return "no reliable edge — the desk shows this horizon as neutral"
    if not quantedge or quantedge.get("auc") is None:
        return "has an edge (QuantEdge not measured)"
    d = (neural["auc"] or 0) - (quantedge["auc"] or 0)
    if d >= BEAT_MARGIN:
        return f"beats QuantEdge (AUC +{d:.3f})"
    return f"has an edge but does not beat QuantEdge (AUC {d:+.3f})"


def calibration_of(parts, P, hi: int, base_final: float) -> dict:
    """Spread from the walk-forward predictions, anchored at the final training base rate."""
    p_raw = np.concatenate([o[1] for o in parts])
    base_train = np.concatenate([np.full(o[0].size, o[3]) for o in parts])
    yy = (P.fwd[np.concatenate([o[0] for o in parts]), hi] > 0).astype(float)
    return {"base": float(base_final), "b": float(qm.calib_fit(p_raw, base_train, yy))}


def search_summary(cfg: TrainConfig, cands: list[dict], chosen: list[dict], final: dict) -> dict:
    return {"candidates": len(cands), "space": {k: list(v) for k, v in SPACE.items()} if len(cands) > 1 else None,
            "chosen_per_fold": chosen, "final": final, "note": (
                "each training window chose its own setting on its own validation slice; the test folds never "
                "took part" if len(cands) > 1 else "defaults only (run with --search N to try N settings)")}


# --------------------------------------------------------------------------- the whole run
def head_cols() -> list[tuple[int, str, int]]:
    """(column of P.fwd / network output, "up" or "beat", horizon) for every head."""
    n = len(ds.HORIZONS)
    return [(i, "up", h) for i, h in enumerate(ds.HORIZONS)] + [(n + i, "beat", h) for i, h in enumerate(ds.HORIZONS)]


def walk_forward(P: ds.Panel, cfg: TrainConfig, dev, say: Say) -> tuple[dict, list[int], dict, list[dict]]:
    """{model: {column: [(rows, p_raw, fold, base_train)]}}, the networks' best epochs, timing, chosen settings."""
    task = DailyTask(P)
    cands = candidates(cfg)
    y, m = _labels(P)
    labelled_any = m.any(axis=1)
    cols = head_cols()
    out = {name: {c: [] for c, _, _ in cols} for name in MODELS}
    epochs: list[int] = []
    chosen: list[dict] = []
    t_nn = t_base = 0.0
    nf = len(P.folds)
    for k, (s, e) in enumerate(P.folds):
        te_all = (P.row_cal >= s) & (P.row_cal < e) & (P.row_cal > P.row_hmm_cal)
        cut = ds.purge_cut(s)
        tr = np.flatnonzero((P.row_cal <= cut) & (P.row_hmm_cal <= cut) & labelled_any)
        te = np.flatnonzero(te_all)
        if tr.size < 1000 or te.size < 50:
            log.info("fold %d skipped (%d train / %d test rows)", k + 1, tr.size, te.size)
            continue
        say(0.10 + 0.70 * k / nf, f"fold {k + 1}/{nf}: training {cfg.seeds} networks on {tr.size:,} rows")
        t0 = time.perf_counter()
        fit_i, val_i = split_val(task, tr, cfg.val_frac)
        nets, hp, eps, trials = fit_ensemble(
            task, fit_i, val_i, cfg, cands, 1000 * k, dev, y, m,
            say=lambda msg: say(0.10 + 0.70 * k / nf, f"fold {k + 1}/{nf}: {msg}"))
        epochs += eps
        chosen.append({"fold": k + 1, "hp": hp, "trials": trials, "train_rows": int(tr.size)})
        p = predict_nets(nets, task, te, dev)
        t_nn += time.perf_counter() - t0
        t0 = time.perf_counter()
        for c, kind, h in cols:
            lab_te = te[m[te, c] > 0]
            if lab_te.size < 50:
                continue
            keep = m[te, c] > 0
            out["neural"][c].append((lab_te, p[keep, c], k, float(y[tr][m[tr, c] > 0, c].mean())))
            cut_h = ds.purge_cut(s, h)
            tr_h = np.flatnonzero((P.row_cal <= cut_h) & (P.row_hmm_cal <= cut_h) & (m[:, c] > 0))
            yh = y[:, c].astype(float)
            base_h = float(yh[tr_h].mean())
            out["quantedge"][c].append((lab_te, _quantedge(P, tr_h, lab_te, h, yh), k, base_h))
            if cfg.gbm:
                say(0.10 + 0.70 * (k + 0.5) / nf, f"fold {k + 1}/{nf}: gradient-boosting baseline, {kind} {h}-day")
                out["gbm"][c].append((lab_te, _gbm(P, tr_h, lab_te, yh, seed=k), k, base_h))
        t_base += time.perf_counter() - t0
    return out, epochs, {"neural_s": round(t_nn, 1), "baselines_s": round(t_base, 1)}, chosen


def rank_verdict(neural: dict, quantedge: dict | None) -> str:
    if not neural.get("edge"):
        return "no reliable ranking edge — the leaderboard does not rank on it"
    if not quantedge or quantedge.get("ic") is None:
        return "ranks stocks with an edge (QuantEdge not measured)"
    d = (neural["ic"] or 0) - (quantedge["ic"] or 0)
    if d >= IC_BEAT_MARGIN:
        return f"ranks stocks better than QuantEdge (IC +{d:.3f})"
    return f"ranks stocks with an edge but not better than QuantEdge (IC {d:+.3f})"


def train(P: ds.Panel, cfg: TrainConfig | None = None, say: Say | None = None):
    """Walk-forward, final fit. Returns (ensemble network, metadata dict)."""
    torch = _torch()
    cfg = cfg or TrainConfig()
    say = say or (lambda f, m: None)
    t0 = time.perf_counter()
    dev = device_of(cfg)
    log.info("training on %s: %d stocks, %d rows, %d test folds", device_name(dev), len(P.syms), P.rows,
             len(P.folds))
    wf, epochs, timing, chosen = walk_forward(P, cfg, dev, say)
    cols = head_cols()
    res: dict[int, dict] = {}
    for c, kind, h in cols:
        res[c] = {name: score(wf[name][c], P, c, h, rank=(kind == "beat")) for name in MODELS
                  if len(wf[name][c]) >= 2}
        if "neural" not in res[c]:
            raise ValueError(f"walk-forward produced too few test folds for {kind} {h}-day")
    horizons = []
    for hi, h in enumerate(ds.HORIZONS):
        up, beat = res[hi], res[len(ds.HORIZONS) + hi]
        horizons.append({"h": h, "label": qm.H_LABELS[h], "metrics": up,
                         "verdict": verdict(up["neural"], up.get("quantedge")),
                         "rank": {"metrics": beat, "verdict": rank_verdict(beat["neural"], beat.get("quantedge"))}})

    say(0.85, f"final fit on all {P.rows:,} rows")
    from .model import Ensemble
    task = DailyTask(P)
    cands = candidates(cfg)
    y, m = _labels(P)
    rows = np.flatnonzero(m.any(axis=1))
    fit_i, val_i = split_val(task, rows, cfg.val_frac)
    n_ep = int(np.median(epochs)) if epochs else cfg.epochs
    nets, hp, _, trials = fit_ensemble(task, fit_i, val_i, cfg, cands, 99_000, dev, y, m,
                                       epochs=cfg.epochs if val_i.size else n_ep,
                                       say=lambda msg: say(0.85, f"final fit: {msg}"))
    ens = Ensemble([n.cpu() for n in nets]).eval()

    for hz in horizons:
        hi = ds.HORIZONS.index(hz["h"])
        rc = len(ds.HORIZONS) + hi
        hz["calibration"] = calibration_of(wf["neural"][hi], P, hi, float(y[m[:, hi] > 0, hi].mean()))
        hz["rank"]["calibration"] = calibration_of(wf["neural"][rc], P, rc, float(y[m[:, rc] > 0, rc].mean()))

    from .model import n_params
    with_minute = "m_on" in P.static_keys
    meta = {
        "model": MODEL_NAME, "version": MODEL_VERSION, "trained_at": time.time(),
        "device": device_name(dev), "torch": torch.__version__,
        "seq_len": ds.SEQ_LEN, "tech_keys": list(ds.TECH_KEYS), "static_keys": P.static_keys,
        "heads": list(ds.HEAD_KEYS),
        "sectors": P.sectors, "horizons": horizons, "symbols": P.syms, "rows": P.rows,
        "params": sum(n_params(n) for n in nets) // max(len(nets), 1),
        "seeds": cfg.seeds, "config": asdict(cfg), "epochs_median": n_ep, "hmm": P.hmm, "hmm_end_ts": P.hmm_end_ts,
        "coverage": P.coverage, "notes": P.notes, "with_minute": with_minute,
        "search": search_summary(cfg, cands, [{"fold": c["fold"], "hp": c["hp"]} for c in chosen],
                                 {"hp": hp, "trials": trials}),
        "evaluation": {"test_start": ds.date_of(P.cal[P.folds[0][0]]), "test_end": ds.date_of(P.cal[P.folds[-1][1] - 1]),
                       "folds": [[ds.date_of(P.cal[a]), ds.date_of(P.cal[b - 1])] for a, b in P.folds],
                       "purge_days": ds.PURGE_DAYS, "hmm_fit_end": ds.date_of(P.hmm_end_ts) if P.hmm_end_ts else None,
                       "baselines": [n for n in MODELS if n != "neural" and (cfg.gbm or n != "gbm")],
                       "train_rows_per_fold": [c.get("train_rows") for c in chosen]},
        "fund_keys": list(fu.FUND_KEYS), "news_keys": list(nw.NEWS_KEYS), "cs_keys": list(ds.CS_KEYS),
        "minute_keys": list(mn.DAY_KEYS) if with_minute else [],
        "timing": {**timing, "total_s": round(time.perf_counter() - t0, 1)},
    }
    return ens, meta


# --------------------------------------------------------------------------- export
def export_onnx(ens, inputs: list[np.ndarray], names: tuple[str, ...], output: str, path: Path) -> float:
    """Write `ens` to `path` as ONNX (dynamic batch) and check it against PyTorch on `inputs`. Returns the max
    difference; raises when it is above 1e-4 (nothing is written then)."""
    import warnings
    torch = _torch()
    xs = tuple(torch.from_numpy(np.ascontiguousarray(x)) for x in inputs)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.onnx.tmp")
    with warnings.catch_warnings():
        # PyTorch warns for every GRU exported with a batch other than 1; the check below runs other batch sizes
        warnings.filterwarnings("ignore", message=".*batch_size other than 1.*")
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(ens, xs, str(tmp), input_names=list(names), output_names=[output],
                          dynamic_axes={**{n: {0: "n"} for n in names}, output: {0: "n"}}, opset_version=17,
                          dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    diff = 0.0
    n = int(xs[0].shape[0])
    for size in sorted({n, 1, min(7, n)}):              # the server scores a different number of stocks
        part = tuple(x[:size] for x in xs)
        with torch.no_grad():
            ref = ens(*part).numpy()
        got = sess.run([output], {nm: x.numpy() for nm, x in zip(names, part)})[0]
        diff = max(diff, float(np.abs(got - ref).max()))
    if diff > 1e-4:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX export does not match PyTorch (max difference {diff:.2e})")
    tmp.replace(path)
    return diff


def write_meta(meta: dict, onnx_path: Path, meta_path: Path, diff: float) -> None:
    meta = dict(meta, onnx_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest(), onnx_check_max_diff=diff)
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    tmp.write_text(json.dumps(meta, separators=(",", ":"), allow_nan=False, default=_json_default))
    tmp.replace(meta_path)


def export(ens, P: ds.Panel, meta: dict, out_dir: Path | str) -> tuple[Path, Path]:
    """model.onnx + model.json in out_dir; the ONNX output is checked against PyTorch on real rows."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path, meta_path = out / "model.onnx", out / "model.json"
    sample = np.linspace(0, P.rows - 1, min(P.rows, 256)).astype(int)
    diff = export_onnx(ens, DailyTask(P).inputs(sample), DailyTask.input_names, "p_up", onnx_path)
    write_meta(meta, onnx_path, meta_path, diff)
    return onnx_path, meta_path


def _json_default(o):
    if isinstance(o, np.generic):
        v = o.item()
        return None if isinstance(v, float) and not math.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)
