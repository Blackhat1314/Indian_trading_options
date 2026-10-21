"""NeuralEdge training: walk-forward against two baselines, calibration, final fit, ONNX export.

Honesty rules (QuantEdge's, so the comparison is fair):
- Same rows, same test folds, same HMM cutoff. Every test fold is predicted by networks trained only on rows that
  end 2·21 trading days before it (one network serves all three horizons, so it takes the longest purge).
- Early stopping looks at a validation slice cut from the END of each fold's own training window (again purged),
  never at the test fold.
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
from . import news as nw

log = logging.getLogger("td.neural.train")

MODEL_NAME = "NeuralEdge"
MODEL_VERSION = 1
MODELS = ("neural", "quantedge", "gbm")
BEAT_MARGIN = 0.003          # the network "beats" QuantEdge at a horizon by at least this much mean fold AUC


@dataclass
class TrainConfig:
    seeds: int = 3
    epochs: int = 15
    patience: int = 3
    batch: int = 2048
    lr: float = 2e-3
    weight_decay: float = 1e-3
    val_frac: float = 0.15
    gbm: bool = True
    device: str = "auto"            # auto | cuda | cpu
    threads: int = 0                # CPU threads for torch (0 = torch's default)


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


def _labels(P: ds.Panel) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(P.fwd)
    return np.where(m, (P.fwd > 0).astype(np.float32), 0.0).astype(np.float32), m.astype(np.float32)


def _split_val(P: ds.Panel, rows: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Train rows → (fit rows, validation rows): the last `frac` of the window's dates validate, with a purge."""
    days = np.unique(P.row_cal[rows])
    if days.size < 200 or frac <= 0:
        return rows, rows[:0]
    edge = days[int(days.size * (1 - frac))]
    va = rows[P.row_cal[rows] >= edge]
    fit = rows[P.row_cal[rows] < edge - ds.PURGE_DAYS]
    if fit.size < 1000 or va.size < 200:
        return rows, rows[:0]
    return fit, va


def fit_net(P: ds.Panel, fit_idx: np.ndarray, val_idx: np.ndarray, cfg: TrainConfig, seed: int, dev,
            y: np.ndarray, m: np.ndarray, epochs: int | None = None):
    """One network. With validation rows: early stopping on validation log loss (best epoch kept); without:
    exactly `epochs` epochs. Returns (net, epochs run to the best state, best validation loss or None)."""
    torch = _torch()
    from .model import NeuralEdgeNet
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = NeuralEdgeNet(P.Z.shape[1], P.static.shape[1], len(ds.HORIZONS)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    n_ep = epochs or cfg.epochs

    def batch_tensors(b):
        return (torch.from_numpy(P.seq(b)).to(dev, non_blocking=True),
                torch.from_numpy(P.static[b]).to(dev, non_blocking=True),
                torch.from_numpy(y[b]).to(dev), torch.from_numpy(m[b]).to(dev))

    def val_loss() -> float:
        net.eval()
        tot, cnt = 0.0, 0.0
        with torch.no_grad():
            for a in range(0, val_idx.size, cfg.batch * 4):
                s, st, yy, mm = batch_tensors(val_idx[a:a + cfg.batch * 4])
                tot += float((bce(net(s, st), yy) * mm).sum())
                cnt += float(mm.sum())
        return tot / max(cnt, 1.0)

    best, best_state, best_ep, bad = math.inf, None, 0, 0
    for ep in range(1, n_ep + 1):
        net.train()
        order = rng.permutation(fit_idx)
        for a in range(0, order.size, cfg.batch):
            s, st, yy, mm = batch_tensors(order[a:a + cfg.batch])
            loss = (bce(net(s, st), yy) * mm).sum() / mm.sum().clamp(min=1.0)
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


def predict_nets(nets: list, P: ds.Panel, idx: np.ndarray, dev, batch: int = 8192) -> np.ndarray:
    torch = _torch()
    out = np.empty((idx.size, len(ds.HORIZONS)), dtype=np.float64)
    with torch.no_grad():
        for a in range(0, idx.size, batch):
            b = idx[a:a + batch]
            s = torch.from_numpy(P.seq(b)).to(dev)
            st = torch.from_numpy(P.static[b]).to(dev)
            p = torch.stack([torch.sigmoid(n(s, st)) for n in nets]).mean(dim=0)
            out[a:a + b.size] = p.float().cpu().numpy()
    return out


# --------------------------------------------------------------------------- baselines
def _quantedge(P: ds.Panel, tr: np.ndarray, te: np.ndarray, h: int, y: np.ndarray) -> np.ndarray:
    X = P.zrow.astype(np.float64)
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
    sd[sd < 1e-9] = 1.0
    b0, w, _ = qm.logistic_fit((X[tr] - mu) / sd, y[tr], qm.ALPHA0 * h)
    return qm._sigmoid(b0 + ((X[te] - mu) / sd) @ w)


def _gbm(P: ds.Panel, tr: np.ndarray, te: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.hstack([P.zrow, P.static]).astype(np.float32)
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=300, l2_regularization=1.0, early_stopping=False,
                                         random_state=seed)
    clf.fit(X[tr], y[tr])
    return clf.predict_proba(X[te])[:, 1]


# --------------------------------------------------------------------------- scoring (QuantEdge's metrics)
def score(parts: list[tuple[np.ndarray, np.ndarray, int, float]], P: ds.Panel, hi: int, h: int) -> dict:
    """Metrics of one model at one horizon from its walk-forward predictions [(rows, p_raw, fold, base_train)]."""
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
    return met


def verdict(neural: dict, quantedge: dict | None) -> str:
    if not neural.get("edge"):
        return "no reliable edge — the desk shows this horizon as neutral"
    if not quantedge or quantedge.get("auc") is None:
        return "has an edge (QuantEdge not measured)"
    d = (neural["auc"] or 0) - (quantedge["auc"] or 0)
    if d >= BEAT_MARGIN:
        return f"beats QuantEdge (AUC +{d:.3f})"
    return f"has an edge but does not beat QuantEdge (AUC {d:+.3f})"


# --------------------------------------------------------------------------- the whole run
def walk_forward(P: ds.Panel, cfg: TrainConfig, dev, say: Say) -> tuple[dict, list[int], dict]:
    """{model: {h: [(rows, p_raw, fold, base_train)]}}, the networks' best epochs, timing."""
    y, m = _labels(P)
    labelled_any = m.any(axis=1)
    out = {name: {h: [] for h in ds.HORIZONS} for name in MODELS}
    epochs: list[int] = []
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
        fit_i, val_i = _split_val(P, tr, cfg.val_frac)
        nets = []
        for sd in range(cfg.seeds):
            net, ep, _ = fit_net(P, fit_i, val_i, cfg, 1000 * k + sd, dev, y, m)
            nets.append(net)
            epochs.append(ep)
        p = predict_nets(nets, P, te, dev)
        t_nn += time.perf_counter() - t0
        t0 = time.perf_counter()
        for hi, h in enumerate(ds.HORIZONS):
            lab_te = te[m[te, hi] > 0]
            if lab_te.size < 50:
                continue
            keep = m[te, hi] > 0
            out["neural"][h].append((lab_te, p[keep, hi], k, float(y[tr][m[tr, hi] > 0, hi].mean())))
            cut_h = ds.purge_cut(s, h)
            tr_h = np.flatnonzero((P.row_cal <= cut_h) & (P.row_hmm_cal <= cut_h) & (m[:, hi] > 0))
            yh = y[:, hi].astype(float)
            base_h = float(yh[tr_h].mean())
            out["quantedge"][h].append((lab_te, _quantedge(P, tr_h, lab_te, h, yh), k, base_h))
            if cfg.gbm:
                say(0.10 + 0.70 * (k + 0.5) / nf, f"fold {k + 1}/{nf}: gradient-boosting baseline, {h}-day")
                out["gbm"][h].append((lab_te, _gbm(P, tr_h, lab_te, yh, seed=k), k, base_h))
        t_base += time.perf_counter() - t0
    return out, epochs, {"neural_s": round(t_nn, 1), "baselines_s": round(t_base, 1)}


def train(P: ds.Panel, cfg: TrainConfig | None = None, say: Say | None = None):
    """Walk-forward, final fit. Returns (ensemble network, metadata dict)."""
    torch = _torch()
    cfg = cfg or TrainConfig()
    say = say or (lambda f, m: None)
    t0 = time.perf_counter()
    dev = device_of(cfg)
    log.info("training on %s: %d stocks, %d rows, %d test folds", device_name(dev), len(P.syms), P.rows,
             len(P.folds))
    wf, epochs, timing = walk_forward(P, cfg, dev, say)
    horizons = []
    for hi, h in enumerate(ds.HORIZONS):
        res = {}
        for name in MODELS:
            if len(wf[name][h]) >= 2:
                res[name] = score(wf[name][h], P, hi, h)
        if "neural" not in res:
            raise ValueError(f"walk-forward produced too few test folds for h={h}")
        horizons.append({"h": h, "label": qm.H_LABELS[h], "metrics": res,
                         "verdict": verdict(res["neural"], res.get("quantedge"))})

    say(0.85, f"final fit on all {P.rows:,} rows")
    from .model import Ensemble
    y, m = _labels(P)
    rows = np.flatnonzero(m.any(axis=1))
    fit_i, val_i = _split_val(P, rows, cfg.val_frac)
    n_ep = int(np.median(epochs)) if epochs else cfg.epochs
    nets = []
    for sd in range(cfg.seeds):
        net, _, _ = fit_net(P, fit_i, val_i, cfg, 99_000 + sd, dev, y, m, epochs=cfg.epochs if val_i.size else n_ep)
        nets.append(net.cpu())
    ens = Ensemble(nets).eval()

    # calibration: spread from the walk-forward predictions, anchored at the final training base rate
    for hz in horizons:
        h = hz["h"]
        hi = ds.HORIZONS.index(h)
        parts = wf["neural"][h]
        p_raw = np.concatenate([o[1] for o in parts])
        base_train = np.concatenate([np.full(o[0].size, o[3]) for o in parts])
        yy = (P.fwd[np.concatenate([o[0] for o in parts]), hi] > 0).astype(float)
        hz["calibration"] = {"base": float(y[m[:, hi] > 0, hi].mean()), "b": float(qm.calib_fit(p_raw, base_train, yy))}

    from .model import n_params
    meta = {
        "model": MODEL_NAME, "version": MODEL_VERSION, "trained_at": time.time(),
        "device": device_name(dev), "torch": torch.__version__,
        "seq_len": ds.SEQ_LEN, "tech_keys": list(ds.TECH_KEYS), "static_keys": P.static_keys,
        "sectors": P.sectors, "horizons": horizons, "symbols": P.syms, "rows": P.rows, "params": n_params(nets[0]),
        "seeds": cfg.seeds, "config": asdict(cfg), "epochs_median": n_ep, "hmm": P.hmm, "hmm_end_ts": P.hmm_end_ts,
        "coverage": P.coverage, "notes": P.notes,
        "evaluation": {"test_start": ds.date_of(P.cal[P.folds[0][0]]), "test_end": ds.date_of(P.cal[P.folds[-1][1] - 1]),
                       "folds": [[ds.date_of(P.cal[a]), ds.date_of(P.cal[b - 1])] for a, b in P.folds],
                       "purge_days": ds.PURGE_DAYS, "hmm_fit_end": ds.date_of(P.hmm_end_ts) if P.hmm_end_ts else None,
                       "baselines": [n for n in MODELS if n != "neural" and (cfg.gbm or n != "gbm")]},
        "fund_keys": list(fu.FUND_KEYS), "news_keys": list(nw.NEWS_KEYS), "cs_keys": list(ds.CS_KEYS),
        "timing": {**timing, "total_s": round(time.perf_counter() - t0, 1)},
    }
    return ens, meta


# --------------------------------------------------------------------------- export
def export(ens, P: ds.Panel, meta: dict, out_dir: Path | str) -> tuple[Path, Path]:
    """model.onnx + model.json in out_dir; the ONNX output is checked against PyTorch on real rows."""
    torch = _torch()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path, meta_path = out / "model.onnx", out / "model.json"
    sample = np.linspace(0, P.rows - 1, min(P.rows, 256)).astype(int)
    seq = torch.from_numpy(P.seq(sample))
    st = torch.from_numpy(P.static[sample])
    tmp = onnx_path.with_name(f"model.{os.getpid()}.onnx.tmp")
    torch.onnx.export(ens, (seq, st), str(tmp), input_names=["seq", "static"], output_names=["p_up"],
                      dynamic_axes={"seq": {0: "n"}, "static": {0: "n"}, "p_up": {0: "n"}}, opset_version=17,
                      dynamo=False)
    with torch.no_grad():
        ref = ens(seq, st).numpy()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    got = sess.run(["p_up"], {"seq": seq.numpy(), "static": st.numpy()})[0]
    diff = float(np.abs(got - ref).max())
    if diff > 1e-4:
        raise RuntimeError(f"ONNX export does not match PyTorch (max difference {diff:.2e})")
    tmp.replace(onnx_path)
    meta = dict(meta, onnx_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest(), onnx_check_max_diff=diff)
    meta_path.write_text(json.dumps(meta, separators=(",", ":"), allow_nan=False, default=_json_default))
    return onnx_path, meta_path


def _json_default(o):
    if isinstance(o, np.generic):
        v = o.item()
        return None if isinstance(v, float) and not math.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)
