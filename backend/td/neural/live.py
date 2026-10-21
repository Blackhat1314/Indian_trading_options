"""The 09:30 model inside the analytics worker: neural picks at the lock, next to the TOP5.

- Model files: data/neural/intraday.onnx + intraday.json (trained on a PC by research/neural_train.py). Re-read when
  they change; nothing happens without them.
- Before the open (at the worker's day start) the daily context of every stock as of the previous close is built
  in a separate process — it is the training code (td.neural.dataset, last_only=True) and takes minutes of CPU,
  which must not stall the tick loop.
- At the lock the worker hands over its 09:30 state: each stock's first 15 one-minute bars, its context (ATR, ADV,
  beta, usual-volume curve — the same load_context numbers the training rows were built with) and NIFTY's bars.
  td.neural.intraday.day_features turns them into the network's inputs exactly as in training.
- A model without a measured edge makes no picks (its verdict says why); the TOP5 stays in charge either way.
- The training chose what scores the picks (meta["deploy"]): the network, gradient-boosted trees (intraday-gbm.json,
  evaluated with numpy by td.neural.trees — no scikit-learn here) or their blend. All three read the same 09:30
  inputs (td.neural.intraday.flat_inputs for the trees).
- A stock is skipped when its daily history is a day behind the others or the feed's previous close disagrees with
  the lake's by more than 0.5 % (a split or bonus the bars don't adjust for).
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np

from ..config import settings
from ..timeutil import IST
from . import intraday as it
from . import minute as mn

log = logging.getLogger("td.neural.live")

MODEL_NAME, MODEL_VERSION = "NeuralEdge-0930", 1
PDC_TOLERANCE = 0.005
CHECK_FILES_S = 60.0


def model_files(folder: Path | None = None) -> tuple[Path, Path]:
    d = Path(folder) if folder else Path(settings.data_dir) / "neural"
    return d / "intraday.onnx", d / "intraday.json"


def read_meta(path: Path) -> dict:
    meta = json.loads(path.read_text())
    if meta.get("model") != MODEL_NAME or int(meta.get("version", 0)) != MODEL_VERSION:
        raise ValueError("not a 09:30 NeuralEdge model of this version — retrain with the current code")
    if list(meta.get("bar_keys") or []) != list(it.BAR_KEYS) or list(meta.get("intra_keys") or []) != list(it.INTRA_KEYS):
        raise ValueError("the 09:30 model was trained on a different input layout — retrain with the current code")
    return meta


def cached_fundamentals(syms: list[str]) -> dict[str, dict]:
    cache = Path(settings.data_dir) / "cache" / "screener"
    out = {}
    for s in syms:
        for view in ("auto", "consolidated", "standalone"):
            f = cache / f"{quote(s, safe='')}.{view}.json"
            if f.exists():
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    out[s] = d.get("fundamentals") or d
                    break
                except (OSError, ValueError):
                    pass
    return out


def prepare_context(day_iso: str, meta_path: str) -> dict:
    """Daily context of every stock as of the close before `day_iso` (runs in a worker process)."""
    from ..lake import Lake
    from ..research import quant as qm
    from ..research.data import clean_daily
    from . import dataset as ds
    from . import news as nw
    from .runtime import _archive_first_day
    t0 = time.time()
    meta = read_meta(Path(meta_path))
    day = date.fromisoformat(day_iso)
    cutoff = int(datetime(day.year, day.month, day.day, tzinfo=IST).timestamp())
    lake = Lake(settings.lake_dir)
    frames = {}
    for s in lake.symbols("D"):
        if qm.is_index(s):
            continue
        df = clean_daily(lake.read("D", s))
        df = df[df["ts"] < cutoff].reset_index(drop=True)
        if len(df):
            frames[s] = df
    index_df = clean_daily(lake.read("D", "NIFTY"))
    index_df = index_df[index_df["ts"] < cutoff].reset_index(drop=True)
    with_minute = bool(meta.get("with_minute"))
    tables = {}
    if with_minute:
        for s in frames:
            m = lake.read("1m", s, last_n=mn.SESSION * 4)
            if len(m):
                tables[s] = mn.day_table(m[m["ts"] < cutoff])
    root = settings.recorder_dir
    since = (day - timedelta(days=10)).isoformat()
    news_df = nw.load_archive(root, since)
    P = ds.build(frames, index_df, fundamentals=cached_fundamentals(sorted(frames)), news=news_df,
                 news_start=_archive_first_day(root), minute=tables, with_minute=with_minute,
                 hmm=meta.get("hmm") or {}, sectors=meta["sectors"], last_only=True)
    if P.static_keys != meta["static_keys"]:
        raise ValueError("the 09:30 model expects a different daily input layout — retrain it with the current code")
    rows = {}
    seq = P.seq(np.arange(P.rows))
    days = mn.ist_day(P.row_ts)
    for i in range(P.rows):
        rows[P.syms[P.row_sym[i]]] = (seq[i], P.static[i], int(days[i]))
    return {"day": day_iso, "rows": rows, "seconds": round(time.time() - t0, 1)}


class LiveIntraday:
    def __init__(self, folder: Path | str | None = None):
        self.folder = Path(folder) if folder else None
        self.meta: dict | None = None
        self.sess = None
        self.gbm: dict | None = None
        self.error: str | None = None
        self._stamp = None
        self._checked = 0.0
        self.ctx: dict | None = None              # prepare_context() result
        self._task: asyncio.Task | None = None
        self._pool: ProcessPoolExecutor | None = None

    # ------------------------------------------------------------------ model files
    def load(self) -> None:
        now = time.monotonic()
        if now - self._checked < CHECK_FILES_S and self._stamp is not None:
            return
        self._checked = now
        onnx_p, meta_p = model_files(self.folder)
        if not onnx_p.exists() or not meta_p.exists():
            self.meta, self.sess, self.error, self._stamp = None, None, None, ("missing",)
            return
        stamp = (onnx_p.stat().st_mtime_ns, onnx_p.stat().st_size, meta_p.stat().st_mtime_ns)
        if stamp == self._stamp:
            return
        self._stamp = stamp
        try:
            meta = read_meta(meta_p)
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(str(onnx_p), sess_options=so, providers=["CPUExecutionProvider"])
            dep = meta.get("deploy") or {}
            self.gbm = None
            if dep.get("model") in ("gbm", "blend"):
                gp = onnx_p.with_name(dep.get("gbm_file") or "intraday-gbm.json")
                if not gp.exists():
                    raise FileNotFoundError(f"{gp.name} is missing — the 09:30 model was trained to use the trees; "
                                            "upload it with intraday.onnx and intraday.json")
                self.gbm = json.loads(gp.read_text())
            self.meta, self.error, self.ctx = meta, None, None
            log.info("09:30 model loaded (trained %s, edge %s)",
                     time.strftime("%Y-%m-%d %H:%M", time.localtime(meta["trained_at"])), meta.get("edge"))
        except Exception as e:
            self.meta, self.sess = None, None
            self.error = f"{type(e).__name__}: {e}"
            log.warning("09:30 model unusable: %s", self.error)

    @property
    def state(self) -> str:
        if self.error:
            return "error"
        if self.meta is None:
            return "missing"
        if not self.meta.get("edge"):
            return "no_edge"
        return "ready"

    def summary(self) -> dict:
        m = self.meta or {}
        return {"status": self.state, "error": self.error, "trained_at": m.get("trained_at"),
                "verdict": m.get("verdict"), "edge": bool(m.get("edge")), "edge_note": m.get("edge_note"),
                "uses": (m.get("deploy") or {}).get("model", "neural") if m else None}

    # ------------------------------------------------------------------ context before the open
    def ready_for(self, day: date) -> bool:
        return bool(self.ctx and self.ctx.get("day") == day.isoformat())

    async def prepare(self, day: date) -> None:
        self.load()
        if self.state != "ready" or self.ready_for(day) or (self._task and not self._task.done()):
            return
        self._task = asyncio.create_task(self._prepare(day), name="neural-0930-context")

    async def _prepare(self, day: date) -> None:
        _, meta_p = model_files(self.folder)
        try:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
            ctx = await asyncio.get_running_loop().run_in_executor(self._pool, prepare_context, day.isoformat(),
                                                                   str(meta_p))
            self.ctx = ctx
            log.info("09:30 model: context for %d stocks as of the close before %s (%.0f s)", len(ctx["rows"]),
                     day, ctx["seconds"])
        except Exception as e:
            self.error = f"context failed: {type(e).__name__}: {e}"
            log.exception("09:30 model context failed")

    # ------------------------------------------------------------------ at the lock
    def score(self, day: date, eqs: list, nifty, news: np.ndarray | None, min_turnover_cr: float) -> dict:
        """Neural picks from the worker's 09:30 state. `eqs`: InstStates; `nifty`: NIFTY's InstState;
        `news`: {symbol: (3,)} overnight news features or None."""
        meta, sess = self.meta, self.sess
        if meta is None or sess is None or not self.ready_for(day):
            return {"status": "pending"}
        rows = self.ctx["rows"]
        latest = max((r[2] for r in rows.values()), default=0)
        use, skipped = [], 0
        for s in eqs:
            r = rows.get(s.symbol)
            c = s.ctx
            if r is None or not (c.pdc > 0) or not s.ltp:
                continue
            if r[2] < latest or (s.pc and abs(s.pc / c.pdc - 1) > PDC_TOLERANCE):
                skipped += 1
                continue
            use.append(s)
        if len(use) < it.MIN_STOCKS or nifty is None or not nifty.ctx.pdc:
            return {"status": "error", "note": f"only {len(use)} stocks have a current daily context"}
        n = len(use)
        bars = np.full((n, it.LOCK_MIN, 5), np.nan)
        for i, s in enumerate(use):
            for m_, b in s.bars.items():
                if 0 <= int(m_) < it.LOCK_MIN:
                    bars[i, int(m_)] = b[:5]
        nb = np.full((it.LOCK_MIN, 5), np.nan)
        for m_, b in nifty.bars.items():
            if 0 <= int(m_) < it.LOCK_MIN:
                nb[int(m_)] = b[:5]
        nb[:, 4] = 0.0
        di = it.DayInputs(syms=[s.symbol for s in use], bars=bars,
                          base=np.stack([it.base_from_curve(s.ctx.base_cum, s.ctx.adv) for s in use]),
                          pdc=np.array([s.ctx.pdc for s in use]), atr=np.array([s.ctx.atr for s in use]),
                          adv=np.array([s.ctx.adv for s in use]), beta=np.array([s.ctx.beta or 1.0 for s in use]),
                          nifty=nb, nifty_pdc=float(nifty.ctx.pdc), nifty_atr=float(nifty.ctx.atr),
                          news=news if news is not None and len(news) == n else np.zeros((n, 3)),
                          vol=np.array([float(s.vol) for s in use]))
        from ..analytics.intraday import RankConfig
        cfg = RankConfig(min_turnover_cr=min_turnover_cr)
        cfg.weights.update({k: float(v) for k, v in (meta.get("ranker_weights") or {}).items() if k in cfg.weights})
        res = it.day_features(di, cfg)
        k = len(res["sym"])
        if k < it.MIN_STOCKS:
            return {"status": "error", "note": f"only {k} stocks had enough 1-minute bars by 09:30"}
        seq = np.stack([rows[s][0] for s in res["sym"]]).astype(np.float32)
        static = np.stack([rows[s][1] for s in res["sym"]]).astype(np.float32)
        p = self.raw(seq, static, res["bars"], res["intra"])
        from ..research import quant as qm
        dep = meta.get("deploy") or {}
        cals = dep.get("calibration") or [hz.get("calibration") or {} for hz in meta["heads"]]
        for hi in range(p.shape[1]):
            cal = cals[hi] or {}
            p[:, hi] = qm.calib_apply(p[:, hi].astype(float), cal.get("base", 0.5), cal.get("b", 1.0))
        from ..universe_data import SECTOR_OF
        sectors = np.array([SECTOR_OF.get(s, "OTHER") for s in res["sym"]])
        lo, sh = it.top_bottom(p[:, 1], res["liquid"], sectors, meta.get("top_n", it.TOP_N),
                               meta.get("per_sector", it.MAX_PER_SECTOR))
        f = lambda x: round(float(x), 4) if math.isfinite(float(x)) else None  # noqa: E731
        return {"status": "ok", "computed_at": time.time(), "n": k, "skipped": skipped,
                "long": [res["sym"][i] for i in lo], "short": [res["sym"][i] for i in sh],
                "p": {s: [f(p[i, 0]), f(p[i, 1])] for i, s in enumerate(res["sym"])},
                **{key: v for key, v in self.summary().items() if key not in ("error", "status")}}

    def raw(self, seq: np.ndarray, static: np.ndarray, bars: np.ndarray, intra: np.ndarray) -> np.ndarray:
        """(n, 2) uncalibrated probabilities of the deployed model: the network, the trees or their mean."""
        model = ((self.meta or {}).get("deploy") or {}).get("model", "neural")
        p_nn = None
        if model in ("neural", "blend"):
            p_nn = self.sess.run(["p"], {"seq": seq, "static": static, "bars": bars, "intra": intra})[0].astype(float)
        if model == "neural":
            return p_nn
        from . import trees
        X = it.flat_inputs(seq, static, bars, intra)
        p_gb = np.column_stack([trees.predict_proba(self.gbm[h], X) for h in it.HEADS])
        return p_gb if model == "gbm" else (p_nn + p_gb) / 2

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)


def overnight_news(eqs: list, day: date, rows: dict | None) -> dict:
    """{symbol: (3,)} headlines between each stock's previous session close and 09:30 (blocking: run in a thread)."""
    from . import news as nw
    from .runtime import _archive_first_day
    root = settings.recorder_dir
    df = nw.load_archive(root, (day - timedelta(days=6)).isoformat())
    start = _archive_first_day(root)
    out = {}
    today = mn.ist_day(int(datetime(day.year, day.month, day.day, tzinfo=IST).timestamp()))
    for s in eqs:
        r = (rows or {}).get(s.symbol)
        prev = r[2] if r else today - 1
        out[s.symbol] = it.overnight_news(df, s.symbol, np.array([today]), np.array([prev]), start)[0]
    return out
