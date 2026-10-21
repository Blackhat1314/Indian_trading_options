"""NeuralEdge on the server: loads data/neural/model.onnx + model.json (trained on a PC), scores the latest day
of every stock with onnxruntime (no PyTorch here), and serves the results to the NEURAL desk.

- The model files are re-read when they change (upload new ones; no restart needed).
- Inputs are built by td.neural.dataset with `last_only=True` — the training code itself — using the model's
  stored per-stock HMMs and sector list, so serving cannot drift from training.
- Fundamentals come from the Screener.in cache the SCREENER desk already keeps; a background job tops up stocks
  that have none or an old copy, one polite request at a time. News comes from the recorder's archive.
- A horizon without a walk-forward edge is not ranked: its probabilities are the base rate, and the leaderboard
  says so instead of sorting noise.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from ..config import settings
from ..research import quant as qm
from ..timeutil import IST
from . import dataset as ds
from . import news as nw

log = logging.getLogger("td.neural.runtime")

MODEL_NAME, MODEL_VERSION = "NeuralEdge", 1
FUND_MAX_AGE_S = 3 * 86400        # fundamentals older than this are refetched in the background
CHECK_FILES_S = 30.0
PREDICT_TTL_S = 600.0


def model_dir() -> Path:
    return Path(settings.data_dir) / "neural"


def _f(x, nd: int = 4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


class NeuralService:
    def __init__(self, data, screener=None, *, folder: Path | str | None = None):
        self.data = data
        self.screener = screener
        self.folder = Path(folder) if folder else model_dir()
        self._sess = None
        self._meta: dict | None = None
        self._stamp: tuple | None = None
        self._checked = 0.0
        self._error: str | None = None
        self._pred: dict | None = None             # {"at", "as_of", "rows": {...}, "key"}
        self._task: asyncio.Task | None = None
        self._progress, self._message = 0.0, ""
        self._fund_task: asyncio.Task | None = None

    # ------------------------------------------------------------ model files
    def _files(self) -> tuple[Path, Path]:
        return self.folder / "model.onnx", self.folder / "model.json"

    def _load(self) -> None:
        """(Re)load the model when its files appeared or changed."""
        now = time.monotonic()
        if now - self._checked < CHECK_FILES_S and self._stamp is not None:
            return
        self._checked = now
        onnx_p, meta_p = self._files()
        if not onnx_p.exists() or not meta_p.exists():
            self._sess, self._meta, self._stamp, self._error = None, None, None, None
            return
        stamp = (onnx_p.stat().st_mtime_ns, onnx_p.stat().st_size, meta_p.stat().st_mtime_ns)
        if stamp == self._stamp:
            return
        try:
            meta = json.loads(meta_p.read_text())
            if meta.get("model") != MODEL_NAME or int(meta.get("version", 0)) != MODEL_VERSION:
                raise ValueError("not a NeuralEdge model of this version — retrain with the current code")
            if list(meta.get("tech_keys") or []) != list(ds.TECH_KEYS):
                raise ValueError("model was trained on a different technical feature set — retrain")
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            sess = ort.InferenceSession(str(onnx_p), sess_options=so, providers=["CPUExecutionProvider"])
            self._sess, self._meta, self._error = sess, meta, None
            self._pred = None
            log.info("NeuralEdge model loaded (trained %s on %s)",
                     time.strftime("%Y-%m-%d %H:%M", time.localtime(meta["trained_at"])), meta.get("device"))
        except Exception as e:
            self._sess, self._meta = None, None
            self._error = f"{type(e).__name__}: {e}"
            log.warning("NeuralEdge model unusable: %s", self._error)
        self._stamp = stamp

    # ------------------------------------------------------------ public
    async def info(self) -> dict:
        self._load()
        if self._error:
            return {"status": "error", "error": self._error, "folder": str(self.folder)}
        m = self._meta
        if m is None:
            return {"status": "missing", "folder": str(self.folder)}
        out = {k: m.get(k) for k in ("trained_at", "device", "torch", "rows", "params", "seeds", "seq_len", "coverage",
                                     "notes", "evaluation", "timing", "epochs_median")}
        out["symbols"] = len(m.get("symbols") or [])
        out["horizons"] = [{"h": hz["h"], "label": hz["label"], "verdict": hz["verdict"], "metrics": hz["metrics"],
                            "calibration": hz.get("calibration")} for hz in m["horizons"]]
        p = self._pred
        if not p:
            self._kick()
        out.update(status="ready" if p else "computing", predicted_at=p["at"] if p else None,
                   as_of=p["as_of"] if p else None, progress=self._progress, message=self._message)
        return out

    async def leaderboard(self, h: int, n: int = 15) -> dict:
        info = await self.info()
        if info["status"] in ("missing", "error"):
            return {"model": info, "h": h, "top": [], "bottom": [], "ranked": False}
        p = await self._predictions(wait=False)
        if p is None:
            return {"model": info, "h": h, "top": [], "bottom": [], "ranked": False, "computing": True}
        hz = self._horizon(h)
        edge = bool(hz["metrics"]["neural"].get("edge"))
        rows = sorted(p["rows"].values(), key=lambda r: r["p"][str(h)], reverse=True)

        def pick(rs: list[dict]) -> list[dict]:
            return [{"symbol": r["symbol"], "sector": r["sector"], "p_up": r["p"][str(h)], "close": r["close"]}
                    for r in rs]
        out = {"model": info, "h": h, "as_of": p["as_of"], "ranked": edge, "n": len(rows),
               "verdict": hz["verdict"], "edge_note": hz["metrics"]["neural"].get("edge_note")}
        # without a measured edge the probabilities are the base rate for everyone: sorting them would rank noise
        out["top"], out["bottom"] = (pick(rows[:n]), pick(rows[::-1][:n])) if edge else ([], [])
        return out

    async def result(self, symbol: str) -> dict:
        info = await self.info()
        if info["status"] in ("missing", "error"):
            return {"symbol": symbol, "status": info["status"], "model": info}
        p = await self._predictions(wait=False)
        if p is None:
            return {"symbol": symbol, "status": "computing", "model": info}
        r = p["rows"].get(symbol.upper())
        if r is None:
            raise KeyError(f"{symbol}: no NeuralEdge prediction (not enough daily history in the lake, or an index)")
        hs = []
        for hz in self._meta["horizons"]:
            h = hz["h"]
            nm = hz["metrics"]["neural"]
            edge = bool(nm.get("edge"))
            hs.append({"h": h, "label": hz["label"], "p_up": r["p"][str(h)], "edge": edge,
                       "edge_note": nm.get("edge_note"), "oos_auc": nm.get("auc"), "verdict": hz["verdict"],
                       "rank_pct": r["rank"][str(h)] if edge else None})
        return {"symbol": r["symbol"], "status": "ready", "as_of": p["as_of"], "close": r["close"],
                "sector": r["sector"], "horizons": hs, "inputs": r["inputs"], "model": info}

    # ------------------------------------------------------------ predictions
    @property
    def _busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def _horizon(self, h: int) -> dict:
        for hz in self._meta["horizons"]:
            if hz["h"] == h:
                return hz
        raise ValueError(f"the model has no {h}-day horizon")

    def _kick(self) -> None:
        if not self._busy and self._sess is not None:
            self._task = asyncio.create_task(self._compute(), name="neural-predict")

    async def _predictions(self, wait: bool) -> dict | None:
        p = self._pred
        if p and time.time() - p["at"] < PREDICT_TTL_S:
            return p
        self._kick()
        if wait and self._task is not None:
            await asyncio.shield(self._task)
            return self._pred
        return p                                    # the previous table while a new one is computed

    async def _frames(self) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        lake = getattr(self.data, "lake", None)
        if lake is None:
            from ..lake import Lake
            lake = Lake(settings.lake_dir)
        syms = [s for s in await asyncio.to_thread(lake.symbols, "D") if not qm.is_index(s)]
        frames = {}
        for i, s in enumerate(syms):
            if i % 25 == 0:
                self._progress, self._message = 0.3 * i / max(len(syms), 1), f"loading daily bars {i}/{len(syms)}"
            try:
                df, _ = await self.data.daily(s, fetch=False)
            except Exception:
                continue
            if len(df):
                frames[s] = df
        index_df = await self.data.index_daily("NIFTY")
        return frames, index_df

    def _fundamentals(self, syms: list[str]) -> dict[str, dict]:
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

    async def _compute(self) -> None:
        meta, sess = self._meta, self._sess
        if meta is None or sess is None:
            return
        t0 = time.perf_counter()
        try:
            frames, index_df = await self._frames()
            self._progress, self._message = 0.35, "fundamentals and news"
            fund = await asyncio.to_thread(self._fundamentals, sorted(frames))
            root = settings.recorder_dir
            since = (datetime.now(tz=IST) - timedelta(days=10)).date().isoformat()
            news_df = await asyncio.to_thread(nw.load_archive, root, since)
            start = _archive_first_day(root)
            self._progress, self._message = 0.45, "technical features"
            P = await asyncio.to_thread(ds.build, frames, index_df, fundamentals=fund, news=news_df,
                                        news_start=start, hmm=meta.get("hmm") or {}, sectors=meta["sectors"],
                                        last_only=True)
            if P.static_keys != meta["static_keys"]:
                raise ValueError("the model expects a different input layout — retrain it with the current code")
            idx = np.arange(P.rows)
            probs = await asyncio.to_thread(lambda: sess.run(["p_up"], {"seq": P.seq(idx),
                                                                        "static": P.static})[0])
            rows = {}
            ranks = {}
            for hi, hz in enumerate(meta["horizons"]):
                cal = hz.get("calibration") or {}
                p_cal = qm.calib_apply(probs[:, hi].astype(float), cal.get("base", 0.5), cal.get("b", 1.0))
                probs[:, hi] = p_cal
                order = np.argsort(np.argsort(-p_cal))
                ranks[hz["h"]] = 100.0 * (1 - order / max(P.rows - 1, 1))
            fund_cols = {k: meta["static_keys"].index(f"has_{k}") for k in meta["fund_keys"]}
            for i in range(P.rows):
                sym = P.syms[P.row_sym[i]]
                rows[sym] = {"symbol": sym, "sector": _sector(meta, P, i), "close": _f(P.close[i], 2),
                             "p": {str(hz["h"]): _f(probs[i, hi]) for hi, hz in enumerate(meta["horizons"])},
                             "rank": {str(h): _f(v[i], 1) for h, v in ranks.items()},
                             "inputs": {"fundamentals": int(sum(P.static[i, c] for c in fund_cols.values())),
                                        "fundamentals_of": len(fund_cols),
                                        "news_on": bool(P.static[i, meta["static_keys"].index("news_on")])}}
            as_of = ds.date_of(int(max(P.row_ts))) if P.rows else None
            self._pred = {"at": time.time(), "as_of": as_of, "rows": rows}
            self._progress, self._message = 1.0, "ready"
            log.info("NeuralEdge scored %d stocks as of %s in %.0f s", len(rows), as_of, time.perf_counter() - t0)
            self._top_up_fundamentals(sorted(frames), fund)
        except Exception as e:
            self._error = f"prediction failed: {type(e).__name__}: {e}"
            log.exception("NeuralEdge prediction failed")

    # ------------------------------------------------------------ fundamentals top-up
    def _top_up_fundamentals(self, syms: list[str], have: dict[str, dict]) -> None:
        if self.screener is None or (self._fund_task is not None and not self._fund_task.done()):
            return
        now = time.time()
        todo = [s for s in syms if s not in have or now - float(have[s].get("fetched_at") or 0) > FUND_MAX_AGE_S]
        if not todo:
            return

        async def run():
            got = 0
            for s in todo:
                try:
                    await self.screener.fundamentals(s)
                    got += 1
                except Exception as e:                  # not on Screener, queue busy, offline: next time
                    log.debug("neural fundamentals %s: %s", s, e)
                await asyncio.sleep(1.0)
            log.info("NeuralEdge: fundamentals refreshed for %d/%d stocks", got, len(todo))
            self._pred = None                            # next request recomputes with them
        self._fund_task = asyncio.create_task(run(), name="neural-fundamentals")


def _sector(meta: dict, P: ds.Panel, i: int) -> str:
    secs = meta["sectors"]
    one = P.static[i, len(P.static_keys) - len(secs):]
    j = int(np.argmax(one)) if one.size and one.max() > 0 else -1
    return secs[j] if j >= 0 else "OTHER"


def _archive_first_day(root: Path) -> float | None:
    """Epoch of the first recorded day that holds news (the archive's start, for the news_on mask)."""
    if not Path(root).exists():
        return None
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if any(d.glob("news-*.parquet")):
            try:
                day = datetime.strptime(d.name, "%Y-%m-%d").replace(tzinfo=IST)
            except ValueError:
                continue
            return day.timestamp()
    return None
