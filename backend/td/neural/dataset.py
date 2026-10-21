"""The panel NeuralEdge learns from: one row per stock and trading day.

Each row has
  seq     the stock's last SEQ_LEN days of QuantEdge's 26 causal technical features (td.research.quant), each
          standardised with the stock's own expanding mean/sd — the price-history input of the network;
  static  what is known that day besides the stock's own history: cross-sectional ranks of momentum / volume /
          trend features and of point-in-time fundamentals (rank among all stocks that day, −1 … +1, with a
          "has value" mask), news features, ranks of the day's 1-minute summaries (td.neural.minute, with an
          "m_on" mask; only when the 1-minute tables were given), a lender flag and the sector;
  fwd     forward log returns over HORIZONS (1, 5, 21 trading days) — the labels (P(up) = fwd > 0, as QuantEdge) —
          followed by the same returns minus that day's median over the panel's stocks (P(beat) = > 0).

The rows, the walk-forward test periods, the HMM fit cutoff and the purge rules are QuantEdge's own, so a
NeuralEdge AUC and a QuantEdge AUC on this panel are measured on the same test days.

`build(...)` makes the training panel; `build(..., last_only=True)` the inputs for the latest day of every stock
(the server's daily predictions) with the same code, so training and serving cannot drift apart;
`build(..., walk_forward=False, hmm_end_ts=…)` every complete row without folds — the daily context of the
09:30 model (td.neural.intraday), which has its own test periods and therefore its own HMM cutoff.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..research import quant as qm
from ..universe_data import SECTOR_OF
from . import fundamentals as fu
from . import minute as mn
from . import news as nw

log = logging.getLogger("td.neural.dataset")

SEQ_LEN = 40
HORIZONS = qm.HORIZONS
# the network's outputs: "up" = the stock's own h-day return > 0; "beat" = it beats that day's median stock over
# h days (a cross-sectional bet: the market's direction cancels out of it)
HEAD_KEYS: tuple[str, ...] = tuple(f"up{h}" for h in HORIZONS) + tuple(f"beat{h}" for h in HORIZONS)
TECH_KEYS = qm.FEATURE_KEYS
CS_KEYS: tuple[str, ...] = ("mom_21", "mom_126", "mom_12_1", "resid_mom", "volume_z", "vol_regime", "dist_high",
                            "ma_200")
PURGE_DAYS = 2 * max(HORIZONS)      # one network predicts every horizon: train rows end 2·21 days before a test fold


def static_keys(sectors: list[str], with_minute: bool = False) -> list[str]:
    mk = [f"cs_{k}" for k in mn.DAY_KEYS] + ["m_on"] if with_minute else []
    return ([f"cs_{k}" for k in CS_KEYS] + [f"f_{k}" for k in fu.FUND_KEYS] + [f"has_{k}" for k in fu.FUND_KEYS]
            + list(nw.NEWS_KEYS) + mk + ["is_fin"] + [f"sec_{s}" for s in sectors])


@dataclass
class Panel:
    syms: list[str]
    sectors: list[str]
    static_keys: list[str]
    Z: np.ndarray                   # (sum of bars, F) float32, every stock's z features, NaN → 0
    offset: np.ndarray              # first row of each stock in Z
    row_sym: np.ndarray             # rows ↓
    row_pos: np.ndarray             # bar index within the stock
    row_ts: np.ndarray
    row_cal: np.ndarray             # index in the union calendar
    row_hmm_cal: np.ndarray         # calendar index of the last return the stock's HMM saw
    zrow: np.ndarray                # (rows, F) z features at the row's bar (finite) — the QuantEdge inputs
    static: np.ndarray              # (rows, S) float32
    fwd: np.ndarray                 # (rows, len(HEAD_KEYS)) forward log returns then relative ones, NaN = no label
    close: np.ndarray               # close at the row's bar
    cal: np.ndarray                 # union calendar (ts)
    hmm: dict[str, dict] = field(default_factory=dict)
    hmm_end_ts: int | None = None
    test_start: int | None = None
    folds: list[tuple[int, int]] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return int(self.row_sym.size)

    def seq(self, idx: np.ndarray) -> np.ndarray:
        """(len(idx), SEQ_LEN, F) float32 sequences ending at each row's bar."""
        idx = np.asarray(idx)
        start = self.offset[self.row_sym[idx]] + self.row_pos[idx] - (SEQ_LEN - 1)
        return self.Z[start[:, None] + np.arange(SEQ_LEN)[None, :]]


def _cs_rank(values: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Per-day cross-sectional rank of each column, mapped to −1 … +1; NaN stays NaN."""
    df = pd.DataFrame(values)
    df["_d"] = day
    r = df.groupby("_d").rank(pct=True, method="average")
    cnt = df.groupby("_d").transform("count")
    out = (2.0 * r - 1.0).where(cnt > 1, 0.0)
    return out.to_numpy(dtype=float)


def eligible(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """QuantEdge's panel rule: stocks (no indices) with ≥ 3 years of daily bars and enough to warm up."""
    return {s: df for s, df in frames.items()
            if not qm.is_index(s) and len(df) > qm.FEATURE_WARMUP + max(HORIZONS)
            and (df["ts"].iat[-1] - df["ts"].iat[0]) / qm.YEAR_S >= qm.MIN_TRAIN_YEARS}


def build(frames: dict[str, pd.DataFrame], index_df: pd.DataFrame | None, *,
          fundamentals: dict[str, dict | None] | None = None, news: pd.DataFrame | None = None,
          news_start: float | None = None, minute: dict[str, pd.DataFrame] | None = None,
          with_minute: bool | None = None, hmm: dict[str, dict] | None = None, hmm_end_ts: int | None = None,
          sectors: list[str] | None = None, last_only: bool = False, walk_forward: bool = True,
          progress: Callable[[float, str], None] | None = None) -> Panel:
    """The panel. Training (last_only=False): QuantEdge's eligible stocks, every complete row, walk-forward folds.
    Serving (last_only=True): every stock with enough history, its latest row only, using the trained model's
    per-stock HMMs (`hmm`) and sector list so the inputs line up with the network's.
    `minute` ({symbol: td.neural.minute.day_table}) adds the 1-minute day summaries; `with_minute` forces the
    input layout either way (serving follows the trained model's layout, with or without the tables)."""
    say = progress or (lambda f, m: None)
    fundamentals = fundamentals or {}
    minute = minute or {}
    with_minute = bool(minute) if with_minute is None else with_minute
    if index_df is None or len(index_df) < 300:
        raise ValueError("NIFTY daily history is missing from the lake — the neural model needs it")
    if last_only or not walk_forward:
        panel = {s: df for s, df in frames.items()
                 if not qm.is_index(s) and len(df) > qm.FEATURE_WARMUP + SEQ_LEN}
    else:
        panel = eligible(frames)
    if len(panel) < 3:
        raise ValueError(f"only {len(panel)} stocks have enough daily history (need ≥ 3 with "
                         f"{qm.MIN_TRAIN_YEARS:g}+ years)")
    syms = sorted(panel)
    cal = np.unique(np.concatenate([panel[s]["ts"].to_numpy(dtype=np.int64) for s in syms]))
    test_start, folds = None, []
    if not last_only and walk_forward:
        first_row_ts = min(int(panel[s]["ts"].iat[qm.FEATURE_WARMUP]) for s in syms)
        test_start, folds = qm._fold_bounds(cal.size, int(np.searchsorted(cal, first_row_ts)))
        auto = int(cal[max(test_start - 2 * max(HORIZONS) - 1, 0)])
        hmm_end_ts = min(int(hmm_end_ts), auto) if hmm_end_ts else auto
    elif not last_only and not hmm_end_ts:
        raise ValueError("walk_forward=False needs hmm_end_ts (the HMM must not see the caller's test periods)")
    secs = sectors or sorted({SECTOR_OF.get(s, "OTHER") for s in syms})
    skeys = static_keys(secs, with_minute)

    Zs, offs, parts = [], [], []
    hmm_out: dict[str, dict] = {}
    off = 0
    n_fund = 0
    for i, s in enumerate(syms):
        if i % 10 == 0:
            say(0.05 + 0.75 * i / len(syms), f"features {i}/{len(syms)}")
        df = panel[s]
        params = (hmm or {}).get(s)
        fe = qm.compute_features(df, index_df, hmm=params, hmm_end_ts=None if params else hmm_end_ts)
        if fe.hmm is not None:
            hmm_out[s] = qm.hmm_params(fe.hmm, fe.hmm_fit_end_ts)
        n = fe.ts.size
        z = np.nan_to_num(fe.z, nan=0.0).astype(np.float32)
        ok = np.isfinite(fe.z).all(axis=1) & (np.arange(n) >= SEQ_LEN - 1)
        if last_only:
            ok[:-1] = False
        if not ok.any():
            Zs.append(z)
            offs.append(off)
            off += n
            continue
        pos = np.flatnonzero(ok)
        f = fundamentals.get(s)
        fv = fu.daily_values(f, fe.ts, fe.close)[pos]
        if f:
            n_fund += 1
        nv = nw.daily_features(news, s, fe.ts, news_start)[pos]
        mv = mn.daily_values(minute.get(s), fe.ts[pos]) if with_minute else np.zeros((pos.size, 0))
        cs = fe.raw[pos][:, [qm.FI[k] for k in CS_KEYS]]
        fit_end = fe.hmm_fit_end_ts if fe.hmm_fit_end_ts is not None else (hmm_end_ts or int(fe.ts[0]))
        parts.append({
            "sym": np.full(pos.size, i), "pos": pos, "ts": fe.ts[pos],
            "cal": np.searchsorted(cal, fe.ts[pos]),
            "hmm_cal": np.full(pos.size, int(np.searchsorted(cal, fit_end, side="right")) - 1),
            "z": fe.z[pos].astype(np.float32), "fund": fv, "news": nv, "min": mv, "cs": cs,
            "fwd": np.column_stack([fe.fwd[h][pos] for h in HORIZONS]), "close": fe.close[pos],
            "fin": np.full(pos.size, 1.0 if f and fu.is_financial(f) else 0.0),
            "sec": np.full(pos.size, secs.index(SECTOR_OF.get(s, "OTHER")) if SECTOR_OF.get(s, "OTHER") in secs
                           else -1),
        })
        Zs.append(z)
        offs.append(off)
        off += n
    if not parts:
        raise ValueError("no stock has a complete feature row")
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    say(0.85, "cross-sectional ranks")
    day = cat["cal"]
    absr = cat["fwd"]
    med = pd.DataFrame(absr).groupby(day).transform("median").to_numpy()
    cat["fwd"] = np.hstack([absr, absr - med])
    cs_r = np.nan_to_num(_cs_rank(cat["cs"], day), nan=0.0)
    fund_r = _cs_rank(cat["fund"], day)
    has = np.isfinite(cat["fund"]).astype(float)
    fund_r = np.nan_to_num(fund_r, nan=0.0)
    sec_1h = np.zeros((day.size, len(secs)))
    good = cat["sec"] >= 0
    sec_1h[np.flatnonzero(good), cat["sec"][good]] = 1.0
    blocks = [cs_r, fund_r, has, cat["news"]]
    m_on = np.zeros(day.size)
    if with_minute:
        m_on = np.isfinite(cat["min"]).all(axis=1).astype(float)
        m_r = np.nan_to_num(_cs_rank(np.where(m_on[:, None] > 0, cat["min"], np.nan), day), nan=0.0)
        blocks += [m_r, m_on[:, None]]
    static = np.column_stack(blocks + [cat["fin"], sec_1h]).astype(np.float32)
    assert static.shape[1] == len(skeys)

    cover = {
        "stocks": len(syms), "rows": int(day.size),
        "fundamentals_stocks": n_fund,
        "fundamentals_rows": round(float(has[:, fu.FUND_KEYS.index("a_roe")].mean()), 3),
        "quarterly_rows": round(float(has[:, fu.FUND_KEYS.index("q_np_yoy")].mean()), 3),
        "news_rows": round(float(cat["news"][:, nw.NEWS_KEYS.index("news_on")].mean()), 3),
        "news_start": news_start,
        "minute_rows": round(float(m_on.mean()), 3) if with_minute else None,
    }
    notes = []
    if cover["news_rows"] < 0.1:
        notes.append(f"news covers {cover['news_rows'] * 100:.0f}% of rows — the news inputs carry almost no "
                     "information yet; they start to count after a few months of recorded headlines")
    if n_fund == 0:
        notes.append("no fundamentals were available — the fundamental inputs are all masked")
    if with_minute and cover["minute_rows"] < 0.1:
        notes.append(f"1-minute summaries cover {cover['minute_rows'] * 100:.0f}% of rows — backfill 1-minute "
                     "history (ADMIN → history) for them to count")
    return Panel(syms=syms, sectors=secs, static_keys=skeys, Z=np.concatenate(Zs), offset=np.array(offs),
                 row_sym=cat["sym"], row_pos=cat["pos"], row_ts=cat["ts"], row_cal=cat["cal"],
                 row_hmm_cal=cat["hmm_cal"], zrow=np.nan_to_num(cat["z"], nan=0.0), static=static,
                 fwd=cat["fwd"], close=cat["close"], cal=cal, hmm=hmm_out, hmm_end_ts=hmm_end_ts,
                 test_start=test_start, folds=folds, coverage=cover, notes=notes)


def date_of(ts: int) -> str:
    return qm._date(int(ts))


def purge_cut(fold_start: int, h: int | None = None) -> int:
    """Last calendar index a training row may have for a test fold starting at `fold_start`."""
    return fold_start - (2 * h if h else PURGE_DAYS) - 1


def is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
