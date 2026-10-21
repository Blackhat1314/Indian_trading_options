#!/usr/bin/env python3
"""Train NeuralEdge on your PC (GPU if PyTorch sees one) from the server's training-data zip.

  1. on the server:  docker compose run --rm tools python /app/research/neural_export.py
                     → data/neural/training-data-YYYYMMDD.zip   (download it)
  2. on the PC:      python research/neural_train.py --data C:\\path\\training-data-20261021.zip --fetch-fundamentals
                     → <data folder>/neural/model.onnx + model.json        (daily model: 1 day, 1 week, 1 month)
                       <data folder>/neural/intraday.onnx + intraday.json  (09:30 → close model; + intraday-gbm.json
                                                                           when the walk-forward chose the trees)
  3. upload the files to the server's data/neural/ folder; the NEURAL desk picks them up within a minute, the
     09:30 picks start the next trading morning.

Two models are trained (--model daily|intraday|both, default both when the zip has 1-minute tables):
  daily     P(up) and P(beats the median stock) after 1 day / 1 week / 1 month, from price history, technicals,
            point-in-time fundamentals, news and the 1-minute day summaries
  intraday  P(up by the close) and P(beats the day's median stock) at 09:30, from the daily context as of the
            previous close plus the first 15 one-minute bars — tested against the live TOP5 ranker on the same days

Time is the price of accuracy here, not of a better backtest: --search N tries N network settings inside every
training window (chosen on that window's own validation slice, never on the test data) and --seeds K averages K
networks. An overnight run: --search 16 --seeds 10. Defaults: --search 0 (the defaults only), --seeds 5.

Fundamentals: the zip carries the server's Screener.in cache (stocks someone opened on the SCREENER desk).
--fetch-fundamentals fetches the rest from Screener.in politely (one page every 3 s, ~20 minutes for the F&O list
the first time, cached after that). Without fundamentals the networks still train; those inputs are masked.

Other options: --epochs 15, --no-gbm (skip the gradient-boosting baseline, faster), --device auto|cuda|cpu,
--out <folder>.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from td.config import settings  # noqa: E402

log = logging.getLogger("neural_train")


def _data_root(arg: str) -> Path:
    p = Path(arg).expanduser().resolve()
    if p.suffix.lower() == ".zip":
        dest = p.with_suffix("")
        stamp = dest / ".extracted"
        if not stamp.exists() or stamp.stat().st_mtime < p.stat().st_mtime:
            print(f"extracting {p.name} → {dest}")
            with zipfile.ZipFile(p) as z:
                for m in z.infolist():                  # no absolute paths or .. from the archive
                    target = (dest / m.filename).resolve()
                    if dest.resolve() not in target.parents and target != dest.resolve():
                        raise SystemExit(f"refusing unsafe path in the zip: {m.filename}")
                z.extractall(dest)
            stamp.write_text(str(time.time()))
        return dest
    if not p.is_dir():
        raise SystemExit(f"{p} is neither the training-data zip nor a data folder")
    return p


def _frames(root: Path):
    from td.lake import Lake
    from td.research.data import clean_daily
    from td.research.quant import is_index
    lake = Lake(root / "lake" / "dhan")
    syms = lake.symbols("D")
    if not syms:
        raise SystemExit(f"no daily bars in {root / 'lake' / 'dhan' / 'D'} — is this the right folder?")
    frames = {}
    for s in syms:
        if is_index(s):
            continue
        df = clean_daily(lake.read("D", s))
        if len(df):
            frames[s] = df
    index_df = clean_daily(lake.read("D", "NIFTY"))
    return frames, index_df


def _tables(root: Path) -> dict:
    import pandas as pd

    from td.lake import _unsafe
    d = root / "intraday"
    out = {}
    for f in sorted(d.glob("*.parquet")) if d.exists() else []:
        try:
            out[_unsafe(f.stem)] = pd.read_parquet(f)
        except Exception as e:
            log.warning("unreadable 1-minute table %s: %s", f.name, e)
    return out


def _cached_fundamentals(caches: list[Path], syms: list[str]) -> dict[str, dict]:
    """Newest cached copy per stock across the caches (the server's, in the zip, and the PC's own)."""
    from urllib.parse import quote
    out = {}
    for s in syms:
        found = [c / f"{quote(s, safe='')}.{view}.json" for c in caches for view in ("auto", "consolidated", "standalone")]
        found = sorted((f for f in found if f.exists()), key=lambda f: f.stat().st_mtime, reverse=True)
        for f in found:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                out[s] = d.get("fundamentals") or d
                break
            except (OSError, ValueError):
                pass
    return out


async def _fetch_fundamentals(cache: Path, syms: list[str], have: dict[str, dict]) -> dict[str, dict]:
    from td.research.screener import ScreenerClient
    client = ScreenerClient(cache_dir=cache, ttl_hours=24 * 30)
    todo = [s for s in syms if s not in have]
    print(f"fetching fundamentals for {len(todo)} stocks from Screener.in (≈{len(todo) * 6 / 60:.0f} min, "
          "cached afterwards) …")
    for i, s in enumerate(todo, 1):
        try:
            have[s] = await client.fundamentals(s)
        except Exception as e:                         # not on Screener, renamed, blocked: train without it
            log.info("fundamentals %s: %s", s, e)
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} done")
    await client.aclose()
    return have


def _n(v, fmt: str) -> str:
    return format(v, fmt) if isinstance(v, (int, float)) else "—"


def daily_table(meta: dict) -> str:
    rows = [f"{'horizon':<10}{'model':<11}{'AUC up':>8}{'± SE':>7}{'IC beat':>9}{'± SE':>7}{'top−bot':>9}"
            f"{'edge':>9}   verdict"]
    for hz in meta["horizons"]:
        verdicts = [f"up/down: {hz['verdict']}", f"ranking: {(hz.get('rank') or {}).get('verdict', '—')}"]
        k = 0
        for name in ("neural", "quantedge", "gbm"):
            m = hz["metrics"].get(name)
            if not m:
                continue
            r = ((hz.get("rank") or {}).get("metrics") or {}).get(name) or {}
            edge = ("up" if m.get("edge") else "") + ("+rank" if r.get("edge") else "")
            spread = f"{r['spread']:+.2f}%" if r.get("spread") is not None else "—"
            rows.append(f"{hz['label'] if k == 0 else '':<10}{name:<11}{_n(m.get('auc'), '.3f'):>8}"
                        f"{_n(m.get('auc_se'), '.3f'):>7}{_n(r.get('ic'), '+.3f'):>9}{_n(r.get('ic_se'), '.3f'):>7}"
                        f"{spread:>9}{edge.strip('+') or 'no':>9}   {verdicts[k] if k < 2 else ''}")
            k += 1
        if k < 2:
            rows.append(f"{'':<70}{verdicts[1]}")
    return "\n".join(rows)


def intraday_table(meta: dict) -> str:
    rows = [f"{'model':<9}{'AUC beat':>9}{'IC':>8}{'± SE':>7}{'long':>9}{'short':>9}{'L/S':>9}{'t':>6}"
            f"{'green':>7}   (09:30 → close, % net of {meta['cost_pct']:.2f}% costs, test days only)"]
    beat = next(h for h in meta["heads"] if h["head"] == "beat")["metrics"]
    for name in ("neural", "ranker", "linear", "gbm", "blend", "selected"):
        m, t = beat.get(name), (meta.get("trading") or {}).get(name)
        if not m or not t:
            continue
        if name == "selected":
            rows.append("")
        rows.append(f"{name:<9}{_n(m.get('auc'), '.3f'):>9}{_n(t.get('ic'), '+.3f'):>8}{_n(m.get('ic_se'), '.3f'):>7}"
                    f"{_n(t.get('long_avg_net'), '+.3f'):>9}{_n(t.get('short_avg_net'), '+.3f'):>9}"
                    f"{_n(t.get('ls_avg'), '+.3f'):>9}{_n(t.get('ls_t'), '.1f'):>6}"
                    f"{_n(t.get('long_green_pct'), '.0f'):>6}%")
    dep = meta.get("deploy") or {}
    if dep.get("selection"):
        trail = ", ".join(f"fold {s['fold']} → {s['chosen']}" for s in dep["selection"])
        rows.append(f"\nselected = each fold scored by the model that did best on the folds before it ({trail})")
        ls = dep.get("fold_ls") or {}
        rows.append("long/short per fold: " + " · ".join(
            f"{n} " + "/".join(_n(v, '+.3f') for v in vals) for n, vals in ls.items()))
    rows.append(f"deployed: {dep.get('model', 'neural')}  (the same rule with every fold seen)")
    rows.append(f"\nverdict: {meta['verdict']}")
    if meta.get("edge_note"):
        rows.append(f"         {meta['edge_note']}")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="training-data zip from the server, or its extracted folder")
    ap.add_argument("--out", default=None, help="output folder (default: <data>/neural)")
    ap.add_argument("--model", choices=("daily", "intraday", "both"), default="both")
    ap.add_argument("--fetch-fundamentals", action="store_true", help="fetch missing fundamentals from Screener.in")
    ap.add_argument("--seeds", type=int, default=5, help="networks averaged (more = steadier, slower)")
    ap.add_argument("--search", type=int, default=0, help="network settings tried per training window")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--no-gbm", action="store_true", help="skip the gradient-boosting baseline")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    root = _data_root(a.data)
    settings.data_dir = root
    settings.data_source = "dhan"
    from td.neural import dataset as ds
    from td.neural import intraday as it
    from td.neural import news as nw
    from td.neural import train as tr
    from td.neural import train_intraday as ti

    t0 = time.time()
    frames, index_df = _frames(root)
    tables = _tables(root)
    print(f"{len(frames)} stocks with daily bars, NIFTY {len(index_df)} days, "
          f"1-minute tables for {len(tables)} symbols")
    if a.model == "intraday" and not tables:
        raise SystemExit("this zip has no 1-minute tables (intraday/) — export it again with the current code")
    pc_cache = root.parent / "screener-cache"       # kept next to the zips: the next export does not start over
    pc_cache.mkdir(parents=True, exist_ok=True)
    fund = _cached_fundamentals([root / "cache" / "screener", pc_cache], sorted(frames))
    if a.fetch_fundamentals:
        fund = asyncio.run(_fetch_fundamentals(pc_cache, sorted(ds.eligible(frames)), fund))
    print(f"fundamentals for {len(fund)} stocks")
    news_df = nw.load_archive(root / "recorder")
    start = nw.archive_start(news_df)
    print(f"news archive: {len(news_df):,} stock headlines"
          + (f" since {time.strftime('%Y-%m-%d', time.localtime(start))}" if start else " (none yet)"))
    cfg = tr.TrainConfig(seeds=a.seeds, epochs=a.epochs, batch=a.batch, gbm=not a.no_gbm, device=a.device,
                         search=a.search)
    out = Path(a.out) if a.out else root / "neural"
    say = lambda f, m: print(f"  {f * 100:3.0f}% {m}", flush=True)  # noqa: E731
    written, reports = [], []

    if a.model in ("daily", "both"):
        print("\n=== daily model ===")
        P = ds.build(frames, index_df, fundamentals=fund, news=news_df, news_start=start, minute=tables,
                     progress=lambda f, m: print(f"  dataset {f * 100:3.0f}% {m}", flush=True))
        print(f"panel: {P.rows:,} rows × ({ds.SEQ_LEN} days × {P.Z.shape[1]} technical + {P.static.shape[1]} other "
              f"inputs), {len(P.folds)} walk-forward folds")
        for n in P.notes:
            print(f"  note: {n}")
        ens, meta = tr.train(P, cfg, say=say)
        written += list(tr.export(ens, P, meta, out))
        reports.append(("daily model (AUC up = up/down; IC beat = ordering stocks against each other; top−bot = "
                        "best 10% minus worst 10% of stocks over the horizon)", daily_table(meta)))
        del P, ens

    if a.model in ("intraday", "both") and tables:
        print("\n=== 09:30 → close model ===")
        weights, conf = {}, {}
        try:
            weights = json.loads((root / "config" / "ranker_weights.json").read_text())
        except (OSError, ValueError):
            pass
        try:
            conf = json.loads((root / "config" / "tdesk.json").read_text())
        except (OSError, ValueError):
            pass
        IP = it.build(frames, index_df, tables, fundamentals=fund, news=news_df, news_start=start, weights=weights,
                      min_turnover_cr=float(conf.get("min_turnover_cr") or settings.min_turnover_cr),
                      progress=lambda f, m: print(f"  dataset {f * 100:3.0f}% {m}", flush=True))
        print(f"panel: {IP.rows:,} stock-days over {IP.cal_days.size} trading days "
              f"({IP.coverage['first_day']} → {IP.coverage['last_day']}), {len(IP.folds)} walk-forward folds")
        for n in IP.notes:
            print(f"  note: {n}")
        ens, meta = ti.train(IP, cfg, say=say)
        written += list(ti.export(ens, IP, meta, out))
        reports.append(("09:30 → close model vs the live TOP5 ranker", intraday_table(meta)))

    for title, table in reports:
        print(f"\n{title}\n{table}")
    print(f"\ntrained in {(time.time() - t0) / 60:.1f} min")
    print("\nmodel files:\n" + "\n".join(f"  {p}" for p in written))
    print("\nnext: upload them to the server's data/neural/ folder (the NEURAL desk loads them within a minute; "
          "the 09:30 picks start the next trading morning):  sudo mv ~/model.* ~/intraday* ~/tradedesk/data/neural/")


if __name__ == "__main__":
    main()
