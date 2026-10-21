/* NEURAL desk — "NeuralEdge", neural networks over price history, technicals, point-in-time fundamentals, news
 * and 1-minute bars, trained on the owner's PC (GPU) and served here as ONNX files:
 *   the daily model   P(up) and P(beats the median stock) after 1 day / 1 week / 1 month
 *   the 09:30 model   picks at the TOP5 lock from the first 15 minutes plus the daily context, run by the analytics
 *                     worker and tracked like the TOP5 (its live record lands in the daily scorecard)
 * Every forecast travels with its walk-forward report card AND with the baselines it was tested against on the
 * same days: a network that cannot beat a straight line — or the TOP5 ranker — says so here. */
import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { ProbBar, inr } from "../components/research/charts";
import { Disclaimer, ErrorNote, Stat, Stats, Warnings } from "../components/research/common";
import { SymbolPicker } from "../components/research/SymbolPicker";
import { Empty, Panel, Seg } from "../components/ui";
import { api } from "../lib/api";
import { ago, cls, dateIST, fx, pct, timeIST } from "../lib/format";
import { research } from "../lib/research";
import type { NeuralInfo, NeuralIntradayInfo, NeuralLeaderboard, NeuralLeader, NeuralMetric, NeuralResult } from "../lib/research";
import { useLive } from "../lib/store";
import type { Pick } from "../lib/types";
import "./neural.css";

const H_SHORT: Record<number, string> = { 1: "1D", 5: "1W", 21: "1M" };
const hShort = (h: number) => H_SHORT[h] ?? `${h}D`;
const ok = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const pctOf = (p: number | null | undefined, d = 0) => (ok(p) ? `${fx(p * 100, d)}%` : "—");
const NO_LEAN: [number, number] = [0, 1];
const MODEL_NAME: Record<string, string> = {
  neural: "NeuralEdge", quantedge: "QuantEdge (logistic)", gbm: "Gradient boosting", ranker: "TOP5 ranker (live)",
  linear: "Logistic (same inputs)",
};
const sgn = (v: number | null | undefined, d = 2) => (ok(v) ? `${v > 0 ? "+" : ""}${fx(v, d)}` : "—");
const sgnPct = (v: number | null | undefined, d = 2) => (ok(v) ? `${sgn(v, d)}%` : "—");

function useLoad<T>(load: () => Promise<T>, deps: unknown[], pollWhile?: (r: T) => boolean) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [n, setN] = useState(0);
  const reload = useCallback(() => setN((x) => x + 1), []);
  const loadRef = useRef(load);
  loadRef.current = load;
  useEffect(() => {
    let alive = true;
    let timer = 0;
    const run = () => {
      setLoading(true);
      loadRef.current()
        .then((r) => {
          if (!alive) return;
          setData(r);
          setError(null);
          if (pollWhile && pollWhile(r)) timer = window.setTimeout(run, 4000);
        })
        .catch((e) => alive && setError(e))
        .finally(() => alive && setLoading(false));
    };
    run();
    return () => { alive = false; window.clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, n]);
  return { data, error, loading, reload };
}

function RankChip({ m }: { m: NeuralMetric | undefined }) {
  if (!m || !ok(m.ic)) return <span className="chip amber">n/a</span>;
  if (!m.edge) return <span className="chip amber" title={m.edge_note ?? "no reliable ranking edge"}>no edge</span>;
  return <span className="chip cyan" title={`IC positive in every fold: ${(m.fold_ic ?? []).map((a) => fx(a, 3)).join(" · ")}`}>edge</span>;
}

function EdgeChip({ m }: { m: NeuralMetric | undefined }) {
  if (!m || !ok(m.auc)) return <span className="chip amber">n/a</span>;
  if (!m.edge) return <span className="chip amber" title={m.edge_note ?? "no reliable edge"}>no edge</span>;
  return <span className={`chip ${m.auc >= 0.55 ? "cyan" : ""}`} title={m.edge_note ?? "edge held in every walk-forward fold"}>{m.auc >= 0.55 ? "edge" : "weak"}</span>;
}

// ------------------------------------------------------------------ setup (no model yet)
function SetupPanel({ info }: { info: NeuralInfo | null }) {
  return (
    <Panel code="SETUP" title="Train the network on your PC" area="call" sub="the server has no GPU, so it only runs the trained file">
      <div className="nn-pad">
        {info?.status === "error" && <div className="rs-error" role="alert">The model files in {info.folder} could not be used: {info.error}</div>}
        <ol className="nn-steps">
          <li><b>On the server</b> (SSH into the VM), package the training data:
            <pre>cd ~/tradedesk && docker compose run --rm tools python /app/research/neural_export.py</pre>
            then download <code>data/neural/training-data-YYYYMMDD.zip</code> (browser SSH: gear menu → <i>Download file</i>).</li>
          <li><b>On your PC</b> (Windows PowerShell, in your clone of the repo), once:
            <pre>py -3.12 -m venv .venv; .venv\Scripts\activate{"\n"}pip install torch --index-url https://download.pytorch.org/whl/cu130{"\n"}pip install -r backend\requirements-train.txt</pre>
            (needs a current NVIDIA driver; check with <code>python -c "import torch; print(torch.cuda.is_available())"</code>), then train — the GPU is used automatically:
            <pre>python research\neural_train.py --data C:\path\to\training-data-YYYYMMDD.zip --fetch-fundamentals</pre></li>
          <li><b>Upload</b> the files it prints — <code>model.onnx</code> + <code>model.json</code> (daily) and <code>intraday.onnx</code> + <code>intraday.json</code> (09:30) — to the server's <code>~/tradedesk/data/neural/</code> folder
            (browser SSH: <i>Upload file</i>, then <code>sudo mv ~/model.* ~/intraday.* ~/tradedesk/data/neural/</code>). This desk loads them within a minute, the 09:30 picks start the next morning; no restart.</li>
        </ol>
        <p className="nn-note">More training time buys steadier networks, not a better backtest: <code>--search 16 --seeds 10</code> runs overnight and tries 16 network settings inside every training window (never on the test data).
          Retrain every few weeks, or after new results seasons: the networks then also learn from the headlines and 1-minute days recorded since.</p>
      </div>
    </Panel>
  );
}

// ------------------------------------------------------------------ the selected stock
function CallPanel({ r, loading, error, reload }: { r: NeuralResult | null; loading: boolean; error: unknown; reload: () => void }) {
  return (
    <Panel code="NEXT" title="Next move" area="call"
      sub={r?.status === "ready" ? `${r.symbol} · data to ${r.as_of} · last close ${inr(r.close ?? null)}` : ""}>
      {error ? <ErrorNote error={error} onRetry={reload} />
        : !r ? <Empty>{loading ? "Scoring…" : "Pick a stock."}</Empty>
          : r.status === "computing" ? <Empty>Scoring every stock with the network… {r.model?.message ?? ""}</Empty>
            : (
              <div className={loading ? "nn-busy" : undefined}>
                <div className="nn-cards">
                  {(r.horizons ?? []).map((h) => (
                    <div key={h.h} className="nn-card">
                      <div className="nn-card-head"><b>{h.label}</b><span className="muted">{hShort(h.h)}</span></div>
                      <ProbBar p={h.p_up} neutral={h.edge ? undefined : NO_LEAN} />
                      <div className="nn-card-line">
                        {h.edge
                          ? <span>ranks <b>{ok(h.rank_pct) ? `${fx(h.rank_pct, 0)}th` : "—"}</b> percentile of {r.model?.symbols ?? "all"} stocks</span>
                          : <span className="amber">no call — no reliable edge at this horizon</span>}
                      </div>
                      <div className="nn-card-line muted" title={h.edge_note ?? undefined}>walk-forward AUC {fx(h.oos_auc, 3)} · {h.verdict}</div>
                      {ok(h.p_beat) && (
                        <div className="nn-card-line" title={h.rank_note ?? h.rank_verdict ?? undefined}>
                          beats the median stock: <b className={h.rank_edge ? undefined : "muted"}>{pctOf(h.p_beat)}</b>
                          <span className="muted"> · rank IC {fx(h.oos_ic, 3)} · {h.rank_edge ? "ranking edge" : "no ranking edge"}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
                {r.inputs && (
                  <p className="nn-note">
                    Inputs for {r.symbol}: 40 days of 26 technical features, fundamentals <b>{r.inputs.fundamentals}/{r.inputs.fundamentals_of}</b> known,
                    news archive {r.inputs.news_on ? "on" : <span className="amber">not covering today</span>}.
                  </p>
                )}
              </div>
            )}
    </Panel>
  );
}

// ------------------------------------------------------------------ report card vs baselines
function QualityPanel({ info, error }: { info: NeuralInfo | null; error: unknown }) {
  const now = Date.now() / 1000;
  const hz = [...(info?.horizons ?? [])].sort((a, b) => a.h - b.h);
  const models = ["neural", "quantedge", "gbm"].filter((k) => hz.some((h) => h.metrics[k as "neural"]));
  const cov = info?.coverage;
  const cell = (m: NeuralMetric | undefined, best: boolean): ReactNode => (
    !m ? <span className="muted">—</span> : (
      <span className="nn-cell">
        <b className={best ? "cyan" : undefined} title={`fold AUCs: ${(m.fold_auc ?? []).map((a) => fx(a, 3)).join(" · ")}`}>{fx(m.auc, 3)}</b>
        <small className="muted"> ±{fx(m.auc_se, 3)}</small>
        <small className="muted"> · IC {fx(m.ic, 3)}</small>
        <EdgeChip m={m} />
      </span>
    )
  );
  return (
    <Panel code="OOS" title="Report card" area="quality" sub="walk-forward, out of sample, same test days for all three models">
      {!info ? (error ? <ErrorNote error={error} /> : <Empty>Loading…</Empty>) : !hz.length ? <Empty>No model yet.</Empty> : (
        <>
          <div className="nn-model-line">
            <span className="chip up">LOADED</span>
            {ok(info.trained_at) && <span>trained {dateIST(info.trained_at)} {timeIST(info.trained_at)} IST <span className="muted">({ago(info.trained_at, now)} ago)</span> on {info.device}</span>}
          </div>
          <div className="nn-scroll"><table className="tbl nn-tbl">
            <thead><tr><th className="l">Model</th>{hz.map((h) => <th key={h.h} className="l">{hShort(h.h)}</th>)}</tr></thead>
            <tbody>
              {models.map((k) => (
                <tr key={k}><td className="l">{MODEL_NAME[k]}</td>{hz.map((h) => {
                  const aucs = models.map((x) => h.metrics[x as "neural"]?.auc).filter(ok);
                  const m = h.metrics[k as "neural"];
                  return <td key={h.h} className="l">{cell(m, !!m && ok(m.auc) && m.auc === Math.max(...aucs))}</td>;
                })}</tr>
              ))}
              <tr><td className="l">Verdict</td>{hz.map((h) => <td key={h.h} className="l nn-verdict">{h.verdict}</td>)}</tr>
            </tbody>
          </table></div>
          {hz.some((h) => h.rank) && (
            <>
              <h3 className="nn-h3">Ordering stocks against each other — beats the median stock</h3>
              <div className="nn-scroll"><table className="tbl nn-tbl">
                <thead><tr><th className="l">Model</th>{hz.map((h) => <th key={h.h} className="l">{hShort(h.h)}</th>)}</tr></thead>
                <tbody>
                  {models.map((k) => (
                    <tr key={k}><td className="l">{MODEL_NAME[k]}</td>{hz.map((h) => {
                      const m = h.rank?.metrics[k as "neural"];
                      return (
                        <td key={h.h} className="l">{!m ? <span className="muted">—</span> : (
                          <span className="nn-cell">
                            <b title={`IC per fold: ${(m.fold_ic ?? []).map((a) => fx(a, 3)).join(" · ")}`}>IC {sgn(m.ic, 3)}</b>
                            <small className="muted"> ±{fx(m.ic_se, 3)}</small>
                            <small className="muted" title={`best 10% minus worst 10% of stocks over ${h.h} day(s), on ${m.spread_periods ?? 0} non-overlapping test periods`}> · top−bottom {sgnPct(m.spread)}</small>
                            <RankChip m={m} />
                          </span>
                        )}</td>
                      );
                    })}</tr>
                  ))}
                  <tr><td className="l">Verdict</td>{hz.map((h) => <td key={h.h} className="l nn-verdict">{h.rank?.verdict ?? "—"}</td>)}</tr>
                </tbody>
              </table></div>
            </>
          )}
          <Stats min={110}>
            <Stat label="Rows" value={fx(info.rows ?? null, 0)} sub={`${info.symbols} stocks`} />
            <Stat label="Network" value={`${fx(info.params ?? null, 0)} weights`} sub={`${info.seeds} seeds averaged`} />
            <Stat label="Tested" value={info.evaluation ? `${info.evaluation.folds.length} folds` : "—"} sub={info.evaluation ? `${info.evaluation.test_start} → ${info.evaluation.test_end}` : ""} />
            <Stat label="Fundamentals" value={cov ? pctOf(cov.fundamentals_rows) : "—"} sub={cov ? `of rows · ${cov.fundamentals_stocks} stocks` : ""}
              title="Share of rows with annual fundamentals known at the time (quarterly: about the last 3 years only)" />
            <Stat label="News" value={cov ? pctOf(cov.news_rows) : "—"} sub="of rows covered" tone={cov && cov.news_rows < 0.1 ? "amber" : undefined}
              title="Share of rows on days the headline archive already existed" />
            {cov && ok(cov.minute_rows) && <Stat label="1-minute" value={pctOf(cov.minute_rows)} sub="of rows with day summaries"
              title="Share of rows whose day has 1-minute bars (volatility, skew, last 30 minutes, close vs VWAP)" />}
          </Stats>
          <Warnings items={info.notes} label="Model notes" />
          <p className="nn-note">
            AUC 0.50 is a coin flip; for daily stock direction 0.52–0.55 is a small real edge. IC is the daily rank correlation between the
            forecast and what each stock then did — the second table asks only "which stocks do better than the others", where the market's
            own direction cancels out; its ± is a block bootstrap over whole days, because overlapping returns make neighbouring days
            near-copies. The network ships forecasts only where an edge passed its test; whether it beats the simpler models is in the verdict rows.
          </p>
        </>
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------ leaderboard
type LbH = "1" | "5" | "21";

function LeaderTable({ title, rows, sel, onPick, beat }: { title: string; rows: NeuralLeader[]; sel: string; onPick: (s: string) => void; beat?: boolean }) {
  const key = (e: KeyboardEvent<HTMLTableRowElement>, s: string) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPick(s); }
  };
  return (
    <div className="nn-lead-col">
      <h3 className="nn-h3">{title}</h3>
      {!rows.length ? <Empty>None.</Empty> : (
        <table className="tbl nn-tbl">
          <thead><tr><th className="l">#</th><th className="l">Stock</th><th className="l">{beat ? "P(beat)" : "P(up)"}</th><th>Close</th></tr></thead>
          <tbody>
            {rows.map((x, i) => (
              <tr key={x.symbol} className={x.symbol === sel ? "sel" : undefined} tabIndex={0} onClick={() => onPick(x.symbol)} onKeyDown={(e) => key(e, x.symbol)}>
                <td className="l muted">{i + 1}</td>
                <td className="sym">{x.symbol}{x.sector && <small>{x.sector}</small>}</td>
                <td className="l"><ProbBar p={beat ? x.p_beat ?? null : x.p_up} label="" width={44} /></td>
                <td>{inr(x.close)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function LeaderboardPanel({ h, setH, b, loading, error, reload, sel, onPick }: {
  h: LbH; setH: (h: LbH) => void; b: NeuralLeaderboard | null; loading: boolean; error: unknown; reload: () => void; sel: string; onPick: (s: string) => void;
}) {
  return (
    <Panel code="RANK" title="Leaderboard" area="lead" sub={b?.as_of ? `data to ${b.as_of} · ${b.n} stocks` : ""}
      actions={<Seg label="Leaderboard horizon" value={h} onChange={setH} options={(["1", "5", "21"] as LbH[]).map((k) => ({ v: k, label: hShort(Number(k)) }))} />}>
      {error ? <ErrorNote error={error} onRetry={reload} />
        : !b ? <Empty>{loading ? "Loading…" : "No leaderboard yet."}</Empty>
          : b.computing ? <Empty>Scoring every stock with the network…</Empty>
            : !b.ranked ? (
              <p className="nn-note warn" role="note">
                No ranking at this horizon: {b.edge_note ?? "the network has no reliable walk-forward edge here"}. Its probabilities are the base rate for every
                stock, so a sorted list would only order noise.
              </p>
            ) : (
              <div className={loading ? "nn-busy" : undefined}>
                <p className="nn-note">{b.verdict}. A small edge across many stocks; any single name can still go the other way — a screen, not a buy list.</p>
                <div className="nn-lead-grid">
                  <LeaderTable title={b.ranked_by === "beat" ? "▲ Most likely to beat the median" : "▲ Highest P(up)"} rows={b.top} sel={sel} onPick={onPick} beat={b.ranked_by === "beat"} />
                  <LeaderTable title={b.ranked_by === "beat" ? "▼ Least likely to beat the median" : "▼ Lowest P(up)"} rows={b.bottom} sel={sel} onPick={onPick} beat={b.ranked_by === "beat"} />
                </div>
              </div>
            )}
    </Panel>
  );
}

// ------------------------------------------------------------------ the 09:30 model: today's picks (live)
function NeuralPickRows({ side, picks, onPick }: { side: "long" | "short"; picks: Pick[]; onPick: (s: string) => void }) {
  return (
    <div className="nn-lead-col">
      <h3 className="nn-h3">{side === "long" ? "▲ Long — most likely to beat the median" : "▼ Short — least likely"}</h3>
      {!picks.length ? <Empty>None.</Empty> : (
        <table className="tbl nn-tbl">
          <thead><tr><th className="l">Stock</th><th className="l">P(beat)</th><th>Since 09:30</th></tr></thead>
          <tbody>
            {picks.map((p) => (
              <tr key={p.symbol} tabIndex={0} onClick={() => onPick(p.symbol)}
                title={`entry ${inr(p.entry)} · target ${inr(p.target)} · stop ${inr(p.invalidation)}`}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPick(p.symbol); } }}>
                <td className="sym">{p.symbol}<small>{p.sector}</small></td>
                <td className="l"><ProbBar p={p.p_beat ?? null} label="" width={40} /></td>
                <td>
                  <span className={cls(p.pnl ?? 0)}>{ok(p.pnl) ? pct(p.pnl, 2) : "—"}</span>
                  <small className={`nn-status ${p.status ?? "LIVE"}`}>{p.status ?? "LIVE"}{p.hit_at ? ` ${p.hit_at}` : ""}</small>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function IntradayPicksPanel({ model, onPick }: { model: NeuralIntradayInfo | null; onPick: (s: string) => void }) {
  const top = useLive((s) => s.top);
  const nn = top?.neural;
  const u = top?.universe_since_lock;
  const n = nn?.since_lock;
  let body: ReactNode;
  if (!top) body = <Empty>Waiting for the analytics worker…</Empty>;
  else if (top.closed) body = <Empty>Market closed — {top.closed}.</Empty>;
  else if (!top.locked) {
    body = model?.status === "ready"
      ? <p className="nn-note">The network picks at the {top.lock_time} lock, from the first 15 one-minute bars and each stock's daily context. Until then there is nothing to show — it never peeks at a provisional state.</p>
      : <p className="nn-note">{model?.status === "no_edge" ? <>No 09:30 picks: {model.verdict}.</> : <>No 09:30 model installed — train it on the PC (<code>--model intraday</code>, needs the 1-minute tables in a fresh export) and upload <code>intraday.onnx</code> + <code>intraday.json</code>.</>}</p>;
  } else if (!nn || nn.status === "missing") {
    body = <p className="nn-note">No 09:30 model was installed at today's lock.</p>;
  } else if (nn.status !== "ok") {
    body = <p className={`nn-note ${nn.status === "error" ? "warn" : ""}`}>{nn.status === "no_edge" ? `No neural picks: ${nn.verdict ?? "the walk-forward showed no reliable edge"}.`
      : nn.status === "pending" ? "Building each stock's daily context — the picks follow within minutes, from the same 09:30 inputs and entry prices."
        : nn.note ?? nn.status}</p>;
  } else {
    body = (
      <>
        {!!nn.late_s && <p className="nn-note warn">Made {Math.round(nn.late_s / 60)} min after the lock (the daily context was still being built) — same 09:30 inputs and entries.</p>}
        <div className="nn-lead-grid">
          <NeuralPickRows side="long" picks={nn.long ?? []} onPick={onPick} />
          <NeuralPickRows side="short" picks={nn.short ?? []} onPick={onPick} />
        </div>
        {n && u && ok(n.long_avg) && (
          <div className="lockbar">
            <span className="amber b">SINCE 09:30</span>
            <span className="kv"><span>Neural long</span><span className={cls(n.long_avg)}>{pct(n.long_avg)}</span></span>
            <span className="kv"><span>short</span><span className={cls(n.short_avg)}>{pct(n.short_avg)}</span></span>
            <span className="kv"><span>TOP5 long</span><span className={cls(u.long_avg)}>{pct(u.long_avg)}</span></span>
            <span className="kv"><span>short</span><span className={cls(u.short_avg)}>{pct(u.short_avg)}</span></span>
            <span className="kv"><span>Universe</span><span className={cls(u.universe_avg)}>{pct(u.universe_avg)}</span></span>
          </div>
        )}
        <p className="nn-note">{nn.n} stocks scored{nn.skipped ? ` · ${nn.skipped} skipped (daily history a day behind, or a split/bonus)` : ""}. {nn.verdict}. Same stop/target rules as the TOP5. Data only — no orders are placed.</p>
      </>
    );
  }
  return (
    <Panel code="0930" title="Neural 09:30 picks" area="intra"
      sub={top?.locked && nn?.status === "ok" ? `locked ${top.lock_time} IST · tracked like the TOP5` : "picks at the TOP5 lock"}>
      <div className="nn-pad">{body}</div>
    </Panel>
  );
}

// ------------------------------------------------------------------ the 09:30 model: report card
interface ScoreSide { mean_ic: number | null; avg_long: number | null; avg_short: number | null; avg_excess: number | null; top5_hits_per_day: number | null; ic_positive_days?: number }
interface NeuralLive { days: number; neural: ScoreSide; top5: ScoreSide }

function IntradayQualityPanel({ info, error }: { info: NeuralIntradayInfo | null; error: unknown }) {
  const [live, setLive] = useState<NeuralLive | null>(null);
  useEffect(() => {
    api<{ neural_summary?: NeuralLive | null }>("/api/scorecard").then((r) => setLive(r.neural_summary ?? null)).catch(() => {});
  }, []);
  const now = Date.now() / 1000;
  const beat = info?.heads?.find((h) => h.head === "beat")?.metrics ?? {};
  const models = ["neural", "ranker", "linear", "gbm"].filter((k) => info?.trading?.[k]);
  return (
    <Panel code="OOS" title="09:30 model — report card" area="iq" sub="walk-forward vs the live TOP5 ranker on the same days, 09:30 → close, net of costs">
      {error ? <ErrorNote error={error} /> : !info ? <Empty>Loading…</Empty>
        : info.status === "missing" ? <Empty>No 09:30 model yet (intraday.onnx + intraday.json in {info.folder}).</Empty>
          : info.status === "error" ? <div className="rs-error" role="alert">The 09:30 model files could not be used: {info.error}</div> : (
            <>
              <div className="nn-model-line">
                <span className={`chip ${info.edge ? "up" : "amber"}`}>{info.edge ? "EDGE" : "NO EDGE"}</span>
                {ok(info.trained_at) && <span>trained {dateIST(info.trained_at)} <span className="muted">({ago(info.trained_at, now)} ago)</span> on {info.device}</span>}
              </div>
              <div className="nn-scroll"><table className="tbl nn-tbl">
                <thead><tr><th className="l">Model</th><th>AUC</th><th>IC</th><th>Long</th><th>Short</th><th>Long/short</th><th>Green</th></tr></thead>
                <tbody>
                  {models.map((k) => {
                    const t = info.trading![k];
                    const m = beat[k];
                    return (
                      <tr key={k}>
                        <td className="l">{MODEL_NAME[k] ?? k}</td>
                        <td title="beats-the-median head, mean fold AUC">{fx(m?.auc ?? null, 3)}</td>
                        <td title={m?.ic_se ? `± ${fx(m.ic_se, 3)} (block bootstrap over days)` : undefined}>{sgn(t.ic, 3)}</td>
                        <td className={cls(t.long_avg_net ?? 0)}>{sgnPct(t.long_avg_net, 3)}</td>
                        <td className={cls(t.short_avg_net ?? 0)}>{sgnPct(t.short_avg_net, 3)}</td>
                        <td className={cls(t.ls_avg ?? 0)} title={`t = ${fx(t.ls_t ?? null, 1)} over ${t.days} test days`}>{sgnPct(t.ls_avg, 3)}{ok(t.ls_t) && <small className="muted"> t {fx(t.ls_t, 1)}</small>}</td>
                        <td>{ok(t.long_green_pct) ? `${fx(t.long_green_pct, 0)}%` : "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table></div>
              <p className="nn-note"><b>{info.verdict}</b>{info.edge_note ? <> — {info.edge_note}</> : null}</p>
              <Stats min={110}>
                <Stat label="Stock-days" value={fx(info.rows ?? null, 0)} sub={`${info.days} days · ${info.symbols} stocks`} />
                <Stat label="Tested" value={info.evaluation ? `${info.evaluation.folds.length} folds` : "—"} sub={info.evaluation ? `${info.evaluation.test_start} → ${info.evaluation.test_end}` : ""} />
                <Stat label="Network" value={`${fx(info.params ?? null, 0)} weights`} sub={`${info.seeds} seeds${info.search && info.search.candidates > 1 ? ` · ${info.search.candidates} settings searched` : ""}`} />
                <Stat label="Costs" value={`${fx(info.cost_pct ?? null, 2)}%`} sub="round trip, per pick" />
              </Stats>
              {live && (
                <>
                  <h3 className="nn-h3">Live record — {live.days} day{live.days === 1 ? "" : "s"} since installed (from the daily scorecard)</h3>
                  <div className="nn-scroll"><table className="tbl nn-tbl">
                    <thead><tr><th className="l">Picks</th><th>Mean IC</th><th>Long avg</th><th>Short avg</th><th>Excess</th><th>Top-5 hits</th></tr></thead>
                    <tbody>
                      {(["neural", "top5"] as const).map((k) => (
                        <tr key={k}><td className="l">{k === "neural" ? "Neural 09:30" : "TOP5 (same days)"}</td>
                          <td>{sgn(live[k].mean_ic, 3)}</td><td className={cls(live[k].avg_long ?? 0)}>{sgnPct(live[k].avg_long)}</td>
                          <td className={cls(live[k].avg_short ?? 0)}>{sgnPct(live[k].avg_short)}</td>
                          <td className={cls(live[k].avg_excess ?? 0)}>{sgnPct(live[k].avg_excess)}</td><td>{fx(live[k].top5_hits_per_day ?? null, 2)}</td></tr>
                      ))}
                    </tbody>
                  </table></div>
                  {live.days < 60 && <p className="nn-note">Fewer than 60 days: too few to tell skill from luck either way.</p>}
                </>
              )}
              <Warnings items={info.notes} label="Model notes" />
            </>
          )}
    </Panel>
  );
}

// ------------------------------------------------------------------ desk
export default function Neural() {
  const symbol = useLive((s) => s.researchSymbol);
  const setResearchSymbol = useLive((s) => s.setResearchSymbol);
  const info = useLoad(() => research.neuralModel(), [], (r) => r.status === "computing");
  const status = info.data?.status;
  const ready = status === "ready" || status === "computing";
  const res = useLoad<NeuralResult | null>(() => (symbol && ready ? research.neural(symbol) : Promise.resolve(null)),
    [symbol, ready], (r) => r?.status === "computing");
  const [lbH, setLbH] = useState<LbH>("21");
  const board = useLoad<NeuralLeaderboard | null>(() => (ready ? research.neuralLeaderboard(Number(lbH), 15) : Promise.resolve(null)),
    [lbH, ready], (r) => !!r?.computing);
  const intra = useLoad(() => research.neuralIntraday(), []);
  return (
    <div className="desk desk-neural rs-desk">
      <SymbolPicker area="pick" actions={
        <button type="button" className="btn" onClick={() => { info.reload(); res.reload(); board.reload(); intra.reload(); }} title="Reload the model status and forecasts">REFRESH</button>
      } />
      {!ready ? <SetupPanel info={info.data} /> : <CallPanel r={res.data} loading={res.loading} error={res.error} reload={res.reload} />}
      <QualityPanel info={ready ? info.data : null} error={info.error} />
      {ready && <LeaderboardPanel h={lbH} setH={setLbH} b={board.data} loading={board.loading} error={board.error} reload={board.reload} sel={symbol} onPick={setResearchSymbol} />}
      <IntradayPicksPanel model={intra.data} onPick={setResearchSymbol} />
      <IntradayQualityPanel info={intra.data} error={intra.error} />
      <Disclaimer style={{ gridArea: "disc" }} />
    </div>
  );
}
