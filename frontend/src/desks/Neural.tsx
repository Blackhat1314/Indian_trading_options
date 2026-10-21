/* NEURAL desk — "NeuralEdge", a neural network over price history, technicals, point-in-time fundamentals and
 * news, trained on the owner's PC (GPU) and served here as an ONNX file. Every forecast travels with its
 * walk-forward report card AND with the two baselines it was tested against on the same days (QuantEdge's
 * logistic regression, gradient-boosted trees): a network that cannot beat a straight line says so here. */
import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { ProbBar, inr } from "../components/research/charts";
import { Disclaimer, ErrorNote, Stat, Stats, Warnings } from "../components/research/common";
import { SymbolPicker } from "../components/research/SymbolPicker";
import { Empty, Panel, Seg } from "../components/ui";
import { ago, dateIST, fx, timeIST } from "../lib/format";
import { research } from "../lib/research";
import type { NeuralInfo, NeuralLeaderboard, NeuralLeader, NeuralMetric, NeuralResult } from "../lib/research";
import { useLive } from "../lib/store";
import "./neural.css";

const H_SHORT: Record<number, string> = { 1: "1D", 5: "1W", 21: "1M" };
const hShort = (h: number) => H_SHORT[h] ?? `${h}D`;
const ok = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const pctOf = (p: number | null | undefined, d = 0) => (ok(p) ? `${fx(p * 100, d)}%` : "—");
const NO_LEAN: [number, number] = [0, 1];
const MODEL_NAME: Record<string, string> = { neural: "NeuralEdge", quantedge: "QuantEdge (logistic)", gbm: "Gradient boosting" };

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
          <li><b>Upload</b> the two files it prints — <code>model.onnx</code> and <code>model.json</code> — to the server's <code>~/tradedesk/data/neural/</code> folder
            (browser SSH: <i>Upload file</i>, then <code>sudo mv ~/model.* ~/tradedesk/data/neural/</code>). This desk loads them within a minute; no restart.</li>
        </ol>
        <p className="nn-note">Retrain every few weeks, or after new results seasons: the network then also learns from the headlines recorded since.</p>
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
          <table className="tbl nn-tbl">
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
          </table>
          <Stats min={110}>
            <Stat label="Rows" value={fx(info.rows ?? null, 0)} sub={`${info.symbols} stocks`} />
            <Stat label="Network" value={`${fx(info.params ?? null, 0)} weights`} sub={`${info.seeds} seeds averaged`} />
            <Stat label="Tested" value={info.evaluation ? `${info.evaluation.folds.length} folds` : "—"} sub={info.evaluation ? `${info.evaluation.test_start} → ${info.evaluation.test_end}` : ""} />
            <Stat label="Fundamentals" value={cov ? pctOf(cov.fundamentals_rows) : "—"} sub={cov ? `of rows · ${cov.fundamentals_stocks} stocks` : ""}
              title="Share of rows with annual fundamentals known at the time (quarterly: about the last 3 years only)" />
            <Stat label="News" value={cov ? pctOf(cov.news_rows) : "—"} sub="of rows covered" tone={cov && cov.news_rows < 0.1 ? "amber" : undefined}
              title="Share of rows on days the headline archive already existed" />
          </Stats>
          <Warnings items={info.notes} label="Model notes" />
          <p className="nn-note">
            AUC 0.50 is a coin flip; for daily stock direction 0.52–0.55 is a small real edge. IC is the daily rank correlation between the
            forecast and what each stock then did. The network ships its forecasts only where it has an edge; whether it beats the simpler
            models is in the verdict row.
          </p>
        </>
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------ leaderboard
type LbH = "1" | "5" | "21";

function LeaderTable({ title, rows, sel, onPick }: { title: string; rows: NeuralLeader[]; sel: string; onPick: (s: string) => void }) {
  const key = (e: KeyboardEvent<HTMLTableRowElement>, s: string) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPick(s); }
  };
  return (
    <div className="nn-lead-col">
      <h3 className="nn-h3">{title}</h3>
      {!rows.length ? <Empty>None.</Empty> : (
        <table className="tbl nn-tbl">
          <thead><tr><th className="l">#</th><th className="l">Stock</th><th className="l">P(up)</th><th>Close</th></tr></thead>
          <tbody>
            {rows.map((x, i) => (
              <tr key={x.symbol} className={x.symbol === sel ? "sel" : undefined} tabIndex={0} onClick={() => onPick(x.symbol)} onKeyDown={(e) => key(e, x.symbol)}>
                <td className="l muted">{i + 1}</td>
                <td className="sym">{x.symbol}{x.sector && <small>{x.sector}</small>}</td>
                <td className="l"><ProbBar p={x.p_up} label="" width={44} /></td>
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
                  <LeaderTable title="▲ Highest P(up)" rows={b.top} sel={sel} onPick={onPick} />
                  <LeaderTable title="▼ Lowest P(up)" rows={b.bottom} sel={sel} onPick={onPick} />
                </div>
              </div>
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
  return (
    <div className="desk desk-neural rs-desk">
      <SymbolPicker area="pick" actions={
        <button type="button" className="btn" onClick={() => { info.reload(); res.reload(); board.reload(); }} title="Reload the model status and forecasts">REFRESH</button>
      } />
      {!ready ? <SetupPanel info={info.data} /> : <CallPanel r={res.data} loading={res.loading} error={res.error} reload={res.reload} />}
      <QualityPanel info={ready ? info.data : null} error={info.error} />
      {ready && <LeaderboardPanel h={lbH} setH={setLbH} b={board.data} loading={board.loading} error={board.error} reload={board.reload} sel={symbol} onPick={setResearchSymbol} />}
      <Disclaimer style={{ gridArea: "disc" }} />
    </div>
  );
}
