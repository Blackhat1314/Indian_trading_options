/* Basic LEARN: five rules worth more than any indicator, how to read this app, and a plain-English glossary. */
import { useMemo, useState } from "react";
import { Card } from "./ui";
import { GLOSSARY, RULES } from "./words";

export default function Learn() {
  const [q, setQ] = useState("");
  const terms = useMemo(() => {
    const s = q.trim().toLowerCase();
    return s ? GLOSSARY.filter((t) => t.term.toLowerCase().includes(s) || t.text.toLowerCase().includes(s)) : GLOSSARY;
  }, [q]);
  return (
    <div className="b-learn">
      <div className="b-intro">
        <h1>Learn</h1>
        <p>Most beginners don't lose money because they pick the wrong stocks — they lose it by risking too much on each one and holding on when they're wrong. These rules come first.</p>
      </div>
      <ol className="b-rules">
        {RULES.map((r) => <li key={r.title}><b>{r.title}</b><span>{r.text}</span></li>)}
      </ol>
      <Card title="How to read this app" id="read">
        <dl className="b-howto">
          <div><dt>Ideas</dt><dd>Simple filters such as “volume twice its usual”. They find stocks worth a look; they don't predict.</dd></div>
          <div><dt>Today's strongest</dt><dd>Five stocks the intraday ranker locks at 9:30, each with a stop and a target for the day. The track record under the list shows how such picks have done — trust it only after 60+ days.</dd></div>
          <div><dt>What our model thinks</dt><dd>A neural network's ranking of a stock against all others, shown only where years of tests found a real edge. “No reliable view” is an honest answer, not a failure.</dd></div>
          <div><dt>Business quality</dt><dd>A 0–100 score from the company's results and balance sheet: growth, profitability, debt, valuation and ownership.</dd></div>
          <div><dt>How many shares?</dt><dd>The calculator on every stock page: enter your capital, your risk and your stop, and it tells you the size that keeps a loss small.</dd></div>
        </dl>
      </Card>
      <Card title="Glossary" id="gloss" action={
        <input className="b-input" placeholder="Find a word…" aria-label="Find a word in the glossary" value={q} onChange={(e) => setQ(e.target.value)} />}>
        <dl className="b-glossary">
          {terms.map((t) => <div key={t.term}><dt>{t.term}</dt><dd>{t.text}</dd></div>)}
          {!terms.length && <p className="b-note">No word matches “{q}”.</p>}
        </dl>
      </Card>
    </div>
  );
}
