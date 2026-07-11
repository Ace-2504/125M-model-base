import { CORPUS } from "@/lib/model";

/**
 * Corpus composition as neutral single-accent bars (no multi-color donut), so
 * it sits in the same visual register as the dedup table and spec lists.
 */
export default function CorpusChart() {
  return (
    <div>
      <div className="mb-4">
        <span className="tag">index.json</span>
        <h3 className="text-base font-medium mt-1">Corpus composition</h3>
      </div>
      <ul className="space-y-4">
        {CORPUS.map((s) => (
          <li key={s.name}>
            <div className="flex items-baseline justify-between mb-1.5">
              <span className="text-sm text-[var(--fg)]">{s.name}</span>
              <span className="mono text-sm text-[var(--fg-muted)]">
                {s.pct.toFixed(1)}% · {(s.tokens / 1e6).toFixed(0)}M
              </span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-[var(--border-soft)] overflow-hidden">
              <div
                className="h-full rounded-full bg-[var(--accent)]"
                style={{ width: `${s.pct}%` }}
              />
            </div>
            <div className="mono text-[0.7rem] text-[var(--fg-dim)] mt-1.5">{s.hf}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
