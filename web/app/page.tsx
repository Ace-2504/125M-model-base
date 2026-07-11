import {
  HEADLINE_STATS,
  ARCHITECTURE,
  TRAINING,
  DEDUP,
  TOTAL_KEPT,
  TOTAL_REMOVED,
} from "@/lib/model";
import Demo from "@/components/Demo";
import CorpusChart from "@/components/CorpusChart";
import ThemePicker from "@/components/ThemePicker";

function fmt(n: number) {
  return n.toLocaleString("en-US");
}

function SpecList({ items }: { items: { k: string; v: string }[] }) {
  return (
    <dl className="divide-y divide-[var(--border-soft)]">
      {items.map((it) => (
        <div key={it.k} className="flex items-baseline justify-between gap-6 py-2.5">
          <dt className="text-sm text-[var(--fg-muted)] shrink-0">{it.k}</dt>
          <dd className="mono text-sm text-[var(--fg)] text-right">{it.v}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function Home() {
  return (
    <main className="mx-auto max-w-5xl px-5 sm:px-8">
      {/* Nav — links only (logo removed), larger type */}
      <header className="flex items-center justify-between py-6">
        <nav className="hidden sm:flex items-center gap-8 text-[1.05rem] text-[var(--fg-muted)]">
          <a href="#model" className="link-underline hover:text-[var(--fg)] transition-colors">model</a>
          <a href="#corpus" className="link-underline hover:text-[var(--fg)] transition-colors">corpus</a>
          <a href="#training" className="link-underline hover:text-[var(--fg)] transition-colors">training</a>
        </nav>
        <ThemePicker />
      </header>

      {/* Hero — centered */}
      <section className="pt-12 pb-8 text-center">
        <div className="inline-flex items-center gap-2 rounded-full border border-[var(--border)] bg-[var(--panel)] px-3.5 py-1.5 mb-7">
          <span className="live-dot inline-block h-1.5 w-1.5 rounded-full bg-[var(--accent-2)]" />
          <span className="tag font-bold !text-[var(--fg)]">Trained from scratch by Harman Sandhu</span>
        </div>
        <h1 className="text-5xl sm:text-7xl font-semibold tracking-tight">SLM-125M</h1>
        <p className="mt-5 mx-auto max-w-xl text-lg text-[var(--fg-muted)] leading-relaxed">
          A 125.8M-parameter, Llama-style legal language model — pretrained from scratch
          on 2.04B tokens, and running live below.
        </p>

        {/* Stat tiles — single horizontal row */}
        <div className="mt-10 grid grid-cols-3 sm:grid-cols-6 gap-3">
          {HEADLINE_STATS.map((s) => (
            <div key={s.label} className="panel px-3 py-4 text-center">
              <div
                className={`mono text-xl sm:text-2xl font-semibold ${
                  s.pending ? "text-[var(--accent)]" : "text-[var(--fg)]"
                }`}
              >
                {s.value}
              </div>
              <div className="tag mt-1.5">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Demo — centered & large */}
      <section id="model" className="pt-10 pb-4 scroll-mt-8">
        <div className="mx-auto max-w-2xl">
          <Demo />
        </div>
      </section>

      {/* What the sampling knobs mean */}
      <section className="pb-0">
        <div className="mx-auto max-w-2xl panel p-5">
          <span className="tag">what the controls mean</span>
          <dl className="mt-3 grid sm:grid-cols-2 gap-x-8 gap-y-3">
            {[
              ["temperature", "How random the output is. Low stays safe and predictable; high gets more varied — and more likely to wander."],
              ["top-p", "Each step, sample only from the most likely words whose probabilities add up to p. Lower = more focused."],
              ["top-k", "Each step, only consider the k most likely next words. Lower = more focused."],
              ["max tokens", "How long the completion can get, in tokens (roughly ¾ of a word each)."],
            ].map(([k, v]) => (
              <div key={k}>
                <dt className="mono text-sm font-semibold text-[var(--accent)]">{k}</dt>
                <dd className="text-[0.83rem] text-[var(--fg-muted)] leading-snug mt-0.5">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      {/* What this is + Architecture + Training — combined into one box */}
      <section id="architecture" className="pt-10 pb-0 scroll-mt-8">
        <div className="panel p-6 sm:p-8">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-center justify-center gap-2.5">
              <h3 className="text-2xl sm:text-3xl font-semibold tracking-tight">What this is</h3>
              <span className="inline-flex items-center rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 py-0.5 mono text-[0.68rem] tracking-wide text-[var(--fg-muted)]">
                base completer
              </span>
            </div>

            <p className="mt-3 text-[var(--fg-muted)] leading-relaxed text-[0.95rem]">
              This is a <b className="font-semibold text-[var(--fg)]">base model</b>, not a
              chatbot. It only ever learned to predict the next token, so it{" "}
              <b className="font-semibold text-[var(--fg)]">continues text</b> instead of
              answering questions — hand it the opening of a sentence and it carries the
              thought forward.
            </p>

            <p className="mt-3 text-[var(--fg-muted)] leading-relaxed text-[0.95rem]">
              The honest quality number is{" "}
              <b className="font-semibold text-[var(--fg)]">held-out validation perplexity: 11.35</b>{" "}
              (lower is better), reached in a{" "}
              <b className="font-semibold text-[var(--fg)]">single epoch</b> over 2.04B
              tokens. It has picked up the{" "}
              <b className="font-semibold text-[var(--fg)]">legal register</b> — citation
              phrasing, procedural language — but at{" "}
              <b className="font-semibold text-[var(--fg)]">125M parameters</b> and a{" "}
              <b className="font-semibold text-[var(--fg)]">1,024-token context</b> it holds
              almost no world knowledge, so anything factual would need retrieval (RAG).
            </p>

            <div className="mt-5 flex flex-wrap gap-2">
              {[
                ["stream", "3 datasets"],
                ["clean", "rule chain"],
                ["dedup", "+ decontaminate"],
                ["BPE", "16K byte-level"],
                ["pack", "1,024-token windows"],
                ["pretrain", "A100 · 1 epoch"],
              ].map(([k, v]) => (
                <span
                  key={k}
                  className="inline-flex items-center gap-1.5 rounded-md border border-[var(--border)] bg-[var(--panel-2)] px-2.5 py-1 text-[0.75rem]"
                >
                  <span className="mono font-semibold text-[var(--fg)]">{k}</span>
                  <span className="text-[var(--fg-muted)]">{v}</span>
                </span>
              ))}
            </div>

            <p className="mt-4 text-[0.8rem] text-[var(--fg-dim)] leading-relaxed">
              Corpus: SEC filings (~42%), US case law (~35%), educational web (~23%). The
              first request can take ~15–30s while the model wakes from idle.
            </p>
          </div>

          <div className="hairline my-7" />

          <div className="grid md:grid-cols-2 gap-x-10 gap-y-8">
            <div>
              <SectionHeadInline title="Architecture" />
              <SpecList items={ARCHITECTURE} />
            </div>
            <div id="training" className="scroll-mt-8">
              <SectionHeadInline title="Training recipe" />
              <SpecList items={TRAINING} />
            </div>
          </div>
        </div>
      </section>

      {/* Corpus — one box, like "what this is" */}
      <section id="corpus" className="pt-10 pb-14 scroll-mt-8">
        <div className="panel p-6 sm:p-8">
          <div className="flex items-center justify-center gap-2.5">
            <h3 className="text-2xl sm:text-3xl font-semibold tracking-tight">What it read</h3>
            <span className="inline-flex items-center rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 py-0.5 mono text-[0.68rem] tracking-wide text-[var(--fg-muted)]">
              training corpus
            </span>
          </div>
          <p className="mt-3 mx-auto max-w-2xl text-center text-[var(--fg-muted)] leading-relaxed text-[0.95rem]">
            A legal-first mix. The realized split — from the final token index, not the
            target ratios — is ~42% SEC / 35% case law / 23% web.
          </p>

          <div className="hairline my-7" />

          <div className="grid md:grid-cols-2 gap-x-10 gap-y-8">
            <CorpusChart />

            {/* Dedup + decontamination */}
            <div>
              <div className="mb-4">
                <span className="tag">phase2_report.json</span>
                <h3 className="text-base font-medium mt-1">Dedup + decontamination</h3>
              </div>
              <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="tag text-left">
                    <th className="font-normal pb-2">source</th>
                    <th className="font-normal pb-2 text-right">kept</th>
                    <th className="font-normal pb-2 text-right">near-dup</th>
                    <th className="font-normal pb-2 text-right">exact</th>
                    <th className="font-normal pb-2 text-right">contam.</th>
                  </tr>
                </thead>
                <tbody className="mono">
                  {DEDUP.map((d) => (
                    <tr key={d.source} className="border-t border-[var(--border-soft)]">
                      <td className="py-2.5 text-[var(--fg)]">{d.source}</td>
                      <td className="py-2.5 text-right text-[var(--fg)]">{fmt(d.kept)}</td>
                      <td className="py-2.5 text-right text-[var(--fg-muted)]">{d.nearDup || "–"}</td>
                      <td className="py-2.5 text-right text-[var(--fg-muted)]">{d.exactDup || "–"}</td>
                      <td className="py-2.5 text-right text-[var(--fg-muted)]">{fmt(d.contaminated) || "–"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="hairline my-4" />
            <div className="flex items-center justify-between text-sm">
              <span className="text-[var(--fg-muted)]">
                <span className="mono text-[var(--fg)]">{fmt(TOTAL_KEPT)}</span> docs kept
              </span>
              <span className="text-[var(--fg-muted)]">
                <span className="mono text-[var(--accent)]">{fmt(TOTAL_REMOVED)}</span> removed
              </span>
            </div>
          </div>
          </div>
        </div>
      </section>

      {/* Footer — gradient divider + status line + byline */}
      <footer className="pt-14 pb-16 mt-4 text-center">
        <div
          className="mx-auto mb-6 h-1.5 w-56 rounded-full"
          style={{
            background:
              "linear-gradient(90deg,#22c55e 0%,#3b82f6 38%,#8b5cf6 68%,#ef4444 100%)",
          }}
        />
        <p className="text-sm text-[var(--fg-muted)]">
          Trained from scratch on Modal. Completions run live on a CPU endpoint.
        </p>
        <p className="tag mt-3">by Harman Sandhu</p>
      </footer>
    </main>
  );
}

function SectionHeadInline({ kicker, title }: { kicker?: string; title: string }) {
  return (
    <div className="mb-4">
      {kicker && <span className="tag">{kicker}</span>}
      <h3 className={`text-lg font-medium ${kicker ? "mt-1" : ""}`}>{title}</h3>
    </div>
  );
}
