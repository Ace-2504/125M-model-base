"use client";

import { useState } from "react";
import { DEFAULT_GEN, EXAMPLE_PROMPTS } from "@/lib/model";

type Status = "idle" | "loading" | "unavailable";

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  fmt = (v: number) => v.toString(),
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  fmt?: (v: number) => string;
}) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="tag">{label}</span>
        <span className="mono text-sm text-[var(--accent)]">{fmt(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-[var(--accent)] cursor-pointer"
      />
    </label>
  );
}

export default function Demo() {
  const [prompt, setPrompt] = useState(EXAMPLE_PROMPTS[0]);
  const [temperature, setTemperature] = useState(DEFAULT_GEN.temperature);
  const [maxTokens, setMaxTokens] = useState(DEFAULT_GEN.maxTokens);
  const [topP, setTopP] = useState(DEFAULT_GEN.topP);
  const [topK, setTopK] = useState(DEFAULT_GEN.topK);
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState<string>("");

  async function generate() {
    setStatus("loading");
    setMessage("");
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, temperature, maxTokens, topP, topK }),
      });
      const data = await res.json();
      if (res.ok && data.completion) {
        setStatus("idle");
        setMessage(data.completion);
      } else {
        setStatus("unavailable");
        setMessage(data.message ?? "Inference is not available yet.");
      }
    } catch {
      setStatus("unavailable");
      setMessage("Could not reach the inference endpoint.");
    }
  }

  return (
    <div className="panel p-5 sm:p-6">
      <div className="flex items-center justify-between gap-4 mb-5">
        <div className="flex items-center gap-2.5">
          <span className="tag">interactive</span>
          <h3 className="text-base font-medium">Text completion</h3>
        </div>
        <div className="flex items-center gap-2">
          <span className="live-dot inline-block h-2 w-2 rounded-full bg-[var(--accent-2)]" />
          <span className="tag" style={{ color: "var(--accent-2)" }}>live</span>
        </div>
      </div>

      <label className="block mb-4">
        <span className="tag">your prompt — a prefix to continue</span>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={5}
          className="mono mt-2 w-full resize-none rounded-lg bg-[var(--bg)] border border-[var(--border)] px-4 py-3.5 text-base leading-relaxed text-[var(--fg)] outline-none focus:border-[var(--accent)]/60 transition-colors"
        />
      </label>

      <div className="mb-4 flex flex-wrap gap-1.5">
        {EXAMPLE_PROMPTS.map((p) => (
          <button
            key={p}
            onClick={() => setPrompt(p)}
            className="mono rounded-md border border-[var(--border)] bg-[var(--bg)] px-2.5 py-1 text-[0.7rem] text-[var(--fg-muted)] hover:border-[var(--accent)]/50 hover:text-[var(--fg)] transition-colors truncate max-w-[15rem]"
          >
            {p}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-4 mb-5">
        <Slider label="temperature" value={temperature} min={0} max={1.5} step={0.05} onChange={setTemperature} fmt={(v) => v.toFixed(2)} />
        <Slider label="top-p" value={topP} min={0} max={1} step={0.01} onChange={setTopP} fmt={(v) => v.toFixed(2)} />
        <Slider label="top-k" value={topK} min={0} max={100} step={1} onChange={setTopK} />
        <Slider label="max tokens" value={maxTokens} min={10} max={256} step={1} onChange={setMaxTokens} />
      </div>

      <button
        onClick={generate}
        disabled={status === "loading"}
        className="mono w-full rounded-lg bg-[var(--accent)] px-4 py-2.5 text-sm font-semibold text-[#1a1204] hover:brightness-110 active:brightness-95 disabled:opacity-60 transition-all"
      >
        {status === "loading" ? "generating…" : "Generate"}
      </button>

      <div className="panel-inset mt-4 min-h-[6rem] px-4 py-3.5">
        {message ? (
          <p className={`mono text-sm leading-relaxed ${status === "unavailable" ? "text-[var(--fg-muted)]" : "text-[var(--fg)]"}`}>
            {status === "unavailable" && (
              <span className="mr-1.5 text-amber-400">◈</span>
            )}
            {status !== "unavailable" && (
              <span className="text-[var(--fg-muted)]">{prompt} </span>
            )}
            {message}
          </p>
        ) : (
          <p className="mono text-sm text-[var(--fg-dim)]">
            The completion will appear here.
          </p>
        )}
      </div>
    </div>
  );
}
