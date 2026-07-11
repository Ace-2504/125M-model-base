import { PIPELINE } from "@/lib/model";

export default function Pipeline() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      {PIPELINE.map((stage, i) => (
        <div key={stage.id} className="panel-inset relative p-4 pl-5 overflow-hidden">
          <span
            className="mono absolute right-3 top-2 text-[3rem] font-bold leading-none text-white/[0.03] select-none"
            aria-hidden
          >
            {String(i + 1).padStart(2, "0")}
          </span>
          <div className="absolute left-0 top-0 h-full w-[3px]" style={{ background: "var(--accent)", opacity: 0.5 }} />
          <div className="tag mb-1.5">step {i + 1}</div>
          <h4 className="text-sm font-medium text-[var(--fg)] mb-1">{stage.title}</h4>
          <p className="text-[0.8rem] leading-relaxed text-[var(--fg-muted)]">{stage.desc}</p>
        </div>
      ))}
    </div>
  );
}
