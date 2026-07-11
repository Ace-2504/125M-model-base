"use client";

import { useEffect, useState } from "react";
import { THEMES, DEFAULT_THEME, STORAGE_KEY } from "@/lib/themes";

export default function ThemePicker() {
  const [active, setActive] = useState(DEFAULT_THEME);

  // The inline script in layout.tsx sets data-theme before paint; read it back.
  useEffect(() => {
    const current =
      document.documentElement.getAttribute("data-theme") ?? DEFAULT_THEME;
    setActive(current);
  }, []);

  function choose(id: string) {
    setActive(id);
    document.documentElement.setAttribute("data-theme", id);
    try {
      localStorage.setItem(STORAGE_KEY, id);
    } catch {
      /* private mode — ignore */
    }
  }

  return (
    <div
      role="radiogroup"
      aria-label="Color theme"
      className="flex items-center gap-1.5 rounded-full border border-[var(--border)] bg-[var(--panel)] px-2 py-1.5"
    >
      {THEMES.map((t) => {
        const isActive = t.id === active;
        return (
          <button
            key={t.id}
            role="radio"
            aria-checked={isActive}
            aria-label={t.name}
            title={t.name}
            onClick={() => choose(t.id)}
            className="group relative grid place-items-center rounded-full outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
            style={{ width: 20, height: 20 }}
          >
            <span
              className="block rounded-full transition-transform group-hover:scale-110"
              style={{
                width: 14,
                height: 14,
                background: `conic-gradient(${t.accents[0]} 0 33.3%, ${t.accents[1]} 0 66.6%, ${t.accents[2]} 0)`,
                boxShadow: isActive
                  ? "0 0 0 2px var(--bg), 0 0 0 3.5px var(--fg)"
                  : "0 0 0 1px rgba(0,0,0,0.25)",
              }}
            />
          </button>
        );
      })}
    </div>
  );
}
