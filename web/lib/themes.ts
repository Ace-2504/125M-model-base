/**
 * Theme registry. Each id maps to a [data-theme="..."] block in globals.css.
 * `accents` are the three corpus colors, shown as a tri-color swatch dot in
 * the picker. Keep this list in sync with the CSS blocks.
 */
export type Theme = {
  id: string;
  name: string;
  accents: [string, string, string];
};

export const THEMES: Theme[] = [
  { id: "parchment", name: "Parchment", accents: ["#b45309", "#0f766e", "#6d28d9"] },
  { id: "amber", name: "Amber Lab", accents: ["#f5b342", "#38d9c4", "#a78bfa"] },
  { id: "cobalt", name: "Cobalt", accents: ["#5b93ff", "#22d3ee", "#9d7bff"] },
  { id: "phosphor", name: "Phosphor", accents: ["#4ade80", "#eab308", "#22d3ee"] },
  { id: "ember", name: "Ember", accents: ["#fb7185", "#fb923c", "#fbbf24"] },
  { id: "nord", name: "Nord Frost", accents: ["#88c0d0", "#a3be8c", "#b48ead"] },
];

export const DEFAULT_THEME = "parchment";
export const STORAGE_KEY = "slm-theme";
