// Observable Framework site config for the MarinDNA leaderboard.
// See: https://observablehq.com/framework/config

import {readFileSync} from "node:fs";

export default {
  title: "MarinDNA Leaderboard",
  root: "src",
  output: "dist",

  // Pin to the light `air` theme: the heatmap + forest plot encode meaning in
  // color (sequential YlGn, diverging RdYlGn) and only read against a light
  // page, but the default theme flips to dark on prefers-color-scheme. Handled
  // natively by both `preview` and `build`.
  theme: "air",

  // Shared global CSS injected into every page's <head>: the family color
  // palette + the family-selector pill widget. Inlined (not linked stylesheets)
  // so they need no base-path-relative href and behave identically in `preview`
  // and `build`. NB: read once at config load — editing these files needs a
  // `preview` restart to take effect.
  head: `<style>\n${["family-colors.css", "family-pills.css"]
    .map((f) => readFileSync(new URL("./" + f, import.meta.url), "utf8"))
    .join("\n")}</style>`,

  // Sidebar navigation. eQTL was retired in PR #194 — see #172.
  pages: [
    {
      name: "Leaderboards",
      pages: [
        {name: "Mendelian traits", path: "/leaderboards/mendelian"},
        {name: "Complex traits", path: "/leaderboards/complex"},
      ],
    },
    {
      name: "Protocols",
      pages: [
        {name: "MarinDNA", path: "/protocols/marin_dna"},
        {name: "Evo 2", path: "/protocols/evo2"},
        {name: "GPN-Star", path: "/protocols/gpn-star"},
      ],
    },
    {name: "Models", path: "/models"},
    {name: "About", path: "/about"},
  ],

  // Python data loaders run via `uv run python` so they pick up the project
  // venv (polars + boto3 + the local `marin_dna` package).
  interpreters: {
    ".py": ["uv", "run", "python"],
  },

  // Suppress Observable's automatic header (we render the page title in
  // the markdown body instead, alongside dataset metadata).
  header: "",
  footer: ({path}) =>
    `Source: <a href="https://github.com/Open-Athena/marin-dna/blob/main/dashboard/src${path}.md">dashboard/src${path}.md</a> · <a href="https://github.com/Open-Athena/marin-dna/blob/main/dashboard/models.yaml">models.yaml</a>`,
};
