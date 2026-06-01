// Observable Framework site config for the MarinDNA leaderboard.
// See: https://observablehq.com/framework/config

export default {
  title: "MarinDNA Leaderboard",
  root: "src",
  output: "dist",

  // Custom global stylesheet (supersedes `theme`). It re-imports the built-in
  // `air` theme and adds the one thing every page must share — the family
  // color palette. See src/style.css for why `air` specifically.
  style: "style.css",

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
