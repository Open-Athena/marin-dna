---
title: Accessibility QTL
toc: false
wide: true
---

# Accessibility QTL

Per-variant predictions on two chromatin-accessibility QTL benchmarks — **caQTL** (ATAC; DeGorter et al. 2023) and **dsQTL** (DNase-I sensitivity; Degner et al. 2012) — scored with the supervised official metrics (#311). Two metrics per model: **causality auPRC** (significant-QTL vs control, ranking variants by |score|) and **direction Pearson** (signed correlation of the variant-effect score with the measured study effect, over the positives only). Higher = better. Numbers are the **`train` (odd chroms)** dev split; the even-chrom test set is held out for final eval.

```js
import * as d3 from "npm:d3";
const table = await FileAttachment("../data/accessibility_qtl.parquet").parquet();
```

```js
const rows = table.toArray().map((r) => ({
  model: String(r.model),
  display: String(r.display),
  group: String(r.group),
  scope: String(r.scope),
  causality: Number(r.causality_auPRC),
  causality_se: Number(r.causality_se),
  direction: Number(r.direction_pearson),
  direction_se: Number(r.direction_pearson_se),
  n_rows: Number(r.n_rows),
  n_pos: Number(r.n_pos),
}));

// Scope = the SGE-style selector: "macro" is the green set-apart overview (mean of the
// two assays), then one pill per assay. The two metric columns are the same in each scope.
const SCOPES = [
  {key: "macro", label: "Macro"},
  {key: "caqtl", label: "caQTL"},
  {key: "dsqtl", label: "dsQTL"},
];
const METRICS = [
  {key: "causality", se: "causality_se", label: "Causality", sub: "auPRC"},
  {key: "direction", se: "direction_se", label: "Direction", sub: "Pearson r"},
];
// Coarse model group → swatch color + legend label (self-contained; these supervised
// baselines aren't gLM "families"). A future fine-tuned gLM (#243) lands in `glm`.
const GROUP = {
  supervised: {label: "Supervised baseline", color: "#6b7280"},
  glm: {label: "Fine-tuned gLM", color: "#7c3aed"},
  other: {label: "Other", color: "#9ca3af"},
};
const groupOf = (g) => GROUP[g] ?? GROUP.other;
// Legend shows only the groups actually present — no empty "Fine-tuned gLM"
// entry until a gLM is actually scored (keeps the canonical GROUP order).
const presentGroups = Object.keys(GROUP).filter((k) => rows.some((r) => r.group === k));

// Inlined dataset provenance for the card (these two HF datasets are now read only by
// this page; #310 build, #311 metrics). Counts are the train/odd-chroms split shown.
const DATASETS = {
  caqtl: {
    name: "caQTL — chromatin accessibility (ATAC)",
    hf_repo: "bolinas-dna/evals_caqtl",
    hf_commit: "27a24296",
    study: "DeGorter et al. 2023",
    n_pos: 3173,
    n_rows: 38616,
  },
  dsqtl: {
    name: "dsQTL — DNase-I sensitivity",
    hf_repo: "bolinas-dna/evals_dsqtl",
    hf_commit: "4a3bf152",
    study: "Degner et al. 2012 (hg19→GRCh38)",
    n_pos: 309,
    n_rows: 15018,
  },
};
```

## Dataset

```js
display(html`<div class="card aqtl-card">
  <table class="aqtl-meta">
    <thead><tr><th>Assay</th><th>HF dataset</th><th>Study</th><th>train (odd chroms)</th></tr></thead>
    <tbody>
      ${Object.entries(DATASETS).map(([key, d]) => html`<tr>
        <td><b>${key}</b></td>
        <td><a href=${`https://huggingface.co/datasets/${d.hf_repo}/tree/${d.hf_commit}`}><code>${d.hf_repo} @ ${d.hf_commit}</code></a></td>
        <td>${d.study}</td>
        <td>${d.n_pos.toLocaleString("en-US")} significant QTLs / ${d.n_rows.toLocaleString("en-US")} variants
            (${(100 * d.n_pos / d.n_rows).toFixed(1)}% positive)</td>
      </tr>`)}
    </tbody>
  </table>
  <div class="aqtl-notes">
    <div><b>Causality</b> = auPRC over <em>all</em> variants (significant QTLs vs control), ranking by |score|; the random baseline is the positive rate above. <b>Direction</b> = signed Pearson of the score vs the measured study effect, over the <em>positives only</em> — small-positive dsQTL gives it a wide SE.</div>
    <div><b>Macro</b> averages the two assays (equal weight; SE combined as independent). Even-chrom <code>test</code> is held out for final eval (no split selector). The AG-test slice reproduces AlphaGenome Suppl Table 4 ≤0.005 — see <a href="https://github.com/Open-Athena/marin-dna/issues/311">#311</a>.</div>
  </div>
</div>`);
```

## Leaderboard

```js
// Scope selector — the macro overview is set apart in green (SGE's "All genes" look),
// then one pill per assay. All options visible at a glance; no dropdown.
function ScopePills(options, initial) {
  let value = initial;
  const node = html`<span class="aqtl-scopebar" role="radiogroup" aria-label="QTL scope"></span>`;
  Object.defineProperty(node, "value", {get: () => value});
  const fire = () => node.dispatchEvent(new Event("input", {bubbles: true}));
  function render() {
    node.replaceChildren(
      ...options.map((o) => {
        const isMacro = o.key === "macro";
        return html`<button type="button" role="radio" aria-checked=${value === o.key}
          class=${`aqtl-scope-btn${value === o.key ? " active" : ""}${isMacro ? " aqtl-scope-macro" : ""}`}
          onclick=${() => { if (value !== o.key) { value = o.key; render(); fire(); } }}
        >${o.label}</button>`;
      }),
    );
  }
  render();
  return node;
}
const scope = view(ScopePills(SCOPES, "macro"));
```

```js
// Heatmap: models × {Causality auPRC, Direction Pearson} for the selected scope. Each
// metric column is colored on its own [0, max] YlGn ramp (the two metrics aren't on a
// shared scale — read the number; color ranks models within a column). Click a header
// to sort; ±SE + n on hover.
function aqtlHeatmap(scope) {
  const scoped = rows.filter((r) => r.scope === scope);
  if (scoped.length === 0) return html`<div class="aqtl-empty">No models for this scope.</div>`;
  const maxByMetric = new Map(
    METRICS.map((m) => [m.key, Math.max(...scoped.map((r) => r[m.key]).filter(Number.isFinite))]),
  );
  const ramp = (v, max) =>
    v == null || !Number.isFinite(v) ? "#ffffff"
      : d3.interpolateYlGn(0.1 + 0.85 * Math.max(0, Math.min(1, max > 0 ? v / max : 0)));
  const textColor = (v, max) =>
    v == null || !Number.isFinite(v) ? "#666" : d3.lab(ramp(v, max)).l > 60 ? "#000" : "#fff";

  let sortKey = "causality";
  const root = html`<div></div>`;
  const cmp = () => (a, b) =>
    (b[sortKey] - a[sortKey]) || a.display.localeCompare(b.display);
  function render() {
    const ordered = [...scoped].sort(cmp());
    // Same n across models in a scope (one variant set); show it once per column header.
    const headerN = (m) =>
      m.key === "causality"
        ? `n=${ordered[0].n_rows.toLocaleString("en-US")}`
        : `n=${ordered[0].n_pos.toLocaleString("en-US")} pos`;
    const table = html`<table class="aqtl-heatmap">
      <colgroup><col style="width:230px"></col>${METRICS.map(() => html`<col style="width:150px"></col>`)}</colgroup>
      <thead><tr>
        <th class="aqtl-method-header">Model</th>
        ${METRICS.map(
          (m) => html`<th class=${`aqtl-col${m.key === sortKey ? " aqtl-col-sorted" : ""}`}
            title="Click to sort by this column"
            onclick=${() => { sortKey = m.key; root.replaceChildren(render()); }}>
            <span class="aqtl-col-label">${m.label}</span><br><small>${m.sub} · ${headerN(m)}</small>
          </th>`,
        )}
      </tr></thead>
      <tbody>
        ${ordered.map((r) => {
          const g = groupOf(r.group);
          return html`<tr>
            <td class="aqtl-method">
              <span class="aqtl-swatch" style=${`background:${g.color}`} title=${g.label}></span>
              <code>${r.display}</code>
            </td>
            ${METRICS.map((m) => {
              const v = r[m.key], se = r[m.se], max = maxByMetric.get(m.key);
              const sortedCls = m.key === sortKey ? " aqtl-col-sorted" : "";
              if (!Number.isFinite(v)) return html`<td class=${`aqtl-na${sortedCls}`}>—</td>`;
              return html`<td class=${`aqtl-cell${sortedCls}`}
                style=${`background-color:${ramp(v, max)};color:${textColor(v, max)}`}
                title=${`${m.label} ${m.sub} ${v.toFixed(3)} ± ${Number.isFinite(se) ? se.toFixed(3) : "—"}`}>
                ${v.toFixed(3)}
              </td>`;
            })}
          </tr>`;
        })}
      </tbody>
    </table>`;
    return table;
  }
  root.replaceChildren(render());
  return root;
}
```

<div class="aqtl-legend">
  <span>Cells are the metric value; color ranks models within each column (own 0→max scale — the two metrics are not on a shared scale). Click a header to sort; hover for ± SE.</span>
  <span class="aqtl-grouplegend">
    ${presentGroups.map((k) => groupOf(k)).map((g) => html`<span class="aqtl-glab"><span class="aqtl-swatch" style=${`background:${g.color}`}></span>${g.label}</span>`)}
  </span>
</div>

```js
display(aqtlHeatmap(scope));
```

<style>
:root { --observablehq-max-width: 1400px; }
main > p, main > h1, main > h2, main > .card { max-width: none; }
main > h1, main > h2, main > p { max-width: 900px; }
.aqtl-card { max-width: 900px; }

.aqtl-heatmap {
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  font-size: 0.9em;
  margin: 0.5em 0;
  table-layout: fixed;
}
.aqtl-heatmap th, .aqtl-heatmap td { padding: 7px 6px; border: 1px solid #ddd; }
.aqtl-heatmap thead th { background: #f7f7f7; text-align: center; user-select: none; cursor: pointer; }
.aqtl-heatmap thead tr { height: 42px; }
.aqtl-heatmap tbody tr { height: 30px; }
.aqtl-method-header { text-align: left !important; }
.aqtl-col-label { white-space: nowrap; font-weight: 600; }
.aqtl-heatmap thead th.aqtl-col-sorted { background: #d6e8d6; border: 2px solid #5c8a5c; }
.aqtl-heatmap td.aqtl-col-sorted { border-left: 2px solid #5c8a5c; border-right: 2px solid #5c8a5c; }
.aqtl-cell { text-align: center; font-feature-settings: "tnum"; }
.aqtl-na { text-align: center; color: #aaa; }
.aqtl-method { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.aqtl-method code { font-size: 0.95em; }
.aqtl-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 7px; vertical-align: middle; }
.aqtl-empty { color: #888; margin: 1em 0; font-size: 0.9em; }

.aqtl-meta { border-collapse: collapse; font-size: 0.9em; width: 100%; }
.aqtl-meta th, .aqtl-meta td { padding: 5px 10px; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; }
.aqtl-meta th { color: #555; font-weight: 600; border-bottom: 1px solid #ccc; }
.aqtl-meta code { font-size: 0.92em; }
.aqtl-notes { margin-top: 0.7em; color: #555; font-size: 0.86em; }
.aqtl-notes div { margin: 4px 0; }

.aqtl-legend { display: flex; align-items: center; gap: 18px; margin: 0.5em 0 1em; font-size: 0.85em; color: #444; max-width: 1100px; flex-wrap: wrap; }
.aqtl-grouplegend { display: inline-flex; gap: 14px; }
.aqtl-glab { display: inline-flex; align-items: center; gap: 4px; }

/* Scope selector — segmented control; macro set apart in green (SGE's look). */
.aqtl-scopebar {
  display: inline-flex; flex-wrap: wrap; gap: 4px;
  padding: 4px; background: #f1f3f5; border: 1px solid #e6e9ec; border-radius: 11px;
}
.aqtl-scope-btn {
  appearance: none; border: none; background: transparent; color: #495057;
  font-size: 0.9em; font-weight: 500; padding: 6px 16px; border-radius: 8px;
  cursor: pointer; transition: background .12s, color .12s, box-shadow .12s;
}
.aqtl-scope-btn:hover:not(.active) { background: rgba(0, 0, 0, 0.05); color: #212529; }
.aqtl-scope-btn.active {
  background: #fff; color: #111; font-weight: 600;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.14), 0 0 0 1px rgba(0, 0, 0, 0.04);
}
.aqtl-scope-btn.aqtl-scope-macro { color: #2f6f3e; font-weight: 600; }
.aqtl-scope-btn.aqtl-scope-macro.active {
  background: #2f6f3e; color: #fff; box-shadow: 0 1px 3px rgba(47, 111, 62, 0.35);
}
</style>
