---
title: SGE Leaderboard
toc: false
wide: true
---

# SGE Leaderboard

Saturation genome editing: per-variant endogenous-locus function scores. Both metrics are **rank-based** (so conservation tracks and gLMs compare on the same footing): **Spearman** of the deleteriousness score vs `−function_score_aligned`, and **AUPRC** for the ClinGen/ExCALIBR `calibrated_class` abnormal-vs-normal call. Scores are non-comparable across studies, so everything is computed **per accession** (MaveDB study) then macro-averaged.

```js
const sgeTable = await FileAttachment("../data/sge.parquet").parquet();
const methods = await FileAttachment("../data/models.json").json();
const datasets = await FileAttachment("../data/datasets.json").json();
import {
  FAMILY_LABEL,
  PROTOCOL_OPTIONS,
  PROTOCOL_DEFAULTS,
  FamilyProtocolToggle,
} from "../components/controls.js";
```

```js
// Arrow → plain JS rows (sge.parquet schema; see leaderboard.sge_normalized_rows).
const rows = sgeTable.toArray().map((r) => ({
  method_id: String(r.method_id),
  method_display: String(r.method_display),
  family: String(r.family),
  score_type: String(r.score_type),
  metric: String(r.metric),
  subset: String(r.subset),
  accession: String(r.accession),
  gene: String(r.gene),
  value: Number(r.value),
  se: Number(r.se),
  n: Number(r.n),
  n_pos: Number(r.n_pos),
}));
const meta = datasets.sge;
// MarinDNA protocol → the score_type column it selects (signed LLR default; the
// assayed ALT's direction is informative, so not abs). Conservation has one score.
const SGE_SCORE_TYPE = {
  marin_dna: {LLR: "minus_llr_avg", JSD: "jsd_avg"},
  conservation: {score: "score"},
};
const SUBSETS = [
  {key: "missense_variant", label: "Missense"},
  {key: "splicing", label: "Splicing"},
  {key: "both", label: "Both (pooled)"},
  {key: "_macro_avg_", label: "Subset macro"},
];
```

## Dataset

```js
const hfUrl = `https://huggingface.co/datasets/${meta.hf_repo}/tree/${meta.hf_commit}`;
display(html`<div class="card">
  <table class="dataset-meta">
    <tr><td><b>HF dataset</b></td><td><a href=${hfUrl}><code>${meta.hf_repo} @ ${meta.hf_commit}</code></a></td></tr>
    <tr><td><b>Split</b></td><td><code>${meta.split}</code> (development split; test held out for final-eval)</td></tr>
  </table>
  <div class="dataset-bullets">
    <div><b>AUPRC positives:</b> ${meta.positives}</div>
    <div><b>AUPRC negatives:</b> ${meta.negatives}</div>
    <div><b>Aggregation:</b> ${meta.matching}</div>
    <div><b>Metric:</b> ${meta.metric}</div>
  </div>
  <div class="dataset-notes">
    ${meta.notes.map((n) => html`<div>${n}</div>`)}
  </div>
</div>`);
```

## Leaderboard

```js
// Gene-scope: macro across accessions, or one accession (labeled by gene).
const accPairs = Array.from(
  new Map(
    rows.filter((r) => r.accession !== "_macro_avg_").map((r) => [r.accession, r.gene]),
  ).entries(),
).sort((a, b) => a[1].localeCompare(b[1]));
const geneScopeLabel = new Map([
  ["_macro_avg_", "All genes (macro)"],
  ...accPairs.map(([urn, gene]) => [urn, gene]),
]);
const geneScopeOptions = ["_macro_avg_", ...accPairs.map(([urn]) => urn)];
const geneScope = view(
  Inputs.select(geneScopeOptions, {
    label: "Gene",
    value: "_macro_avg_",
    format: (o) => geneScopeLabel.get(o) ?? o,
  }),
);
```

```js
// Only families with SGE data appear; order follows FAMILY_LABEL.
const present = new Set(rows.map((r) => r.family));
const families = Object.keys(FAMILY_LABEL).filter((f) => present.has(f));
const sel = view(FamilyProtocolToggle(families, PROTOCOL_OPTIONS, PROTOCOL_DEFAULTS));
```

```js
const search = view(
  Inputs.text({
    label: "Model name",
    placeholder: "filter by method (e.g. exp135, phyloP)",
  }),
);
```

```js
function activeScoreType(family, protocol) {
  const m = SGE_SCORE_TYPE[family];
  if (!m) return null;
  return m[protocol] ?? Object.values(m)[0];
}

// One row per method for a subset board: {method, family, spearman, AUPRC}.
function boardRows(subsetKey) {
  const byMethod = new Map();
  for (const r of rows) {
    if (r.accession !== geneScope) continue;
    if (r.subset !== subsetKey) continue;
    if (!sel.families.includes(r.family)) continue;
    const wantProto = sel.protocols[r.family] ?? PROTOCOL_DEFAULTS[r.family];
    if (r.score_type !== activeScoreType(r.family, wantProto)) continue;
    if (search && !r.method_display.toLowerCase().includes(search.toLowerCase())) continue;
    if (!byMethod.has(r.method_id)) {
      byMethod.set(r.method_id, {
        method_id: r.method_id,
        method_display: r.method_display,
        family: r.family,
      });
    }
    byMethod.get(r.method_id)[r.metric] = {value: r.value, se: r.se, n: r.n};
  }
  // Sort by Spearman desc (methods missing it sink to the bottom).
  return [...byMethod.values()].sort(
    (a, b) => (b.spearman?.value ?? -Infinity) - (a.spearman?.value ?? -Infinity),
  );
}

function fmtCell(cell) {
  if (cell == null || !Number.isFinite(cell.value)) return html`<span class="lb-na">—</span>`;
  return html`${cell.value.toFixed(3)} <span class="muted">± ${Number.isFinite(cell.se) ? cell.se.toFixed(3) : "—"}</span>`;
}

function renderBoard(subset) {
  const data = boardRows(subset.key);
  return html`<div class="card sge-board">
    <h3>${subset.label}</h3>
    ${
      data.length === 0
        ? html`<div class="lb-forest-empty">No methods for this selection.</div>`
        : html`<table class="sge-heatmap">
            <thead><tr>
              <th class="sge-method-header">Model</th>
              <th>Spearman</th>
              <th>AUPRC</th>
            </tr></thead>
            <tbody>${data.map(
              (d) => html`<tr>
                <td class="sge-method" title=${d.method_display}>
                  <span class=${`lb-family lb-family-${d.family}`}></span>${d.method_display}
                </td>
                <td class="sge-cell">${fmtCell(d.spearman)}</td>
                <td class="sge-cell">${fmtCell(d.AUPRC)}</td>
              </tr>`,
            )}</tbody>
          </table>`
    }
  </div>`;
}
```

<div class="legend-row">
  <span>Per the selected gene scope, one board per consequence subset; rows are methods, sorted by Spearman. <b>±</b> = bootstrap SE. Spearman is oriented so a good deleteriousness predictor scores positive; AUPRC random-baseline is the per-cell abnormal rate.</span>
</div>

```js
display(html`<div class="sge-board-grid">${SUBSETS.map(renderBoard)}</div>`);
```

<style>
:root { --observablehq-max-width: 2200px; }
main > p, main > h1, main > h2, main > h3, main > .card { max-width: none; }
main > h1, main > h2, main > p { max-width: 1200px; }
.card.sge-board { max-width: 560px; margin: 0; }

.sge-board-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  align-items: flex-start;
}
.sge-board h3 { margin: 0 0 0.5em; font-size: 1em; }

.sge-heatmap {
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  font-size: 0.85em;
  table-layout: fixed;
  width: 420px;
}
.sge-heatmap thead th { width: 110px; }
.sge-heatmap th.sge-method-header { width: 200px; text-align: left; }
.sge-heatmap th, .sge-heatmap td { padding: 6px 4px; border: 1px solid #ddd; }
.sge-heatmap thead th { background: #f7f7f7; text-align: center; user-select: none; }
.sge-cell { text-align: center; font-feature-settings: "tnum"; }
.sge-method { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lb-family {
  display: inline-block;
  width: 10px; height: 10px;
  border-radius: 2px;
  margin-right: 6px;
  vertical-align: middle;
}
.lb-na { text-align: center; color: #aaa; }
.lb-forest-empty { color: #888; margin: 0.5em 0; font-size: 0.9em; }
.muted { color: #aaa; }
.dataset-meta td { padding: 2px 8px; }
.dataset-meta td:first-child { white-space: nowrap; }
.dataset-bullets { margin: 0.5em 0 0.25em; }
.dataset-bullets div { margin: 2px 0; }
.dataset-notes { margin-top: 0.5em; color: #666; font-size: 0.9em; }
.dataset-notes div { margin: 2px 0; }
.legend-row {
  display: flex; align-items: center; gap: 12px;
  margin: 0.5em 0 1em;
  font-size: 0.85em; color: #444;
  max-width: 1200px;
}
</style>
