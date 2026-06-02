---
title: dsQTL Leaderboard
toc: false
wide: true
---

# dsQTL Leaderboard

```js
const leaderboard = await FileAttachment("../data/leaderboard.parquet").parquet();
const methods = await FileAttachment("../data/models.json").json();
const datasets = await FileAttachment("../data/datasets.json").json();
import {rowsFromLeaderboard} from "../components/heatmap.js";
import {qtlTable, QTL_METRICS, QTL_METRIC_LABEL} from "../components/qtl.js";
import {
  FAMILY_LABEL,
  PROTOCOL_OPTIONS,
  PROTOCOL_DEFAULTS,
  FamilyProtocolToggle,
  PillSelect,
} from "../components/controls.js";
```

```js
const allRows = rowsFromLeaderboard(leaderboard);
const dsqtl = allRows.filter(r => r.dataset === "dsqtl");
const modelById = new Map(methods.map(m => [m.id, m]));
const meta = datasets.dsqtl;
```

## Dataset

```js
const hfUrl = `https://huggingface.co/datasets/${meta.hf_repo}/tree/${meta.hf_commit}`;
```

```js
display(html`<div class="card">
  <table class="dataset-meta">
    <tr><td><b>HF dataset</b></td><td><a href=${hfUrl}><code>${meta.hf_repo} @ ${meta.hf_commit}</code></a></td></tr>
    <tr><td><b>Split</b></td><td><code>${meta.split}</code> (used for both training and development; test held out for final-eval)</td></tr>
  </table>
  <div class="dataset-bullets">
    <div><b>Positives:</b> ${meta.positives}</div>
    <div><b>Negatives:</b> ${meta.negatives}</div>
    <div><b>Matching:</b> ${meta.matching}</div>
    <div><b>Metric:</b> ${meta.metric}</div>
  </div>
  <div class="dataset-notes">
    ${meta.notes.map((n) => html`<div>${n}</div>`)}
  </div>
</div>`);
```

## Leaderboard

```js
// Only families with dsQTL data appear (marin_dna exp136, the 5 conservation
// tracks, AlphaGenome). Derive the pills from the families actually present
// in the rows so GPN-Star / Evo 2 (no QTL data) don't render dead pills.
// Order follows FAMILY_LABEL.
const present = new Set(dsqtl.map(r => r.family));
const families = Object.keys(FAMILY_LABEL).filter(f => present.has(f));
// On the QTL benchmark MarinDNA defaults to JSD (the internal protocol key;
// displayed as "NucDep" per #222), not the global LLR — QTL effects are
// unsigned, so a symmetric distributional distance fits better here. LLR
// stays one click away via the toggle.
const QTL_DEFAULTS = {...PROTOCOL_DEFAULTS, marin_dna: "JSD"};
const sel = view(FamilyProtocolToggle(families, PROTOCOL_OPTIONS, QTL_DEFAULTS));
```

```js
// QTL has no consequence subsets — pick which global metric to rank by.
// AUPRC is the DART-Eval Task-5 ranking metric; Pearson/Spearman correlate
// the score with the measured effect_size over positives only.
const metric = view(PillSelect(QTL_METRICS, "AUPRC", (m) => QTL_METRIC_LABEL[m]));
```

```js
const search = view(
  Inputs.text({
    label: "Model name",
    placeholder: "filter by method (e.g. exp136, phyloP)",
  }),
);
```

```js
const filtered = dsqtl.filter(r => {
  if (!sel.families.includes(r.family)) return false;
  // One protocol per family. Falls back to DEFAULTS for families that
  // don't appear in `sel.protocols` (single-option families). qtlTable
  // then selects the active metric row (AUPRC / pearson / spearman).
  const wantedProtocol = sel.protocols[r.family] ?? QTL_DEFAULTS[r.family];
  if (r.protocol !== wantedProtocol) return false;
  if (search && !r.method_display.toLowerCase().includes(search.toLowerCase())) return false;
  return true;
});
```

<style>
/* Observable Framework's default theme caps prose elements at 640px and
   constrains main to ~1072px even on a wide page. Override both so the
   table and the side-by-side forest plot fit on one row. */
:root { --observablehq-max-width: 2200px; }
main > p, main > h1, main > h2, main > h3, main > .card {
  max-width: none;
}
main > h1, main > h2, main > h3, main > p { max-width: 1200px; }
.card { max-width: 1200px; }
.lb-heatmap-row { width: max-content; max-width: 100%; }
.lb-heatmap, .lb-forest { flex: 0 0 auto; }

.lb-heatmap {
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  font-size: 0.85em;
  margin: 1em 0;
  /* Two columns only (Model + selected metric), so a narrow fixed width —
     not the matched-pair heatmap's 1290px. Fixed layout + explicit col
     widths so long model names truncate with an ellipsis. */
  table-layout: fixed;
  width: 360px;
}
.lb-heatmap thead th:not(.lb-method-header) { width: 150px; }
.lb-heatmap th.lb-method-header { width: 210px; }
.lb-heatmap th, .lb-heatmap td {
  padding: 6px 4px;
  border: 1px solid #ddd;
}
/* Explicit row heights so the forest plot to the right (which uses fixed
   pixel rowH) lines up dot-for-row with the table. Keep in sync with
   HEATMAP_HEADER_PX / HEATMAP_ROW_PX in dashboard/src/components/qtl.js. */
.lb-heatmap thead tr { height: 40px; }
.lb-heatmap tbody tr { height: 28px; }
.lb-heatmap-row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  flex-wrap: nowrap;
  overflow-x: auto;
}
/* Drop the table's outer top margin inside the side-by-side row so the
   table's top edge lines up with the forest plot's. */
.lb-heatmap-row .lb-heatmap { margin: 0; }
.lb-heatmap thead th {
  background: #f7f7f7;
  text-align: center;
  user-select: none;
}
.lb-heatmap td.lb-cell { font-variant-numeric: tabular-nums; padding: 6px 4px; }
.lb-col-label { white-space: nowrap; }
.lb-heatmap thead th.lb-col-sorted { background: #d6e8d6; font-weight: 600; }
.lb-heatmap td.lb-col-sorted { border-left: 2px solid #5c8a5c; border-right: 2px solid #5c8a5c; }
.lb-heatmap thead th.lb-col-sorted { border-left: 2px solid #5c8a5c; border-right: 2px solid #5c8a5c; border-bottom: 2px solid #5c8a5c; }
.lb-method-header { text-align: left !important; }
.lb-method {
  /* Long model names truncate with an ellipsis. Hover popover surfaces
     the full name + family + links, so info is one hover away. */
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.lb-method a {
  text-decoration: none;
  display: inline-block;
  max-width: calc(100% - 22px); /* room for the family swatch + margin */
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: middle;
}
.lb-method a:hover { text-decoration: underline; }
.lb-desc { color: #666; font-size: 0.92em; }
.lb-family {
  display: inline-block;
  width: 10px; height: 10px;
  border-radius: 2px;
  margin-right: 6px;
  vertical-align: middle;
}
.lb-cell {
  text-align: center;
  font-feature-settings: "tnum";
}
.lb-na { text-align: center; color: #aaa; }
.lb-forest { display: block; }
.lb-forest-empty { color: #888; margin: 1em 0; font-size: 0.9em; }
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
}

/* Metric selector (PillSelect) — single-choice segmented pill. */
.lb-protocol-segmented {
  display: inline-flex;
  border: 1px solid #ccc;
  border-radius: 6px;
  overflow: hidden;
}
.lb-protocol-btn {
  appearance: none;
  background: #fff;
  border: none;
  border-left: 1px solid #ccc;
  padding: 3px 11px;
  font: inherit;
  font-size: 0.95em;
  color: #555;
  cursor: pointer;
  transition: background 80ms, color 80ms;
}
.lb-protocol-btn:first-child { border-left: none; }
.lb-protocol-btn:hover:not(.active) { background: #f4f4f4; color: #000; }
.lb-protocol-btn.active { background: #333; color: #fff; }

/* Model popover on table method-name hover */
.lb-method-popover {
  background: #fff;
  border: 1px solid #ccc;
  border-radius: 6px;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
  font-size: 0.85em;
  line-height: 1.4;
  padding: 10px 12px;
  min-width: 280px;
  max-width: 360px;
}
.lb-pop-header { display: flex; flex-direction: column; gap: 3px; margin-bottom: 4px; }
.lb-pop-family {
  display: inline-block;
  font-size: 0.7em;
  font-weight: 500;
  padding: 1px 7px;
  border-radius: 9999px;
  color: #fff;
  width: fit-content;
}
.lb-pop-display { font-size: 0.98em; font-weight: 600; }
.lb-pop-desc { color: #555; margin: 4px 0 6px; font-size: 0.92em; }
.lb-pop-specs { margin: 6px 0; }
.lb-pop-row {
  display: grid;
  grid-template-columns: 70px 1fr;
  gap: 10px;
  align-items: baseline;
  margin: 2px 0;
}
.lb-pop-key {
  color: #888; text-transform: uppercase; font-size: 0.72em;
  letter-spacing: 0.04em;
}
.lb-pop-val { font-size: 0.92em; }
.lb-pop-links { margin: 6px 0 2px; font-size: 0.9em; }
.lb-pop-more {
  display: inline-block;
  margin-top: 6px;
  font-size: 0.82em;
  color: #3a7bd5;
}
.muted { color: #aaa; }
</style>

<div class="legend-row">
  <span>Color &amp; forest axis scale to the selected metric, anchored at random performance (dashed line) — AUPRC = positive rate, correlation = 0. Whisker = ± SE.</span>
</div>

```js
display(qtlTable({rows: filtered, metric, modelById}));
```
