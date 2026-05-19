---
title: GPN-Star protocol comparison
toc: false
wide: true
---

# GPN-Star: protocol comparison

```js
const leaderboard = await FileAttachment("../data/leaderboard.parquet").parquet();
const methods = await FileAttachment("../data/models.json").json();
const datasets = await FileAttachment("../data/datasets.json").json();
import {heatmap, colorLegend, leadingAggregateSubset, rowsFromLeaderboard} from "../components/heatmap.js";
import {PillSelect, DirectionPicker, labeledRow} from "../components/controls.js";
```

```js
const FAMILY = "gpn_star";
const PROTOCOLS = ["cLLR", "LLR"];
const DATASETS = ["mendelian_traits", "complex_traits"];
const DATASET_LABEL = {
  mendelian_traits: "Mendelian traits",
  complex_traits: "Complex traits",
};

const allRows = rowsFromLeaderboard(leaderboard);

const modelById = new Map(methods.map(m => [m.id, m]));
```

```js
const dataset = view(
  labeledRow("Dataset", PillSelect(DATASETS, "mendelian_traits", (d) => DATASET_LABEL[d])),
);
```

```js
const direction = view(
  labeledRow(
    "Compare",
    DirectionPicker(PROTOCOLS, "cLLR", "LLR"),
    html`Cells show <b>right − left</b>, in pp.`,
  ),
);
```

```js
// `direction` from the cell above is the latest yielded value of the
// view's Generator — accessible only from a downstream cell, not from
// the cell that calls `view(...)`. Split so `.from` / `.to` resolve.
const baseline = direction.from;
const alternative = direction.to;
```

```js
const meta = datasets[dataset];
const leadingAggregate = leadingAggregateSubset(meta);
```

```js
const sortKeyState = Mutable("_macro_avg_");
const setSortKey = (k) => { sortKeyState.value = k; };
```

```js
{
  setSortKey(leadingAggregate);
}
```

```js
const grouped = new Map();
for (const r of allRows) {
  if (r.family !== FAMILY || r.dataset !== dataset) continue;
  const key = r.method_id + "\0" + r.subset;
  if (!grouped.has(key)) grouped.set(key, {method_id: r.method_id, method_display: r.method_display, subset: r.subset});
  grouped.get(key)[r.protocol] = r;
}

const deltaRows = [];
if (baseline !== alternative) {
  for (const cell of grouped.values()) {
    const d = cell[baseline];
    const a = cell[alternative];
    if (!d || !a) continue;
    deltaRows.push({
      method_id: cell.method_id,
      method_display: cell.method_display,
      family: FAMILY,
      protocol: alternative,
      subset: cell.subset,
      value: a.value - d.value,
      se: 0,
      n: d.n,
      n_positives: d.n_positives,
      dataset: dataset,
    });
  }
}
```

Each cell below is **${alternative} AUPRC − ${baseline} AUPRC**, in percentage points. Green = ${alternative} scores higher than ${baseline}; red = the reverse. cLLR is the producer's recommended protocol on this leaderboard ([Benegas et al. #145](https://github.com/Open-Athena/bolinas-dna/issues/145)) — calibration subtracts pentanucleotide-context background, `llr_calibrated = llr − E[llr | 5-mer, mut]`.

Same matched groups as the [${DATASET_LABEL[dataset]} leaderboard](../leaderboards/${dataset === "mendelian_traits" ? "mendelian" : "complex"}); only the `score_type` filter changes.

<style>
:root { --observablehq-max-width: 1920px; }
main > p, main > h1, main > h2, main > h3, main > .card { max-width: none; }
main > p, main > h1, main > h2, main > h3 { max-width: 1100px; }

.lb-heatmap {
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  font-size: 0.85em;
  margin: 1em 0;
  table-layout: fixed;
  width: 1290px;
}
.lb-heatmap thead tr { height: 40px; }
.lb-heatmap tbody tr { height: 28px; }
.lb-heatmap-row { display: block; }
.lb-heatmap thead th:not(.lb-method-header) { width: 108px; }
.lb-heatmap th.lb-method-header { width: 210px; }
.lb-heatmap th, .lb-heatmap td {
  padding: 6px 4px;
  border: 1px solid #ddd;
}
.lb-heatmap thead th {
  background: #f7f7f7;
  text-align: center;
  cursor: pointer;
  user-select: none;
}
.lb-heatmap thead th:hover { background: #eee; }
.lb-col-label { white-space: nowrap; }
.lb-heatmap thead th.lb-col-sorted { background: #d6e8d6; font-weight: 600; }
.lb-heatmap td.lb-col-sorted { border-left: 2px solid #5c8a5c; border-right: 2px solid #5c8a5c; }
.lb-heatmap thead th.lb-col-sorted { border-left: 2px solid #5c8a5c; border-right: 2px solid #5c8a5c; border-bottom: 2px solid #5c8a5c; }
.lb-method-header { text-align: left !important; cursor: default !important; }
.lb-method {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.lb-method a {
  text-decoration: none;
  display: inline-block;
  max-width: calc(100% - 22px);
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: middle;
}
.lb-method a:hover { text-decoration: underline; }
.lb-family {
  display: inline-block;
  width: 10px; height: 10px;
  border-radius: 2px;
  margin-right: 6px;
  vertical-align: middle;
}
.lb-family-bolinas      { background: #1f77b4; }
.lb-family-conservation { background: #7f7f7f; }
.lb-family-alphagenome  { background: #d62728; }
.lb-family-gpn_star     { background: #9467bd; }
.lb-cell { text-align: center; font-feature-settings: "tnum"; }
.lb-na { text-align: center; color: #aaa; }

/* Control row (dataset + direction pickers) */
.lb-control-row {
  display: inline-flex; align-items: center; gap: 10px;
  margin: 0.25em 1.5em 0.25em 0;
  font-size: 0.85em;
}
.lb-control-label {
  color: #555; text-transform: uppercase; font-size: 0.72em;
  letter-spacing: 0.04em;
}
.lb-control-hint { color: #888; font-size: 0.85em; }

/* Segmented pill toggle */
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

/* Model popover on heatmap method-name hover */
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
  padding: 1px 7px;
  border-radius: 9999px;
  color: #fff;
  width: fit-content;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.lb-pop-family.family-bolinas      { background: #1f77b4; }
.lb-pop-family.family-conservation { background: #7f7f7f; }
.lb-pop-family.family-alphagenome  { background: #d62728; }
.lb-pop-family.family-gpn_star     { background: #9467bd; }
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
.lb-pop-key { color: #888; text-transform: uppercase; font-size: 0.72em; letter-spacing: 0.04em; }
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

```js
display(
  heatmap({
    rows: deltaRows,
    modelById,
    sortKey: sortKeyState,
    onSortChange: setSortKey,
    leadingAggregate,
    palette: "delta",
    showForest: false,
  }),
);
```

<small>Color scale clamps at ±10 percentage points. Sort by clicking any column header — sort follows the dataset's natural rank axis after a dataset switch. The model column links to the model card on the [Models](/models) page.</small>
