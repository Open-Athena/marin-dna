---
title: Mendelian Traits Leaderboard
toc: false
wide: true
---

# Mendelian Traits Leaderboard

```js
const leaderboard = await FileAttachment("../data/leaderboard.parquet").parquet();
const methods = await FileAttachment("../data/models.json").json();
const datasets = await FileAttachment("../data/datasets.json").json();
import {heatmap, colorLegend, leadingAggregateSubset, rowsFromLeaderboard} from "../components/heatmap.js";
import {
  FAMILY_LABEL,
  PROTOCOL_OPTIONS,
  PROTOCOL_DEFAULTS,
  FamilyProtocolToggle,
  PillToggle,
} from "../components/controls.js";
```

```js
const allRows = rowsFromLeaderboard(leaderboard);
const mendelian = allRows.filter(r => r.dataset === "mendelian_traits");
const modelById = new Map(methods.map(m => [m.id, m]));
const meta = datasets.mendelian_traits;
```

```js
// Sort column state — lives outside the heatmap so it survives re-mounts
// when family / protocol / search filters change. Mutable persists
// across all later heatmap re-renders.
const sortKeyState = Mutable(leadingAggregateSubset(meta));
const setSortKey = (k) => { sortKeyState.value = k; };
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
// Single source of truth: add/remove a key in `FAMILY_LABEL` (controls.js)
// to surface a new family pill. Selecting a family reveals its protocol
// chips inset in the pill (multi-protocol families only).
const families = Object.keys(FAMILY_LABEL);
const sel = view(FamilyProtocolToggle(families, PROTOCOL_OPTIONS, PROTOCOL_DEFAULTS));
```

```js
const search = view(
  Inputs.text({
    label: "Model name",
    placeholder: "filter by method (e.g. exp166, phyloP)",
  }),
);
```

```js
const bestOnly = view(PillToggle("Best per family", false));
```

```js
const filtered = mendelian.filter(r => {
  if (!sel.families.includes(r.family)) return false;
  // One protocol per family. Falls back to DEFAULTS for families that
  // don't appear in `sel.protocols` (single-option families).
  const wantedProtocol = sel.protocols[r.family] ?? PROTOCOL_DEFAULTS[r.family];
  if (r.protocol !== wantedProtocol) return false;
  if (search && !r.method_display.toLowerCase().includes(search.toLowerCase())) return false;
  return true;
});
```

```js
// When the "Best per family" toggle is on, keep only the top-scoring
// method per family at the currently-sorted column. The dependency on
// `sortKeyState` is fine — the heatmap already re-mounts on sort clicks
// (header-click → onSortChange → Mutable write → heatmap-cell re-run).
const displayRows = (() => {
  if (!bestOnly) return filtered;
  const valueAtSort = new Map();
  for (const r of filtered) {
    if (r.subset === sortKeyState) valueAtSort.set(r.method_id, r.value);
  }
  const bestByFamily = new Map();
  for (const r of filtered) {
    const v = valueAtSort.get(r.method_id);
    if (v === undefined) continue;
    const cur = bestByFamily.get(r.family);
    if (cur === undefined || v > cur.value) {
      bestByFamily.set(r.family, {method_id: r.method_id, value: v});
    }
  }
  const keep = new Set([...bestByFamily.values()].map(b => b.method_id));
  return filtered.filter(r => keep.has(r.method_id));
})();
```

<style>
/* Observable Framework's default theme caps prose elements at 640px and
   constrains main to ~1072px even on a wide page. Override both so the
   heatmap and the side-by-side forest plot fit on one row. */
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
  /* Fixed layout with an explicit overall width so every data column
     shares one identical 90px width. 90px fits the widest header
     ("Synonymous") with padding to spare; the model column takes the
     remainder. */
  table-layout: fixed;
  width: 1290px;
}
.lb-heatmap thead th:not(.lb-method-header) { width: 108px; }
.lb-heatmap th.lb-method-header { width: 210px; }
.lb-heatmap th, .lb-heatmap td {
  padding: 6px 4px;
  border: 1px solid #ddd;
}
/* Explicit row heights so the forest plot to the right (which uses fixed
   pixel rowH) lines up dot-for-row with the heatmap. Keep in sync with
   HEATMAP_HEADER_PX / HEATMAP_ROW_PX in dashboard/src/components/heatmap.js. */
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
   heatmap's top edge lines up with the forest plot's. */
.lb-heatmap-row .lb-heatmap { margin: 0; }
.lb-heatmap thead th {
  background: #f7f7f7;
  text-align: center;
  cursor: pointer;
  user-select: none;
}
.lb-heatmap td.lb-cell { font-variant-numeric: tabular-nums; padding: 6px 4px; }
.lb-heatmap thead th:hover { background: #eee; }
.lb-col-label { white-space: nowrap; }
.lb-heatmap thead th.lb-col-sorted { background: #d6e8d6; font-weight: 600; }
/* Bold left/right borders + slight inset shadow mark the sorted column
   on the body rows (not just the header). */
.lb-heatmap td.lb-col-sorted { border-left: 2px solid #5c8a5c; border-right: 2px solid #5c8a5c; }
.lb-heatmap thead th.lb-col-sorted { border-left: 2px solid #5c8a5c; border-right: 2px solid #5c8a5c; border-bottom: 2px solid #5c8a5c; }
.lb-method-header { text-align: left !important; cursor: default !important; }
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
.lb-family-marin_dna      { background: #1f77b4; }
.lb-family-conservation { background: #7f7f7f; }
.lb-family-alphagenome  { background: #d62728; }
.lb-family-gpn_star     { background: #9467bd; }
.lb-family-evo2         { background: #ff7f0e; }
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

/* Compound family pill: the family name is the selection toggle; a selected
   multi-protocol family shows its protocol chips inset inside the same colored
   pill (active chip = white, family-colored text). Single-protocol or
   deselected families render as a plain pill. */
.lb-family-toggle-row {
  display: flex; align-items: center; flex-wrap: wrap; gap: 20px;
  margin: 0.25em 0 0.75em;
}
.lb-cpill {
  display: inline-flex;
  align-items: stretch;
  border: 1.5px solid transparent;
  border-radius: 9999px;
  overflow: hidden;
  font-size: 0.85em;
  line-height: 1;
  transition: background 80ms, border-color 80ms;
}
.lb-cpill:not(.active) { border-color: #ddd; }
.lb-cpill:not(.active):hover { border-color: #999; }
.lb-cpill-name {
  display: inline-flex;
  align-items: center;
  appearance: none;
  background: transparent;
  border: none;
  padding: 4px 12px;
  font: inherit;
  font-size: inherit;
  font-weight: 500;
  color: #999;
  cursor: pointer;
  transition: color 80ms;
}
.lb-cpill:not(.active):hover .lb-cpill-name { color: #555; }
.lb-cpill.active .lb-cpill-name { color: #fff; }
.lb-cpill.active.family-marin_dna      { background: #1f77b4; }
.lb-cpill.active.family-conservation { background: #7f7f7f; }
.lb-cpill.active.family-alphagenome  { background: #d62728; }
.lb-cpill.active.family-gpn_star     { background: #9467bd; }
.lb-cpill.active.family-evo2         { background: #ff7f0e; }
/* Inset protocol chips — only rendered for selected multi-protocol families. */
.lb-cpill-protos {
  display: inline-flex;
  align-items: stretch;
  gap: 1px;
  padding: 2px;
  background: rgba(255, 255, 255, 0.22);
}
.lb-cpill-proto {
  display: inline-flex;
  align-items: center;
  appearance: none;
  background: transparent;
  border: none;
  border-radius: 9999px;
  padding: 1px 8px;
  font: inherit;
  font-size: 0.78em;
  color: rgba(255, 255, 255, 0.92);
  cursor: pointer;
  transition: background 80ms, color 80ms;
}
.lb-cpill-proto:hover:not(.active) { background: rgba(255, 255, 255, 0.18); }
.lb-cpill-proto.active { background: #fff; }
.lb-cpill.family-marin_dna      .lb-cpill-proto.active { color: #1f77b4; }
.lb-cpill.family-conservation .lb-cpill-proto.active { color: #7f7f7f; }
.lb-cpill.family-alphagenome  .lb-cpill-proto.active { color: #d62728; }
.lb-cpill.family-gpn_star     .lb-cpill-proto.active { color: #9467bd; }
.lb-cpill.family-evo2         .lb-cpill-proto.active { color: #ff7f0e; }
.lb-toggle-actions {
  margin-left: 6px;
  color: #888;
  font-size: 0.82em;
}
.lb-toggle-actions .lb-link {
  appearance: none;
  background: transparent;
  border: none;
  padding: 0 2px;
  font: inherit;
  font-size: inherit;
  color: #3a7bd5;
  cursor: pointer;
}
.lb-toggle-actions .lb-link:hover { text-decoration: underline; }

/* Standalone on/off pill (Best per family). Lives on its own row; tap
   to toggle. */
.lb-pill-toggle {
  appearance: none;
  background: #fff;
  border: 1px solid #ccc;
  border-radius: 9999px;
  padding: 3px 12px;
  font: inherit;
  font-size: 0.85em;
  color: #555;
  cursor: pointer;
  transition: background 80ms, color 80ms, border-color 80ms;
}
.lb-pill-toggle:hover:not(.active) { background: #f4f4f4; color: #000; }
.lb-pill-toggle.active {
  background: #333;
  color: #fff;
  border-color: #333;
}

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
.lb-pop-family.family-marin_dna      { background: #1f77b4; }
.lb-pop-family.family-conservation { background: #7f7f7f; }
.lb-pop-family.family-alphagenome  { background: #d62728; }
.lb-pop-family.family-gpn_star     { background: #9467bd; }
.lb-pop-family.family-evo2         { background: #ff7f0e; }
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
  <span>Color: AUPRC, fixed scale</span>
  ${colorLegend({width: 240, height: 14})}
  <span>· Click a column header to re-sort</span>
</div>

```js
// Reactive on `filtered` (family/protocol/search) AND on `sortKeyState`
// (column click). Observable Framework auto-unwraps a Mutable when read
// from another cell — `sortKeyState` here is the current string value,
// not the Mutable instance.
display(
  heatmap({
    rows: displayRows,
    modelById,
    sortKey: sortKeyState,
    onSortChange: setSortKey,
    leadingAggregate: meta.leading_aggregate === "macro_avg" ? "_macro_avg_" : "_global_",
  }),
);
```
