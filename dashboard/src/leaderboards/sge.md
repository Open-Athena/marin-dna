---
title: Saturation Genome Editing
toc: false
wide: true
---

# Saturation Genome Editing

Per-variant endogenous-locus function scores (MaveDB SGE studies). The leaderboard metric is **AUPRC** for the ClinGen/ExCALIBR `calibrated_class` abnormal-vs-normal call — computed **per accession** (study) then macro-averaged over consequence subsets and accessions (scores are non-comparable across studies). Higher = better; the abnormal base rate varies per gene/subset (~5–16%).

```js
import * as d3 from "npm:d3";
const sgeTable = await FileAttachment("../data/sge.parquet").parquet();
const methods = await FileAttachment("../data/models.json").json();
const datasets = await FileAttachment("../data/datasets.json").json();
import {modelHref, attachModelPopover} from "../components/model-cards.js";
import {
  FAMILY_LABEL,
  PROTOCOL_OPTIONS,
  PROTOCOL_DEFAULTS,
  FamilyProtocolToggle,
} from "../components/controls.js";
```

```js
// SGE v3 computes AUPRC only; the metric filter below is defensive.
const rows = sgeTable
  .toArray()
  .map((r) => ({
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
  }))
  .filter((r) => r.metric === "AUPRC");
const modelById = new Map(methods.map((m) => [m.id, m]));
const meta = datasets.sge;

// MarinDNA protocol → score_type column (signed LLR default; ALT direction is
// informative, so not abs). Conservation has its single per-position score.
const SGE_SCORE_TYPE = {
  marin_dna: {LLR: "minus_llr_avg", JSD: "jsd_avg"},
  conservation: {score: "score"},
};
// Columns: subset-macro leftmost (the headline), then the per-subset scopes.
const SUBSET_COLS = [
  {key: "_macro_avg_", label: "Macro"},
  {key: "missense_variant", label: "Missense"},
  {key: "splicing", label: "Splicing"},
  {key: "both", label: "Both"},
];
// AUPRC color domain: anchor at 0 (SGE's abnormal base rate is low, ~5–16%, well
// below the matched-pair 0.10) and run to the metric ceiling 1.0, so the color
// encodes absolute AUPRC. This matches the matched-pair heatmap's upper end
// (components/heatmap.js maps [0.10, 1.0]); same YlGn ramp and `0.1 + 0.85·t`
// mapping, so a given green reads as the same AUPRC across leaderboards.
const AUPRC_DOMAIN = [0, 1.0];
const auprcColor = (v) => {
  if (v == null || !Number.isFinite(v)) return "#ffffff";
  const t = Math.max(0, Math.min(1, (v - AUPRC_DOMAIN[0]) / (AUPRC_DOMAIN[1] - AUPRC_DOMAIN[0])));
  return d3.interpolateYlGn(0.1 + 0.85 * t);
};
const auprcText = (v) => (v == null || !Number.isFinite(v) ? "#666" : d3.lab(auprcColor(v)).l > 60 ? "#000" : "#fff");
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
// Gene scope — all options visible at a glance (macro across accessions, or one
// accession labeled by gene).
const accPairs = Array.from(
  new Map(
    rows.filter((r) => r.accession !== "_macro_avg_").map((r) => [r.accession, r.gene]),
  ).entries(),
).sort((a, b) => a[1].localeCompare(b[1]));
const geneOptions = ["_macro_avg_", ...accPairs.map(([urn]) => urn)];
const geneLabel = new Map([
  ["_macro_avg_", "All genes"],
  ...accPairs.map(([urn, gene]) => [urn, gene]),
]);

// Modern segmented control: a single grey track of pills, active = white pill
// with a soft lift (the iOS look). "All genes" (the overview) is set apart in
// green. All options visible at a glance; no dropdown.
function GenePills(options, initial, labelFor) {
  let value = initial;
  const node = html`<span class="sge-genebar" role="radiogroup" aria-label="Gene scope"></span>`;
  Object.defineProperty(node, "value", {get: () => value});
  const fire = () => node.dispatchEvent(new Event("input", {bubbles: true}));
  function render() {
    node.replaceChildren(
      ...options.map((o) => {
        const isAll = o === "_macro_avg_";
        return html`<button type="button" role="radio" aria-checked=${value === o}
          class=${`sge-gene-btn${value === o ? " active" : ""}${isAll ? " sge-gene-all" : ""}`}
          onclick=${() => { if (value !== o) { value = o; render(); fire(); } }}
        >${labelFor(o)}</button>`;
      }),
    );
  }
  render();
  return node;
}
const geneScope = view(GenePills(geneOptions, "_macro_avg_", (o) => geneLabel.get(o) ?? o));
```

```js
const present = new Set(rows.map((r) => r.family));
const families = Object.keys(FAMILY_LABEL).filter((f) => present.has(f));
const sel = view(FamilyProtocolToggle(families, PROTOCOL_OPTIONS, PROTOCOL_DEFAULTS));
```

```js
const search = view(
  Inputs.text({label: "Model name", placeholder: "filter by method (e.g. exp135, phyloP)"}),
);
```

```js
// Dedicated SGE heatmap: methods × {Macro, Missense, Splicing, Both}, colored by
// AUPRC. Self-managing sort (click a header). Mirrors the matched-pair heatmap's
// look (YlGn cells, family swatch, model popover) but on SGE's columns + domain.
function sgeHeatmap({rows, geneScope}) {
  const byMethod = new Map();
  for (const r of rows) {
    if (!byMethod.has(r.method_id)) {
      byMethod.set(r.method_id, {
        method_id: r.method_id,
        method_display: r.method_display,
        family: r.family,
        cells: new Map(),
      });
    }
    byMethod.get(r.method_id).cells.set(r.subset, {value: r.value, se: r.se, n: r.n, n_pos: r.n_pos});
  }
  if (byMethod.size === 0) {
    return html`<div class="sge-empty">No methods for this selection.</div>`;
  }

  let sortKey = "_macro_avg_";
  const root = html`<div></div>`;
  const cmp = () => (a, b) => {
    const va = a.cells.get(sortKey)?.value ?? -Infinity;
    const vb = b.cells.get(sortKey)?.value ?? -Infinity;
    return vb !== va ? vb - va : a.method_display.localeCompare(b.method_display);
  };
  function render() {
    const ordered = [...byMethod.values()].sort(cmp());
    // Per-column sub-label (same across methods — identical scored rows).
    //  • All-genes macro: K accessions averaged → show coverage (fewer genes
    //    qualify on splicing, where small genes miss the 30-per-class gate).
    //  • Specific gene, Macro column: mean of the qualifying subsets → K subsets.
    //  • Specific gene, a real subset (leaf cell): the AUPRC class balance,
    //    positives vs negatives (n_neg = n − n_pos, since every scored row is
    //    one or the other).
    const colSub = (key) => {
      let c = null;
      for (const m of ordered) { const cc = m.cells.get(key); if (cc != null) { c = cc; break; } }
      if (c == null) return "";
      if (geneScope === "_macro_avg_") return `${c.n} ${c.n === 1 ? "gene" : "genes"}`;
      if (key === "_macro_avg_") return `${c.n} ${c.n === 1 ? "subset" : "subsets"}`;
      return `n=${c.n_pos} vs. ${c.n - c.n_pos}`;
    };
    const table = html`<table class="sge-heatmap">
      <colgroup><col style="width:220px"></col>${SUBSET_COLS.map(() => html`<col style="width:108px"></col>`)}</colgroup>
      <thead><tr>
        <th class="sge-method-header">Model</th>
        ${SUBSET_COLS.map(
          (c) => html`<th class=${`sge-col${c.key === sortKey ? " sge-col-sorted" : ""}`}
            title="Click to sort by this column"
            onclick=${() => { sortKey = c.key; root.replaceChildren(render()); }}>
            <span class="sge-col-label">${c.label}</span><br><small>${colSub(c.key)}</small>
          </th>`,
        )}
      </tr></thead>
      <tbody>
        ${ordered.map((m) => {
          const meta = modelById.get(m.method_id);
          const anchor = html`<a href=${modelHref(m.method_id)}><code>${m.method_display}</code></a>`;
          if (meta) attachModelPopover(anchor, meta);
          return html`<tr>
            <td class="sge-method"><span class=${`lb-family lb-family-${m.family}`} title=${m.family}></span>${anchor}</td>
            ${SUBSET_COLS.map((c) => {
              const cell = m.cells.get(c.key);
              const sortedCls = c.key === sortKey ? " sge-col-sorted" : "";
              if (cell == null || !Number.isFinite(cell.value))
                return html`<td class=${`sge-na${sortedCls}`}>—</td>`;
              return html`<td class=${`sge-cell${sortedCls}`}
                style=${`background-color:${auprcColor(cell.value)};color:${auprcText(cell.value)}`}
                title=${`AUPRC ${cell.value.toFixed(3)} ± ${Number.isFinite(cell.se) ? cell.se.toFixed(3) : "—"}`}>
                ${(cell.value * 100).toFixed(1)}
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

// AUPRC color legend (YlGn on the SGE domain).
function auprcLegend({width = 240, height = 14} = {}) {
  const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
  const stops = d3.range(0, 1.001, 1 / 40).map((t) => ({
    offset: `${t * 100}%`,
    color: auprcColor(AUPRC_DOMAIN[0] + t * (AUPRC_DOMAIN[1] - AUPRC_DOMAIN[0])),
  }));
  return svg`<svg viewBox=${`0 0 ${width} ${height + 18}`} width=${width} style="overflow:visible">
    <defs><linearGradient id="sge-grad" x1="0" x2="1">
      ${stops.map((s) => svg`<stop offset=${s.offset} stop-color=${s.color}></stop>`)}
    </linearGradient></defs>
    <rect x="0" y="0" width=${width} height=${height} fill="url(#sge-grad)" stroke="#888"></rect>
    ${ticks.map((v) => {
      const x = ((v - AUPRC_DOMAIN[0]) / (AUPRC_DOMAIN[1] - AUPRC_DOMAIN[0])) * width;
      return svg`<g transform=${`translate(${x},0)`}><line y1=${height} y2=${height + 4} stroke="#666"></line>
        <text y=${height + 16} text-anchor="middle" font-size="10" fill="#555">${(v * 100).toFixed(0)}</text></g>`;
    })}
  </svg>`;
}
```

```js
function scoreTypeFor(family, protocol) {
  const m = SGE_SCORE_TYPE[family];
  return m ? m[protocol] ?? Object.values(m)[0] : null;
}
const filtered = rows.filter(
  (r) =>
    r.accession === geneScope &&
    sel.families.includes(r.family) &&
    r.score_type === scoreTypeFor(r.family, sel.protocols[r.family] ?? PROTOCOL_DEFAULTS[r.family]) &&
    (!search || r.method_display.toLowerCase().includes(search.toLowerCase())),
);
```

<div class="legend-row">
  <span><b>AUPRC</b> ×100, colored on the 0→100 scale below (anchored at 0, the metric's full range — abnormal base rate ~5–16% varies per gene). <b>Macro</b> = mean of the missense + splicing subsets; <b>Both</b> = the two pooled. Column sub-labels: <b>positives vs. negatives</b> for a selected gene, or genes-averaged for the all-genes macro. Click a column header to sort. Hover a model for its card; hover a cell for ± SE.</span>
  ${auprcLegend()}
</div>

```js
display(sgeHeatmap({rows: filtered, geneScope}));
```

<style>
:root { --observablehq-max-width: 2000px; }
main > p, main > h1, main > h2, main > .card { max-width: none; }
main > h1, main > h2, main > p { max-width: 1100px; }
.card { max-width: 1100px; }

.sge-heatmap {
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  font-size: 0.85em;
  margin: 0.5em 0;
  table-layout: fixed;
}
.sge-heatmap th, .sge-heatmap td { padding: 6px 4px; border: 1px solid #ddd; }
.sge-heatmap thead th { background: #f7f7f7; text-align: center; user-select: none; cursor: pointer; }
.sge-heatmap thead tr { height: 40px; }
.sge-heatmap tbody tr { height: 28px; }
.sge-method-header { text-align: left !important; }
.sge-col-label { white-space: nowrap; }
.sge-heatmap thead th.sge-col-sorted { background: #d6e8d6; font-weight: 600; border: 2px solid #5c8a5c; }
.sge-heatmap td.sge-col-sorted { border-left: 2px solid #5c8a5c; border-right: 2px solid #5c8a5c; }
.sge-cell { text-align: center; font-feature-settings: "tnum"; }
.sge-na { text-align: center; color: #aaa; }
.sge-method { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sge-method a { text-decoration: none; }
.sge-method a:hover { text-decoration: underline; }
.lb-family { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.sge-empty { color: #888; margin: 1em 0; font-size: 0.9em; }
.dataset-meta td { padding: 2px 8px; }
.dataset-meta td:first-child { white-space: nowrap; }
.dataset-bullets { margin: 0.5em 0 0.25em; }
.dataset-bullets div { margin: 2px 0; }
.dataset-notes { margin-top: 0.5em; color: #666; font-size: 0.9em; }
.dataset-notes div { margin: 2px 0; }
.legend-row { display: flex; align-items: center; gap: 16px; margin: 0.5em 0 1em; font-size: 0.85em; color: #444; max-width: 1100px; flex-wrap: wrap; }

/* Gene-scope segmented control */
.sge-genebar {
  display: inline-flex; flex-wrap: wrap; gap: 4px;
  padding: 4px; background: #f1f3f5; border: 1px solid #e6e9ec; border-radius: 11px;
  max-width: 100%;
}
.sge-gene-btn {
  appearance: none; border: none; background: transparent; color: #495057;
  font-size: 0.86em; font-weight: 500; padding: 5px 13px; border-radius: 8px;
  cursor: pointer; transition: background .12s, color .12s, box-shadow .12s;
}
.sge-gene-btn:hover:not(.active) { background: rgba(0, 0, 0, 0.05); color: #212529; }
.sge-gene-btn.active {
  background: #fff; color: #111; font-weight: 600;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.14), 0 0 0 1px rgba(0, 0, 0, 0.04);
}
.sge-gene-btn.sge-gene-all { color: #2f6f3e; font-weight: 600; }
.sge-gene-btn.sge-gene-all.active {
  background: #2f6f3e; color: #fff; box-shadow: 0 1px 3px rgba(47, 111, 62, 0.35);
}

/* Model popover on method-name hover (the element is appended to <body> by
   attachModelPopover, so this CSS must live on every page that uses it — the
   matched-pair/QTL leaderboards carry an identical block). */
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
</style>
