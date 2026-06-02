// QTL leaderboard: single selected-metric table + forest plot.
//
// caQTL / dsQTL use the `qtl_global` eval path (PR #217) — no consequence
// subsets, no matched negatives. The metrics are global AUPRC (the DART-Eval
// Task-5 ranking metric, significant-QTL vs control) plus Pearson/Spearman of
// the variant score vs the measured `effect_size` over positives only. A
// single shared color scale can't serve AUPRC and the correlations (different
// ranges), so the page puts a metric selector (AUPRC | Pearson | Spearman) on
// top and this component renders ONE column of the chosen metric + a forest
// plot, with the color scale AND forest x-axis derived from the displayed
// values — so the range adapts per metric and never clamps a real value
// (e.g. dsQTL AlphaGenome AUPRC ≈ 0.40 vs caQTL's ≈ 0.28).
//
// Deliberately separate from heatmap.js: that component is built around the
// matched-pair consequence-subset model (SUBSET_DISPLAY, N_POSITIVES_MIN
// gating, GLOBAL/MACRO aggregates, caller-owned sort state). Bending it to a
// single metric-indexed column with a per-view domain would add a conditional
// at each of those points — more entangled, not less. CSS classes
// (`lb-heatmap`, `lb-cell`, `lb-method`, `lb-forest`, …) are shared so the
// page's <style> block styles both; the forest-plot draw is copied here and
// parameterized by (domain, ticks, label).

import * as d3 from "npm:d3";
import {html, svg} from "npm:htl";

import {attachModelPopover, modelHref} from "./model-cards.js";

// The three metric rows (the `subset` overload from
// leaderboard.fetch_method_metrics' QTL branch), in display order.
export const QTL_METRICS = ["AUPRC", "pearson", "spearman"];
export const QTL_METRIC_LABEL = {
  AUPRC: "AUPRC",
  pearson: "Pearson r",
  spearman: "Spearman ρ",
};

// Static estimates of the rendered <thead> / tbody-row heights; the forest
// plot uses them for its initial draw, then a rAF callback re-measures the
// real heights and redraws so each dot lands in its row's vertical center.
// Match the CSS row heights in the QTL pages' <style> block.
const HEATMAP_HEADER_PX = 48;
const HEATMAP_ROW_PX = 30;

// Random-performance baseline per metric. AUPRC's is the positive rate
// (prevalence = n_pos / n_rows) — data-driven, so it's the dataset's own value
// (≈0.077 caQTL, ≈0.021 dsQTL) with no hard-coded constant. The correlations'
// is 0 (no association). Color and the forest axis are anchored here so a
// random method reads as "no signal", mirroring the matched-pair heatmap's
// 0.10 floor.
function metricBaseline(metric, rows) {
  if (metric !== "AUPRC") return 0;
  // Every AUPRC row carries the dataset's n (= n_rows) and n_positives, so any
  // row gives the prevalence; they're identical across methods.
  const r = rows[0];
  return r && r.n > 0 ? r.n_positives / r.n : 0;
}

// Two domains, both anchored at `baseline` (random performance) and recomputed
// per render (so switching metric or filtering families rescales).
//   - colorDomain spans [baseline, max value]: color encodes the value (not the
//     whisker, so a cell's shade doesn't move with its SE), and a method at or
//     below random clamps to the palest shade.
//   - axisDomain widens to fit the ± SE whiskers AND the baseline, so the
//     forest plot never clamps a whisker and always shows the random marker.
// The printed value + forest axis are the source of truth; color is an aid.
function colorDomain(rows, baseline) {
  if (rows.length === 0) return [baseline, baseline + 1];
  const hi = Math.max(...rows.map((r) => r.value));
  return [baseline, Math.max(hi, baseline + 1e-6)];
}

function axisDomain(rows, baseline) {
  if (rows.length === 0) return [baseline, baseline + 1];
  const hi = Math.max(...rows.map((r) => r.value + r.se));
  const lo = Math.min(baseline, ...rows.map((r) => r.value - r.se));
  const pad = (hi - lo) * 0.06 || 0.01;
  return [lo, hi + pad];
}

function niceTicks([lo, hi]) {
  const t = d3.ticks(lo, hi, 5);
  return t.length ? t : [lo, hi];
}

function qtlColor(v, [lo, hi]) {
  if (v == null) return "#ffffff";
  const t = hi > lo ? Math.max(0, Math.min(1, (v - lo) / (hi - lo))) : 0;
  return d3.interpolateYlGn(0.1 + 0.85 * t);
}

function qtlTextColor(v, domain) {
  if (v == null) return "#666";
  return d3.lab(qtlColor(v, domain)).l > 60 ? "#000" : "#fff";
}

const fmtVal = (v) => v.toFixed(3);
const fmtN = (n) => n.toLocaleString("en-US");
// Forest x-axis tick labels: round to ≤6 decimals and trim trailing zeros, so
// d3.ticks float artifacts (e.g. 0.15000000000000002) render clean.
const fmtTick = d3.format("~f");

// Header count differs by metric: AUPRC is over all variants, the
// correlations over positives only.
function headerCount(metric, n) {
  return metric === "AUPRC" ? `n=${fmtN(n)}` : `n=${fmtN(n)} pos`;
}

/**
 * Single-metric QTL table + forest plot.
 *
 * @param {object} opts
 * @param {Array}  opts.rows long-form QTL rows for ONE dataset; `subset`
 *   holds the metric name (AUPRC / pearson / spearman). Output of
 *   leaderboard.normalized_rows, filtered to one dataset (+ family/protocol/search).
 * @param {string} opts.metric "AUPRC" | "pearson" | "spearman".
 * @param {Map}    opts.modelById id → model metadata (links + hover popover).
 * @returns {HTMLElement}
 */
export function qtlTable({rows, metric, modelById}) {
  const label = QTL_METRIC_LABEL[metric] ?? metric;

  // One value per method at the selected metric, sorted descending.
  const byMethod = new Map();
  for (const r of rows) {
    if (r.subset !== metric) continue;
    byMethod.set(r.method_id, {
      method_id: r.method_id,
      method_display: r.method_display,
      family: r.family,
      value: r.value,
      se: r.se,
      n: r.n,
      n_positives: r.n_positives,
    });
  }
  const sorted = [...byMethod.values()].sort(
    (a, b) => b.value - a.value || a.method_display.localeCompare(b.method_display),
  );
  if (sorted.length === 0) {
    return html`<div class="lb-forest-empty">No methods match the current filters for ${label}.</div>`;
  }

  const baseline = metricBaseline(metric, sorted);
  const colorDom = colorDomain(sorted, baseline);
  const axisDom = axisDomain(sorted, baseline);
  const ticks = niceTicks(axisDom);
  const n = sorted[0].n; // same across methods (one dataset, one metric)

  const table = html`<table class="lb-heatmap lb-qtl">
    <colgroup>
      <col style="width: 210px"></col>
      <col style="width: 150px"></col>
    </colgroup>
    <thead>
      <tr>
        <th class="lb-method-header">Model</th>
        <th class="lb-col lb-col-sorted">
          <span class="lb-col-label">${label}</span><br><small>${headerCount(metric, n)}</small>
        </th>
      </tr>
    </thead>
    <tbody>
      ${sorted.map((m) => {
        const meta = modelById.get(m.method_id);
        const anchor = html`<a href=${modelHref(m.method_id)}><code>${m.method_display}</code></a>`;
        if (meta) attachModelPopover(anchor, meta);
        const bg = qtlColor(m.value, colorDom);
        const fg = qtlTextColor(m.value, colorDom);
        return html`<tr>
          <td class="lb-method">
            <span class=${`lb-family lb-family-${m.family}`} title=${m.family}></span>
            ${anchor}
          </td>
          <td class="lb-cell lb-col-sorted" style=${`background-color: ${bg}; color: ${fg};`}>
            ${fmtVal(m.value)}
          </td>
        </tr>`;
      })}
    </tbody>
  </table>`;

  // Forest plot adjacent on the right. Initial draw uses the static height
  // estimates; a rAF callback then re-measures and redraws so dots align to
  // the browser's actually-rendered rows.
  const forestSlot = html`<div class="lb-forest-slot"></div>`;
  forestSlot.appendChild(qtlForestPlot(sorted, {axisDom, colorDom, ticks, label, baseline}));
  const root = html`<div class="lb-heatmap-wrap"><div class="lb-heatmap-row">${table}${forestSlot}</div></div>`;
  requestAnimationFrame(() =>
    realignForest(table, forestSlot, sorted, {axisDom, colorDom, ticks, label, baseline}),
  );
  return root;
}

function realignForest(table, forestSlot, sorted, opts) {
  const thead = table.querySelector("thead");
  const firstRow = table.querySelector("tbody tr");
  const lastRow = table.querySelector("tbody tr:last-child");
  if (!thead || !firstRow) return;
  const headerPx = thead.getBoundingClientRect().height;
  const firstRect = firstRow.getBoundingClientRect();
  const lastRect = lastRow.getBoundingClientRect();
  const nRows = sorted.length;
  const rowPx =
    nRows > 1
      ? (lastRect.top + lastRect.height / 2 - (firstRect.top + firstRect.height / 2)) /
        (nRows - 1)
      : firstRect.height;
  forestSlot.replaceChildren(qtlForestPlot(sorted, opts, headerPx, rowPx));
}

// Forest plot: one row per method (current sort order), a colored dot at the
// value with a ± SE whisker, and a dashed `baseline` line marking random
// performance. axisDom (x-axis, fits whiskers + baseline), colorDom (dot
// color, point values), ticks, label, baseline are passed in (per-metric,
// data-driven) — unlike heatmap.js's forestPlot, hard-wired to the
// matched-pair [0.1, 1.0] AUPRC scale.
function qtlForestPlot(
  sorted,
  {axisDom: [xMin, xMax], colorDom, ticks, label, baseline},
  headerPx,
  rowPx,
) {
  if (sorted.length === 0) {
    return html`<div class="lb-forest-empty">No values for ${label}.</div>`;
  }
  const width = 254;
  const margin = {top: headerPx ?? HEATMAP_HEADER_PX, right: 46, bottom: 32, left: 16};
  const rowH = rowPx ?? HEATMAP_ROW_PX;
  const height = margin.top + sorted.length * rowH + margin.bottom;
  const innerW = width - margin.left - margin.right;
  const span = xMax - xMin || 1;
  const xPx = (v) =>
    margin.left + ((Math.max(xMin, Math.min(xMax, v)) - xMin) / span) * innerW;

  return svg`<svg class="lb-forest" viewBox=${`0 0 ${width} ${height}`} width=${width} style="flex: 0 0 auto;">
    ${ticks.map(
      (t) => svg`<g>
        <line x1=${xPx(t)} x2=${xPx(t)} y1=${margin.top} y2=${height - margin.bottom}
              stroke="#eee"></line>
        <text x=${xPx(t)} y=${margin.top - 8} text-anchor="middle" font-size="10" fill="#666">${fmtTick(t)}</text>
      </g>`,
    )}
    <line x1=${xPx(baseline)} x2=${xPx(baseline)} y1=${margin.top} y2=${height - margin.bottom}
          stroke="#c66" stroke-width="1" stroke-dasharray="3 2"></line>
    <text x=${xPx(baseline)} y=${margin.top - 19}
          text-anchor=${xPx(baseline) < margin.left + 22 ? "start" : "middle"}
          font-size="8.5" fill="#c66">random</text>
    ${sorted.map((cell, i) => {
      const y = margin.top + i * rowH + rowH / 2;
      const cx = xPx(cell.value);
      const lo = xPx(cell.value - cell.se);
      const hi = xPx(cell.value + cell.se);
      const fill = qtlColor(cell.value, colorDom);
      return svg`<g>
        <line x1=${lo} x2=${hi} y1=${y} y2=${y} stroke="#666" stroke-width="1"></line>
        <circle cx=${cx} cy=${y} r="4.5" fill=${fill} stroke="#333" stroke-width="0.5"></circle>
        <text x=${hi + 5} y=${y} dy="0.32em" font-size="10.5" fill="#444"
              font-variant-numeric="tabular-nums">${cell.value.toFixed(3)}</text>
      </g>`;
    })}
    <text x=${margin.left + innerW / 2} y=${height - 14} text-anchor="middle"
          font-size="10.5" font-weight="600" fill="#444">${label}</text>
    <text x=${margin.left + innerW / 2} y=${height - 2} text-anchor="middle"
          font-size="9.5" fill="#888">dot = value · whisker = ± SE</text>
  </svg>`;
}
