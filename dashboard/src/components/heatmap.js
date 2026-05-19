// Leaderboard heatmap: method × subset → AUPRC color cell.
//
// Sequential YlGn color scale fixed at [0.1, 1.0]:
//   - 0.1 (random baseline)  = pale yellow (neutral)
//   - 1.0 (perfect)          = deep green
// 0.1 is the positive rate under 1:9 matching by design — random ranking
// yields AUPRC = 0.10. AUPRC values below 0.1 are clamped to 0.1; they're
// treated as noise rather than highlighted as anti-predictive. Repick
// the floor if the matching ratio changes.

import * as d3 from "npm:d3";
import {html, svg} from "npm:htl";

import {attachModelPopover} from "./model-cards.js";

const SUBSET_DISPLAY = {
  missense_variant: "Missense",
  splicing: "Splicing",
  "5_prime_UTR_variant": "5' UTR",
  distal: "Distal",
  "3_prime_UTR_variant": "3' UTR",
  tss_proximal: "Promoter",
  non_coding_transcript_exon_variant: "ncRNA",
  synonymous_variant: "Synonymous",
};

export const GLOBAL = "_global_";
export const MACRO = "_macro_avg_";
// Per-subset display threshold: a subset's column is rendered only if at
// least one method has ≥30 positives in it. Matches the pipeline-time
// macro-avg filter (`n_min=30` in metrics.compute_auprc_metrics), so a
// subset that doesn't qualify here also doesn't contribute to Macro Avg.
const N_POSITIVES_MIN = 30;

// Resolve the dashboard's chosen leading aggregate (string "macro_avg" /
// "global", as emitted by `datasets.json.py`) to the subset key the
// heatmap and parquet rows use. Single source of truth — page-level
// cells read this instead of re-mapping the string inline.
export const leadingAggregateSubset = (meta) =>
  meta.leading_aggregate === "macro_avg" ? MACRO : GLOBAL;

// ---- Color palettes -------------------------------------------------------
//
// `absolute` (the default): sequential YlGn on [0.5, 1.0]. Below-random PA
// clamps to 0.5 so it reads as "no signal" rather than a separate regime.
// `delta`: diverging RdYlGn on [-0.10, +0.10]. Negative → red, zero → yellow,
// positive → green. Used by the protocol-comparison pages where cells
// carry "JSD − LLR"-style differences rather than absolute PAs.

const PALETTE_ABSOLUTE = "absolute";
const PALETTE_DELTA = "delta";
const DELTA_DOMAIN = 0.1; // ±10 percentage points

function paColorAbsolute(v) {
  if (v == null) return "#ffffff";
  const clamped = Math.max(0.1, Math.min(1.0, v));
  return d3.interpolateYlGn(0.1 + 0.85 * ((clamped - 0.1) / 0.9));
}

function paColorDelta(v) {
  if (v == null) return "#ffffff";
  const clamped = Math.max(-DELTA_DOMAIN, Math.min(DELTA_DOMAIN, v));
  // Map [-D, +D] to [0, 1] for d3.interpolateRdYlGn (red → green).
  return d3.interpolateRdYlGn((clamped + DELTA_DOMAIN) / (2 * DELTA_DOMAIN));
}

function paColor(v, palette = PALETTE_ABSOLUTE) {
  return palette === PALETTE_DELTA ? paColorDelta(v) : paColorAbsolute(v);
}

function textColor(v, palette = PALETTE_ABSOLUTE) {
  if (v == null) return "#666";
  return d3.lab(paColor(v, palette)).l > 60 ? "#000" : "#fff";
}

function cellLabel(v, palette = PALETTE_ABSOLUTE) {
  const pp = v * 100;
  if (palette === PALETTE_DELTA) {
    const sign = pp > 0 ? "+" : "";
    return `${sign}${pp.toFixed(1)}`;
  }
  return pp.toFixed(1);
}

// Sort: numeric desc, with method_display as a stable tiebreaker so toggle
// behavior is predictable.
function makeComparator(sortKey) {
  return (a, b) => {
    const va = a.cells.get(sortKey)?.value ?? -Infinity;
    const vb = b.cells.get(sortKey)?.value ?? -Infinity;
    if (va !== vb) return vb - va;
    return a.method_display.localeCompare(b.method_display);
  };
}

/**
 * Build the heatmap.
 *
 * @param {object} opts
 * @param {Array} opts.rows long-form (method × subset) rows; output of
 *   `bolinas.pipelines.evals.leaderboard.normalized_rows`.
 * @param {Map}   opts.modelById metadata for each method id (for links + tooltips).
 * @param {string} opts.leadingAggregate "_macro_avg_" or "_global_" — drives
 *   the default sort + which aggregate appears leftmost.
 * @param {string} [opts.sortKey] Currently-sorted column. Optional — defaults
 *   to `leadingAggregate`. Pass this in (and `onSortChange`) to make the
 *   sort selection survive parent re-mounts when filters/protocols change.
 * @param {(col: string) => void} [opts.onSortChange] Invoked when a header
 *   is clicked. The component does not re-render itself in this case — the
 *   caller is expected to update its sortKey state and pass it back in.
 * @returns {HTMLElement}
 */
export function heatmap({
  rows,
  modelById,
  leadingAggregate = MACRO,
  sortKey: initialSortKey,
  onSortChange,
  palette = PALETTE_ABSOLUTE,
  showForest = true,
}) {
  // Group by method_id; collect cells in a Map keyed by subset.
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
    byMethod
      .get(r.method_id)
      .cells.set(r.subset, {value: r.value, se: r.se, n: r.n, n_positives: r.n_positives});
  }

  // Per-subset max counts across methods. `subsetMaxN` (total variants =
  // 10× positives by design under 1:9) drives the "(n=X)" header label;
  // `subsetMaxNPositives` drives the N_POSITIVES_MIN filter — the
  // dashboard hides any subset where no method has at least 30
  // positives, matching the macro-avg gate at pipeline time.
  const subsetMaxN = new Map();
  const subsetMaxNPositives = new Map();
  for (const m of byMethod.values()) {
    for (const [subset, cell] of m.cells) {
      if (subset === GLOBAL || subset === MACRO) continue;
      subsetMaxN.set(subset, Math.max(subsetMaxN.get(subset) ?? 0, cell.n));
      subsetMaxNPositives.set(
        subset,
        Math.max(subsetMaxNPositives.get(subset) ?? 0, cell.n_positives),
      );
    }
  }
  // Columns: aggregates first (leading aggregate leftmost), then per-subset
  // sorted by descending positives count. Subsets with fewer than
  // N_POSITIVES_MIN positives across every method are dropped.
  const subsetCols = [...subsetMaxNPositives.entries()]
    .filter(([s, n_pos]) => n_pos >= N_POSITIVES_MIN && s in SUBSET_DISPLAY)
    .sort((a, b) => b[1] - a[1])
    .map(([s]) => s);
  const aggCols =
    leadingAggregate === MACRO ? [MACRO, GLOBAL] : [GLOBAL, MACRO];
  const columns = [...aggCols, ...subsetCols];

  // Sample any method that has both aggregate rows to read the per-column
  // header counts (n for global, K for macro_avg). Aggregates are constant
  // across methods (same match_groups).
  const sample = [...byMethod.values()].find(
    (m) => m.cells.has(GLOBAL) && m.cells.has(MACRO),
  );
  const globalN = sample?.cells.get(GLOBAL)?.n ?? 0;
  // The macro_avg cell's `n` carries K (qualifying subsets), not variant
  // count — overload set in `leaderboard.fetch_method_metrics`.
  const macroK = sample?.cells.get(MACRO)?.n ?? 0;
  const headerLabel = (col) => {
    // Wrap the label so multi-word names ("Macro Avg") stay on a single row
    // instead of breaking across two lines when the column is tight.
    if (col === GLOBAL)
      return html`<span class="lb-col-label">Global</span><br><small>n=${globalN}</small>`;
    if (col === MACRO)
      return html`<span class="lb-col-label">Macro Avg</span><br><small>${macroK} subsets</small>`;
    return html`<span class="lb-col-label">${SUBSET_DISPLAY[col]}</span><br><small>n=${subsetMaxN.get(col)}</small>`;
  };

  // sortKey is owned by the caller when `onSortChange` is provided — pass
  // the current value as a prop and re-mount the heatmap on change. Without
  // onSortChange, sortKey is internal-only (set by header clicks) and
  // resets to `leadingAggregate` on every re-mount.
  let sortKey = initialSortKey ?? leadingAggregate;
  const sortedMethods = () =>
    [...byMethod.values()].sort(makeComparator(sortKey));

  function colLabelText(col) {
    if (col === GLOBAL) return `Global (n=${globalN})`;
    if (col === MACRO) return `Macro Avg (${macroK} subsets)`;
    return `${SUBSET_DISPLAY[col]} (n=${subsetMaxN.get(col)})`;
  }

  // After mount, read the actual rendered thead height + tbody row height
  // and rebuild the forest plot inside `forestSlot` so each dot lands in
  // the middle of its corresponding heatmap row, even when the browser's
  // computed row height drifts from `HEATMAP_ROW_PX` by sub-pixel amounts
  // (which adds up to ~10px over 26 rows).
  function realignForestPlot(table, forestSlot, methods, columnKey) {
    const thead = table.querySelector("thead");
    const firstRow = table.querySelector("tbody tr");
    const lastRow = table.querySelector("tbody tr:last-child");
    if (!thead || !firstRow) return;
    const headerPx = thead.getBoundingClientRect().height;
    // Measure as the average row spacing across the whole tbody, so any
    // accumulated sub-pixel drift is absorbed.
    const firstRect = firstRow.getBoundingClientRect();
    const lastRect = lastRow.getBoundingClientRect();
    const nRows = methods.filter((m) => m.cells.get(columnKey) != null).length;
    const rowPx =
      nRows > 1
        ? (lastRect.top + lastRect.height / 2 - (firstRect.top + firstRect.height / 2)) /
          (nRows - 1)
        : firstRect.height;
    const fresh = forestPlot(methods, columnKey, colLabelText(columnKey), headerPx, rowPx);
    forestSlot.replaceChildren(fresh);
  }

  function render() {
    const methods = sortedMethods();

    const root = html`<div class="lb-heatmap-wrap"></div>`;

    const table = html`<table class="lb-heatmap">
      <colgroup>
        <col style="width: 210px"></col>
        ${columns.map(() => html`<col style="width: 108px"></col>`)}
      </colgroup>
      <thead>
        <tr>
          <th class="lb-method-header">Model</th>
          ${columns.map(
            (col) => html`<th
              class=${`lb-col${col === sortKey ? " lb-col-sorted" : ""}`}
              onclick=${() => {
                if (onSortChange) {
                  // Caller owns sortKey; it'll re-mount us with the new
                  // value, so don't double-render here.
                  onSortChange(col);
                } else {
                  sortKey = col;
                  const next = render();
                  root.replaceChildren(next);
                }
              }}
              title="Click to sort by this column"
            >
              ${headerLabel(col)}
            </th>`,
          )}
        </tr>
      </thead>
      <tbody>
        ${methods.map((m) => {
          const meta = modelById.get(m.method_id);
          const family = m.family;
          const anchor = html`<a href=${`/models#${encodeURIComponent(m.method_id)}`}><code>${m.method_display}</code></a>`;
          if (meta) attachModelPopover(anchor, meta);
          const modelCell = html`<td class="lb-method">
            <span class=${`lb-family lb-family-${family}`} title=${family}></span>
            ${anchor}
          </td>`;
          return html`<tr>
            ${modelCell}
            ${columns.map((col) => {
              const c = m.cells.get(col);
              const sortedCls = col === sortKey ? " lb-col-sorted" : "";
              if (c == null) return html`<td class=${`lb-na${sortedCls}`}>—</td>`;
              const bg = paColor(c.value, palette);
              const fg = textColor(c.value, palette);
              return html`<td
                class=${`lb-cell${sortedCls}`}
                style=${`background-color: ${bg}; color: ${fg};`}
              >
                ${cellLabel(c.value, palette)}
              </td>`;
            })}
          </tr>`;
        })}
      </tbody>
    </table>`;

    // Heatmap on the left, forest plot adjacent on the right (when
    // showForest). Initial forest plot uses the static `HEATMAP_HEADER_PX`
    // / `HEATMAP_ROW_PX` estimates; a `requestAnimationFrame` callback
    // then measures the browser's actually-rendered row positions and
    // replaces the SVG inside `forestSlot` so each dot lands in the
    // middle of its corresponding heatmap row.
    if (showForest) {
      const forestSlot = html`<div class="lb-forest-slot"></div>`;
      forestSlot.appendChild(forestPlot(methods, sortKey, colLabelText(sortKey)));
      root.append(html`<div class="lb-heatmap-row">${table}${forestSlot}</div>`);
      requestAnimationFrame(() =>
        realignForestPlot(table, forestSlot, methods, sortKey),
      );
    } else {
      root.append(html`<div class="lb-heatmap-row">${table}</div>`);
    }
    return root;
  }

  // Top-level wrapper that swaps in re-rendered tables on column-click.
  // We return one stable node so reactive consumers don't need to re-mount.
  const initial = render();
  return initial;
}

// ---- Forest plot (PA ± SE for the currently sorted column) -----------------
//
// One row per method, in current sort order. A colored dot at the PA value
// with a horizontal whisker spanning [value − SE, value + SE] makes SE
// width explicit — Macro Avg whiskers are narrow (~0.02), per-subset
// whiskers on small-n subsets can be ~0.07-0.10.

// Side-by-side with the heatmap: the heatmap's leftmost column already
// labels the models, so the forest plot drops its left margin and shares
// the same row order/height. `HEATMAP_HEADER_PX` + `HEATMAP_ROW_PX`
// match the *rendered* heights of the heatmap's <thead> and tbody rows
// — the CSS `height` declarations are only a minimum, browsers expand
// rows to fit content, so these constants reflect what the browser
// actually paints (measured at 14px base font in Observable Framework's
// default theme).
const HEATMAP_HEADER_PX = 48;
const HEATMAP_ROW_PX = 30;

function forestPlot(methods, columnKey, columnText, headerPx, rowPx) {
  const visible = methods
    .map((m) => ({m, cell: m.cells.get(columnKey)}))
    .filter((x) => x.cell != null);
  if (visible.length === 0) {
    return html`<div class="lb-forest-empty">No values for ${columnText}.</div>`;
  }

  const width = 254;
  const margin = {top: headerPx ?? HEATMAP_HEADER_PX, right: 42, bottom: 32, left: 16};
  const rowH = rowPx ?? HEATMAP_ROW_PX;
  const height = margin.top + visible.length * rowH + margin.bottom;
  const xMin = 0.1;
  const xMax = 1.0;
  const innerW = width - margin.left - margin.right;
  const xPx = (v) =>
    margin.left + ((Math.max(xMin, Math.min(xMax, v)) - xMin) / (xMax - xMin)) * innerW;
  const xTicks = [0.1, 0.25, 0.5, 0.75, 1.0];
  const tickLabel = (t) => (t * 100).toFixed(0);

  return svg`<svg class="lb-forest" viewBox=${`0 0 ${width} ${height}`} width=${width} style="flex: 0 0 auto;">
    ${xTicks.map(
      (t) => svg`<g>
        <line x1=${xPx(t)} x2=${xPx(t)} y1=${margin.top} y2=${height - margin.bottom}
              stroke=${t === xMin ? "#999" : "#eee"}></line>
        <text x=${xPx(t)} y=${margin.top - 8} text-anchor="middle" font-size="10" fill="#666">${tickLabel(t)}</text>
      </g>`,
    )}
    ${visible.map(({cell}, i) => {
      const y = margin.top + i * rowH + rowH / 2;
      const cx = xPx(cell.value);
      const lo = xPx(cell.value - cell.se);
      const hi = xPx(cell.value + cell.se);
      const fill = paColor(cell.value);
      return svg`<g>
        <line x1=${lo} x2=${hi} y1=${y} y2=${y} stroke="#666" stroke-width="1"></line>
        <circle cx=${cx} cy=${y} r="4.5" fill=${fill} stroke="#333" stroke-width="0.5"></circle>
        <text x=${hi + 5} y=${y} dy="0.32em" font-size="10.5" fill="#444"
              font-variant-numeric="tabular-nums">${(cell.value * 100).toFixed(1)}</text>
      </g>`;
    })}
    <text x=${margin.left + innerW / 2} y=${height - 14} text-anchor="middle"
          font-size="10.5" font-weight="600" fill="#444">AUPRC on ${columnText}</text>
    <text x=${margin.left + innerW / 2} y=${height - 2} text-anchor="middle"
          font-size="9.5" fill="#888">dot = value · whisker = ± SE</text>
  </svg>`;
}

// ---- Color legend ----------------------------------------------------------

export function colorLegend({width = 280, height = 16} = {}) {
  // Domain endpoints + intermediates.
  const ticks = [0.1, 0.25, 0.5, 0.75, 1.0];
  const domainMin = 0.1;
  const domainMax = 1.0;
  const x = (v) => ((v - domainMin) / (domainMax - domainMin)) * width;
  const stops = d3.range(0, 1.001, 1 / 40).map((t) => {
    const v = domainMin + t * (domainMax - domainMin);
    return {offset: `${t * 100}%`, color: paColor(v)};
  });
  return svg`<svg viewBox=${`0 0 ${width} ${height + 18}`} width=${width} style="overflow: visible;">
    <defs>
      <linearGradient id="lb-legend-grad" x1="0" x2="1">
        ${stops.map(
          (s) => svg`<stop offset=${s.offset} stop-color=${s.color}></stop>`,
        )}
      </linearGradient>
    </defs>
    <rect x="0" y="0" width=${width} height=${height} fill="url(#lb-legend-grad)" stroke="#888"></rect>
    ${ticks.map(
      (v) => svg`<g transform=${`translate(${x(v)},0)`}>
        <line y1=${height} y2=${height + 4} stroke="#666"></line>
        <text y=${height + 16} text-anchor="middle" font-size="10" fill="#555">${(v * 100).toFixed(0)}</text>
      </g>`,
    )}
  </svg>`;
}
