---
title: Categorical Jacobian / nucleotide dependency
toc: false
wide: true
---

# Categorical Jacobian / nucleotide dependency maps

A **categorical Jacobian / nucleotide dependency** map measures how substituting the base at one position shifts the model's predicted nucleotide distribution at every other position — collapsed to an `L×L` heatmap over a locus. The method was discovered independently for protein language models (the *categorical Jacobian* — [Zhang et al., *PNAS* 2024](https://doi.org/10.1073/pnas.2406285121)) and for genomic language models (*nucleotide dependency* analysis — [Tomaz da Silva et al., *Nat. Genet.* 2025](https://www.nature.com/articles/s41588-025-02347-3)). Strong off-diagonal blocks flag positions the model treats as coupled (splice sites, structured elements, …).

Pick a locus to see **every model's map for it, stacked** — they share one genomic coordinate axis, so you can read a position straight down across models, against the annotated reference panel (cited beneath it).

Our gLMs are **causal**, so each strand populates only one triangle; the maps stitch a forward and a reverse-complement pass, then symmetrize (`mean`). See [#237](https://github.com/Open-Athena/marin-dna/issues/237) for the method and the autoregressive correctness argument. The visible dependency range is bounded by the model's context window (255 bp), so kilobase-scale structure is not shown here. Color encodes dependency strength (`coolwarm`, per-map robust scaling; the diagonal is masked).

```js
const arch = await FileAttachment("../data/nuc_dep.zip").zip();
const manifest = JSON.parse(await arch.file("manifest.json").text());
const methods = await FileAttachment("../data/models.json").json();
const modelById = new Map(methods.map((m) => [m.id, m]));
import {PillSelect, labeledRow} from "../components/controls.js";
import {modelHref, attachModelPopover} from "../components/model-cards.js";
```

```js
const loci = [...new Set(manifest.map((e) => e.locus))];
const combines = [...new Set(manifest.map((e) => e.combine))];
const locusTitle = new Map(manifest.map((e) => [e.locus, e.title]));
```

```js
const locus = view(
  labeledRow("Locus", PillSelect(loci, loci[0], (l) => locusTitle.get(l) ?? l)),
);
```

```js
// Symmetrization picker appears only when more than one is shipped (we ship the
// mean-symmetrized map); otherwise it stays hidden and `combine` is fixed.
const combine =
  combines.length > 1
    ? view(
        labeledRow(
          "Symmetrization",
          PillSelect(combines, combines.includes("mean") ? "mean" : combines[0]),
        ),
      )
    : combines[0];
```

```js
// Every model's map for the selected locus. The manifest is locus-major then
// model (config order), so `entries` is already a sensible model order; the
// per-locus context (coords / UCSC / paper) is shared, taken from the first.
const entries = manifest.filter((e) => e.locus === locus && e.combine === combine);
const ctx = entries[0];
const svgByModel = new Map(
  await Promise.all(entries.map(async (e) => [e.model, await arch.file(e.svg).text()])),
);
const paperImgUrl =
  ctx && ctx.paper && ctx.paper.image ? await arch.file(ctx.paper.image).url() : null;
```

```js
// Inline the heatmap SVG (matplotlib emits an XML prolog before <svg>; strip
// it), drop the fixed pt width/height so it scales to a shared max width — so
// every model's map renders at the same size and the genomic axes line up.
function renderMap(text) {
  const i = text.indexOf("<svg");
  const div = html`<div class="nd-map"></div>`;
  div.innerHTML = i >= 0 ? text.slice(i) : text;
  const svg = div.querySelector("svg");
  if (svg) {
    svg.removeAttribute("width");
    svg.removeAttribute("height");
    svg.style.width = "100%";
    svg.style.height = "auto";
  }
  return div;
}
```

```js
display(
  ctx
    ? html`<div class="nd-page">
        <aside class="nd-context">
          <h2 class="nd-title">${ctx.title}</h2>
          ${ctx.description ? html`<p class="nd-desc">${ctx.description}</p>` : null}
          <table class="nd-meta">
            <tr><td>Region</td><td><code>${ctx.display_region}</code></td></tr>
            <tr><td>Strand</td><td>${ctx.strand} · ${ctx.span} bp</td></tr>
            <tr><td>Models</td><td>${entries.length}</td></tr>
          </table>
          <p><a href=${ctx.ucsc_url} target="_blank" rel="noopener">View in the UCSC Genome Browser ↗</a></p>
          ${ctx.note ? html`<p class="nd-note">${ctx.note}</p>` : null}
        </aside>
        <div class="nd-maps">
          ${
            paperImgUrl
              ? html`<figure class="nd-panel nd-ref">
                  <figcaption class="nd-panel-name">Annotated reference — <a class="nd-model-link" href=${ctx.paper.url} target="_blank" rel="noopener">${ctx.paper.citation}</a>${ctx.paper.figure ? ` (${ctx.paper.figure})` : ""}</figcaption>
                  <img src=${paperImgUrl} alt=${`Paper reference figure for ${ctx.title}`} />
                </figure>`
              : null
          }
          ${entries.map((e) => {
            // Model name → its Models card, with the same hover popover as the
            // leaderboard (attachModelPopover from model-cards.js).
            const method = modelById.get(e.model);
            const label = method
              ? html`<a class="nd-model-link" href=${modelHref(e.model)}>${e.model_display}</a>`
              : html`<span>${e.model_display}</span>`;
            if (method) attachModelPopover(label, method);
            return html`<figure class="nd-panel">
              <figcaption class="nd-panel-name">${label}</figcaption>
              ${renderMap(svgByModel.get(e.model))}
            </figure>`;
          })}
        </div>
      </div>`
    : html`<div class="nd-empty">No dependency maps are materialized for that selection yet.</div>`,
);
```

<style>
:root { --observablehq-max-width: 1400px; }
main > h1, main > p { max-width: 900px; }

.nd-page { display: flex; gap: 28px; align-items: flex-start; flex-wrap: wrap; margin-top: 0.5em; }
/* Context sticks alongside the scrolling map stack. */
.nd-context { flex: 0 0 280px; max-width: 320px; position: sticky; top: 1rem; }
.nd-title { margin: 0 0 6px; }
.nd-desc { color: #444; margin: 0 0 10px; font-size: 0.92em; }
.nd-meta { border-collapse: collapse; font-size: 0.9em; margin: 0 0 10px; }
.nd-meta td { padding: 2px 12px 2px 0; vertical-align: top; }
.nd-meta td:first-child {
  color: #888; text-transform: uppercase; font-size: 0.72em;
  letter-spacing: 0.04em; white-space: nowrap;
}
.nd-note {
  background: #fff7e6; border-left: 3px solid #f0b429;
  padding: 8px 10px; font-size: 0.88em; color: #5c4813; border-radius: 0 4px 4px 0;
}
.nd-paper { font-size: 0.85em; color: #555; }

/* The stacked maps: one model (or the reference) per row, all the same width so
   the shared genomic axis lines up down the column. */
.nd-maps { flex: 1 1 520px; display: flex; flex-direction: column; gap: 22px; min-width: 320px; }
.nd-panel { margin: 0; max-width: 560px; }
.nd-panel-name {
  font-weight: 600; font-size: 0.9em; margin-bottom: 4px; color: #222;
}
.nd-panel .nd-map { width: 100%; }
.nd-ref { border-bottom: 1px dashed #ccc; padding-bottom: 18px; }
.nd-ref .nd-panel-name { color: #666; font-weight: 500; }
.nd-ref img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
.nd-empty { color: #888; margin: 1em 0; }
.nd-model-link { color: #1c6fb3; text-decoration: none; }
.nd-model-link:hover { text-decoration: underline; }

/* Model popover on model-name hover (mirrors the leaderboard; see model-cards.js). */
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

/* Control rows + segmented pills (mirrored per-page; see components/controls.js). */
.lb-control-row {
  display: inline-flex; align-items: center; gap: 10px;
  margin: 0.25em 1.5em 0.25em 0;
  font-size: 0.85em;
}
.lb-control-label {
  color: #555; text-transform: uppercase; font-size: 0.72em;
  letter-spacing: 0.04em;
}
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
</style>
