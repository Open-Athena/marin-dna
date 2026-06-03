---
title: Nucleotide dependency
toc: false
wide: true
---

# Nucleotide dependency maps

A *nucleotide dependency map* (the **categorical Jacobian**) measures how substituting the base at one position shifts the model's predicted nucleotide distribution at every other position — collapsed to an `L×L` heatmap over a locus. Strong off-diagonal blocks flag positions the model treats as coupled (splice sites, structured elements, …).

Our gLMs are **causal**, so each strand populates only one triangle; the maps below stitch a forward and a reverse-complement pass, then symmetrize. See [#237](https://github.com/Open-Athena/marin-dna/issues/237) for the method and the autoregressive correctness argument. The visible dependency range is bounded by the model's context window (255 bp for exp135), so kilobase-scale structure is not shown here.

Color encodes dependency strength (`coolwarm`, per-map robust scaling; the diagonal is masked). **Symmetrization** — `mean` vs `max` of the stitched forward/RC map — is a toggle; the two are near-identical (Spearman ≈ 0.98).

```js
const arch = await FileAttachment("../data/nuc_dep.zip").zip();
const manifest = JSON.parse(await arch.file("manifest.json").text());
import {PillSelect, labeledRow} from "../components/controls.js";
```

```js
// Option lists + label maps, derived from the manifest (only materialized
// (locus, model, combine) combinations appear — e.g. a locus whose SVG isn't
// rendered yet is simply absent).
const loci = [...new Set(manifest.map((e) => e.locus))];
const models = [...new Set(manifest.map((e) => e.model))];
const combines = [...new Set(manifest.map((e) => e.combine))];
const locusTitle = new Map(manifest.map((e) => [e.locus, e.title]));
const modelDisp = new Map(manifest.map((e) => [e.model, e.model_display]));
```

```js
const locus = view(
  labeledRow("Locus", PillSelect(loci, loci[0], (l) => locusTitle.get(l) ?? l)),
);
```

```js
const model = view(
  labeledRow("Model", PillSelect(models, models[0], (m) => modelDisp.get(m) ?? m)),
);
```

```js
const combine = view(
  labeledRow(
    "Symmetrization",
    PillSelect(combines, combines.includes("mean") ? "mean" : combines[0]),
  ),
);
```

```js
const entry = manifest.find(
  (e) => e.locus === locus && e.model === model && e.combine === combine,
);
const svgText = entry ? await arch.file(entry.svg).text() : null;
```

```js
// Inline the heatmap SVG (matplotlib emits an XML prolog before <svg>; strip
// it), drop the fixed pt width/height so it scales to the column via its
// viewBox. The embedded raster keeps `image-rendering:pixelated`, so cells stay
// crisp at any size.
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
    svg.style.maxWidth = "560px";
  }
  return div;
}
```

```js
display(
  entry
    ? html`<div class="nd-layout">
        ${renderMap(svgText)}
        <div class="nd-context">
          <h2 class="nd-title">${entry.title}</h2>
          ${entry.description ? html`<p class="nd-desc">${entry.description}</p>` : null}
          <table class="nd-meta">
            <tr><td>Region</td><td><code>${entry.display_region}</code> (${entry.strand} strand), ${entry.span} bp</td></tr>
            <tr><td>Model</td><td>${entry.model_display}</td></tr>
            <tr><td>Symmetrization</td><td>${entry.combine}</td></tr>
          </table>
          <p><a href=${entry.ucsc_url} target="_blank" rel="noopener">View region in the UCSC Genome Browser ↗</a></p>
          ${entry.note ? html`<p class="nd-note">${entry.note}</p>` : null}
          ${
            entry.paper
              ? html`<p class="nd-paper">Reference: <a href=${entry.paper.url} target="_blank" rel="noopener">${entry.paper.citation}</a>${entry.paper.figure ? html` (${entry.paper.figure})` : ""}.</p>`
              : null
          }
          ${
            entry.paper && entry.paper.image
              ? html`<figure class="nd-figure">
                  <img src=${entry.paper.image} alt=${`Paper reference figure for ${entry.title}`} />
                  <figcaption>Paper reference — ${entry.paper.citation}${entry.paper.figure ? ` (${entry.paper.figure})` : ""}.</figcaption>
                </figure>`
              : null
          }
        </div>
      </div>`
    : html`<div class="nd-empty">No dependency map is materialized for that selection yet.</div>`,
);
```

<style>
:root { --observablehq-max-width: 1400px; }
main > h1, main > p { max-width: 900px; }

.nd-layout { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; margin-top: 0.5em; }
.nd-map { flex: 0 0 auto; }
.nd-context { flex: 1 1 300px; min-width: 280px; max-width: 520px; }
.nd-title { margin: 0 0 6px; }
.nd-desc { color: #444; margin: 0 0 10px; }
.nd-meta { border-collapse: collapse; font-size: 0.9em; margin: 0 0 10px; }
.nd-meta td { padding: 2px 12px 2px 0; vertical-align: top; }
.nd-meta td:first-child {
  color: #888; text-transform: uppercase; font-size: 0.72em;
  letter-spacing: 0.04em; white-space: nowrap;
}
.nd-note {
  background: #fff7e6; border-left: 3px solid #f0b429;
  padding: 8px 10px; font-size: 0.9em; color: #5c4813; border-radius: 0 4px 4px 0;
}
.nd-paper { font-size: 0.88em; color: #555; }
.nd-figure { margin: 12px 0 0; }
.nd-figure img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
.nd-figure figcaption { font-size: 0.8em; color: #888; margin-top: 4px; }
.nd-empty { color: #888; margin: 1em 0; }

/* Control rows + segmented pills (shared widget styles, mirrored per-page like
   the leaderboard/protocol pages — see components/controls.js). */
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
