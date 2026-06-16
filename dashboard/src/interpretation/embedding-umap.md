---
title: Embedding UMAP
toc: false
wide: true
---

# Embedding UMAP

A **UMAP of model embeddings** over 111,329 labeled 100 bp genomic windows — coding (CDS), 5′/3′ UTRs, lncRNA, promoters and enhancers (ENCODE cCREs), and background — asking whether a model's representations segregate functional elements *without supervision*. Ported from **GPN-Star** ([Ye, Benegas et al., bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.09.21.677619v1), Fig 4A/4B); the window set is their published [`songlab/gpn-star-umap-regions`](https://huggingface.co/datasets/songlab/gpn-star-umap-regions), so these plots are directly comparable to the paper. See [#246](https://github.com/Open-Athena/marin-dna/issues/246) for the method.

Each window's embedding is the model's **last-layer hidden state**, mean-pooled over the central 100 bp of a context-sized window and averaged across the forward and reverse-complement strands (for our causal models the strand average also corrects the left-context bias). Every window is embedded — none dropped — so the point set is identical across models of different context sizes. **Left:** colored by annotated region. **Right:** by conservation (75th-percentile phastCons). Points are rasterized; the legends carry the color key.

```js
const arch = await FileAttachment("../data/umap.zip").zip();
const manifest = JSON.parse(await arch.file("manifest.json").text());
const methods = await FileAttachment("../data/models.json").json();
const modelById = new Map(methods.map((m) => [m.id, m]));
import {modelHref, attachModelPopover} from "../components/model-cards.js";
```

```js
// One panel per model (config order); each shows region + conservation side by
// side. Pull both SVG texts from the archive up front.
const models = [...new Set(manifest.map((e) => e.model))];
const displayByModel = new Map(manifest.map((e) => [e.model, e.model_display]));
const svgByKey = new Map(
  await Promise.all(
    manifest.map(async (e) => [`${e.model}/${e.color_by}`, await arch.file(e.svg).text()]),
  ),
);
```

```js
// Inline an SVG: strip matplotlib's XML prolog and drop the fixed pt
// width/height so it scales to its column.
function renderSvg(text) {
  const i = text.indexOf("<svg");
  const div = html`<div class="um-svg"></div>`;
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
  models.length
    ? html`<div class="um-models">
        ${models.map((model) => {
          // Model name → its Models card, with the leaderboard hover popover.
          const method = modelById.get(model);
          const label = method
            ? html`<a class="um-model-link" href=${modelHref(model)}>${displayByModel.get(model)}</a>`
            : html`<span>${displayByModel.get(model)}</span>`;
          if (method) attachModelPopover(label, method);
          const region = svgByKey.get(`${model}/region`);
          const cons = svgByKey.get(`${model}/conservation`);
          return html`<figure class="um-panel">
            <figcaption class="um-panel-name">${label}</figcaption>
            <div class="um-row">
              ${
                region
                  ? html`<div class="um-cell"><div class="um-cell-name">By region (Fig 4A)</div>${renderSvg(region)}</div>`
                  : null
              }
              ${
                cons
                  ? html`<div class="um-cell"><div class="um-cell-name">By conservation (Fig 4B)</div>${renderSvg(cons)}</div>`
                  : null
              }
            </div>
          </figure>`;
        })}
      </div>`
    : html`<div class="um-empty">No embedding UMAPs are materialized yet.</div>`,
);
```

<style>
:root { --observablehq-max-width: 1400px; }
main > h1, main > p { max-width: 900px; }

.um-models { display: flex; flex-direction: column; gap: 30px; margin-top: 0.5em; }
.um-panel { margin: 0; }
.um-panel-name { font-weight: 600; font-size: 0.95em; margin-bottom: 6px; color: #222; }
/* Region + conservation side by side, wrapping on narrow screens. */
.um-row { display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }
.um-cell { flex: 1 1 360px; max-width: 520px; }
.um-cell-name {
  color: #888; text-transform: uppercase; font-size: 0.72em;
  letter-spacing: 0.04em; margin-bottom: 2px;
}
.um-svg { width: 100%; }
.um-empty { color: #888; margin: 1em 0; }
.um-model-link { color: #1c6fb3; text-decoration: none; }
.um-model-link:hover { text-decoration: underline; }

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
</style>
