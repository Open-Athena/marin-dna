---
title: Models
toc: false
wide: true
---

# Models

Every entry on the leaderboard, with its family, training metadata, and links out to wandb / source / HF / GCS / tracking issues. Links from the heatmap on the [Mendelian page](./leaderboards/mendelian) deep-link to a method's anchor here.

```js
const methods = await FileAttachment("data/models.json").json();
import {modelCards} from "./components/model-cards.js";
import {FAMILY_LABEL, FamilyProtocolToggle} from "./components/controls.js";
```

```js
// Family filter rendered as colored pills, matching the leaderboard's family
// selector. FamilyProtocolToggle with no protocol options renders plain family
// pills (no protocol chips); it labels via FAMILY_LABEL and colors via the
// global palette. Order families like the leaderboard. The widget value is
// {families, protocols}; only `.families` is used here.
const familyOrder = Object.keys(FAMILY_LABEL);
const families = [...new Set(methods.map(m => m.family))].sort(
  (a, b) => familyOrder.indexOf(a) - familyOrder.indexOf(b),
);
const familyChoice = view(FamilyProtocolToggle(families, {}, {}));
const search = view(
  Inputs.text({label: "Search", placeholder: "name, description, training data, …"}),
);
const dataset = view(
  Inputs.select(["all", "mendelian_traits", "complex_traits"], {
    label: "Evaluated on",
    value: "all",
  }),
);
```

```js
function matches(m) {
  if (!familyChoice.families.includes(m.family)) return false;
  if (dataset !== "all" && !m.datasets.includes(dataset)) return false;
  if (search) {
    const q = search.toLowerCase();
    const haystack = [
      m.id, m.display, m.description,
      m.training?.data, m.training?.objective,
      m.family,
    ].filter(Boolean).join(" ").toLowerCase();
    if (!haystack.includes(q)) return false;
  }
  return true;
}
const filtered = methods.filter(matches);
```

<small>${filtered.length} of ${methods.length} models shown.</small>

<style>
/* Lift OF's 640px prose cap so the cards grid can use the full page width. */
main > p, main > h1, main > h2, main > h3, main > small { max-width: none; }

.method-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 14px;
  margin: 1em 0 2em;
}
.method-card {
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 12px 14px;
  background: #fff;
  font-size: 0.9em;
  scroll-margin-top: 1em;
}
.method-card:target { box-shadow: 0 0 0 3px #c7e0ff; border-color: #4a8edc; }
.method-card-header { display: flex; flex-direction: column; gap: 2px; margin-bottom: 6px; }
.method-card-header h3 { margin: 0; font-size: 1em; }
.method-card-family {
  display: inline-block;
  font-size: 0.74em;
  font-weight: 500;
  padding: 1px 9px;
  border-radius: 9999px;
  color: #fff;
  width: fit-content;
}
.method-card-step { color: #888; font-family: var(--monospace); }
.method-card-desc { color: #444; margin: 4px 0 8px; }
.method-card-row {
  display: grid; grid-template-columns: 90px 1fr;
  align-items: baseline;
  gap: 8px; margin: 3px 0; font-size: 0.88em;
}
.method-card-row .label { color: #888; text-transform: uppercase; font-size: 0.78em; letter-spacing: 0.04em; }
.method-card-row code { font-size: 0.95em; }
.dataset-tag {
  display: inline-block;
  background: #f0f0f0;
  border-radius: 3px;
  padding: 1px 6px;
  margin-right: 4px;
  font-size: 0.88em;
  font-family: var(--monospace);
}
.muted { color: #aaa; }
</style>

```js
display(modelCards(filtered));
```
