// Shared filter / picker widgets for the leaderboard + protocol pages.
//
// All widgets return a DOM element with a `.value` getter, and dispatch
// `input` events on state change — so they're drop-in `view(...)` targets.

import {html} from "npm:htl";

// Family display labels — single source of truth. Shared across the
// leaderboard pills/toggles, the Models page cards, and the heatmap hover
// popover (model-cards.js imports this).
export const FAMILY_LABEL = {
  marin_dna: "MarinDNA",
  conservation: "Conservation",
  alphagenome: "AlphaGenome",
  gpn_star: "GPN-Star",
  evo2: "Evo 2",
};

// Top-level supervision mode for the Mendelian leaderboard — the score-world a row was
// computed in. "Unsupervised" = zero-shot likelihood metrics; "Supervised" = the trained
// frozen-embedding linear probe (#347/#348). The two use different, non-level-comparable
// metrics (matched-pair AUPRC vs per-chromosome-weighted AUPRC), so the page renders one at
// a time — never mixed in a single ranked view. Keys match the parquet `supervision` column.
export const SUPERVISION_LABEL = {
  unsupervised: "Unsupervised",
  supervised: "Supervised",
};

// Leaderboard-visible protocol options per family. This is the subset of
// `PROTOCOLS` (in snakemake/analysis/evals_v2/src/marin_dna_evals/leaderboard.py) that the
// leaderboards' `FamilyProtocolToggle` exposes — additional protocols can live in
// `PROTOCOLS` (e.g. `LLR-FWD`, `JSD-FWD` for the AVG-vs-FWD exploration on
// the Protocols pages) without showing up as leaderboard toggles. Defaults
// match `DEFAULT_PROTOCOL`.
export const PROTOCOL_OPTIONS = {
  marin_dna: ["LLR", "JSD"],
  gpn_star: ["cLLR", "LLR"],
  evo2: ["LLR", "JSD"],
};
export const PROTOCOL_DEFAULTS = {
  marin_dna: "LLR",
  conservation: "score",
  alphagenome: "L2",
  gpn_star: "cLLR",
  evo2: "LLR",
};

// Protocol *display* labels — render layer only. Internal protocol keys
// (the parquet `protocol` column / `PROTOCOLS` in
// snakemake/analysis/evals_v2/src/marin_dna_evals/leaderboard.py) are unchanged; this maps a
// key to how it reads in the UI. JSD is surfaced as "NucDep" (nucleotide
// dependency). Unlisted keys fall back to themselves.
export const PROTOCOL_LABEL = {
  JSD: "NucDep",
  "JSD-FWD": "NucDep-FWD",
};
export const protocolLabel = (p) => PROTOCOL_LABEL[p] ?? p;

// Optional hover tooltip per protocol. Surfaces the underlying quantity for
// renamed protocols on the leaderboard pills, which — unlike the Protocols
// pages — carry no inline definition. Unlisted keys get no `title` (htl omits
// the attribute when the value is null).
export const PROTOCOL_TITLE = {
  JSD: "Jensen-Shannon divergence (JSD)",
  "JSD-FWD": "Jensen-Shannon divergence, forward strand only (JSD-FWD)",
};
export const protocolTitle = (p) => PROTOCOL_TITLE[p] ?? null;

// Combined family selector + per-family protocol toggle. Each family renders
// one compound pill; selecting a family reveals its protocol chips inset inside
// the same colored pill (only for families with ≥2 protocols — single-protocol
// families stay plain pills and fall back to `defaults`). Protocol choices
// persist across deselect/reselect. Renders `all · none` quick actions after
// the pills. Value is `{families: string[], protocols: {family: protocol}}`.
export function FamilyProtocolToggle(allFamilies, options, defaults, initial = allFamilies) {
  const selected = new Set(initial);
  const protocols = {...defaults};
  const node = html`<div class="lb-family-toggle"></div>`;
  const compute = () => ({families: [...selected], protocols: {...protocols}});
  let _value = compute();
  Object.defineProperty(node, "value", {get: () => _value});
  function fire() {
    _value = compute();
    node.dispatchEvent(new Event("input", {bubbles: true}));
  }
  function setAll(next) {
    selected.clear();
    next.forEach((f) => selected.add(f));
    render();
    fire();
  }
  function render() {
    node.replaceChildren(html`<div class="lb-family-toggle-row">
      ${allFamilies.map((f) => {
        const protos = options[f] ?? [];
        const active = selected.has(f);
        return html`<span class=${`lb-cpill family-${f}${active ? " active" : ""}`}>
          <button
            type="button"
            class="lb-cpill-name"
            aria-pressed=${active ? "true" : "false"}
            onclick=${() => {
              selected.has(f) ? selected.delete(f) : selected.add(f);
              render();
              fire();
            }}
          >${FAMILY_LABEL[f] ?? f}</button>
          ${active && protos.length >= 2
            ? html`<span class="lb-cpill-protos">${protos.map((p) => html`<button
                type="button"
                class=${`lb-cpill-proto${protocols[f] === p ? " active" : ""}`}
                aria-pressed=${protocols[f] === p ? "true" : "false"}
                title=${protocolTitle(p)}
                onclick=${() => { protocols[f] = p; render(); fire(); }}
              >${protocolLabel(p)}</button>`)}</span>`
            : null}
        </span>`;
      })}
      <span class="lb-toggle-actions">
        <button type="button" class="lb-link" onclick=${() => setAll(allFamilies)}>all</button>
        <span aria-hidden="true">·</span>
        <button type="button" class="lb-link" onclick=${() => setAll([])}>none</button>
      </span>
    </div>`);
  }
  render();
  return node;
}

// Standalone on/off pill (e.g. "Best per family"). Boolean value.
export function PillToggle(label, initial = false) {
  let value = initial;
  const node = html`<button type="button" class=${`lb-pill-toggle${value ? " active" : ""}`}>${label}</button>`;
  Object.defineProperty(node, "value", {get: () => value});
  node.addEventListener("click", () => {
    value = !value;
    node.className = `lb-pill-toggle${value ? " active" : ""}`;
    node.dispatchEvent(new Event("input", {bubbles: true}));
  });
  return node;
}

// Single-choice segmented pill row. Returns one of the given `options`.
// Optional `formatter(o)` renders the button label.
export function PillSelect(options, initial, formatter = (o) => o) {
  let value = initial;
  const node = html`<span class="lb-protocol-segmented"></span>`;
  Object.defineProperty(node, "value", {get: () => value});
  function fire() { node.dispatchEvent(new Event("input", {bubbles: true})); }
  function render() {
    node.replaceChildren(...options.map(o => html`<button
      type="button"
      class=${`lb-protocol-btn${value === o ? " active" : ""}`}
      onclick=${() => { if (value !== o) { value = o; render(); fire(); } }}
    >${formatter(o)}</button>`));
  }
  render();
  return node;
}

// "A → B" comparison picker. Each `pairs` entry is an explicit `[from, to]`
// pair; the protocol-comparison heatmap renders `to AUPRC − from AUPRC` so
// cells read as "improvement over from". Value is `{from, to}`.
export function ComparisonPicker(pairs, initialIdx = 0) {
  let [from, to] = pairs[initialIdx];
  const node = html`<span class="lb-protocol-segmented"></span>`;
  Object.defineProperty(node, "value", {get: () => ({from, to})});
  function fire() { node.dispatchEvent(new Event("input", {bubbles: true})); }
  function render() {
    node.replaceChildren(...pairs.map(([a, b]) => html`<button
      type="button"
      class=${`lb-protocol-btn${from === a && to === b ? " active" : ""}`}
      onclick=${() => {
        if (from === a && to === b) return;
        from = a; to = b;
        render();
        fire();
      }}
    >${protocolLabel(a)} → ${protocolLabel(b)}</button>`));
  }
  render();
  return node;
}

// Wrap an inner input-style element in a labeled row. Forwards the inner
// element's `value` getter (input events bubble) so the wrapper is a
// drop-in `view()` target.
export function labeledRow(label, inner, hint) {
  const wrapper = html`<span class="lb-control-row">
    <span class="lb-control-label">${label}</span>
    ${inner}
    ${hint ? html`<span class="lb-control-hint">${hint}</span>` : null}
  </span>`;
  Object.defineProperty(wrapper, "value", {get: () => inner.value});
  return wrapper;
}
