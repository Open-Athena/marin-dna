// Shared filter / picker widgets for the leaderboard + protocol pages.
//
// All widgets return a DOM element with a `.value` getter, and dispatch
// `input` events on state change — so they're drop-in `view(...)` targets.

import {html} from "npm:htl";

// Short family labels used in pills/toggles. Distinct from the longer
// `model-cards.js` labels (`"bolinas gLM"` etc.) — those go on cards.
export const FAMILY_LABEL = {
  bolinas: "bolinas",
  conservation: "conservation",
  alphagenome: "AlphaGenome",
  gpn_star: "GPN-Star",
};

// Protocol options per family. Mirror of `PROTOCOLS` in
// src/bolinas/pipelines/evals/leaderboard.py — keep in sync when adding
// a protocol. Defaults match `DEFAULT_PROTOCOL`.
export const PROTOCOL_OPTIONS = {
  bolinas: ["LLR", "JSD"],
  gpn_star: ["cLLR", "LLR"],
};
export const PROTOCOL_DEFAULTS = {
  bolinas: "LLR",
  conservation: "score",
  alphagenome: "L2",
  gpn_star: "cLLR",
};

// Pill-row toggle for selecting which families to include. Value is the
// array of currently-selected family slugs. Renders `all · none` quick
// actions next to the pills.
export function FamilyToggle(allFamilies, initial = allFamilies) {
  const state = new Set(initial);
  const node = html`<div class="lb-family-toggle"></div>`;
  let _value = [...state];
  Object.defineProperty(node, "value", {get: () => _value});
  function fire() {
    _value = [...state];
    node.dispatchEvent(new Event("input", {bubbles: true}));
  }
  function setAll(next) {
    state.clear();
    next.forEach((f) => state.add(f));
    render();
    fire();
  }
  function render() {
    node.replaceChildren(html`<div class="lb-family-toggle-row">
      ${allFamilies.map(
        (f) => html`<button
          type="button"
          class=${`lb-pill family-${f}${state.has(f) ? " active" : ""}`}
          aria-pressed=${state.has(f) ? "true" : "false"}
          onclick=${() => {
            state.has(f) ? state.delete(f) : state.add(f);
            render();
            fire();
          }}
        >${FAMILY_LABEL[f] ?? f}</button>`,
      )}
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

// Per-family protocol selector. Only families with ≥2 protocols render
// a toggle row (single-protocol families fall back to PROTOCOL_DEFAULTS).
// Value is `{family: protocol}` for every family in `options`.
export function ProtocolPicker(options, defaults) {
  const state = {...defaults};
  const node = html`<div class="lb-protocol-picker"></div>`;
  Object.defineProperty(node, "value", {get: () => ({...state})});
  function fire() {
    node.dispatchEvent(new Event("input", {bubbles: true}));
  }
  function render() {
    const groups = [];
    for (const [fam, protos] of Object.entries(options)) {
      if (protos.length < 2) continue;
      groups.push(html`<span class="lb-protocol-group">
        <span class=${`lb-family-tag family-${fam}`}>${FAMILY_LABEL[fam] ?? fam}</span>
        <span class="lb-protocol-segmented">${protos.map(p => html`<button
          type="button"
          class=${`lb-protocol-btn${state[fam] === p ? " active" : ""}`}
          onclick=${() => { state[fam] = p; render(); fire(); }}
        >${p}</button>`)}</span>
      </span>`);
    }
    node.replaceChildren(html`<div class="lb-protocol-picker-row">${groups}</div>`);
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

// "A → B" direction picker. Each option is an ordered pair of protocols;
// the protocol-comparison heatmap renders `B PA − A PA` so cells read as
// "improvement over A". Value is `{from, to}`.
export function DirectionPicker(protos, initialFrom, initialTo) {
  const pairs = [];
  for (const a of protos) for (const b of protos) if (a !== b) pairs.push([a, b]);
  let from = initialFrom, to = initialTo;
  const node = html`<span class="lb-protocol-segmented"></span>`;
  Object.defineProperty(node, "value", {get: () => ({from, to})});
  function fire() { node.dispatchEvent(new Event("input", {bubbles: true})); }
  function render() {
    node.replaceChildren(...pairs.map(([a, b]) => html`<button
      type="button"
      class=${`lb-protocol-btn${from === a && to === b ? " active" : ""}`}
      onclick=${() => { from = a; to = b; render(); fire(); }}
    >${a} → ${b}</button>`));
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
