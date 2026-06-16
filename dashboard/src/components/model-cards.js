// Models index — one card per registered model.
//
// Each card surfaces every field from models.yaml that would otherwise be
// buried in the rendered tooltip / table cell: training metadata, all
// outgoing links (wandb, issue, source, paper, HF, GCS), the datasets the
// method is evaluated on. Card grid is responsive (CSS grid auto-fill).

import {html} from "npm:htl";
// Family labels are shared with the leaderboard pills — single source of truth.
import {FAMILY_LABEL} from "./controls.js";

// Link to a model's full card on the Models page. Relative (`../models`) so it
// resolves under the production `/marin-dna/` base path AND locally at `/`.
// Both callers — the heatmap row anchor and the hover popover below — render
// only on one-level-deep pages (/leaderboards/*, /protocols/*), so `../`
// reaches the root-level Models page.
export const modelHref = (id) => `../models#${encodeURIComponent(id)}`;

function paramsLabel(params) {
  if (params == null) return null;
  if (params >= 1e9) return `${(params / 1e9).toFixed(1).replace(/\.0$/, "")}B`;
  if (params >= 1e6) return `${(params / 1e6).toFixed(0)}M`;
  return params.toString();
}

function link(href, label) {
  if (!href) return null;
  return html`<a href=${href} target="_blank" rel="noopener">${label}</a>`;
}

function joinLinks(parts) {
  const present = parts.filter((p) => p != null);
  if (present.length === 0) return html`<span class="muted">—</span>`;
  const out = [];
  for (let i = 0; i < present.length; i++) {
    if (i > 0) out.push(" · ");
    out.push(present[i]);
  }
  return html`${out}`;
}

function modelCard(m) {
  const training = m.training ?? {};
  // Always render Size + Context rows; "—" for entries that don't have a
  // model (conservation tracks) or where the value isn't known.
  const sizeLabel = paramsLabel(training.params) ?? "—";
  const contextLabel = training.window_size ? `${training.window_size} bp` : "—";
  // Training data / objective only when populated (marin_dna entries).
  const trainingExtras = [training.data, training.objective].filter(Boolean);
  const checkpointLinks = [];
  if (m.checkpoint?.hf) {
    checkpointLinks.push(
      link(
        `https://huggingface.co/${m.checkpoint.hf}`,
        html`HF <code>${m.checkpoint.hf}</code>`,
      ),
    );
  }
  if (m.checkpoint?.gcs) {
    checkpointLinks.push(html`<code title="GCS checkpoint">${m.checkpoint.gcs}</code>`);
  }
  const externalLinks = [
    link(m.wandb, "wandb"),
    link(m.issue, m.experiment ? `issue #${m.experiment}` : "issue"),
    link(m.source_code, "source"),
    link(m.paper, "paper"),
  ];
  return html`<article class="method-card" id=${m.id}>
    <header class="method-card-header">
      <span class=${`method-card-family family-${m.family}`}>${FAMILY_LABEL[m.family] ?? m.family}</span>
      <h3><code>${m.display}</code></h3>
      ${m.id !== m.display ? html`<small class="method-card-step">${m.id}</small>` : ""}
    </header>
    <p class="method-card-desc">${m.description}</p>
    <div class="method-card-row"><span class="label">Size</span><span>${sizeLabel}</span></div>
    <div class="method-card-row"><span class="label">Context</span><span>${contextLabel}</span></div>
    ${
      trainingExtras.length
        ? html`<div class="method-card-row"><span class="label">Training</span><span>${trainingExtras.join(" · ")}</span></div>`
        : ""
    }
    ${
      m.msa
        ? html`<div class="method-card-row"><span class="label">MSA</span><span><code>${m.msa}</code></span></div>`
        : ""
    }
    <div class="method-card-row">
      <span class="label">Datasets</span>
      <span>${m.datasets.map((d) => html`<span class="dataset-tag">${d}</span>`)}</span>
    </div>
    ${
      checkpointLinks.length
        ? html`<div class="method-card-row"><span class="label">Checkpoint</span><span>${joinLinks(checkpointLinks)}</span></div>`
        : ""
    }
    <div class="method-card-row">
      <span class="label">Links</span>
      <span>${joinLinks(externalLinks)}</span>
    </div>
  </article>`;
}

export function modelCards(methods) {
  return html`<div class="method-card-grid">${methods.map(modelCard)}</div>`;
}

// ---- Compact popover for hover/click on heatmap method names ---------------

function popoverLinks(m) {
  const items = [
    link(m.wandb, "wandb"),
    link(m.issue, m.experiment ? `issue #${m.experiment}` : "issue"),
    link(m.source_code, "source"),
    link(m.paper, "paper"),
    m.checkpoint?.hf
      ? link(`https://huggingface.co/${m.checkpoint.hf}`, "HF")
      : null,
  ].filter((p) => p != null);
  return items;
}

function specRow(key, value) {
  return html`<div class="lb-pop-row"><span class="lb-pop-key">${key}</span><span class="lb-pop-val">${value}</span></div>`;
}

export function modelPopoverContent(m) {
  const training = m.training ?? {};
  const sizeLabel = paramsLabel(training.params) ?? "—";
  const contextLabel = training.window_size ? `${training.window_size} bp` : "—";
  const rows = [
    specRow("Size", sizeLabel),
    specRow("Context", contextLabel),
    training.data ? specRow("Data", training.data) : null,
    m.msa ? specRow("MSA", html`<code>${m.msa}</code>`) : null,
  ].filter(Boolean);
  return html`<div class="lb-pop-inner">
    <header class="lb-pop-header">
      <span class=${`lb-pop-family family-${m.family}`}>${FAMILY_LABEL[m.family] ?? m.family}</span>
      <code class="lb-pop-display">${m.display}</code>
    </header>
    ${m.description ? html`<div class="lb-pop-desc">${m.description}</div>` : ""}
    <div class="lb-pop-specs">${rows}</div>
    ${(() => {
      const links = popoverLinks(m);
      if (links.length === 0) return "";
      const out = [];
      links.forEach((l, i) => {
        if (i > 0) out.push(" · ");
        out.push(l);
      });
      return html`<div class="lb-pop-links">${out}</div>`;
    })()}
    <a class="lb-pop-more" href=${modelHref(m.id)}>full card →</a>
  </div>`;
}

// Singleton popover element. Lazily created on first show; reused across
// cells. Lives on `document.body` so the heatmap can re-render without
// destroying our popover state.
let _popEl = null;
let _hideTimer = null;
let _lastAnchor = null;

function ensurePopover() {
  if (_popEl) return _popEl;
  _popEl = document.createElement("div");
  _popEl.className = "lb-method-popover";
  _popEl.style.cssText =
    "position: fixed; display: none; z-index: 1000; max-width: 360px;";
  _popEl.addEventListener("mouseenter", () => {
    clearTimeout(_hideTimer);
  });
  _popEl.addEventListener("mouseleave", () => {
    scheduleHidePopover();
  });
  document.body.appendChild(_popEl);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") hidePopoverNow();
  });
  return _popEl;
}

function hidePopoverNow() {
  if (_popEl) _popEl.style.display = "none";
  _lastAnchor = null;
}

function scheduleHidePopover() {
  clearTimeout(_hideTimer);
  _hideTimer = setTimeout(hidePopoverNow, 220);
}

function positionPopover(anchorEl) {
  const pop = _popEl;
  // Reset to compute natural size before measuring.
  pop.style.left = "-9999px";
  pop.style.top = "0";
  pop.style.display = "block";
  const a = anchorEl.getBoundingClientRect();
  const p = pop.getBoundingClientRect();
  const margin = 8;
  // Default: anchored to the right of the method-name cell.
  let left = a.right + margin;
  let top = a.top - 4;
  // If it would overflow on the right, mirror to the left.
  if (left + p.width > window.innerWidth - margin) {
    left = a.left - p.width - margin;
  }
  // Clamp vertical to viewport.
  if (top + p.height > window.innerHeight - margin) {
    top = window.innerHeight - p.height - margin;
  }
  if (top < margin) top = margin;
  pop.style.left = `${Math.max(margin, left)}px`;
  pop.style.top = `${top}px`;
}

export function attachModelPopover(anchorEl, method) {
  let pendingShow = null;
  anchorEl.addEventListener("mouseenter", () => {
    clearTimeout(_hideTimer);
    clearTimeout(pendingShow);
    pendingShow = setTimeout(() => {
      const pop = ensurePopover();
      if (_lastAnchor !== anchorEl) {
        pop.replaceChildren(modelPopoverContent(method));
        _lastAnchor = anchorEl;
      }
      positionPopover(anchorEl);
    }, 120);
  });
  anchorEl.addEventListener("mouseleave", () => {
    clearTimeout(pendingShow);
    scheduleHidePopover();
  });
}
