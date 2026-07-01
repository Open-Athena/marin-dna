#!/usr/bin/env bash
# Reach the final MarinDNA issue-label taxonomy (Type x Area + meta).
# See AGENTS.md > GitHub Communication > Issue labels for the scheme.
# Idempotent: safe to re-run. Does NOT delete `enhancement` (that happens in
# teardown_enhancement.sh, after the retroactive relabel moves issues off it).
set -euo pipefail

# --- New Area labels: create-or-update (color + description) ---------------
gh label create data \
  --color 9d1bf3 \
  --description "Area: training-data construction — projection, labeling, filtering (eval sets go under evals)" \
  --force
gh label create modeling \
  --color fbca04 \
  --description "Area: core gLM recipe — architecture, objective, tokenization, loss weighting, context" \
  --force
gh label create baselines \
  --color 006b75 \
  --description "Area: competitor/reference models — Evo2, GPN-Star, AlphaGenome, conservation, ChromBPNet" \
  --force
gh label create hyperparameter-optimization \
  --color 0052cc \
  --description "Area: hyperparameter/optimizer/schedule sweeps, incl. scaling (size/compute) sweeps" \
  --force

# --- Existing labels: add/refresh one-line descriptions (color preserved) ---
gh label edit research-question \
  --description "Type: durable, revisitable question pursued across many experiments over time (a north-star)"
gh label edit experiment \
  --description "Type: one preregistered, scoped run — hypothesis/goal fixed before running"
gh label edit eda \
  --description "Type: bounded exploratory analysis of data/results; no preregistered hypothesis"
gh label edit infrastructure \
  --description "Type: tooling, pipelines, CI, migrations, compute plumbing"
gh label edit evals \
  --description "Area: work on the eval apparatus — new eval dataset, scoring protocol, or metric"
gh label edit interpretation \
  --description "Area: model interpretation — UMAP, nucleotide-dependency maps, SAEs, TF-MoDISco"
gh label edit epic \
  --description "Engineering decomposition only — a build split into parts (not for organizing research)"
gh label edit agent-generated \
  --description "Created by an agent"

echo "Labels created/updated. Run 'gh label list' to verify."
