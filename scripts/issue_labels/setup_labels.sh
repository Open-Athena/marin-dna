#!/usr/bin/env bash
# Reach the final MarinDNA issue-label taxonomy (Type x Area + meta).
# See AGENTS.md > GitHub Communication > Issue labels for the scheme.
#
# Source of truth for the label set: every label is created-or-updated with
# `gh label create --force` (color + description), so this runs cleanly on a
# fresh repo AND is idempotent on re-run. Does NOT delete `enhancement` — that
# is teardown_enhancement.sh, run after the retroactive relabel moves issues
# off it.
set -euo pipefail

label() { gh label create "$1" --color "$2" --description "$3" --force; }

# --- Type ---
label research-question 5319e7 "Type: durable, revisitable question pursued across many experiments over time (a north-star)"
label experiment        e20f5b "Type: one preregistered, scoped run — hypothesis/goal fixed before running"
label eda               1d76db "Type: bounded exploratory analysis of data/results; no preregistered hypothesis"
label infrastructure    9a210c "Type: tooling, pipelines, CI, migrations, compute plumbing"
label bug               d73a4a "Type: something is broken"

# --- Area ---
label evals                       aaaaaa "Area: work on the eval apparatus — new eval dataset, scoring protocol, or metric"
label data                        9d1bf3 "Area: training-data construction — projection, labeling, filtering (eval sets go under evals)"
label modeling                    fbca04 "Area: core gLM recipe — architecture, objective, tokenization, loss weighting, context"
label baselines                   006b75 "Area: reference models — Evo2, GPN-Star, AlphaGenome, conservation, ChromBPNet"
label hyperparameter-optimization 0052cc "Area: hyperparameter/optimizer/schedule sweeps and model scaling (size/compute)"
label interpretation              0e8a16 "Area: model interpretation — UMAP, nucleotide-dependency maps, SAEs, TF-MoDISco"

# --- Meta / structural ---
label agent-generated ea6df9 "Created by an agent"
label marin           1f6feb "Change really belongs upstream in marin/levanter"
label epic            7f155a "Engineering decomposition only — a build split into parts (not for organizing research)"

echo "Labels created/updated. Run 'gh label list' to verify."
