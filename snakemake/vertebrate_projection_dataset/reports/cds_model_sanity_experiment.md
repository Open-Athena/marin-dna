# CDS mammals-only vs. combined-vertebrate sanity experiment

Status: **preregistered; not launched**. A full projection/QC pass and explicit
user approval for paid training are required first.

## Question and directional expectation

Does adding family-deduplicated non-mammalian MultiZ projections to otherwise
identical human-plus-Zoonomia CDS training data improve coding-variant effect
prediction?

The preregistered directional expectation is that the combined-vertebrate arm
outperforms the mammals-only arm because it observes deeper evolutionary
constraint. A null or reversed result triggers a projection, duplication,
species-balance, and exposure audit; it does not authorize tuning the analysis
after seeing results.

## Matched arms

1. `mammals_only`: human reference plus Zoonomia mammalian projections.
2. `combined_vertebrates`: the identical rows plus selected non-mammalian
   MultiZ projections.

Both arms must pin the same:

- producing pipeline commit and species manifests;
- CDS anchor definition and chromosome-18 split policy;
- 16,384-row, original-orientation validation cap and seed;
- architecture, tokenizer, optimizer/schedule, batch construction, training
  token budget, initialization policy, and evaluation cadence; and
- coding-variant VEP harness revision and evaluation inputs.

The only intended treatment difference is the presence of non-mammalian
training sequences. Record realized rows/tokens and per-species exposure for
both arms so that accidental compute or sampling differences are visible.

## Required execution record

Fill this before launch:

| Field | Mammals only | Combined vertebrates |
|---|---|---|
| Pipeline commit | pending | pending |
| HF dataset + revision | pending | pending |
| Species manifest commit | pending | pending |
| Train rows/tokens | pending | pending |
| Validation rows/tokens | pending | pending |
| Model config | pending | pending |
| Random seed(s) | pending | pending |
| W&B run (`dna-exp417` in name) | pending | pending |
| VEP harness commit | pending | pending |

## Required results

Report overall and consequence-level coding-variant metrics for both arms, with
uncertainty where the harness supports it. Include paired deltas, training
curves, realized compute, and failures. Do not report only the favorable metric.

| Metric / consequence | Mammals only | Combined | Delta | Uncertainty |
|---|---:|---:|---:|---:|
| Overall | pending | pending | pending | pending |
| Consequence-level rows | pending | pending | pending | pending |

## Audit if the direction is null or reversed

- confirm identical human/CDS anchors and chromosome-18 membership;
- check projection rejection rates, bounds, strand, source case, and duplicate
  `(anchor, species)` rows by backend;
- compare per-species/per-clade exposure and RC augmentation;
- inspect CDS recovery breadth and the ZRS positive control;
- verify the two runs used equal training-token budgets and evaluation inputs;
  and
- document the audit before proposing a follow-up experiment.
