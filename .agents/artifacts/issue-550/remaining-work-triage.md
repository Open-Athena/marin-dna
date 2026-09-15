# Remaining-work triage after experiment #550

Proposed dispositions following the user's request to triage experiment-specific versus mainline work.
No PR closure, merge, or scope change is performed by this record.
Experiment #550 is complete and its interpretation merged through #576.

| PR | Proposed disposition | Rationale | Context needed |
| --- | --- | --- | --- |
| [#552](https://github.com/Open-Athena/marin-dna/pull/552) | Keep experimental; recommend closing the PR without merging. | The producer hard-codes 255-base windows, 40 species, 10,240 tokens, the five regions, and chr18 validation. Reuse it on the next RAG experiment branch; promote a stable producer only after its contract settles. | Recipe and data-construction context matter; preserved below. |
| [#554](https://github.com/Open-Athena/marin-dna/pull/554) | Keep the RAG adapter experimental; recommend closing the PR without merging. | The consumer fixes the same geometry, human-last layout, and three development cohorts. Optional attention-mask support in the shared cached scorer is a separable reusable candidate; it does not require adopting the full adapter. | Scoring and embedding contracts matter; preserved below. |
| [#565](https://github.com/Open-Athena/marin-dna/pull/565) | Keep the registrations experimental; recommend closing the PR without merging. | Pins only the four #550 checkpoints and combined harness; also brings RAG-specific offline CI scaffolding. It depends on #554 and #559. It does not add dashboard/models.yaml entries. | Checkpoint pins and completed result receipts are already durable. |
| [#559](https://github.com/Open-Athena/marin-dna/pull/559) | Keep as a standalone mainline candidate; defer to another session. | Small Transformers export compatibility fix with isolated regression tests. Its utility is independent of the RAG recipe. | Low: issue #558 and tests reproduce the failure. |
| [#562](https://github.com/Open-Athena/marin-dna/pull/562) | Defer; separate the narrow Trainer fix from the fp32 workflow policy before deciding what to merge. | Reapplying explicit TF32 flags after Trainer construction fixes a general behavior. The remaining diff introduces an optional fp32 fallback and provenance policy that the completed A10G BF16 evaluation does not need. | Low to moderate: issue #561, regression tests, and numeric pilot evidence are recorded. |
| [#556](https://github.com/Open-Athena/marin-dna/pull/556) | Keep as standalone guidance; defer to another session. | Authorization persistence and execution-access preflight apply across tasks. The user explicitly identifies this as work that does not require the present session. | Low: the PR and runbook contain the behavior, limits, and checks. |
| [#567](https://github.com/Open-Athena/marin-dna/pull/567) | Keep as standalone guidance; defer to another session. | Five-line standing preference for EC2 A10G VEP, independent of the experiment. | Low: the PR body records the preference and validation. |

## Preserve for the next RAG experiment

The permanent experiment branch is `codex/issue-550-rag-five-regions`; the completed evaluation source is pinned at b435fa352a150835cc7d22eaaf3640bb3aec85d6.
The accepted interpretation is at 45a125eed9cd25dd11ce8449649a202a552af1ad and links the full result/provenance record in #550.
Keep the producer and consumer geometry consistent when adding non-human primates: both currently assume at most 40 species, 255 bases per species, and 10,240 model tokens.
Training uses fixed per-document species permutations including human and right-padding; VEP places human last, uses full left-padding, and fixes the human variant position.
VEP must retain the standard `compute_variant_score_bundle` shared-prefix cache, BF16 on EC2 A10G, and REF/ALT human-token embeddings; the completed run used measured batch size 8.
The combined adapter restores canonical row identities for all three development cohorts and writes the existing Snakemake S3 score, metric, and probe outputs.
Compare checkpoints within the same metric protocol: matched-data zero-shot pools within consequence subsets, while probes aggregate per chromosome; do not interpret their difference as a direct probe gain.
All requested outputs are preserved, paid evaluation workers were terminated, and no additional run is needed to preserve this experiment.

## Published PR snapshots

| PR | Head commit | Base branch at triage |
| --- | --- | --- |
| #552 | [0a5e3130](https://github.com/Open-Athena/marin-dna/tree/0a5e31300fd94f179e6478e376604c7b18b08194) | `main` |
| #554 | [ea37af3a](https://github.com/Open-Athena/marin-dna/tree/ea37af3a9cf6de75b158a4b6fff0ecf687920c90) | `main` |
| #565 | [0253beb8](https://github.com/Open-Athena/marin-dna/tree/0253beb83294f3c060df30fa6bf3a21be80c1870) | `codex/issue-553-combined-rag-evaluation` |
| #559 | [ddc8f9aa](https://github.com/Open-Athena/marin-dna/tree/ddc8f9aa85ebd20515e11aee2e432e437dc71741) | `main` |
| #562 | [92bc85b3](https://github.com/Open-Athena/marin-dna/tree/92bc85b30944a44c5959d4caf8a320eb2b7a9cb6) | `main` |
| #556 | [2299ad04](https://github.com/Open-Athena/marin-dna/tree/2299ad04ec6a2230b31ecf942b25859616a1c522) | `main` |
| #567 | [3886e2c5](https://github.com/Open-Athena/marin-dna/tree/3886e2c577847bf303e83e285aea4b7ce407432d) | `main` |

The #552 producer source, rules, and tests are byte-identical in the permanent experiment branch; its README includes further experiment context.
The #554 shared scorer and RAG module are byte-identical there; routing/configuration/rule differences include integration with the other fixes and registrations.
The #565 registration, offline CI helper, workflow test, and CI change are present in the permanent experiment branch.
The #559 compatibility implementation and tests are present there; #562 is integrated with the RAG workflow.
Preserve the published snapshots when retiring experimental PRs, since the standalone and integrated diffs are not identical.
