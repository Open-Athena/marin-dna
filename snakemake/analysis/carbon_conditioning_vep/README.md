# Carbon species conditioning on Mendelian VEP

This isolated Snakemake project implements [issue #486](https://github.com/Open-Athena/marin-dna/issues/486).
It measures whether frozen Carbon-3B changes human promoter-variant rankings when its inference prompt is untagged or correctly tagged as mammalian.

The analysis is exploratory.
It reports every condition and paired comparison without a winner gate or testing hierarchy.

## Fixed contract

- Model: `HuggingFaceBio/Carbon-3B` at revision `95c3c68fc77fdf70b1582031bacf9d7753f72cf2`.
- Data: the 2,050-row `tss_proximal` promoter subset from the `marin-dna/evals_mendelian_traits` train split at revision `4aed58e50c5dea0b878a665007af2ef9e5108e9f`.
- Reference: Ensembl release 115 GRCh38 soft-masked primary assembly with Ensembl sequence names.
- Coordinates: source `pos` is converted from 1-based to 0-based at window extraction; materialized windows use 0-based half-open bounds.
- Context: 8,192 bp centered on each SNV.
- Carbon input: each strand is truncated to 8,190 bp at the deterministic 6-mer boundary before tokenization.
- Score: bf16 masked mean causal token log likelihood on REF and ALT for FWD and reverse-complement strands.
- Inference batch: one variant, or four allele-strand prompts, validated on both an NVIDIA A10G and a Lambda GH200.
- Derived score: `llr = ((LL_alt_fwd - LL_ref_fwd) + (LL_alt_rc - LL_ref_rc)) / 2`, then `score = -llr`.
- Metric: promoter AUPRC for the untagged and correct conditions.
- Uncertainty: 1,000 seeded paired match-group bootstrap draws.

The project does not import `marin_dna_evals`, modify `evals_v2`, write into the `evals_v2` artifact prefix, or register Carbon in the dashboard.

## Prompt preflight gate

Carbon's pinned model card documents `<{species}><dna>`.
Its pinned pretraining-corpus card documents `<species>{species}<dna>`.

`prompt_preflight` compares the two grammars on the same deterministic slice of fungi, protozoa, and invertebrate rows from `GenerTeam/sequence-recovery` revision `3ac1de1be0e4c55dd180c719c1f3805a2cdb9be9`.
The selected grammar must have a positive correct-tag minus untagged continuation-accuracy delta and must exceed the other grammar.
The workflow stops if neither grammar has a positive delta or if they tie within the configured tolerance.

The Mendelian window rule depends on the successful preflight JSON.
No Mendelian labels are loaded before a grammar is selected.

## Validation and exclusions

The Mendelian loader downloads exactly `train.parquet` from the pinned dataset revision before parsing it.
It rejects every other split or filename so a repository-backed dataset builder cannot discover or materialize held-out labels.
The dataset loader asserts the exact row, positive, group, chromosome, SNV, unique-key, and 1:9 group contracts of the pinned development split.
It then selects `tss_proximal` and asserts 2,050 rows, 205 positives, 205 match groups, and ten rows per group.
The reference loader asserts exact GRCh38 lengths for every development chromosome.

Window extraction rejects out-of-bound coordinates, missing contigs, noncanonical sequence, length mismatches, and reference-allele mismatches.
Any row failure excludes its complete `match_group`.
Every failed row and group is written to `results/promoter_pilot/windows/exclusions.parquet`.

The smoke sample is selected by a stable hash of `variant_id`.
Labels, subsets, match groups, and consequence columns are removed before the smoke scorer sees the rows.

## Outputs

```text
results/
├── preflight/
│   ├── prompt_grammar.json
│   └── prompt_grammar.parquet
└── promoter_pilot/
    ├── windows/
    │   ├── mendelian.parquet
    │   └── exclusions.parquet
    ├── smoke/
    │   ├── windows.parquet
    │   ├── scores/{untagged,correct}.parquet
    │   └── runtime/{untagged,correct}.json
    ├── scores/Carbon-3B/{untagged,correct}.parquet
    ├── metrics/Carbon-3B/{untagged,correct}.parquet
    ├── paired/Carbon-3B/correct_minus_untagged.parquet
    ├── runtime/Carbon-3B/{untagged,correct}.json
    └── summary.md
```

Each full score parquet contains variant keys, labels, subsets, match groups, four per-allele/per-strand log likelihoods, per-strand LLRs, the FWD/RC average LLR, and `score = -llr`.

## Local validation

Run from this project root.

```bash
uv sync --locked --group dev
uv run --locked pytest
uv run --locked snakemake -n --profile workflow/profiles/default
```

The dry-run performs no remote inference.
The default storage profile maps result paths to `s3://oa-bolinas/snakemake/analysis/carbon_conditioning_vep/`.

## Remote execution

Paid remote compute requires explicit approval.
The prompt preflight and corrected label-blind four-condition smoke passed on an AWS `g5.2xlarge` A10G.
The smoke's maximum allocated GPU memory was 11.26 GiB.

A subsequent Lambda GH200 benchmark scored the same eight label-blind rows in 1.55 seconds with batch size one.
Batch size eight was slower per row and raised peak allocation to 44.76 GiB, so the pilot keeps batch size one.
At the 2026-08-20 catalog price of $2.29 per hour, the two-condition promoter pilot is expected to use about 13–15 GPU-minutes and cost about $0.50–$0.60.

The GH200 task has a 20-minute command timeout and three-minute autodown, giving a conservative approval ceiling of $1.00 after ordinary sync overhead.
Check the current [Lambda instance price](https://lambda.ai/instances) immediately before launch.

Stage the already-validated development-only artifacts on the coordinator before launch.
This reads the canonical S3 artifacts locally, filters the window parquet to the 2,050 promoter rows with provenance checks, and does not forward AWS credentials to Lambda.

```bash
bash snakemake/analysis/carbon_conditioning_vep/sky/stage-gh200-pilot.sh
```

Launch from the repository root only after explicit approval of the current price and cap.

```bash
sky launch snakemake/analysis/carbon_conditioning_vep/sky/run-gh200-pilot.yaml \
  -c carbon-conditioning-vep-gh200-pilot
```

Retrieve the results immediately after the job succeeds and before the three-minute autodown.

```bash
rsync -a \
  carbon-conditioning-vep-gh200-pilot:sky_workdir/snakemake/analysis/carbon_conditioning_vep/results/promoter_pilot/ \
  snakemake/analysis/carbon_conditioning_vep/results/promoter_pilot/
sky down carbon-conditioning-vep-gh200-pilot -y
```
