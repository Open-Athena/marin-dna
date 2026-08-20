# Carbon species conditioning on Mendelian VEP

This isolated Snakemake project implements [issue #486](https://github.com/Open-Athena/marin-dna/issues/486).
It measures whether frozen Carbon-3B changes human development-set variant rankings when its inference prompt is untagged, correctly tagged as mammalian, or incorrectly tagged as fungal.

The analysis is exploratory.
It reports every condition and paired comparison without a winner gate or testing hierarchy.

## Fixed contract

- Model: `HuggingFaceBio/Carbon-3B` at revision `95c3c68fc77fdf70b1582031bacf9d7753f72cf2`.
- Data: the 16,140-row development-only `marin-dna/evals_mendelian_traits` train split at revision `4aed58e50c5dea0b878a665007af2ef9e5108e9f`.
- Analysis scopes: the retained `config/config.yaml` promoter pilot contains 2,050 `tss_proximal` rows, while `config/full_development.yaml` contains all 16,140 development rows.
- Reference: Ensembl release 115 GRCh38 soft-masked primary assembly with Ensembl sequence names.
- Coordinates: source `pos` is converted from 1-based to 0-based at window extraction; materialized windows use 0-based half-open bounds.
- Context: 8,192 bp centered on each SNV.
- Carbon input: each strand is truncated to 8,190 bp at the deterministic 6-mer boundary before tokenization.
- Score: bf16 masked mean causal token log likelihood on REF and ALT for FWD and reverse-complement strands.
- Inference batch: one variant, or four allele-strand prompts, validated on both an NVIDIA A10G and a Lambda GH200.
- Derived score: `llr = ((LL_alt_fwd - LL_ref_fwd) + (LL_alt_rc - LL_ref_rc)) / 2`, then `score = -llr`.
- Metric: per-consequence-subset and eligible-subset macro AUPRC for the untagged, correct, and far-wrong fungal conditions.
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
The promoter configuration selects `tss_proximal` and asserts 2,050 rows, 205 positives, 205 match groups, and ten rows per group.
The full-development configuration selects every train row and asserts 16,140 rows, 1,614 positives, 1,614 match groups, and ten rows per group.
The reference loader asserts exact GRCh38 lengths for every development chromosome.

Window extraction rejects out-of-bound coordinates, missing contigs, noncanonical sequence, length mismatches, and reference-allele mismatches.
Any row failure excludes its complete `match_group`.
Every failed row and group is written under the selected output namespace's `windows/exclusions.parquet`.

The smoke sample is selected by a stable hash of `variant_id`.
Labels, subsets, match groups, and consequence columns are removed before the smoke scorer sees the rows.

## Outputs

```text
results/
├── preflight/
│   ├── prompt_grammar.json
│   └── prompt_grammar.parquet
├── promoter_pilot/
│   ├── windows/
│   │   ├── mendelian.parquet
│   │   └── exclusions.parquet
│   ├── smoke/
│   │   ├── windows.parquet
│   │   ├── scores/{untagged,correct}.parquet
│   │   └── runtime/{untagged,correct}.json
│   ├── scores/Carbon-3B/{untagged,correct}.parquet
│   ├── metrics/Carbon-3B/{untagged,correct}.parquet
│   ├── paired/Carbon-3B/correct_minus_untagged.parquet
│   ├── runtime/Carbon-3B/{untagged,correct}.json
│   └── summary.md
└── full_development/
    ├── windows/{mendelian,exclusions}.parquet
    ├── scores/Carbon-3B/{untagged,correct,far_wrong}.parquet
    ├── metrics/Carbon-3B/{untagged,correct,far_wrong}.parquet
    ├── paired/Carbon-3B/{correct_minus_untagged,far_wrong_minus_untagged,far_wrong_minus_correct}.parquet
    ├── runtime/Carbon-3B/{untagged,correct,far_wrong}.json
    └── summary.md
```

Each full score parquet contains variant keys, labels, subsets, match groups, four per-allele/per-strand log likelihoods, per-strand LLRs, the FWD/RC average LLR, and `score = -llr`.

The retained three-arm bundle is stored at `s3://oa-bolinas/snakemake/analysis/carbon_conditioning_vep/snapshots/carbon-conditioning-vep-full-three-arm-20260820/`.
It contains one score row per variant for all three conditions, the complete checksum manifest, derived metrics, paired tables, runtimes, report, exclusions, and staged development windows.

## Per-variant score-shift analysis

The compact follow-up analysis compares each conditioned score with the same variant's untagged score.
For each condition and consequence subset, it reports the mean and standard deviation of these deltas separately for pathogenic positives and matched negatives.
It also reports a matched label-separation shift: the positive delta minus the mean delta of that positive's nine matched negatives.
Positive values mean that conditioning moves positives upward relative to their matched negatives.

The variability statistic is the standard deviation of positive deltas divided by the standard deviation of negative deltas and is plotted on a base-two logarithmic scale.
Both statistics use 1,000 seeded match-group bootstrap draws for 95% intervals.
The per-variant follow-up excludes all 40 mature-miRNA rows, leaving 16,100 variants and 1,610 complete match groups across eight consequence subsets.
Three pairwise scatter figures compare untagged, correct mammalian, and far-wrong fungal scores for the complete analyzed population and every retained subset; each panel reports Pearson correlation overall and by label.
All subset findings are exploratory and have no multiplicity correction or testing hierarchy.

Run the analysis from this project root after retrieving all three per-variant score parquets.

```bash
uv run --locked python \
  ../../../.agents/artifacts/carbon-species-conditioning-vep/analyze_score_shifts.py
```

The script writes these compact, branch-retained artifacts under `.agents/artifacts/carbon-species-conditioning-vep/`:

- `three_approach_summary.{parquet,tsv}`: macro AUPRC and positive/negative score summaries for all three retained approaches.
- `score_pairwise_correlations.{parquet,tsv}`: Pearson correlations for every approach pair, retained subset, and label stratum.
- `score_shift_by_subset_label.{parquet,tsv}`: descriptive score shifts by condition, subset, and label.
- `score_shift_matched_bootstrap.{parquet,tsv}`: matched label-separation and variability estimates with bootstrap intervals.
- `score_shift_summary.md`: the three-approach comparison and overall shift estimates.
- `score_shift_by_subset.{svg,png}`: the rendered subset comparison.
- `score_scatter_untagged_vs_correct.{svg,png}`: all retained subsets for untagged versus correct mammalian scores.
- `score_scatter_untagged_vs_far_wrong.{svg,png}`: all retained subsets for untagged versus far-wrong fungal scores.

## Local validation

Run from this project root.

```bash
uv sync --locked --group dev
uv run --locked pytest
uv run --locked snakemake -n --profile workflow/profiles/default
uv run --locked snakemake -n --configfile config/full_development.yaml --profile workflow/profiles/default
```

The dry-run performs no remote inference.
The default storage profile maps result paths to `s3://oa-bolinas/snakemake/analysis/carbon_conditioning_vep/`.

## Remote execution

Paid remote compute requires explicit approval.
The prompt preflight and corrected label-blind four-condition smoke passed on an AWS `g5.2xlarge` A10G.
The smoke's maximum allocated GPU memory was 11.26 GiB.

A subsequent Lambda GH200 benchmark scored the same eight label-blind rows in 1.55 seconds with batch size one.
Batch size eight was slower per row and raised peak allocation to 44.76 GiB, so the pilot keeps batch size one.
The retained promoter execution scored both conditions in 7.6 inference minutes.
The retained two-condition full-development run kept the Lambda GH200 instance up for 59 minutes 49 seconds and cost an estimated $2.28 at $2.29 per hour.

The two-arm full-development GH200 task has a 70-minute command timeout and two-minute autodown, giving an approval ceiling of $3.00 after ordinary sync overhead.
The additive far-wrong task targets only the fungal arm, reuses the retained untagged score locally, and has a 40-minute command timeout plus two-minute autodown for an approximately $1.60 ceiling at $2.29 per hour.
The retained far-wrong run kept the instance up for 33 minutes 53 seconds and cost an estimated $1.29.
Its 16,140-row scoring command took 1,698.9 seconds.
Check the current [Lambda instance price](https://lambda.ai/instances) immediately before launch.

Stage the already-validated development-only artifacts on the coordinator before launch.
This reads the canonical S3 artifacts locally, streams the configured scope with provenance checks, and does not forward AWS credentials to Lambda.

```bash
bash snakemake/analysis/carbon_conditioning_vep/sky/stage-gh200-full.sh
```

Launch from the repository root only after explicit approval of the current price and cap.

```bash
sky launch snakemake/analysis/carbon_conditioning_vep/sky/run-gh200-full.yaml \
  -c carbon-conditioning-vep-gh200-full
```

Retrieve the results immediately after the job succeeds and before the two-minute autodown.
SkyPilot stores each cluster's SSH configuration separately from `~/.ssh/config`, so pass that generated configuration to rsync explicitly.

```bash
rsync -a \
  -e 'ssh -F /home/ubuntu/.sky/generated/ssh/carbon-conditioning-vep-gh200-full' \
  carbon-conditioning-vep-gh200-full:sky_workdir/snakemake/analysis/carbon_conditioning_vep/results/full_development/ \
  snakemake/analysis/carbon_conditioning_vep/results/full_development/
sky down carbon-conditioning-vep-gh200-full -y
```

The far-wrong task computes only the new score and absolute metric on Lambda.
After retrieval, direct metric commands add the paired far-wrong contrasts and render the combined report from the retained baseline artifacts.

```bash
sky launch snakemake/analysis/carbon_conditioning_vep/sky/run-gh200-far-wrong.yaml \
  -c carbon-conditioning-vep-gh200-far-wrong
rsync -a \
  -e 'ssh -F /home/ubuntu/.sky/generated/ssh/carbon-conditioning-vep-gh200-far-wrong' \
  carbon-conditioning-vep-gh200-far-wrong:sky_workdir/snakemake/analysis/carbon_conditioning_vep/results/full_development/ \
  snakemake/analysis/carbon_conditioning_vep/results/full_development/
sky down carbon-conditioning-vep-gh200-far-wrong -y
```

Finalize the paired comparisons and combined report locally after verifying that the new score was retrieved.
These direct commands cannot invoke a scoring rule.

```bash
cd snakemake/analysis/carbon_conditioning_vep
uv run --locked carbon-conditioning-vep paired-deltas \
  --config config/full_development.yaml \
  --comparison far_wrong_minus_untagged \
  --score-a results/full_development/scores/Carbon-3B/far_wrong.parquet \
  --score-b results/full_development/scores/Carbon-3B/untagged.parquet \
  --output results/full_development/paired/Carbon-3B/far_wrong_minus_untagged.parquet
uv run --locked carbon-conditioning-vep paired-deltas \
  --config config/full_development.yaml \
  --comparison far_wrong_minus_correct \
  --score-a results/full_development/scores/Carbon-3B/far_wrong.parquet \
  --score-b results/full_development/scores/Carbon-3B/correct.parquet \
  --output results/full_development/paired/Carbon-3B/far_wrong_minus_correct.parquet
uv run --locked carbon-conditioning-vep report \
  --config config/full_development.yaml \
  --preflight results/preflight/prompt_grammar.json \
  --absolute-metrics \
    results/full_development/metrics/Carbon-3B/untagged.parquet \
    results/full_development/metrics/Carbon-3B/correct.parquet \
    results/full_development/metrics/Carbon-3B/far_wrong.parquet \
  --paired-deltas \
    results/full_development/paired/Carbon-3B/correct_minus_untagged.parquet \
    results/full_development/paired/Carbon-3B/far_wrong_minus_untagged.parquet \
    results/full_development/paired/Carbon-3B/far_wrong_minus_correct.parquet \
  --exclusions results/full_development/windows/exclusions.parquet \
  --runtimes \
    results/full_development/runtime/Carbon-3B/untagged.json \
    results/full_development/runtime/Carbon-3B/correct.json \
    results/full_development/runtime/Carbon-3B/far_wrong.json \
  --output results/full_development/summary.md
```
