# Evaluation Datasets Pipeline

This pipeline curates two variant-classification benchmark datasets from primary
sources, matches negatives to positives within (chrom, consequence_final) strata
at a 1:9 ratio on continuous features, and writes the resulting parquet
datasets locally (plus their eval-harness sequence variants). HuggingFace upload
is a separate target invoked explicitly after reviewing the per-subset matching
diagnostics — see [Usage](#usage).

The curation is a from-scratch reimplementation of the TraitGym pipeline
(Benegas et al. 2025; songlab-cal/TraitGym) at the parent of the HGMD-removal
commit `e59d612e9`, so HGMD pathogenic SNVs are included as a positive source.

## Datasets

| Name | Description | Positives | Negatives |
|---|---|---|---|
| `mendelian_traits` | Mendelian disease pathogenic SNVs | HGMD ∪ OMIM ∪ Smedley et al. 2016 (de-duped, AF<0.001) | gnomAD common (AN≥25k, AF>0.001) |
| `complex_traits` | UKBB fine-mapped complex-trait variants | SuSiE+FINEMAP `max(PIP across the traits where this variant was fine-mapped) > 0.9` | `max(PIP) < 0.01` AND no SuSiE/FINEMAP combine-step null PIP among those traits (`label_variants_by_pip(use_null_pip_guard=True)`) |
| `caqtl` | DART-Eval African caQTLs (ATAC), GRCh38 | Significant caQTLs in peaks | Control variants in peaks (no matching) |
| `dsqtl` | DART-Eval Yoruban dsQTLs (DNase), hg19→GRCh38 | Significant dsQTLs in peaks | Control variants in peaks (no matching) |

Only `mendelian_traits` has a corresponding `_harness_255` eval-harness
variant. A 255 bp window centered on each variant is materialized into
`context` / `ref_completion` / `alt_completion` columns; models that prepend a
BOS token see 256 tokens of context (the rest of the codebase uses 255 bp
windows for the same reason). Each input variant produces **two output rows**
— one for `strand="+"` (FWD), one for `strand="-"` (RC of the same window) —
for the online lm_eval VEP scorer to average per variant (#179, #175 conclusion 2).
`complex_traits` has no harness variant: it's scored offline only via
`snakemake/analysis/evals_v2/`, which already does FWD+RC averaging in the
batched VEP path.

### DART-Eval caQTL / dsQTL

`caqtl` and `dsqtl` are the [DART-Eval](https://github.com/kundajelab/DART-Eval)
Task-5 chromatin-accessibility QTL benchmarks, brought in to give them
train/test splits (DART-Eval ships none for this task — see issue #139).
Unlike the two matched datasets above they are **not matched and not
subsampled**: every significant QTL (positive) and control variant (negative)
within accessible peaks is kept at its natural ratio (≈1:11 for caQTL, ≈1:48
for dsQTL). The `consequence*` and `distance_*` columns are added as
**reference annotations only** — they are not used to filter or match. The signed study `effect` is oriented to the
`alt` allele (positive ⇒ alt increases accessibility, matching an alt-vs-ref /
LLR model score; sign-flipped where ref/alt were swapped to the reference), and
`effect_size` is its unsigned magnitude (`|effect|`); caQTL additionally carries
the raw `pval` and `se` for the record. For dsQTL `effect`/`effect_size` are
present for the significant variants only (controls have no measured effect).

Evaluation follows DART-Eval Task 5 / ARSENAL, with the two metrics computed
over **different variant sets**: a binary **AUROC/AUPRC over all variants**
(significant QTLs vs controls, via `label`), and a **Pearson correlation over
the positive variants only** (a model's signed alt-vs-ref score vs the signed
`effect`). The dataset cards document this.

- **caqtl** — African caQTLs (DeGorter et al. 2023), Synapse `syn60756043`
  (`Afr.CaQTLS.tsv`). Native **GRCh38**.
- **dsqtl** — Yoruban dsQTLs (Degner et al. 2012), Synapse `syn60756039`
  (`yoruban.dsqtls.benchmarking.tsv`). **hg19**, lifted to GRCh38 via
  `lift_hg19_to_hg38`.

These two skip the matching pipeline entirely: `dart_eval_dataset_unsplit` (in
`workflow/rules/dart_eval.smk`) parses + annotates the raw TSV straight into
`results/dataset_unsplit/{caqtl,dsqtl}.parquet`, which the generic
`split_dataset_by_chrom` rule then splits. They have **no** `results/qc/`
artifact (no matching to diagnose).

Building them needs **Synapse auth**: the raw TSVs live only on Synapse, so
create a free Synapse account + a [Personal Access Token](https://www.synapse.org/#!PersonalAccessTokens:)
(scopes: View, Download) and export it — the download is plain HTTPS via
`curl`, no `synapseclient` needed. (AWS credentials with S3 read access are
also required, as for the rest of the pipeline — see [Storage](#storage).)

The `dart_eval_stage_genome` rule downloads the GRCh38 reference (+ `.fai`/`.gzi`
indexes) to local disk once via boto3, so `check_ref_alt` reads from disk
rather than doing a per-variant S3 round-trip.

```bash
export SYNAPSE_AUTH_TOKEN=...   # Synapse PAT
uv run snakemake \
  results/dataset/caqtl/{train,test}.parquet \
  results/dataset/dsqtl/{train,test}.parquet
```

## Matching scheme

For each dataset, every positive is matched to 9 negatives via TraitGym's
greedy-nearest-neighbor matcher (`marin_dna.pipelines.evals.matching.match_features`).
Matching is exact on `(chrom, consequence_final)` plus subset-targeted
distance bins (see below), then Euclidean-nearest on the (RobustScaler-scaled)
continuous features, without replacement within a stratum. See
`src/marin_dna/pipelines/evals/matching.py` for the algorithm.

**Gene-ID columns** (`tss_closest_pc_gene_id`, `tss_closest_nc_gene_id`,
`exon_closest_pc_gene_id`, `exon_closest_nc_gene_id`) are *not* part of the
categorical match key — exact gene matching dropped too many positives. They
remain in the output parquets as passthrough metadata.

**Per-(subset, feature) distance bins** as exact-match categoricals on top of
the continuous nearest-neighbor step, applied via
`add_subset_distance_bins_v2(df, scheme)`. Each entry adds a `{feature}_bin`
column whose value is `"{subset[:8]}:b{i}"` for rows in that subset, `BIN_NA`
otherwise — multiple subsets can share a feature column with disjoint
subset-prefixed labels. Bin edges target only the AUPRC-leak cells flagged by
the diagnostic (see [Matching diagnostics](#matching-diagnostics)); they're
intentionally coarse to leave per-stratum negative pools intact at k=9.

Mendelian scheme:

| subset | feature | edges |
|---|---|---|
| `tss_proximal` | `distance_tss_pc` | `[0, 100, 1000, ∞]` |
| `tss_proximal` | `distance_exon_pc` | `[0, 100, 1000, ∞]` |
| `splicing` | `distance_exon_pc` | `[0, 5, 30, ∞]` |
| `distal` | `distance_exon_pc` | `[0, 100, 1000, 5000, 10000, ∞]` |

Complex-traits scheme mirrors mendelian's `tss_proximal` and `splicing`
entries; `distal` is not binned (already at baseline).

**Missense cap** (mendelian only): positives in the `missense_variant`
consequence_group are subsampled (seed `42`) before matching, with the cap
configured by `mendelian_traits.max_positives_per_subset.missense_variant` in
`config/config.yaml`. The default `1000` is a ~5× VEP-inference speedup over
the uncapped ~12.7k missense positives, with negligible loss of subset-AUPRC
CI tightness; macro-average reporting handles imbalance for the global metric.

Per-dataset specifics:

- **mendelian_traits** (no MAF column in `dataset_all`):
  - continuous = `[distance_tss_pc, distance_tss_nc, distance_exon_pc, distance_exon_nc]`
  - categorical = `[chrom, consequence_final, distance_tss_pc_bin, distance_exon_pc_bin]`
- **complex_traits**:
  - continuous = mendelian's + `MAF`
  - categorical = `[chrom, consequence_final, distance_tss_pc_bin, distance_exon_pc_bin]`

The bin columns also serve as passthrough metadata on the output so
downstream consumers can stratify by the same bins used for matching.

## Matching diagnostics

Each rebuild produces `results/qc/{dataset}.parquet` with one row per
`consequence_group` subset, via the `dataset_matching_qc` rule (see
`workflow/rules/common.smk` and `src/marin_dna/pipelines/evals/matching_qc.py`).
Two diagnostics:

1. **Subsampling drops** — `n_positives_input`, `n_positives_kept`,
   `n_dropped`, `frac_dropped`. At k=9 a positive is dropped when its
   `(chrom, consequence_final)` stratum has fewer than 9 negatives, so
   `_match_single_group` subsamples positives.
2. **Per-feature AUPRC leak** — for each continuous matching feature `f`,
   `{f}_auprc` is `max(AP(label, +f), AP(label, −f))` against the matched
   pos/neg labels within the subset (sign flip handles either-direction
   leaks); `{f}_auprc_sign` reports which direction. The `baseline_auprc`
   column is the positive prevalence in the subset (≈ 0.1 for 1:9). A
   feature whose AUPRC is at-or-near baseline did not separate positives
   from negatives — i.e. matching controlled for it. AUPRC well above
   baseline signals residual leak; if confirmed, reach for a bin on that
   feature.

The QC artifact is local-only; it is not uploaded to HuggingFace.

## Pipeline structure

`workflow/rules/common.smk` consolidates all shared infrastructure:

- `download_genome` — fetches the reference FASTA.
- `split_dataset_by_chrom` — generic chrom-based train/test split. Reads
  `results/dataset_unsplit/{dataset}.parquet` and emits both
  `results/dataset/{dataset}/train.parquet` and `.../test.parquet`. Train uses
  odd chromosomes (1, 3, …, X), test uses even (2, 4, …, Y).
- `materialize_eval_harness_dataset` — materializes the `_harness_{window_size}`
  variant of any dataset (`{base}_harness_{window_size}` naming convention).
- `hf_upload` — uploads `results/dataset/{dataset}/{train,test}.parquet` to
  `f"{output_hf_prefix}_{dataset}"` on HuggingFace.

`workflow/rules/intervals.smk`, `consequence.smk`, `gnomad_common.smk`,
`hgmd.smk`, `omim.smk`, `smedley.smk` build the shared per-source files
(GTF-derived TSS/exon intervals, per-chrom Ensembl VEP consequences, gnomAD
common variants, and the three Mendelian positive sources).

`workflow/rules/mendelian_traits.smk` produces:

```
positives.parquet              (HGMD+OMIM+Smedley, deduped, AF<0.001, consequences attached)
dataset_all.parquet            (positives ∪ gnomAD common, with build_dataset annotations)
dataset_unsplit/mendelian_traits.parquet  (1:9 matched)
```

`workflow/rules/ldscore.smk` + `workflow/rules/complex_traits.smk` produce the
complex-trait dataset along the same lines, plus per-trait fine-mapping
downloads and aggregation across 119 traits.

The generic `split_dataset_by_chrom` rule then turns each
`results/dataset_unsplit/{name}.parquet` into the train/test pair. The
`dataset_matching_qc` rule emits `results/qc/{name}.parquet` from the same
unsplit parquet so diagnostics see the full dataset before the chrom-based
split. `hf_upload` is wired up but **not** in `rule all`; it must be
invoked explicitly via `rule upload_all` (see [Usage](#usage)).

## Setup

Python dependencies are managed by the main project (see `../../README.md`).

Authenticate with HuggingFace before uploading:

```bash
huggingface-cli login
```

### Storage

Pipeline results are stored in S3 (`s3://oa-bolinas/snakemake/evals/`). A
default Snakemake profile at `workflow/profiles/default/config.yaml` configures
S3 storage and cores automatically.

You need AWS credentials with S3 access:
- **On EC2**: attach an IAM role with `AmazonS3FullAccess` to the instance.
- **On your laptop**: run `aws configure` with an IAM user's access key.

`ldscore_download` shells out to `aws s3 cp`, so install the `aws-cli` group
to get the `aws` binary on `uv run`'s PATH:

```bash
uv sync --group aws-cli
```

### Singularity (LD score)

`ldscore.smk` runs Hail inside a Docker image via Singularity to convert the
UKBB LD score HailTable to TSV. The host must have Singularity available and
be able to pull `hailgenetics/hail:0.2.130.post1-py3.11`. If Singularity isn't
available, the rule will fail; you can hand-build the LD-score parquet
elsewhere and stage it into `results/ldscore/UKBB.EUR.ldscore.parquet`.

### HGMD redistribution

HGMD pathogenic SNVs are downloaded from `sei-files.s3.amazonaws.com`. Including
HGMD-derived variants in a public HF dataset has license implications — check
before pushing the upload.

## Configuration (`config/config.yaml`)

Top-level keys:

| Key | Purpose |
|---|---|
| `genome_url` | Reference FASTA URL (GRCh38). |
| `annotation_url` | Ensembl GTF URL (release 107). |
| `consequences_repo` | HF repo with per-chrom Ensembl VEP consequences (`{chrom}.parquet`). |
| `gnomad_full_repo` | HF repo for the full gnomAD test parquet. |
| `gnomad_min_AN`, `gnomad_common_min_AF` | Filter for "common" gnomAD variants. |
| `tss_proximal_dist`, `exon_proximal_dist` | Distance thresholds for the `consequence_final` overrides. |
| `exclude_consequences` | High-impact VEP consequences dropped from the dataset. |
| `consequence_groups` | Mapping from collapsed-group key (`splicing`, `distal`) to the VEP consequences merged into that group. Single-consequence categories (`missense_variant`, `tss_proximal`, etc.) are not listed — they pass through unchanged. |
| `consequence_group_order` | Display name + plot order for each value that ends up in the `consequence_group` column. |
| `output_hf_prefix` | HF repo prefix; final repo is `f"{prefix}_{dataset}"`. |
| `datasets` | Which datasets `rule all` builds + uploads. |
| `mendelian_traits.*` | HGMD URL, Smedley URL, ClinVar release pin, submission summary date, AF threshold. |
| `complex_traits.*` | Fine-mapping repo, LD-score S3 path, PIP thresholds. |
| `dart_eval.*` | Synapse FileEntity IDs for the caQTL / dsQTL TSVs. |

`config/complex_traits.csv` lists the 119 UKBB traits used for `complex_traits`.

## Usage

```bash
# Build local parquets + matching diagnostics (does NOT upload to HF):
uv run snakemake --directory snakemake/evals

# Review:
#   results/dataset/{dataset}/{train,test}.parquet
#   results/qc/{dataset}.parquet
# After approving the datasets, push to HuggingFace:
uv run snakemake --directory snakemake/evals upload_all
```

To build a single dataset locally without uploading:

```bash
uv run snakemake --directory snakemake/evals \
  results/dataset/mendelian_traits/train.parquet \
  results/dataset/mendelian_traits/test.parquet \
  results/qc/mendelian_traits.parquet
```

## Output

Datasets are uploaded to HuggingFace Hub at `f"{output_hf_prefix}_{dataset}"`.

Examples:

- `bolinas-dna/evals_mendelian_traits`
- `bolinas-dna/evals_complex_traits`
- `bolinas-dna/evals_mendelian_traits_harness_255`
- `bolinas-dna/evals_caqtl`
- `bolinas-dna/evals_dsqtl`

Locally, files live in `results/dataset/{dataset}/{train,test}.parquet`, and
matching diagnostics in `results/qc/{dataset}.parquet` (matched datasets only —
`caqtl`/`dsqtl` have none).

### Eval-harness columns

Datasets materialized with `_harness_{window_size}` add the following columns
and emit **two output rows per input variant** — one per strand:

| Column | Description |
|---|---|
| `context` | Left flank up to (but not including) the variant position, on the strand named in `strand`. |
| `ref_completion` | Reference allele (in-strand) + right flank. |
| `alt_completion` | Alternate allele (in-strand) + right flank. |
| `strand` | `"+"` (FWD) or `"-"` (RC of the FWD window; ref/alt complemented). |
| `target` | Binary label (renamed from `label`; identical across the two strand rows). |

Two-row layout exists so the online lm_eval VEP scorer (`marin_dna.pipelines.evals.lm_eval.dna_vep_llr_eval`)
averages each variant's raw LLR across the two strands before computing AUPRC
(migrated from PairwiseAccuracy in #225) — the
FWD+RC averaging documented as #175 conclusion 2 (mirrors `snakemake/analysis/evals_v2/`'s
`inference.rc_avg=true`). Rows are sorted by `(chrom, pos, ref, alt, strand)`
so per-variant strand pairs are adjacent.

Window-length math:
- FWD: `var_pos = window_size // 2`. Context length `window_size // 2`,
  completion length `window_size - window_size // 2`.
- RC:  `var_pos = window_size - 1 - window_size // 2`. For odd `window_size`
  this matches FWD; for even `window_size` the RC context is one bp shorter
  and the RC completion one bp longer.
