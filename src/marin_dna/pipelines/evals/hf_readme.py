"""HuggingFace dataset-card generators for the matched eval datasets.

One function per `{output_hf_prefix}_{dataset}` repo. Each returns the full
markdown README — tag frontmatter, description, splits, columns, retention,
matching scheme, AUPRC-leak diagnostic, provenance, citation.

Computed live from the produced parquets + QC artifact at upload time so the
numbers always match the dataset being pushed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

REPO_ROOT_URL = "https://github.com/Open-Athena/marin-dna"


def _frontmatter(
    extra_tags: Sequence[str] | None = None, *, size_category: str = "10K<n<100K"
) -> str:
    tags = ["biology", "genomics", "dna"]
    if extra_tags:
        tags = tags + [t for t in extra_tags if t not in tags]
    lines = ["---", "license: apache-2.0", "tags:"]
    lines.extend(f"  - {t}" for t in tags)
    lines.append("size_categories:")
    lines.append(f"  - {size_category}")
    lines.append("---")
    return "\n".join(lines)


def _split_counts(train_path: str | Path, test_path: str | Path) -> dict:
    train = pl.read_parquet(train_path)
    test = pl.read_parquet(test_path)
    return {
        "train_pos": train.filter(pl.col("label")).height,
        "train_neg": train.filter(~pl.col("label")).height,
        "train_total": train.height,
        "test_pos": test.filter(pl.col("label")).height,
        "test_neg": test.filter(~pl.col("label")).height,
        "test_total": test.height,
    }


def _retention_table(qc: pl.DataFrame) -> str:
    qc = qc.sort("n_positives_input", descending=True)
    rows = [
        "| Subset | n_pos in `dataset_all` | matched (kept) | retention |",
        "|---|---:|---:|---:|",
    ]
    total_in, total_kept = 0, 0
    for r in qc.iter_rows(named=True):
        ni = r["n_positives_input"] or 0
        nk = r["n_positives_kept"] or 0
        total_in += ni
        total_kept += nk
        ret = f"{nk / ni:.1%}" if ni else "—"
        rows.append(f"| `{r['subset']}` | {ni:,} | {nk:,} | {ret} |")
    overall = f"{total_kept / total_in:.1%}" if total_in else "—"
    rows.append(
        f"| **total** | **{total_in:,}** | **{total_kept:,}** | **{overall}** |"
    )
    return "\n".join(rows)


def _auprc_table(qc: pl.DataFrame, features: list[str]) -> str:
    qc = qc.sort("n_positives_kept", descending=True)
    header = "| subset | n |"
    sep = "|---|---:|"
    for f in features:
        header += f" {f} |"
        sep += "---:|"
    lines = [header, sep]
    for r in qc.iter_rows(named=True):
        n = r["n_positives_kept"] or 0
        if n == 0:
            continue
        cells = [f"| `{r['subset']}` | {n:,} |"]
        for f in features:
            v = r.get(f"{f}_auprc")
            cells.append(f" {v:.3f} |" if v is not None else " — |")
        lines.append("".join(cells))
    return "\n".join(lines)


def _pipeline_link(sha: str, *, label: str = "snakemake/evals") -> str:
    return f"[`{label}`]({REPO_ROOT_URL}/tree/{sha}/snakemake/evals)"


def _file_link(sha: str, path: str) -> str:
    return f"[`{path}`]({REPO_ROOT_URL}/blob/{sha}/{path})"


def render_mendelian(
    sha: str,
    train_path: str | Path,
    test_path: str | Path,
    qc_path: str | Path,
) -> str:
    c = _split_counts(train_path, test_path)
    qc = pl.read_parquet(qc_path)
    retention = _retention_table(qc)
    auprc = _auprc_table(
        qc,
        ["distance_tss_pc", "distance_tss_nc", "distance_exon_pc", "distance_exon_nc"],
    )
    return f"""{_frontmatter(["variant-effect-prediction"])}

# evals_mendelian_traits

Variant-effect-prediction benchmark of pathogenic Mendelian SNVs vs gnomAD
common SNVs, 1:9 matched within consequence categories on `(chrom,
consequence_final)` plus subset-targeted distance bins.

## Description

| | |
|---|---|
| Positives | OMIM ∪ Smedley *et al.* 2016 ∪ HGMD (latter via [Sei](https://www.nature.com/articles/s41588-022-01102-2), Chen *et al.* *Nat Genet* 2022), deduplicated, gnomAD AF<0.001 |
| Negatives | gnomAD common: AN≥25 000 and AF>0.001, 1:9 matched per positive |
| Genome build | GRCh38 |
| Variant type | SNVs only |
| Coordinates | 1-based (`pos` is 1-based; `ref`/`alt` are single bases) |
| Matching ratio | 1:9 (positive : matched negatives) |

## Splits

| Split | Variants (pos + 9·neg) | Positives | Chromosomes |
|---|---:|---:|---|
| `train` | {c["train_total"]:,} | {c["train_pos"]:,} | odd: 1, 3, …, X |
| `test` | {c["test_total"]:,} | {c["test_pos"]:,} | even: 2, 4, …, Y |
| **total** | **{c["train_total"] + c["test_total"]:,}** | **{c["train_pos"] + c["test_pos"]:,}** | |

## Columns

| Column | Type | Description |
|---|---|---|
| `chrom`, `pos`, `ref`, `alt` | str / int / str / str | Variant coordinates (1-based, GRCh38) |
| `label` | bool | `True` for pathogenic positive, `False` for matched gnomAD-common negative |
| `subset` | str | Consequence-group label for stratified eval |
| `match_group` | int | Integer ID grouping each positive with its 9 matched negatives |
| `source` | str | Pathogenic source (`omim`, `smedley_et_al`, `hgmd`); positives only |
| `clinvar_id`, `trait` | int / str | ClinVar IDs (OMIM-derived positives) and/or phenotype text |
| `AF`, `AC`, `AN` | float / int / int | gnomAD allele frequency, count, total |
| `consequence`, `consequence_cre`, `consequence_final`, `consequence_group` | str | Ensembl VEP consequence + grouping used by the matcher |
| `distance_tss_pc`, `distance_tss_nc`, `distance_tss` | int | Distance to nearest protein-coding / non-protein-coding transcript TSS (and the minimum, used by the `consequence_group` recategorization) |
| `tss_closest_pc_gene_id`, `tss_closest_nc_gene_id`, `tss_closest_gene_id` | str | Ensembl gene IDs at those distances (passthrough metadata — gene-id was *not* used in matching) |
| `distance_exon_pc`, `distance_exon_nc`, `distance_exon` | int | Same shape, for nearest exon |
| `exon_closest_pc_gene_id`, `exon_closest_nc_gene_id`, `exon_closest_gene_id` | str | Same shape |
| `distance_tss_pc_bin`, `distance_exon_pc_bin` | str | Subset-prefixed bin labels used as exact-match keys during matching (e.g. `"tss_prox:b0"`, `"splicing:b2"`); `BIN_NA` outside the binned subsets |

## Per-subset retention

{retention}

The bulk loss is the deliberate **missense cap** (12.7k → 1k for VEP inference
speed; cap is `mendelian_traits.max_positives_per_subset.missense_variant: 1000`
in the pipeline config). Beyond that, drops come from `(chrom, consequence_final
× bin)` strata with fewer than 9 negatives, which forces `_match_single_group`
to subsample positives.

## Matching design

Matching is exact on every categorical key, then Euclidean-nearest on the
(RobustScaler-scaled) continuous features as a within-group tie-breaker.
Without replacement, k=9.

- **Continuous features**: `distance_tss_pc`, `distance_tss_nc`, `distance_exon_pc`, `distance_exon_nc`.
- **Categorical features**:
  - `chrom`, `consequence_final`
  - `distance_tss_pc_bin` — `tss_proximal`: edges `[0, 100, 1000, ∞]`; `BIN_NA` elsewhere
  - `distance_exon_pc_bin` —
    - `tss_proximal`: edges `[0, 100, 1000, ∞]`
    - `splicing`: edges `[0, 5, 30, ∞]`
    - `distal`: edges `[0, 100, 1000, 5000, 10000, ∞]`
    - `BIN_NA` elsewhere

Gene-ID columns are kept as passthrough metadata but **not** used as match
keys — exact gene matching at 1:9 dropped too many positives. The bin
columns and gene-IDs are still useful for downstream stratification.

## Matched-feature AUPRC diagnostic

Each continuous matching feature `f` is scored as a single-feature predictor
within each subset: `{{f}}_auprc = max(AP(label, +f), AP(label, −f))`.
**Baseline = 0.1 for 1:9 matching**. Values near baseline mean the feature
does not separate positives from negatives within the subset (matching
worked); values well above baseline are residual leak.

<details>
<summary>Per-(subset, feature) AUPRC table</summary>

{auprc}

</details>

## Provenance

Built by the [`marin-dna`]({REPO_ROOT_URL}) eval pipeline at commit
[`{sha[:7]}`]({REPO_ROOT_URL}/tree/{sha}/snakemake/evals).

- Curation pipeline: {_pipeline_link(sha)}
- Matching algorithm: {_file_link(sha, "src/marin_dna/pipelines/evals/matching.py")}
- Diagnostic helper: {_file_link(sha, "src/marin_dna/pipelines/evals/matching_qc.py")}

The curation is a from-scratch reimplementation of the [TraitGym](https://github.com/songlab-cal/TraitGym) pipeline.

## Companion datasets

- **[`bolinas-dna/evals_mendelian_traits_harness_255`](https://huggingface.co/datasets/bolinas-dna/evals_mendelian_traits_harness_255)** — same variants with 255 bp reference-genome windows materialized for direct use as eval-harness inputs.

## Citation

If you use this benchmark, please cite the upstream sources:

- TraitGym — Benegas *et al.* 2025, [bioRxiv 2025.02.11.637758](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v2)
- gnomAD — Karczewski *et al.* (Nature 2020)
- OMIM — [omim.org](https://omim.org)
- Smedley *et al.* — *AJHG* 99(3): 595–606 (2016)
- Sei (HGMD redistribution path) — Chen *et al.* *Nat Genet* 2022
- HGMD — Stenson *et al.* (Hum Genet 2017)
"""


def render_complex(
    sha: str,
    train_path: str | Path,
    test_path: str | Path,
    qc_path: str | Path,
) -> str:
    c = _split_counts(train_path, test_path)
    qc = pl.read_parquet(qc_path)
    retention = _retention_table(qc)
    auprc = _auprc_table(
        qc,
        [
            "distance_tss_pc",
            "distance_tss_nc",
            "distance_exon_pc",
            "distance_exon_nc",
            "MAF",
        ],
    )
    return f"""{_frontmatter(["variant-effect-prediction", "complex-traits", "gwas", "fine-mapping"])}

# evals_complex_traits

Variant-effect-prediction benchmark of UKBB fine-mapped complex-trait SNVs vs
low-PIP SNVs, 1:9 matched within consequence categories on `(chrom,
consequence_final)` plus subset-targeted distance bins, with MAF entering as
a continuous matching feature.

## Description

| | |
|---|---|
| Positives | UKBB SuSiE+FINEMAP fine-mapped variants with max(PIP) > 0.9 across 119 traits |
| Negatives | max(PIP) < 0.01 across 119 traits, 1:9 matched per positive |
| Genome build | GRCh38 (lifted from hg19) |
| Variant type | SNVs only |
| Coordinates | 1-based (`pos` is 1-based; `ref`/`alt` are single bases) |
| Matching ratio | 1:9 |

## Splits

| Split | Variants (pos + 9·neg) | Positives | Chromosomes |
|---|---:|---:|---|
| `train` | {c["train_total"]:,} | {c["train_pos"]:,} | odd: 1, 3, …, X |
| `test` | {c["test_total"]:,} | {c["test_pos"]:,} | even: 2, 4, …, Y |
| **total** | **{c["train_total"] + c["test_total"]:,}** | **{c["train_pos"] + c["test_pos"]:,}** | |

## Columns

| Column | Type | Description |
|---|---|---|
| `chrom`, `pos`, `ref`, `alt` | str / int / str / str | Variant coordinates (1-based, GRCh38) |
| `label` | bool | `True` for high-PIP positive, `False` for low-PIP matched negative |
| `subset` | str | Consequence-group label for stratified eval |
| `match_group` | int | Integer ID grouping each positive with its 9 matched negatives |
| `rsid` | str | dbSNP rsID (when available) |
| `pip` | float | Maximum PIP across the 119 traits |
| `traits` | str | Comma-separated list of traits with PIP > 0.9 (positives only) |
| `MAF` | float | UKBB EUR minor allele frequency |
| `ld_score` | float | UKBB EUR LD score (passthrough, **not** a matching feature) |
| `consequence`, `consequence_cre`, `consequence_final`, `consequence_group` | str | Ensembl VEP consequence + grouping |
| `distance_tss_pc`, `distance_tss_nc`, `distance_tss` | int | Distances to nearest protein-coding / non-protein-coding TSS (and min, used for `consequence_group` recategorization) |
| `tss_closest_pc_gene_id`, `tss_closest_nc_gene_id`, `tss_closest_gene_id` | str | Ensembl gene IDs (passthrough — gene-id was *not* used in matching) |
| `distance_exon_pc`, `distance_exon_nc`, `distance_exon` | int | Same shape, for nearest exon |
| `exon_closest_pc_gene_id`, `exon_closest_nc_gene_id`, `exon_closest_gene_id` | str | Same shape |
| `distance_tss_pc_bin`, `distance_exon_pc_bin` | str | Subset-prefixed bin labels used as exact-match keys; `BIN_NA` outside the binned subsets |

## Per-subset retention

{retention}

## Matching design

Matching is exact on every categorical key, then Euclidean-nearest on the
(RobustScaler-scaled) continuous features as a within-group tie-breaker.
Without replacement, k=9.

- **Continuous features**: `distance_tss_pc`, `distance_tss_nc`, `distance_exon_pc`, `distance_exon_nc`, `MAF`.
- **Categorical features**:
  - `chrom`, `consequence_final`
  - `distance_tss_pc_bin` — `tss_proximal`: edges `[0, 100, 1000, ∞]`; `BIN_NA` elsewhere
  - `distance_exon_pc_bin` —
    - `tss_proximal`: edges `[0, 100, 1000, ∞]`
    - `splicing`: edges `[0, 5, 30, ∞]`
    - `BIN_NA` elsewhere

Gene-ID columns are kept as passthrough metadata but **not** used as match
keys.

## Matched-feature AUPRC diagnostic

Each continuous matching feature `f` is scored as a single-feature predictor
within each subset: `{{f}}_auprc = max(AP(label, +f), AP(label, −f))`.
**Baseline = 0.1 for 1:9 matching**.

<details>
<summary>Per-(subset, feature) AUPRC table</summary>

{auprc}

</details>

## Provenance

Built by the [`marin-dna`]({REPO_ROOT_URL}) eval pipeline at commit
[`{sha[:7]}`]({REPO_ROOT_URL}/tree/{sha}/snakemake/evals).

- Curation pipeline: {_pipeline_link(sha)}
- Matching algorithm: {_file_link(sha, "src/marin_dna/pipelines/evals/matching.py")}
- Diagnostic helper: {_file_link(sha, "src/marin_dna/pipelines/evals/matching_qc.py")}

The curation is a from-scratch reimplementation of the [TraitGym](https://github.com/songlab-cal/TraitGym) complex-traits pipeline.

## License

Released under the same terms as its sources. UKBB summary-level data and
the [Finucane lab fine-mapping release](https://huggingface.co/datasets/gonzalobenegas/finucane-ukbb-finemapping)
are intended for non-commercial research; check upstream license if you plan
to use commercially.

## Citation

- TraitGym — Benegas *et al.* 2025, [bioRxiv 2025.02.11.637758](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v2)
- UKBB fine-mapping — Wang *et al.* (Nat Commun 2021) and the [Finucane lab release](https://huggingface.co/datasets/gonzalobenegas/finucane-ukbb-finemapping)
- LD scores — Bulik-Sullivan *et al.* (Nat Genet 2015)
"""


def render_harness(
    sha: str,
    train_path: str | Path,
    test_path: str | Path,
    base_repo: str = "bolinas-dna/evals_mendelian_traits",
    window_size: int = 255,
) -> str:
    train = pl.read_parquet(train_path)
    test = pl.read_parquet(test_path)
    # Two rows per variant (one per strand).
    train_n = train.height // 2
    test_n = test.height // 2
    half = window_size // 2
    return f"""{_frontmatter()}

# evals_mendelian_traits_harness_{window_size}

Eval-harness ready variant-effect-prediction benchmark — same matched
variants as [`{base_repo}`](https://huggingface.co/datasets/{base_repo}),
with **{window_size} bp** reference-genome windows materialized into
`context` / `ref_completion` / `alt_completion` columns for direct scoring
with autoregressive genomic language models. **Each variant emits two rows,
one per strand**, for FWD+RC averaging during online lm_eval scoring.

## Why {window_size} bp

Models that prepend a `<BOS>` token see {window_size} + 1 = **{window_size + 1} tokens** of context.
The materialized window is intentionally {window_size} bp so that with BOS the eval input
fits a {window_size + 1}-token model context exactly. Other windows can be materialized by
re-running the pipeline with a different `window_size` wildcard.

## Why two rows per variant

Per [issue #175](https://github.com/Open-Athena/marin-dna/issues/175)
conclusion 2, averaging FWD-strand and RC-strand LLR-family scores beats
single-strand on most (model, subset) cells. The two-row layout lets the
online lm_eval scorer
(`marin_dna.pipelines.evals.lm_eval.dna_vep_llr_eval`) compute per-strand LLR
per row and average per `(chrom, pos, ref, alt)` before computing the
metric.

## Splits

| Split | Variants | Rows (2× variants) | Chromosomes |
|---|---:|---:|---|
| `train` | {train_n:,} | {train.height:,} | odd: 1, 3, …, X |
| `test` | {test_n:,} | {test.height:,} | even: 2, 4, …, Y |

## Eval-harness columns

In addition to the columns from [`{base_repo}`](https://huggingface.co/datasets/{base_repo}) (with `label` renamed to `target`):

| Column | Length ({window_size} bp window) | Description |
|---|---:|---|
| `context` | {half} bp | Left flank up to (but not including) the variant position, on the strand named in `strand`. |
| `ref_completion` | {window_size - half} bp | Reference allele (in-strand) + right flank. |
| `alt_completion` | {window_size - half} bp | Alternate allele (in-strand) + right flank. |
| `strand` | — | `"+"` (FWD) or `"-"` (RC of the FWD window; ref/alt complemented). |
| `target` | bool | Binary classification label (renamed from `label`; identical across the two strand rows). |

Rows are sorted by `(chrom, pos, ref, alt, strand)` so per-variant strand
pairs are adjacent.

Consumers that don't want RC averaging can filter to a single strand:

```python
ds = load_dataset("bolinas-dna/evals_mendelian_traits_harness_{window_size}", split="train")
fwd_only = ds.filter(lambda x: x["strand"] == "+")
```

## Provenance

Built by the [`marin-dna`]({REPO_ROOT_URL}) eval pipeline at commit
[`{sha[:7]}`]({REPO_ROOT_URL}/tree/{sha}/snakemake/evals).

- Sequence materialization: {_file_link(sha, "src/marin_dna/pipelines/evals/materialize.py")}
- Reference genome: GRCh38 `dna_sm` primary assembly, Ensembl release 115 (sequence is byte-identical to releases 113/114). Loaded directly from `s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz`.

For matching scheme, full column documentation, retention, and AUPRC-leak
diagnostic, see the companion raw dataset:
**[`{base_repo}`](https://huggingface.co/datasets/{base_repo})**.

## License

Same terms as the raw companion dataset.
"""


_DART_EVAL_META = {
    "caqtl": {
        "assay": "ATAC-seq chromatin accessibility",
        "qtl": "caQTL",
        "study": "African caQTLs, DeGorter *et al.* 2023 (bioRxiv)",
        "synapse": "syn60756043",
        "build": "GRCh38 (native)",
        "extra_tags": ["variant-effect-prediction", "chromatin-accessibility", "caqtl"],
    },
    "dsqtl": {
        "assay": "DNase-seq chromatin accessibility",
        "qtl": "dsQTL",
        "study": "Yoruban dsQTLs, Degner *et al.* *Nature* 2012",
        "synapse": "syn60756039",
        "build": "GRCh38 (lifted from hg19)",
        "extra_tags": ["variant-effect-prediction", "chromatin-accessibility", "dsqtl"],
    },
}


def render_dart_eval(
    dataset: str,
    sha: str,
    train_path: str | Path,
    test_path: str | Path,
) -> str:
    """Dataset card for the DART-Eval caQTL/dsQTL datasets.

    Unlike the matched datasets these are **not** matched or subsampled — all
    positives and negatives are kept at their natural ratio — so there is no
    retention or AUPRC-leak diagnostic (no `qc_path`).
    """
    m = _DART_EVAL_META[dataset]
    c = _split_counts(train_path, test_path)
    total = c["train_total"] + c["test_total"]
    pos = c["train_pos"] + c["test_pos"]
    neg = c["train_neg"] + c["test_neg"]
    pval_se_rows = (
        "| `pval` | float | caQTL association p-value (reference only) |\n"
        "| `se` | float | Standard error of the effect estimate (`beta`); reference only |\n"
        if dataset == "caqtl"
        else ""
    )
    return f"""{_frontmatter(m["extra_tags"])}

# evals_{dataset}

Variant-effect-prediction benchmark of **{m["qtl"]}s** ({m["assay"]}):
statistically significant chromatin-accessibility QTLs vs control variants,
both within accessible peaks. Sourced from
[DART-Eval](https://github.com/kundajelab/DART-Eval) Task 5 and brought into
the `marin-dna` evals pipeline with chromosome-based **train/test splits** for
model development.

**No matching and no subsampling** — every positive and negative is kept at
its natural ratio (≈1:{round(neg / pos) if pos else "?"} positive:negative).

## Description

| | |
|---|---|
| Positives | Significant {m["qtl"]}s within accessible peaks (`label = True`) |
| Negatives | Control variants in peaks, no significant association (`label = False`) |
| Assay | {m["assay"]} |
| Source study | {m["study"]} |
| Source data | DART-Eval, Synapse [`{m["synapse"]}`](https://www.synapse.org/Synapse:{
        m["synapse"]
    }) |
| Genome build | {m["build"]} |
| Variant type | SNVs only |
| Coordinates | 1-based (`pos` is 1-based; `ref`/`alt` are single bases) |
| Matching | none (natural class ratio) |

## Splits

Chromosome-parity split (same convention as the other `evals_*` datasets).

| Split | Variants | Positives | Negatives | Chromosomes |
|---|---:|---:|---:|---|
| `train` | {c["train_total"]:,} | {c["train_pos"]:,} | {
        c["train_neg"]:,} | odd: 1, 3, …, X |
| `test` | {c["test_total"]:,} | {c["test_pos"]:,} | {
        c["test_neg"]:,} | even: 2, 4, …, Y |
| **total** | **{total:,}** | **{pos:,}** | **{neg:,}** | |

## Columns

| Column | Type | Description |
|---|---|---|
| `chrom`, `pos`, `ref`, `alt` | str / int / str / str | Variant coordinates (1-based, GRCh38). `ref`/`alt` oriented against the reference. |
| `label` | bool | `True` for a significant {m["qtl"]}, `False` for a control variant |
| `effect` | float | **Signed** study effect (`{
        "beta" if dataset == "caqtl" else "obs.estimate"
    }`), **oriented to the `alt` allele** — positive ⇒ `alt` increases accessibility; sign-flipped for variants whose ref/alt were swapped to match the reference{
        ", and present for positives only" if dataset == "dsqtl" else ""
    }. This is the value the Pearson correlation uses. |
| `effect_size` | float | Unsigned effect magnitude (absolute value of `effect`). |
{
        pval_se_rows
    }| `consequence`, `consequence_cre`, `consequence_final` | str | Ensembl VEP consequence (raw, with-CRE-class, and after TSS/exon-proximity recategorization); reference annotations |
| `distance_tss_pc`, `distance_tss_nc`, `distance_tss` | int | Distance to nearest protein-coding / non-protein-coding TSS, and the minimum |
| `tss_closest_pc_gene_id`, `tss_closest_nc_gene_id`, `tss_closest_gene_id` | str | Ensembl gene IDs at those distances |
| `distance_exon_pc`, `distance_exon_nc`, `distance_exon` | int | Same shape, for nearest exon |
| `exon_closest_pc_gene_id`, `exon_closest_nc_gene_id`, `exon_closest_gene_id` | str | Same shape |

The consequence/distance columns are **reference annotations only** — they are
not used to match or filter the variant set.

## Evaluation

Following [DART-Eval](https://github.com/kundajelab/DART-Eval) Task 5 (§4.5) and
[ARSENAL](https://www.biorxiv.org/content/10.64898/2026.02.05.703637v1.full),
score each variant with a model's allelic effect (e.g. an alt-vs-ref
log-likelihood ratio) and report two complementary metrics, **each over a
different variant set**:

- **Classification — AUROC / AUPRC over _all_ variants:** significant
  {m["qtl"]}s (`label = True`) vs control variants (`label = False`).
- **Pearson correlation over the _positive_ variants only** (`label = True`):
  between `effect` (signed, oriented to `alt`) and the model's signed
  alt-vs-ref score. DART-Eval Table 6: *"for the positive variants, we computed
  the correlation between measured and predicted allelic effects."* Controls
  are not used in the correlation{
        " (and for dsQTL they have no measured effect_size)"
        if dataset == "dsqtl"
        else ""
    }.

## Provenance

Built by the [`marin-dna`]({REPO_ROOT_URL}) eval pipeline at commit
[`{sha[:7]}`]({REPO_ROOT_URL}/tree/{sha}/snakemake/evals).

- Curation pipeline: {_pipeline_link(sha)}
- Rules: {_file_link(sha, "snakemake/evals/workflow/rules/dart_eval.smk")}
- Parsing + annotation: {_file_link(sha, "src/marin_dna/pipelines/evals/dart_eval.py")}

## License

Released under the terms of its upstream sources. The variant set is
redistributed from [DART-Eval](https://github.com/kundajelab/DART-Eval) (Task 5)
and derives from the original QTL study ({m["study"]}); DART-Eval ships no
explicit license, so consult it and the source study for redistribution and
commercial-use terms.

## Citation

If you use this benchmark, please cite the upstream sources:

- DART-Eval — Patel *et al.* 2024, [arXiv 2412.05430](https://arxiv.org/abs/2412.05430) (NeurIPS D&B 2024)
- {m["qtl"]} study — {m["study"]}
"""


# Per-study metadata for the SGE dataset card, keyed by the per-variant
# `mavedb_urn` accession. Extend as more genes are added. `pmid` may be None for
# MaveDB deposits without a linked publication.
_SGE_STUDY_META = {
    "urn:mavedb:00000097-0-2": {
        "gene": "BRCA1",
        "study": "Findlay et al. 2018, *Nature* 562:217–222",
        "pmid": "30209399",
        "build": "hg19 → GRCh38 (lifted)",
        "score_col": "`author_function_score_mean`",
        "class_col": "`author_func_class` (FUNC / INT / LOF)",
    },
    "urn:mavedb:00001250-a-1": {
        "gene": "BARD1",
        "study": "Saturation genome editing of BARD1 (medRxiv 2025)",
        "pmid": None,
        "build": "GRCh38",
        "score_col": "`author_score`",
        "class_col": "—",
    },
    "urn:mavedb:00001259-a-1": {
        "gene": "PALB2",
        "study": "Saturation genome editing of PALB2 (MaveDB)",
        "pmid": None,
        "build": "GRCh38",
        "score_col": "`author_score`",
        "class_col": "—",
    },
    "urn:mavedb:00001260-a-1": {
        "gene": "RAD51D",
        "study": "Saturation genome editing of RAD51D (MaveDB)",
        "pmid": None,
        "build": "GRCh38",
        "score_col": "`author_score`",
        "class_col": "—",
    },
    "urn:mavedb:00001264-a-1": {
        "gene": "XRCC2",
        "study": "Saturation genome editing of XRCC2 (MaveDB)",
        "pmid": None,
        "build": "GRCh38",
        "score_col": "`author_score`",
        "class_col": "`author_functional_consequence`",
    },
    "urn:mavedb:00001262-a-1": {
        "gene": "CTCF",
        "study": "Saturation genome editing of CTCF (MaveDB)",
        "pmid": None,
        "build": "GRCh38",
        "score_col": "`author_score`",
        "class_col": "—",
    },
    "urn:mavedb:00001265-a-1": {
        "gene": "SFPQ",
        "study": "Saturation genome editing of SFPQ (MaveDB)",
        "pmid": None,
        "build": "GRCh38",
        "score_col": "`author_score`",
        "class_col": "—",
    },
    # Transcript-targeted (c.->g. recoded with pyhgvs + cdot; intronic recovered).
    "urn:mavedb:00001225-a-1": {
        "gene": "BRCA2", "study": "Huang et al. 2025, *Nature* 638:528–537",
        "pmid": "39779857", "build": "GRCh38 (c.→g.)", "score_col": "`author_score`",
        "class_col": "—",
    },
    "urn:mavedb:00000673-0-1": {
        "gene": "RAD51C", "study": "Saturation genome editing of RAD51C (2024)",
        "pmid": "39299233", "build": "GRCh38 (c.→g.)", "score_col": "`author_score`",
        "class_col": "`author_functional_classification`",
    },
    "urn:mavedb:00000662-0-1": {
        "gene": "BAP1", "study": "Waters et al. 2024 — BAP1 SGE",
        "pmid": "38969833", "build": "GRCh38 (c.→g.)", "score_col": "`author_score`",
        "class_col": "—",
    },
    "urn:mavedb:00000658-0-1": {
        "gene": "DDX3X", "study": "Saturation genome editing of DDX3X (2023)",
        "pmid": "38057330", "build": "GRCh38 (c.→g.)", "score_col": "`author_score`",
        "class_col": "—",
    },
    "urn:mavedb:00000675-a-1": {
        "gene": "VHL", "study": "VHL SGE — functional spectrum (2024)",
        "pmid": "38969834", "build": "GRCh38 (c.→g.)", "score_col": "`author_score`",
        "class_col": "—",
    },
    "urn:mavedb:00001226-a-1": {
        "gene": "CARD11", "study": "Multiplexed functional assessment of CARD11 (2020)",
        "pmid": "33202260", "build": "GRCh38 (c.→g.)", "score_col": "`author_score`",
        "class_col": "— (codon-level study; SNV subset)",
    },
}


def render_sge(
    dataset: str,
    sha: str,
    train_path: str | Path,
    test_path: str | Path,
) -> str:
    """Dataset card for the SGE (saturation genome editing) dataset.

    Each row is one assayed SNV with an experimental function score. There is no
    matching/subsampling and no binary `label` (so no pos/neg counts); the
    HIGH-impact `exclude_consequences` are dropped; every original author column is
    preserved under an `author_` prefix. Provenance is per-variant `(gene,
    mavedb_urn)`; per-study citation comes from `_SGE_STUDY_META`.
    """
    train = pl.read_parquet(train_path)
    test = pl.read_parquet(test_path)
    allv = pl.concat([train, test], how="vertical_relaxed")
    total = allv.height
    counts = dict(allv.group_by("mavedb_urn").len().iter_rows())  # urn -> n_variants

    study_rows = []
    for urn, n in sorted(counts.items()):
        m = _SGE_STUDY_META.get(urn, {})
        study_rows.append(
            f"| {m.get('gene', '?')} | [`{urn}`](https://www.mavedb.org/score-sets/{urn}) "
            f"| {m.get('study', '—')} | {m.get('build', '—')} | {n:,} "
            f"| {m.get('score_col', '—')} | {m.get('class_col', '—')} |"
        )
    studies_table = "\n".join(study_rows)
    n_author = sum(c.startswith("author_") for c in allv.columns)

    def _cite(urn: str, m: dict) -> str:
        pmid = (
            f" (PMID [{m['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{m['pmid']}/))"
            if m.get("pmid")
            else ""
        )
        return (
            f"- {m['gene']} — {m['study']}{pmid}; "
            f"MaveDB [`{urn}`](https://www.mavedb.org/score-sets/{urn})"
        )

    citations = "\n".join(
        _cite(u, _SGE_STUDY_META[u]) for u in sorted(counts) if u in _SGE_STUDY_META
    )

    # Minimal tag set (biology, genomics, dna) per the bolinas-dna dataset-card
    # convention — no fine-grained extras.
    return f"""{_frontmatter()}

# evals_sge

Variant-effect-prediction benchmark of **saturation genome editing (SGE)** function
scores. SGE edits the *endogenous genomic locus* (CRISPR-HDR, typically in haploid
HAP1 cells), so every assayed SNV has a **direct experimental functional measurement**
in genomic coordinates — an axis orthogonal to the clinical/population/statistical
labels of the other `evals_*` datasets, and one that covers near-exon noncoding
(splice-region, proximal-intronic) SNVs, not just missense.

**No matching, no subsampling, no binary label** — every assayed SNV is kept with its
continuous author score(s). The trivially-deleterious **HIGH-impact consequences**
(canonical splice, nonsense, frameshift, …) are **dropped** (`exclude_consequences`);
they are not the discriminative signal an SGE benchmark is about. **Every original
author column is preserved** under an `author_` prefix ({n_author} columns), so no
source metadata is lost.

## Studies

One row per (gene × study). `mavedb_urn` is stamped on every variant so `(gene,
mavedb_urn)` identifies the exact source.

| Gene | MaveDB accession | Study | Build | Variants | Function score | Classification |
|---|---|---|---:|---:|---|---|
{studies_table}

## Splits

Chromosome-parity split (same convention as the other `evals_*` datasets): odd
chromosomes + X → `train`, even + Y → `test`. SGE loci sit on whole chromosomes, so
this is a **gene-level holdout** (e.g. BRCA1·chr17 → train).

| Split | Variants | Chromosomes |
|---|---:|---|
| `train` | {train.height:,} | odd: 1, 3, …, X |
| `test` | {test.height:,} | even: 2, 4, …, Y |
| **total** | **{total:,}** | |

## Columns

| Column | Type | Description |
|---|---|---|
| `chrom`, `pos`, `ref`, `alt` | str / int / str / str | Variant coordinates (1-based, **GRCh38**). |
| `gene` | str | Gene symbol. |
| `assay` | str | `sge`. |
| `mavedb_urn` | str | Canonical MaveDB accession for the source study (see the table above). |
| `author_*` | mixed | **Every original column from the source study, verbatim** (slugified, `author_`-prefixed). The headline variables per study are listed in the table above — e.g. for BRCA1 `author_function_score_mean` (continuous) and `author_func_class` (FUNC/INT/LOF). Original coordinates are kept too (e.g. `author_position_hg19`). |
| `consequence`, `consequence_cre`, `consequence_final` | str | Ensembl VEP consequence (raw, with-CRE-class, and after TSS/exon-proximity recategorization); reference annotations. |
| `distance_tss_*`, `distance_exon_*`, `*_closest_gene_id` | int / str | Distances to nearest TSS / exon and the Ensembl gene IDs there; reference annotations. |

No binary `label` is imposed — that (and any score harmonization across studies) is an
eval-time decision; the artifact preserves the authors' continuous scores and discrete
classes as-is.

## Provenance

Built by the [`marin-dna`]({REPO_ROOT_URL}) eval pipeline at commit
[`{sha[:7]}`]({REPO_ROOT_URL}/tree/{sha}/snakemake/evals).

- Curation pipeline: {_pipeline_link(sha)}
- Rules: {_file_link(sha, "snakemake/evals/workflow/rules/sge.smk")}
- Loading + annotation: {_file_link(sha, "src/marin_dna/pipelines/evals/sge.py")}

## License

Released under the terms of its upstream sources; consult each source study (below)
for redistribution and commercial-use terms.

## Citation

If you use this benchmark, please cite the source SGE studies:

{citations}
"""


def render(
    dataset: str,
    sha: str,
    train_path: str | Path,
    test_path: str | Path,
    qc_path: str | Path | None = None,
) -> str:
    if dataset == "mendelian_traits":
        assert qc_path is not None
        return render_mendelian(sha, train_path, test_path, qc_path)
    if dataset == "complex_traits":
        assert qc_path is not None
        return render_complex(sha, train_path, test_path, qc_path)
    if dataset.startswith("mendelian_traits_harness_"):
        window = int(dataset.rsplit("_", 1)[1])
        return render_harness(sha, train_path, test_path, window_size=window)
    if dataset in _DART_EVAL_META:
        return render_dart_eval(dataset, sha, train_path, test_path)
    if dataset == "sge":
        return render_sge(dataset, sha, train_path, test_path)
    raise ValueError(f"no README template for dataset {dataset!r}")
