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


_CHROMBPNET_QTL_META = {
    "caqtl": {
        "assay": "ATAC-seq chromatin accessibility",
        "qtl": "caQTL",
        "study": "African caQTLs (DeGorter *et al.* 2023)",
        "synapse": "syn64126781",
        "build": "GRCh38 (native)",
        "effect_src": "obs.beta",
        "extra_tags": ["variant-effect-prediction", "chromatin-accessibility", "caqtl"],
    },
    "dsqtl": {
        "assay": "DNase-seq chromatin accessibility",
        "qtl": "dsQTL",
        "study": "Yoruba dsQTLs (Degner *et al.* *Nature* 2012)",
        "synapse": "syn64126779",
        "build": "GRCh38 (lifted from hg19)",
        "effect_src": "obs.estimate",
        "extra_tags": ["variant-effect-prediction", "chromatin-accessibility", "dsqtl"],
    },
}

# Carried precomputed baseline scores (canonical column -> card description). Whichever
# are present in the produced parquet are listed in the Columns table. All are signed
# alt-vs-ref allelic scores (sign-flipped alongside `effect` on a ref/alt swap).
_CARRIED_SCORE_DESC = {
    "chrombpnet_atac_ips": "ChromBPNet GM12878 **ATAC** IPS (= logFC × JSD × AAQ) — the recommended **causality** score",
    "chrombpnet_atac_logfc": "ChromBPNet GM12878 **ATAC** log2 fold-change — the recommended **direction** score",
    "chrombpnet_dnase_ips": "ChromBPNet GM12878 **DNase** IPS",
    "chrombpnet_dnase_logfc": "ChromBPNet GM12878 **DNase** log2 fold-change",
    "enformer_dnase_local_logfc": "Enformer GM12878 DNase recomputed local-2 kb log2 fold-change",
}


def render_chrombpnet_qtl(
    dataset: str,
    sha: str,
    train_path: str | Path,
    test_path: str | Path,
) -> str:
    """Dataset card for the standardized caQTL/dsQTL accessibility-QTL benchmarks.

    The canonical supervised official-metrics variant sets (#309/#310/#262), sourced
    from the standardized ChromBPNet benchmark files used by both the ChromBPNet and
    AlphaGenome papers. Each row carries precomputed ChromBPNet/Enformer baseline
    scores so those baselines are free downstream. Unmatched / unsubsampled (natural
    class ratio) — no retention or AUPRC-leak diagnostic (no `qc_path`).
    """
    m = _CHROMBPNET_QTL_META[dataset]
    c = _split_counts(train_path, test_path)
    total = c["train_total"] + c["test_total"]
    pos = c["train_pos"] + c["test_pos"]
    neg = c["train_neg"] + c["test_neg"]
    present = pl.read_parquet_schema(str(train_path))
    carried_rows = "".join(
        f"| `{col}` | float | {desc} |\n"
        for col, desc in _CARRIED_SCORE_DESC.items()
        if col in present
    )
    return f"""{_frontmatter(m["extra_tags"])}

# evals_{dataset}

Supervised variant-effect-prediction benchmark of **{m["qtl"]}s** ({m["assay"]}):
statistically significant chromatin-accessibility QTLs vs control variants. These
are the **standardized** ChromBPNet benchmark variant sets (Synapse
[`syn64126763`](https://www.synapse.org/Synapse:syn64126763)) used by *both* the
ChromBPNet and AlphaGenome papers, brought into the `marin-dna` evals pipeline with
chromosome-parity **train/test splits**. Each row carries **precomputed
ChromBPNet/Enformer baseline scores**, so those baselines reproduce without running
any model.

**No matching and no subsampling** — every positive and negative is kept at its
natural ratio (≈1:{round(neg / pos) if pos else "?"} positive:negative).

## Description

| | |
|---|---|
| Positives | Significant {m["qtl"]}s (`label = True`; −log10p > 5) |
| Negatives | Control variants (`label = False`; −log10p < 3) |
| Assay | {m["assay"]} (GM12878 / LCL) |
| Source study | {m["study"]} |
| Source data | Standardized ChromBPNet benchmark, Synapse [`{
        m["synapse"]
    }`](https://www.synapse.org/Synapse:{m["synapse"]}) |
| Genome build | {m["build"]} |
| Variant type | SNVs only |
| Coordinates | 1-based (`pos` is 1-based; `ref`/`alt` are single bases) |
| Matching | none (natural class ratio) |

## Splits

Chromosome-parity split (same convention as the other `evals_*` datasets).
Chromosome subsets used by the official benchmark — all chroms, the AlphaGenome
test chroms {{3, 6, 9, 12, 16, 18, 19, 21}}, or this train/test parity — are a
**scoring-time slice** of the full variant set, not a re-score.

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
| `chrom`, `pos`, `ref`, `alt` | str / int / str / str | Variant coordinates (1-based, GRCh38). `ref`/`alt` are genome-oriented (`alt` ≠ reference). |
| `label` | bool | `True` for a significant {m["qtl"]}, `False` for a control variant |
| `effect` | float | **Signed** study effect (`{
        m["effect_src"]
    }`), **oriented to the `alt` allele** — positive ⇒ `alt` increases accessibility; sign-flipped for variants whose ref/alt were swapped to match the reference. Present for every positive; controls may be null. |
{carried_rows}
The carried `chrombpnet_*` / `enformer_*` columns are the papers' precomputed
per-variant baseline scores (signed, oriented to `alt`) — supplied so the
ChromBPNet and Enformer baselines need no model run.

## Evaluation

Two official metrics (the ChromBPNet/AlphaGenome accessibility-QTL protocol; see
issue #262), **each over a different variant set**:

- **Causality — AUPRC over _all_ variants** on `|score|`: significant {m["qtl"]}s
  (`label = True`) vs controls (`label = False`).
- **Direction — Pearson over the _positive_ variants only** (`label = True`):
  between `effect` (signed, oriented to `alt`) and the model's signed alt-vs-ref
  score. Controls are not used in the correlation.

ChromBPNet uses **IPS** for causality and **logFC** for direction; AlphaGenome /
Enformer use signed accessibility log-fold-change. Metrics are computed by
`marin_dna.pipelines.chrombpnet_eval.metrics.compute_supervised_qtl_metrics`.

## Provenance

Built by the [`marin-dna`]({REPO_ROOT_URL}) eval pipeline at commit
[`{sha[:7]}`]({REPO_ROOT_URL}/tree/{sha}/snakemake/evals).

- Curation pipeline: {_pipeline_link(sha)}
- Rules: {_file_link(sha, "snakemake/evals/workflow/rules/chrombpnet_qtl.smk")}
- Build + parsing: {_file_link(sha, "src/marin_dna/pipelines/evals/chrombpnet_qtl.py")}

## License

Released under the terms of its upstream sources. The variant set is redistributed
from the standardized ChromBPNet benchmark (Synapse `syn64126763`) and derives from
the original QTL study ({m["study"]}); consult those sources for redistribution and
commercial-use terms.

## Citation

If you use this benchmark, please cite the upstream sources:

- ChromBPNet — Pampari *et al.* 2024, [bioRxiv 2024.12.25.630221](https://www.biorxiv.org/content/10.1101/2024.12.25.630221)
- AlphaGenome — Avsec *et al.* 2025
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
        "class_col": "`author_functional_consequence`",
    },
    "urn:mavedb:00001259-a-1": {
        "gene": "PALB2",
        "study": "Saturation genome editing of PALB2 (MaveDB)",
        "pmid": None,
        "build": "GRCh38",
        "score_col": "`author_score`",
        "class_col": "`author_functional_consequence`",
    },
    "urn:mavedb:00001260-a-1": {
        "gene": "RAD51D",
        "study": "Saturation genome editing of RAD51D (MaveDB)",
        "pmid": None,
        "build": "GRCh38",
        "score_col": "`author_score`",
        "class_col": "`author_functional_consequence`",
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
        "class_col": "`author_functional_consequence`",
    },
    "urn:mavedb:00001265-a-1": {
        "gene": "SFPQ",
        "study": "Saturation genome editing of SFPQ (MaveDB)",
        "pmid": None,
        "build": "GRCh38",
        "score_col": "`author_score`",
        "class_col": "`author_functional_consequence`",
    },
    # Transcript-targeted (c.->g. recoded with pyhgvs + cdot; intronic recovered).
    "urn:mavedb:00001225-a-1": {
        "gene": "BRCA2",
        "study": "Huang et al. 2025, *Nature* 638:528–537",
        "pmid": "39779857",
        "build": "GRCh38 (c.→g.)",
        "score_col": "`author_score`",
        "class_col": "—",
    },
    "urn:mavedb:00000673-0-1": {
        "gene": "RAD51C",
        "study": "Saturation genome editing of RAD51C (2024)",
        "pmid": "39299233",
        "build": "GRCh38 (c.→g.)",
        "score_col": "`author_score`",
        "class_col": "`author_functional_classification`",
    },
    "urn:mavedb:00000662-0-1": {
        "gene": "BAP1",
        "study": "Waters et al. 2024 — BAP1 SGE",
        "pmid": "38969833",
        "build": "GRCh38 (c.→g.)",
        "score_col": "`author_score`",
        "class_col": "—",
    },
    "urn:mavedb:00000658-0-1": {
        "gene": "DDX3X",
        "study": "Saturation genome editing of DDX3X (2023)",
        "pmid": "38057330",
        "build": "GRCh38 (c.→g.)",
        "score_col": "`author_score`",
        "class_col": "`author_sge_prediction_of_variant_function_in_ndd_context`",
    },
    "urn:mavedb:00000675-a-1": {
        "gene": "VHL",
        "study": "VHL SGE — functional spectrum (2024)",
        "pmid": "38969834",
        "build": "GRCh38 (c.→g.)",
        "score_col": "`author_score`",
        "class_col": "—",
    },
}


# Curated subset of the MaveDB assay-fact keyword columns surfaced in the card's
# "Assay characteristics" table (display label -> assay_ column). The *full*
# keyword set is preserved as assay_ columns on the dataset; this is just the
# decision-relevant slice shown inline.
_SGE_ASSAY_FACT_COLS = [
    ("Assay readout", "assay_phenotypic_assay_method"),
    ("Mechanism", "assay_phenotypic_assay_mechanism"),
    ("Molecular mechanism", "assay_molecular_mechanism_assessed"),
    ("Model system", "assay_phenotypic_assay_model_system"),
    ("Library mechanism", "assay_endogenous_locus_library_method_mechanism"),
]

# ACMG functional-evidence strengths, weakest -> strongest, for ordering.
_ACMG_STRENGTH_ORDER = {
    s: i
    for i, s in enumerate(
        ["SUPPORTING", "MODERATE", "MODERATE_PLUS", "STRONG", "VERY_STRONG"]
    )
}


def _render_assay_characteristics(allv: pl.DataFrame) -> str:
    """'Assay characteristics' card section from the per-variant ``assay_*`` keyword
    columns (one summary row per gene). Empty string if no ``assay_*`` columns."""
    present = [(lbl, col) for lbl, col in _SGE_ASSAY_FACT_COLS if col in allv.columns]
    if not present:
        return ""
    cols = [col for _, col in present]
    # assay_ values are constant per gene (joined by accession), so first-non-null
    # collapses each gene to one row.
    by_gene = (
        allv.group_by("gene")
        .agg([pl.col(c).drop_nulls().first().alias(c) for c in cols])
        .sort("gene")
    )
    header = "| Gene | " + " | ".join(lbl for lbl, _ in present) + " |"
    sep = "|---|" + "|".join("---" for _ in present) + "|"
    rows = [header, sep]
    for r in by_gene.iter_rows(named=True):
        cells = " | ".join(str(r[c]) if r[c] is not None else "—" for _, c in present)
        rows.append(f"| {r['gene']} | {cells} |")
    table = "\n".join(rows)
    return f"""## Assay characteristics

MaveDB annotates each experiment with controlled-vocabulary **assay facts**,
captured verbatim on every variant as constant-per-gene `assay_*` columns (the full
keyword set; the table surfaces the decision-relevant ones).

{table}

Every MaveDB-annotated study is a **loss-of-function** assay (the `Mechanism`
column): a depletion / fitness screen reading out a variant's effect on cell
survival or growth in **immortalized human cells**, with the variant library
written into the **endogenous locus** by a CRISPR **nuclease** (SpCas9).
Splice-disrupting and NMD-triggering variants are therefore captured **by
construction** — the endogenous readout reflects mis-splicing and
nonsense-mediated decay — so MaveDB exposes no separate "detects splicing / NMD"
flag. **BRCA2** (`urn:mavedb:00001225-a-1`) is the one study MaveDB leaves
unannotated, so its `assay_*` cells are blank.
"""


def _render_score_calibration(calibration_path: str | Path) -> str:
    """'Score calibration' card section from the long-format calibration companion
    table (per-gene scheme count + ACMG evidence strengths). Empty if no rows."""
    cal = pl.read_parquet(calibration_path)
    if cal.height == 0:
        return ""
    by_gene = (
        cal.group_by("gene")
        .agg(
            pl.col("calibration_title").n_unique().alias("n_schemes"),
            pl.col("acmg_evidence_strength").drop_nulls().unique().alias("strengths"),
        )
        .sort("gene")
    )
    rows = [
        "| Gene | Calibration schemes | ACMG evidence strengths |",
        "|---|---:|---|",
    ]
    for r in by_gene.iter_rows(named=True):
        ss = sorted(r["strengths"], key=lambda s: _ACMG_STRENGTH_ORDER.get(s, 99))
        rows.append(f"| {r['gene']} | {r['n_schemes']} | {', '.join(ss) or '—'} |")
    table = "\n".join(rows)
    n_schemes = cal.select("gene", "calibration_title").unique().height
    return f"""## Score calibration

MaveDB attaches **score calibrations** — threshold schemes mapping the continuous
function score onto functional classes. Two flavors appear here: **investigator-provided**
functional classes (the authors' own normal / abnormal cutoffs) and **ClinGen /
ExCALIBR ACMG calibrations** (clinically-calibrated thresholds that assign each score
bin an ACMG functional-evidence strength — `PS3` pathogenic / `BS3` benign, graded
`SUPPORTING` → `VERY_STRONG`, under a `prior_probability_pathogenicity` OddsPath
prior). {n_schemes} schemes are captured across the {by_gene.height} calibrated genes.

{table}

The full, tidy long-format calibration table — **one row per (gene × calibration ×
functional class)**, with the score range, variant count, GO call
(normal / abnormal / not_specified), ACMG criterion / strength / signed points, the
OddsPath prior, and threshold-source PMIDs — ships as **`calibrations.parquet`**
alongside the splits. **BRCA2** has no MaveDB calibrations.

<details>
<summary><code>calibrations.parquet</code> columns</summary>

| Column | Type | Description |
|---|---|---|
| `gene`, `mavedb_urn` | str | Study identifier (joins to the splits). |
| `calibration_title` | str | Scheme name (e.g. `Investigator-provided functional classes`, `ExCALIBR calibration`). |
| `research_use_only` | bool | MaveDB research-use-only flag for the scheme. |
| `baseline_score` | float | Scheme baseline (often the synonymous/normal anchor); null when not score-range based. |
| `prior_probability_pathogenicity` | float | OddsPath prior for the ACMG schemes; null otherwise. |
| `threshold_source_pmids` | str | Comma-joined PubMed IDs the thresholds derive from. |
| `class_label` | str | Functional class name (e.g. `Functional`, `PS3 Strong (5)`). |
| `go_classification` | str | `normal` / `abnormal` / `not_specified`. |
| `range_lower`, `range_upper` | float | Score range for the class (null = open / not score-range based). |
| `inclusive_lower`, `inclusive_upper` | bool | Whether the range bounds are inclusive. |
| `variant_count` | int | Variants MaveDB places in the class. |
| `acmg_criterion` | str | `PS3` (pathogenic) / `BS3` (benign); null for non-ACMG schemes. |
| `acmg_evidence_strength` | str | `SUPPORTING` … `VERY_STRONG`. |
| `acmg_points` | int | ExCALIBR signed evidence points (negative = benign). |

</details>
"""


def render_sge(
    dataset: str,
    sha: str,
    train_path: str | Path,
    test_path: str | Path,
    calibration_path: str | Path | None = None,
) -> str:
    """Dataset card for the SGE (saturation genome editing) dataset.

    Each row is one assayed SNV with an experimental function score and (v3, #301) a
    binary `label` (True = impactful / calibrated abnormal, False = normal). No
    matching/subsampling; the HIGH-impact `exclude_consequences` are dropped, the
    missense+splicing groups kept, and v3 keeps only label-non-null variants (drops
    intermediate + uncalibrated, incl. BRCA2). Every original author column is
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

    # Study-level MaveDB metadata sections (empty strings if the inputs lack the
    # columns / the calibration companion isn't passed).
    assay_section = _render_assay_characteristics(allv)
    calibration_section = (
        _render_score_calibration(calibration_path) if calibration_path else ""
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

**No matching, no subsampling.** The trivially-deleterious **HIGH-impact consequences**
(canonical splice, nonsense, frameshift, …) are dropped (`exclude_consequences`), and v2
([#297](https://github.com/Open-Athena/marin-dna/issues/297)) further keeps only the
**missense + splicing** consequence groups — the two where the SGE assay actually
measures function. **v3 ([#301](https://github.com/Open-Athena/marin-dna/issues/301))**
adds a binary **`label`** (`True` = impactful = calibrated `abnormal`, `False` =
`normal`) and keeps only label-non-null variants — dropping `intermediate` and
uncalibrated rows (all of BRCA2) — so this is a clean classification benchmark scored by
AUPRC. **Every original author column is preserved** under an `author_` prefix
({n_author} columns), so no source metadata is lost.

## Studies

One row per (gene × study). `mavedb_urn` is stamped on every variant so `(gene,
mavedb_urn)` identifies the exact source.

| Gene | MaveDB accession | Study | Build | Variants | Function score | Classification |
|---|---|---|---:|---:|---|---|
{studies_table}

{assay_section}
{calibration_section}
## Splits

Chromosome-parity split (same convention as the other `evals_*` datasets): odd
chromosomes + X → `train`, even + Y → `test`. SGE loci sit on whole chromosomes, so
this is a **gene-level holdout** (e.g. BRCA1·chr17 → train).

| Split | Variants | Abnormal | Normal | Chromosomes |
|---|---:|---:|---:|---|
| `train` | {train.height:,} | {train.filter(pl.col("label")).height:,} | {train.filter(~pl.col("label")).height:,} | odd: 1, 3, …, X |
| `test` | {test.height:,} | {test.filter(pl.col("label")).height:,} | {test.filter(~pl.col("label")).height:,} | even: 2, 4, …, Y |
| **total** | **{total:,}** | **{allv.filter(pl.col("label")).height:,}** | **{allv.filter(~pl.col("label")).height:,}** | |

## Columns

| Column | Type | Description |
|---|---|---|
| `chrom`, `pos`, `ref`, `alt` | str / int / str / str | Variant coordinates (1-based, **GRCh38**). |
| `gene` | str | Gene symbol. |
| `assay` | str | `sge`. |
| `mavedb_urn` | str | Canonical MaveDB accession for the source study (see the table above). |
| `label` | bool | **The AUPRC target (v3, [#301](https://github.com/Open-Athena/marin-dna/issues/301)):** `True` = impactful (calibrated `abnormal`), `False` = `normal`. The build keeps only label-non-null rows, so this is always a clean bool. |
| `function_score` | float | Each study's headline continuous functional score (raw, **per-study scale** — pool by rank, not raw value). |
| `function_direction`, `function_score_aligned` | int / float | Per-gene assay direction (`+1` / `−1`, null if unresolved — BRCA2) and the direction-corrected score (`function_direction × function_score`) so "higher = more functional" holds across genes. Sourced from the assay (calibration ranges / author class), **not** the model. |
| `calibrated_class` | str | ClinGen/ExCALIBR-calibrated `abnormal` / `intermediate` / `normal` (or null), decided at build with an **ExCALIBR-first policy** (prefers the live calibration; never a dated ClinVar snapshot). |
| `calibration_scheme`, `acmg_strength` | str | The scheme behind `calibrated_class` (an ExCALIBR / investigator title, or `author_class` for the categorical-only gene) and the matched range's ACMG functional-evidence strength (`SUPPORTING` … `VERY_STRONG`). |
| `author_class_harmonized` | str | Each study's discrete functional class mapped to a common `abnormal` / `intermediate` / `normal` axis (null where a study ships none — BAP1, VHL, BRCA2). |
| `author_*` | mixed | **Every original column from the source study, verbatim** (slugified, `author_`-prefixed). The headline variables per study are listed in the table above — e.g. for BRCA1 `author_function_score_mean` (continuous) and `author_func_class` (FUNC/INT/LOF). Original coordinates are kept too (e.g. `author_position_hg19`). |
| `assay_*` | str | **MaveDB 'assay facts'** — the experiment's controlled-vocabulary keywords (assay readout, mechanism, model system, library mechanism, …), constant per gene. See *Assay characteristics* below; blank for the one unannotated study (BRCA2). |
| `subset`, `consequence_group` | str | Coarse consequence grouping the eval stratifies on (`subset` is an alias of `consequence_group`, matching the other `evals_*` datasets). The build keeps only the `missense_variant` + `splicing` groups. |
| `consequence`, `consequence_cre`, `consequence_final` | str | Ensembl VEP consequence (raw, with-CRE-class, and after TSS/exon-proximity recategorization); reference annotations. |
| `distance_tss_*`, `distance_exon_*`, `*_closest_gene_id` | int / str | Distances to nearest TSS / exon and the Ensembl gene IDs there; reference annotations. |

v3 ([#301](https://github.com/Open-Athena/marin-dna/issues/301)) derives the binary
`label` from `calibrated_class` (abnormal → `True`, normal → `False`) and filters to
label-non-null rows. The harmonized continuous columns v2 added —
`function_direction` / `function_score_aligned`, `calibrated_class` with its
`calibration_scheme` + `acmg_strength`, and `author_class_harmonized` — are kept for
provenance (the AUPRC eval reads only `label`). The authors' raw `function_score` and
every `author_` column are preserved verbatim; **per-study scales still differ**, so any
continuous re-analysis should pool by rank, not raw value.

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


def render_dart_task3(
    sha: str,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
) -> str:
    """Dataset card for the DART-Eval Task 3 cell-type-specific peak dataset.

    An **interval** dataset (not variants): each row is a 500 bp ATAC-seq
    consensus-peak window labeled by the cell type it is differentially
    accessible in (5-class). No ref/alt, no consequence annotation, no
    matching/subsampling. Split by file on DART-Eval's canonical 3-way
    chromosome holdout, so this takes a `validation_path` the generic `render()`
    (train/test only) can't supply — the upload rule calls it directly.
    """
    splits = {
        "train": pl.read_parquet(train_path),
        "validation": pl.read_parquet(validation_path),
        "test": pl.read_parquet(test_path),
    }
    totals = {s: df.height for s, df in splits.items()}
    total = sum(totals.values())
    cell_types = sorted(
        set().union(*(set(df["label"].unique().to_list()) for df in splits.values()))
    )

    def _n(split: str, ct: str) -> int:
        return splits[split].filter(pl.col("label") == ct).height

    ct_rows = "\n".join(
        f"| `{ct}` | {_n('train', ct):,} | {_n('validation', ct):,} "
        f"| {_n('test', ct):,} | {_n('train', ct) + _n('validation', ct) + _n('test', ct):,} |"
        for ct in cell_types
    )

    # Minimal tag set (biology, genomics, dna) per the bolinas-dna dataset-card
    # convention — no fine-grained extras.
    return f"""{_frontmatter()}

# evals_dart_task3

Cell-type-specific **chromatin-accessibility peak** dataset from
[DART-Eval](https://github.com/kundajelab/DART-Eval) **Task 3** ("Discriminating
Cell-Type-Specific Elements"). Each row is a **500 bp** ATAC-seq consensus-peak
window (±250 bp around the summit, GRCh38), labeled by the **cell type** it is
differentially accessible in. The benchmark question: can a model embedding
distinguish cell types from the sequence alone?

**Interval dataset, not variants** — no ref/alt, no consequence annotation, **no
matching and no subsampling**. The window set is DART-Eval's
`input_data/top_5000_deseq_peaks.tsv` — the **top 5,000 DESeq2
differentially-accessible peaks per cell type** (the subset they feed their
zero-shot clustering / UMAP), **25,000 windows, 5,000 balanced per cell type**,
with unique peak coordinates.

## Description

| | |
|---|---|
| Element | 500 bp ATAC-seq consensus peak (±250 bp around the summit; window midpoint = summit) |
| Label | One of 5 cell lines: `GM12878`, `H1ESC`, `HEPG2`, `IMR90`, `K562` |
| Selection | Top 5,000 DESeq2 differentially-accessible peaks per cell type (25,000 total, balanced) |
| Assay | ATAC-seq chromatin accessibility (ENCODE) |
| Source data | DART-Eval, Synapse [`syn62161401`](https://www.synapse.org/Synapse:syn62161401) (`top_5000_deseq_peaks.tsv`), project `syn60581042` |
| Genome build | GRCh38 |
| Coordinates | **0-based, half-open** (`end - start == 500`) |
| Matching | none (no subsampling) |

The full 500 bp peak is stored (never pre-cropped to a model's context window),
so the embedding context / pooling choice stays an open downstream decision.

## Splits

DART-Eval's canonical **3-way** chromosome holdout (verbatim from their Task-3
training scripts), shipped one file per split. Per-split counts follow the
peaks' genomic distribution.

| File | Windows | Chromosomes |
|---|---:|---|
| `train.parquet` | {totals["train"]:,} | 1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, X, Y |
| `validation.parquet` | {totals["validation"]:,} | 6, 21 |
| `test.parquet` | {totals["test"]:,} | 5, 10, 14, 18, 20, 22 |
| **total** | **{total:,}** | |

### Windows per cell type

| Cell type | train | validation | test | total |
|---|---:|---:|---:|---:|
{ct_rows}

## Columns

| Column | Type | Description |
|---|---|---|
| `chrom` | str | Chromosome (no `chr` prefix), GRCh38 |
| `start` | int | Window start — 0-based, inclusive |
| `end` | int | Window end — 0-based, exclusive (`end - start == 500`) |
| `label` | str | Cell type the peak is differentially accessible in |

## Provenance

Built by the [`marin-dna`]({REPO_ROOT_URL}) eval pipeline at commit
[`{sha[:7]}`]({REPO_ROOT_URL}/tree/{sha}/snakemake/evals).

- Curation pipeline: {_pipeline_link(sha)}
- Rules: {_file_link(sha, "snakemake/evals/workflow/rules/dart_task3.smk")}
- Parsing + splitting: {_file_link(sha, "src/marin_dna/pipelines/evals/dart_task3.py")}

## License

Released under the terms of its upstream sources. The peak set is redistributed
from [DART-Eval](https://github.com/kundajelab/DART-Eval) (Task 3) and derives
from **ENCODE** ATAC-seq in the 5 cell lines (freely redistributable under the
[ENCODE data-use policy](https://www.encodeproject.org/help/citing-encode/));
DART-Eval ships no explicit license, so consult it and ENCODE for redistribution
and commercial-use terms.

## Citation

If you use this benchmark, please cite the upstream sources:

- DART-Eval — Patel *et al.* 2024, [arXiv 2412.05430](https://arxiv.org/abs/2412.05430) (NeurIPS D&B 2024)
- ENCODE Project Consortium — the ATAC-seq data for GM12878, H1ESC, HEPG2, IMR90, K562
"""


def render(
    dataset: str,
    sha: str,
    train_path: str | Path,
    test_path: str | Path,
    qc_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
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
    if dataset in _CHROMBPNET_QTL_META:
        return render_chrombpnet_qtl(dataset, sha, train_path, test_path)
    if dataset == "sge":
        return render_sge(
            dataset, sha, train_path, test_path, calibration_path=calibration_path
        )
    raise ValueError(f"no README template for dataset {dataset!r}")
