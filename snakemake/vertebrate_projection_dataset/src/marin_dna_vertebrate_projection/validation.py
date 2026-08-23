"""Build per-recipe validation parquets for ``vertebrate_projection_dataset``.

Seven human-only recipes (val_cds, val_utr5, val_utr3, val_ncrna, val_tss_pc,
val_promoter, val_enhancer): tile a region BED into 255 bp non-overlapping
windows, conservation-pre-filter via phyloP_447m, deterministic subsample,
then case-encode sequences (uppercase iff phyloP >= threshold else lowercase;
NaN positions are lowercase).

The annotation-derived recipes (val_cds, val_utr5, val_utr3, val_ncrna,
val_tss_pc) are restricted to canonical transcripts (``tag "Ensembl_canonical"``);
val_promoter and val_enhancer use ENCODE cCRE classes and are
transcript-independent.

Companion to ``snakemake/vertebrate_projection_dataset/workflow/rules/validation.smk``.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pyBigWig

from marin_dna.data.intervals import GenomicSet
from marin_dna.data.utils import (
    ENHANCER_CRE_CLASSES,
    get_cds,
    get_promoters_from_exons,
    load_fasta,
)
from marin_dna_vertebrate_projection import (
    _GITHUB_PIPELINE_PATH,
    _GITHUB_REPO,
)
from marin_dna_vertebrate_projection.projection.tss import (
    get_ensembl_protein_coding_exons,
)

# ============================================================================
# Canonical transcript filter
# ============================================================================


def filter_to_canonical_transcripts(
    ann: pl.DataFrame, *, tag: str = "Ensembl_canonical"
) -> pl.DataFrame:
    """Return only annotation rows whose transcript carries ``tag``.

    Ensembl GTF marks the canonical transcript on the ``transcript`` feature
    row's attribute (e.g. ``tag "Ensembl_canonical"``). Descendant rows
    (``exon``, ``CDS``, ``five_prime_utr``, ``three_prime_utr``) inherit the
    tag via shared ``transcript_id``. Gene-feature rows (no ``transcript_id``)
    are dropped — they aren't used by the downstream extractors.

    Asserts at least one canonical transcript (guards against a wrong-flavor
    GTF that doesn't carry the tag at all). Higher thresholds (e.g. > 10_000
    for human r115) belong in the calling pipeline rule.
    """
    tag_pattern = f'tag "{tag}"'
    canonical_tids = (
        ann.filter(pl.col("feature") == "transcript")
        .filter(pl.col("attribute").str.contains(tag_pattern, literal=True))
        .with_columns(
            pl.col("attribute").str.extract(r'transcript_id "(.*?)"').alias("tid")
        )["tid"]
        .drop_nulls()
        .unique()
        .to_list()
    )
    assert canonical_tids, (
        f"no transcripts tagged {tag!r} found — wrong GTF flavour? "
        "(Ensembl release 115 has tens of thousands)"
    )
    return (
        ann.with_columns(
            pl.col("attribute").str.extract(r'transcript_id "(.*?)"').alias("_tid")
        )
        .filter(pl.col("_tid").is_in(canonical_tids))
        .drop("_tid")
    )


# ============================================================================
# Ensembl-flavored feature extractors
# ============================================================================


def get_ensembl_5_prime_utr(ann: pl.DataFrame) -> GenomicSet:
    """5' UTR derived from Ensembl protein_coding exons and CDS bounds.

    Mirrors ``marin_dna.data.utils.get_5_prime_utr`` but sources mRNA exons
    from ``get_ensembl_protein_coding_exons`` (filters
    ``transcript_biotype == "protein_coding"``) instead of the RefSeq
    ``"mRNA"``-only ``get_mrna_exons``.
    """
    return _utr_from_exons_and_cds(ann, side="5_prime")


def get_ensembl_3_prime_utr(ann: pl.DataFrame) -> GenomicSet:
    """3' UTR derived from Ensembl protein_coding exons and CDS bounds."""
    return _utr_from_exons_and_cds(ann, side="3_prime")


def _utr_from_exons_and_cds(ann: pl.DataFrame, *, side: str) -> GenomicSet:
    assert side in ("5_prime", "3_prime"), side
    exons = get_ensembl_protein_coding_exons(ann)
    cds = (
        ann.with_columns(
            pl.col("attribute")
            .str.extract(r'transcript_id "(.*?)"')
            .alias("transcript_id"),
        )
        .filter(pl.col("feature") == "CDS")
        .select(["chrom", "start", "end", "strand", "transcript_id"])
    )
    if len(cds) == 0 or len(exons) == 0:
        return GenomicSet(
            pl.DataFrame(
                schema={"chrom": pl.String, "start": pl.Int64, "end": pl.Int64}
            )
        )
    cds_bounds = cds.group_by("transcript_id").agg(
        pl.col("start").min().alias("cds_start"),
        pl.col("end").max().alias("cds_end"),
    )
    joined = exons.join(cds_bounds, on="transcript_id", how="inner")
    if side == "5_prime":
        # +strand: exon up to cds_start. -strand: exon from cds_end onward.
        utr = joined.with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("start"))
            .otherwise(pl.max_horizontal("start", "cds_end"))
            .alias("utr_start"),
            pl.when(pl.col("strand") == "+")
            .then(pl.min_horizontal("end", "cds_start"))
            .otherwise(pl.col("end"))
            .alias("utr_end"),
        )
    else:
        # 3' UTR: + strand: exon after cds_end. - strand: exon before cds_start.
        utr = joined.with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.max_horizontal("start", "cds_end"))
            .otherwise(pl.col("start"))
            .alias("utr_start"),
            pl.when(pl.col("strand") == "+")
            .then(pl.col("end"))
            .otherwise(pl.min_horizontal("end", "cds_start"))
            .alias("utr_end"),
        )
    return GenomicSet(
        utr.filter(pl.col("utr_end") > pl.col("utr_start")).select(
            pl.col("chrom"),
            pl.col("utr_start").alias("start"),
            pl.col("utr_end").alias("end"),
        )
    )


def get_ensembl_ncrna_exons(ann: pl.DataFrame, *, biotypes: list[str]) -> GenomicSet:
    """ncRNA exons restricted to ``biotypes`` (Ensembl-flavored vocabulary).

    Mirrors the spirit of ``marin_dna.data.utils._filter_ncrna_exons`` but
    requires an explicit Ensembl biotype list passed by the caller — Ensembl
    uses ``"lncRNA"``, RefSeq uses ``"lnc_RNA"`` (different vocabularies, do
    not mix).

    Pseudogene / partial / "pseudogenic" guards are kept on transcript_biotype
    and gene_biotype.
    """
    assert biotypes, "biotypes must be non-empty"
    return GenomicSet(
        ann.filter(pl.col("feature") == "exon")
        .with_columns(
            pl.col("attribute")
            .str.extract(r'transcript_biotype "(.*?)"')
            .alias("transcript_biotype"),
            pl.col("attribute")
            .str.extract(r'gene_biotype "(.*?)"')
            .alias("gene_biotype"),
            pl.col("attribute").str.contains(r'pseudo "true"').alias("is_pseudo"),
            pl.col("attribute").str.contains(r'partial "true"').alias("is_partial"),
        )
        .filter(pl.col("transcript_biotype").is_in(list(biotypes)))
        .filter(~pl.col("is_pseudo").fill_null(False))
        .filter(~pl.col("is_partial").fill_null(False))
        .filter(
            ~pl.col("transcript_biotype").fill_null("").str.contains("(?i)pseudogenic")
        )
        .filter(~pl.col("gene_biotype").fill_null("").str.contains("(?i)pseudogene"))
        .select(["chrom", "start", "end"])
    )


# ============================================================================
# Region builders
# ============================================================================


_ANNOTATION_EXTRACTORS: dict[str, Callable[[pl.DataFrame], GenomicSet]] = {
    "val_cds": get_cds,
    "val_utr5": get_ensembl_5_prime_utr,
    "val_utr3": get_ensembl_3_prime_utr,
}


# Recipes that use the filter_size + add_flank + expand_min_size chain via
# `build_annotation_region`. val_tss_pc is annotation-derived too but uses
# `build_tss_band_region` (TSS ± flank, no chain) and is dispatched separately
# at the snakemake-rule level — not handled here.
_FILTER_FLANK_EXPAND_RECIPES = ("val_cds", "val_utr5", "val_utr3", "val_ncrna")

ANNOTATION_RECIPES = _FILTER_FLANK_EXPAND_RECIPES + ("val_tss_pc",)
CRE_RECIPES = ("val_promoter", "val_enhancer")
ALL_RECIPES = ANNOTATION_RECIPES + CRE_RECIPES


def build_annotation_region(
    recipe: str,
    ann_canonical: pl.DataFrame,
    defined: GenomicSet,
    *,
    ncrna_biotypes: list[str],
    add_flank: int = 20,
    min_size: int = 20,
    max_size: int = 10_000,
    expand_min: int = 255,
) -> GenomicSet:
    """Annotation-derived region BED via the filter_size+flank+expand chain.

    Pipeline: extractor(ann) → filter_size → add_flank (splice signal) →
    expand_min_size → intersect(defined). No cross-feature subtraction —
    canonical-transcript filtering already disambiguates CDS vs 5'UTR vs
    3'UTR within a transcript, and ncRNA-vs-coding genes are separated by
    biotype.

    val_tss_pc is annotation-derived too but uses ``build_tss_band_region``
    (no chain — just a fixed-width band around each TSS); call that directly
    instead.

    Args:
        recipe: one of ``val_cds``, ``val_utr5``, ``val_utr3``, ``val_ncrna``.
        ann_canonical: annotation DataFrame already filtered via
            ``filter_to_canonical_transcripts``.
        defined: defined-region (genome minus N) GenomicSet for intersection.
        ncrna_biotypes: Ensembl-flavored biotype list (only used for
            ``val_ncrna``).
        add_flank: bp added to both sides of each interval (captures splice
            signal at exon boundaries). Default 20.
        min_size, max_size: keep intervals with raw size in this range.
        expand_min: minimum interval size after extraction; smaller intervals
            are padded equally to reach this size.
    """
    if recipe == "val_ncrna":
        intervals = get_ensembl_ncrna_exons(ann_canonical, biotypes=ncrna_biotypes)
    elif recipe in _ANNOTATION_EXTRACTORS:
        intervals = _ANNOTATION_EXTRACTORS[recipe](ann_canonical)
    else:
        raise ValueError(
            f"unknown filter+flank+expand recipe: {recipe!r} "
            f"(expected one of {_FILTER_FLANK_EXPAND_RECIPES}; "
            "val_tss_pc uses build_tss_band_region instead)"
        )
    intervals = intervals.filter_size(min_size=min_size, max_size=max_size)
    intervals = intervals.add_flank(add_flank)
    intervals = intervals.expand_min_size(expand_min)
    intervals = intervals & defined
    return intervals


def build_tss_band_region(
    ann_canonical: pl.DataFrame,
    defined: GenomicSet,
    *,
    flank: int = 255,
) -> GenomicSet:
    """TSS ± ``flank`` band over canonical protein_coding transcripts.

    For ``val_tss_pc``: gene-centric symmetric window around each canonical
    protein_coding TSS. Captures both the proximal-promoter (upstream) and
    the immediate downstream 5' UTR in one anchor — different probe than
    val_promoter (cCRE chromatin-centric) and val_utr5 (annotation-driven
    full UTR including distal regions).

    No CDS subtraction by design — TSS bands sit *at* the TSS, so any
    overlap with CDS happens only for genes with very short 5' UTRs (where
    CDS extends back into the TSS-proximal region). For consistency with
    val_promoter (which is also unsubtracted), the contamination is
    accepted; downstream consumers can compare val_tss_pc and val_promoter
    side-by-side without confounding subtraction differences.
    """
    exons = get_ensembl_protein_coding_exons(ann_canonical)
    assert len(exons) > 0, (
        "no canonical protein_coding exons found "
        "— wrong GTF flavour or canonical filter?"
    )
    band = get_promoters_from_exons(exons, n_upstream=flank, n_downstream=flank)
    return band & defined


def build_cre_region(
    recipe: str,
    cre_parquet: str | Path,
    defined: GenomicSet,
    *,
    subtract: GenomicSet | None = None,
    target_size: int = 255,
) -> GenomicSet:
    """cCRE-derived region BED for ``val_promoter`` / ``val_enhancer``.

    Filters cCRE parquet (``chrom, start, end, cre_class``) to the recipe's
    class set, resizes each region to ``target_size`` bp (centered on the
    midpoint), intersects with ``defined``, and (only when ``subtract`` is
    given) subtracts ``subtract`` (the all-exon mask, used for val_enhancer).

    Args:
        recipe: ``val_promoter`` -> filter cre_class == ``PLS``;
            ``val_enhancer`` -> filter cre_class in ``ENHANCER_CRE_CLASSES``
            (= ``["dELS", "pELS"]``).
        cre_parquet: path to a parquet with cCRE regions and ``cre_class``
            column.
        defined: defined-region (genome minus N) for intersection.
        subtract: optional GenomicSet to subtract from the resized cCRE set.
            For val_enhancer this is the all-biotype exon mask
            (``get_exons``); for val_promoter this is None (PLS is *defined*
            to overlap 5' UTRs, so subtracting gut the set).
        target_size: window size after centered resize. Default 255.
    """
    if recipe == "val_promoter":
        classes = ["PLS"]
    elif recipe == "val_enhancer":
        classes = list(ENHANCER_CRE_CLASSES)
    else:
        raise ValueError(
            f"unknown cCRE-derived recipe: {recipe!r} (expected one of {CRE_RECIPES})"
        )
    df = pl.read_parquet(cre_parquet).filter(pl.col("cre_class").is_in(classes))
    intervals = GenomicSet(df.select(["chrom", "start", "end"]))
    intervals = intervals.resize(target_size)
    intervals = intervals & defined
    if subtract is not None:
        intervals = intervals - subtract
    return intervals


# ============================================================================
# Subsample + case-encode
# ============================================================================


def subsample_deterministic(
    df: pl.DataFrame, *, max_samples: int, seed: int
) -> pl.DataFrame:
    """Deterministic-shuffle subsample to at most ``max_samples`` rows.

    Always shuffles deterministically, even when ``len(df) <= max_samples``
    (in which case all rows are kept but in a seeded-permuted order).
    This guards against partial-eval bias: without shuffling, recipes that
    don't trigger the cap (e.g. val_utr5, val_promoter) would come out in
    chromosome order from the upstream filter step.

    Reproducibility: same input + same seed produces the same row order
    across runs (polars' ``DataFrame.sample`` is seeded-deterministic).
    """
    n = min(len(df), max_samples)
    if n == 0:
        return df
    return df.sample(n=n, seed=seed, with_replacement=False, shuffle=True)


def _bw_chrom(chrom: str, prefix: str = "chr") -> str:
    """Bridge bare Ensembl chrom names ("1") to UCSC-style ("chr1")."""
    return chrom if chrom.startswith(prefix) else f"{prefix}{chrom}"


def case_encode_sequences(
    fasta_path: str | Path,
    bw_path: str | Path,
    *,
    threshold: float,
    window_size: int,
    chrom_prefix: str = "chr",
) -> pl.DataFrame:
    """Read a FASTA and case-encode each base by per-position phyloP value.

    Expects FASTA headers in ``chrom:start-end`` format (the output of
    ``twoBitToFa -bedPos`` on a 4-column BED). Produces a polars frame
    ``(id, seq)`` where ``id`` is the FASTA header and ``seq`` has uppercase
    bases at positions where the bigWig value is ``>= threshold`` and
    lowercase elsewhere. NaN positions (gaps in the bigWig, e.g. unaligned
    bases) are lowercase because ``NaN >= t`` is ``False``.

    Asserts every sequence is exactly ``window_size`` bp; the
    bigWig-vs-sequence length match is also asserted per row.
    """
    sequences = load_fasta(str(fasta_path))
    ids = sequences.get_column("id").to_list()
    seqs = sequences.get_column("seq").to_list()

    encoded: list[str] = []
    bw = pyBigWig.open(str(bw_path))
    try:
        bw_chroms = set(bw.chroms())
        for fasta_id, seq in zip(ids, seqs):
            assert len(seq) == window_size, (
                f"{fasta_id}: sequence length {len(seq)} != window_size {window_size}"
            )
            chrom, coords = fasta_id.rsplit(":", 1)
            start_str, end_str = coords.split("-")
            start, end = int(start_str), int(end_str)
            assert end - start == window_size, (
                f"{fasta_id}: coord span {end - start} != window_size {window_size}"
            )
            bw_chrom = _bw_chrom(chrom, chrom_prefix)
            if bw_chrom not in bw_chroms:
                # No conservation track for this chrom — degrade gracefully:
                # everything lowercase.
                encoded.append(seq.lower())
                continue
            values = np.asarray(
                bw.values(bw_chrom, start, end, numpy=True), dtype=np.float64
            )
            assert len(values) == len(seq), (
                f"{fasta_id}: {len(values)} bigWig values vs {len(seq)} bases"
            )
            mask = values >= threshold  # NaN -> False -> lowercase
            upper = seq.upper()
            lower = seq.lower()
            encoded.append(
                "".join(u if m else lo for u, lo, m in zip(upper, lower, mask))
            )
    finally:
        bw.close()

    return pl.DataFrame(
        {"id": ids, "seq": encoded},
        schema={"id": pl.String, "seq": pl.String},
    )


# ============================================================================
# HF dataset card (README.md) generator
# ============================================================================


# Per-recipe blurb used in the dataset card "Recipe" section.
_RECIPE_BLURBS: dict[str, str] = {
    "val_cds": (
        "Protein-coding sequence from Ensembl release {ensembl_release}, "
        'restricted to canonical transcripts (`tag "{canonical_tag}"`). Each '
        "canonical CDS region is extended by 20 bp on each side (captures "
        "splice signal at exon boundaries), then expanded to a minimum length "
        "of 255 bp."
    ),
    "val_utr5": (
        "5' UTR (untranslated region upstream of CDS) from Ensembl release "
        "{ensembl_release} protein_coding canonical transcripts (`tag "
        '"{canonical_tag}"`). Each canonical 5\' UTR region is extended by '
        "20 bp on each side, then expanded to a minimum length of 255 bp."
    ),
    "val_utr3": (
        "3' UTR (untranslated region downstream of CDS) from Ensembl release "
        "{ensembl_release} protein_coding canonical transcripts (`tag "
        '"{canonical_tag}"`). Each canonical 3\' UTR region is extended by '
        "20 bp on each side, then expanded to a minimum length of 255 bp."
    ),
    "val_ncrna": (
        "Non-coding RNA exons from Ensembl release {ensembl_release}, "
        'restricted to canonical transcripts (`tag "{canonical_tag}"`) of '
        "functional ncRNA biotypes ({ncrna_biotypes_md}). Each canonical "
        "ncRNA exon region is extended by 20 bp on each side, then expanded "
        "to a minimum length of 255 bp. Pseudogene and partial-transcript "
        "biotypes are excluded."
    ),
    "val_promoter": (
        "Promoter regions defined by ENCODE cCRE V4 Promoter-Like Signature "
        "(PLS) class. Each cCRE is centered and resized to exactly 255 bp. "
        "**No subtraction** — PLS sits at the TSS by definition, so its "
        "overlap with 5' UTR is the intended biology, and any overlap with "
        "CDS in genes with very short 5' UTRs is accepted as part of the "
        "natural PLS distribution rather than filtered out. For comparison, "
        "`val_tss_pc` is a gene-centric annotation-driven alternative that "
        "uses canonical protein_coding TSSes instead of ENCODE cCREs."
    ),
    "val_tss_pc": (
        "Gene-centric promoter / proximal 5' UTR probe: ±255 bp band around "
        "each canonical protein_coding transcript's TSS, as annotated in "
        'Ensembl r{ensembl_release} (`tag "{canonical_tag}"`). One anchor '
        "per gene, ~19k canonical TSSes, tiled into 255 bp windows. No "
        "subtraction (CDS contamination only happens for very-short 5' UTR "
        "genes; consistent with `val_promoter`'s unsubtracted policy). "
        "Complementary to `val_promoter` (chromatin-centric, ~45k cCRE "
        "anchors including non-coding genes and alt-TSSes) and `val_utr5` "
        "(captures distal 5' UTR beyond the TSS-proximal window)."
    ),
    "val_enhancer": (
        "Enhancer regions defined by ENCODE cCRE V4 Enhancer-Like Signature "
        "classes (proximal pELS + distal dELS). Each cCRE is centered and "
        "resized to exactly 255 bp, then **every annotated exon is "
        "subtracted** (`get_exons` from Ensembl r{ensembl_release}, no "
        "biotype filter). Stricter than the corresponding training-set "
        "recipe (`get_exons_for_masking`), which keeps low-quality / "
        "retained-intron exons scannable for new enhancer prediction; for "
        "validation we want a clean enhancer probe, so any exonic annotation "
        "disqualifies a base."
    ),
}


def write_hf_readme(
    recipe: str,
    output_path: str | Path,
    *,
    commit_sha: str,
    hf_owner: str,
    pipeline_version: str,
    ensembl_release: int,
    threshold: float,
    min_proportion_conserved: float,
    max_samples: int,
    seed: int,
    ncrna_biotypes: list[str],
    canonical_tag: str = "Ensembl_canonical",
    github_repo: str = _GITHUB_REPO,
) -> None:
    """Write a per-recipe HuggingFace dataset card (README.md).

    Includes the GitHub permalink (commit-pinned) to the producing pipeline,
    the recipe-specific selection logic, the case-encoding semantics
    (uppercase vs lowercase), and the conservation pre-filter parameters.
    """
    if recipe not in _RECIPE_BLURBS:
        raise ValueError(
            f"unknown recipe {recipe!r}; expected one of {list(_RECIPE_BLURBS)}"
        )

    repo_name = f"{hf_owner}/zoonomia-{pipeline_version}-{recipe}"
    pipeline_permalink = (
        f"https://github.com/{github_repo}/tree/{commit_sha}/{_GITHUB_PIPELINE_PATH}"
    )
    pipeline_main_link = (
        f"https://github.com/{github_repo}/tree/main/{_GITHUB_PIPELINE_PATH}"
    )

    ncrna_biotypes_md = ", ".join(f"`{b}`" for b in ncrna_biotypes)
    blurb = _RECIPE_BLURBS[recipe].format(
        ensembl_release=ensembl_release,
        canonical_tag=canonical_tag,
        ncrna_biotypes_md=ncrna_biotypes_md,
    )

    # Conservation pre-filter: at least `min_pass_bp` of `window_size` bases
    # in a window must score `phyloP_447m >= threshold`. Computed once so the
    # f-string body keeps to one source of truth.
    window_size = 255
    min_pass_bp = round(min_proportion_conserved * window_size)

    body = f"""---
tags:
- biology
- genomics
- DNA
- conservation
- validation
---

# `{repo_name}`

Conservation pre-filtered, case-encoded human-genome validation set from the
[`{_GITHUB_PIPELINE_PATH}`]({pipeline_permalink}) pipeline
(commit [`{commit_sha[:12]}`]({pipeline_permalink})).

This is one of seven per-recipe validation parquets (`val_cds`, `val_utr5`,
`val_utr3`, `val_ncrna`, `val_promoter`, `val_enhancer`, `val_tss_pc`) built
from the same human-anchored phyloP_447m scoring used to create the
cross-mammal training sets
[`{hf_owner}/zoonomia-{pipeline_version}-v1`](https://huggingface.co/datasets/{hf_owner}/zoonomia-{pipeline_version}-v1)
and [`{hf_owner}/zoonomia-{pipeline_version}-v2`](https://huggingface.co/datasets/{hf_owner}/zoonomia-{pipeline_version}-v2).

## Recipe (`{recipe}`)

{blurb}

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `id`   | str  | `chrom:start-end` — 0-based half-open coordinates with bare Ensembl chrom names (e.g. `1:1234567-1234822`, not `chr1`). |
| `seq`  | str  | 255 bp DNA sequence with **case-encoded conservation** (see below). |

### Sequence case encoding

For each base in `seq`:

- **Uppercase** (`A` / `C` / `G` / `T` / `N`): the phyloP_447m score at this
  position is `>= {threshold}` (the calibrated conservation threshold for the
  447-way Cactus alignment, set to match phyloP_241m's passing-base proportion
  at `2.27`).
- **Lowercase** (`a` / `c` / `g` / `t` / `n`): **either** the phyloP score is
  below the threshold, **or** there is no alignment at this position (NaN in
  the bigWig — `NaN >= t` is `False` in NumPy, so unaligned bases are
  encoded as lowercase).

Uppercase positions are confidently conserved across mammals; lowercase
positions are either non-conserved or unaligned, and the encoding does **not**
distinguish those two cases.

## Construction

1. Build region BED from Ensembl release {ensembl_release} (canonical
   transcripts, where applicable) or ENCODE cCRE V4 — see *Recipe* above.
2. Intersect with defined regions (genome minus N).
3. Tile into 255 bp non-overlapping windows
   (`bedtools makewindows -w 255 -s 255`).
4. Score each window against phyloP_447m (UCSC 447-way Cactus, Zoonomia +
   primates).
5. **Conservation pre-filter**: keep only windows with
   `proportion_conserved >= {min_proportion_conserved}` (i.e. at least
   {min_pass_bp} of the {window_size} bases pass the phyloP_447m threshold
   of `{threshold}`).
6. Deterministic subsample to ≤ {max_samples:,} rows (seed `{seed}`).
7. Extract sequences from hg38 (Ensembl r{ensembl_release}) via
   `twoBitToFa -bedPos`.
8. Case-encode each base by its per-position phyloP_447m value.

**No reverse-complement augmentation** — each genomic region appears exactly
once. (RC augmentation is a training concern; for evaluation we want
deterministic per-locus rows.)

## Caveats

- **Recipes are independent probes, not a partition.** A small number of
  bases may appear in multiple recipes (e.g. a `val_promoter` PLS overlapping
  a `val_utr5` 5' UTR; an `add_flank(20)` on CDS extending into intronic
  splice signal that abuts a 3' UTR). Don't concatenate the seven parquets
  without dedup.
- **Lowercase is ambiguous.** Lowercase letters could mean either "phyloP
  score is below threshold" or "no alignment at this position". The encoding
  does not distinguish.
- Only autosomes + chrX + chrY are included; the mitochondrial contig is
  excluded.
- The `val_promoter` and `val_enhancer` cCRE recipes are *not* restricted to
  canonical transcripts — those classes are transcript-independent and come
  directly from ENCODE SCREEN cCRE V4.

## Source code

- Pipeline: [{_GITHUB_PIPELINE_PATH}]({pipeline_main_link}) (latest)
- Pinned to this dataset's build: [commit `{commit_sha[:12]}`]({pipeline_permalink})
- Library: `marin_dna_vertebrate_projection.validation`
- Conservation track: phyloP_447m
  ([`hg38.phyloP447way.bw`](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP447way/hg38.phyloP447way.bw))
"""
    Path(output_path).write_text(body)
