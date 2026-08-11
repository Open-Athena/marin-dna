"""Validation datasets — seven per-recipe human-only HF parquets.

Pipeline per recipe (8 stages):
  region build  → tile (255 bp non-overlap)
              → score (phyloP_447m)
              → filter (proportion_conserved >= validation_min_p)
              → subsample (deterministic, <= validation_max_samples)
              → twoBitToFa -bedPos
              → case-encode (uppercase iff phyloP_447m >= threshold else lowercase)
              → HF upload

Annotation-derived recipes (val_cds / val_utr5 / val_utr3 / val_ncrna /
val_tss_pc) restrict to canonical Ensembl transcripts via tag
"Ensembl_canonical"; cCRE-derived recipes (val_promoter = PLS, val_enhancer =
pELS+dELS) are transcript-independent. val_enhancer subtracts every annotated
exon (``get_exons``, no biotype filter — stricter than training_dataset v30
because this is a *validation* probe, not a training scan).

Library code lives in ``marin_dna_zoonomia_projection.validation``;
``run:`` blocks here are thin glue + pipeline-level assertions.
"""

from marin_dna_zoonomia_projection.conservation.scoring import (
    score_windows as _score_windows,
)
from marin_dna_zoonomia_projection.validation import (
    ANNOTATION_RECIPES,
    ALL_RECIPES,
    CRE_RECIPES,
)

VALIDATION_RECIPES = list(config["validation_recipes"])
VALIDATION_MIN_P = float(config["validation_min_proportion_conserved"])
VALIDATION_MAX_SAMPLES = int(config["validation_max_samples"])
VALIDATION_SEED = int(config["validation_seed"])
VALIDATION_CANONICAL_TAG = str(config["validation_canonical_tag"])
VALIDATION_NCRNA_BIOTYPES = list(config["validation_ncrna_biotypes"])
VALIDATION_TSS_FLANK = int(config["validation_tss_flank"])
CCRE_URL = str(config["ccre_url"])

RECIPE_RE = "|".join(VALIDATION_RECIPES)
ANNOTATION_RECIPES_RE = "|".join(ANNOTATION_RECIPES)
CRE_RECIPES_RE = "|".join(CRE_RECIPES)

assert set(VALIDATION_RECIPES) == set(ALL_RECIPES), (
    f"config['validation_recipes'] must equal ALL_RECIPES={ALL_RECIPES}; "
    f"got {VALIDATION_RECIPES}"
)


# ============================================================================
# Defined regions (genome minus N)
# ============================================================================


rule defined_regions:
    """``defined = standard_chroms - N regions``. BED has bare Ensembl chrom names."""
    input:
        sizes="results/human/chrom.sizes.filtered",
        undefined="results/human/intervals/undefined.bed",
    output:
        "results/human/intervals/defined.bed",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        r"""
        awk 'BEGIN {{OFS="\t"}} {{print $1, 0, $2}}' {input.sizes} \
            | bedtools subtract -a stdin -b {input.undefined} >{output}
        """


# ============================================================================
# ENCODE cCRE Registry V4
# ============================================================================


rule cre_download:
    """ENCODE SCREEN cCRE V4 (GRCh38, UCSC chrom names)."""
    output:
        temp("results/human/intervals/cre/all.bed"),
    params:
        url=CCRE_URL,
    shell:
        "wget -q -O {output} {params.url}"


rule cre_process:
    """Strip ``chr`` prefix; restrict to ``standard_chroms``; preserve cre_class."""
    input:
        "results/human/intervals/cre/all.bed",
    output:
        "results/human/intervals/cre/all.parquet",
    run:
        df = (
            pl.read_csv(
                input[0],
                separator="\t",
                has_header=False,
                # cCRE V4 BED columns: chrom, start, end, accession_v3, accession_v4,
                # cre_class. Take chrom/start/end + cre_class (col 5, 0-indexed).
                columns=[0, 1, 2, 5],
                new_columns=["chrom", "start", "end", "cre_class"],
            )
            .with_columns(pl.col("chrom").str.replace("chr", ""))
            .filter(pl.col("chrom").is_in(STANDARD_CHROMS))
        )
        assert len(df) > 100_000, f"unexpectedly few cCRE rows: {len(df)}"
        df.write_parquet(output[0])


# ============================================================================
# All-exon mask (used only by val_enhancer)
# ============================================================================


rule validation_mask_all_exons:
    """Every annotated exon (no biotype filter) — for val_enhancer subtraction.

    Stricter than ``get_exons_for_masking`` (which excludes pseudogene /
    retained_intron / NMD biotypes for *training* scannability). Validation
    needs a clean enhancer probe: any exonic annotation disqualifies a base.
    """
    input:
        gtf=f"results/annotation/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz",
    output:
        "results/human/intervals/validation/mask/all_exons.bed",
    resources:
        # See validation_region_annotation note below for why mem_mb=16000.
        mem_mb=16000,
    run:
        from marin_dna.data.utils import get_exons, load_annotation

        ann = load_annotation(input.gtf)
        exons = get_exons(ann)
        assert exons.n_intervals() > 100_000, (
            f"too few exons in {input.gtf}: {exons.n_intervals()} "
            "(expected hundreds of thousands for human r115)"
        )
        exons.write_bed(output[0])


# ============================================================================
# Region builders (annotation-derived + cCRE-derived)
# ============================================================================


rule validation_region_annotation:
    """Build region BED for val_cds / val_utr5 / val_utr3 / val_ncrna / val_tss_pc."""
    input:
        gtf=f"results/annotation/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz",
        defined="results/human/intervals/defined.bed",
    output:
        "results/human/intervals/validation/region/{recipe}.bed.gz",
    wildcard_constraints:
        recipe=ANNOTATION_RECIPES_RE,
    resources:
        # GTF-loading rules peak ~7-10 GB each (polars `with_columns` filter
        # transiently doubles the ~3 GB parsed Ensembl r115 frame). On a
        # 30 GB-budget VM, mem_mb=16000 forces serial execution — earlier
        # attempts at lower mem_mb let Snakemake schedule concurrent loads
        # that OOMed.
        mem_mb=16000,
    params:
        ncrna_biotypes=VALIDATION_NCRNA_BIOTYPES,
        canonical_tag=VALIDATION_CANONICAL_TAG,
        tss_flank=VALIDATION_TSS_FLANK,
    run:
        from marin_dna.data.intervals import GenomicSet
        from marin_dna.data.utils import load_annotation
        from marin_dna_zoonomia_projection.validation import (
            build_annotation_region,
            build_tss_band_region,
            filter_to_canonical_transcripts,
        )

        ann = load_annotation(input.gtf)
        canonical = filter_to_canonical_transcripts(ann, tag=params.canonical_tag)
        n_canonical = canonical.filter(pl.col("feature") == "transcript").height
        assert n_canonical > 10_000, (
            f"too few transcripts tagged {params.canonical_tag!r}: "
            f"{n_canonical} (expected >10k for human r115)"
        )
        defined = GenomicSet.read_bed(input.defined)
        if wildcards.recipe == "val_tss_pc":
            intervals = build_tss_band_region(
                canonical, defined, flank=params.tss_flank
            )
        else:
            intervals = build_annotation_region(
                wildcards.recipe,
                canonical,
                defined,
                ncrna_biotypes=params.ncrna_biotypes,
            )
        assert (
            intervals.n_intervals() > 0
        ), f"empty region BED for {wildcards.recipe}"
        intervals.write_bed(output[0])


def _cre_region_inputs(wildcards):
    """val_enhancer subtracts all annotated exons; val_promoter has no subtraction."""
    base = {
        "cre": "results/human/intervals/cre/all.parquet",
        "defined": "results/human/intervals/defined.bed",
    }
    if wildcards.recipe == "val_enhancer":
        base["mask"] = "results/human/intervals/validation/mask/all_exons.bed"
    return base


rule validation_region_cre:
    """Build region BED for val_promoter / val_enhancer."""
    input:
        unpack(_cre_region_inputs),
    output:
        "results/human/intervals/validation/region/{recipe}.bed.gz",
    wildcard_constraints:
        recipe=CRE_RECIPES_RE,
    run:
        from marin_dna.data.intervals import GenomicSet
        from marin_dna_zoonomia_projection.validation import (
            build_cre_region,
        )

        defined = GenomicSet.read_bed(input.defined)
        if wildcards.recipe == "val_enhancer":
            subtract = GenomicSet.read_bed(input.mask)
        else:
            subtract = None
        intervals = build_cre_region(
            wildcards.recipe, input.cre, defined, subtract=subtract
        )
        assert (
            intervals.n_intervals() > 0
        ), f"empty cCRE region BED for {wildcards.recipe}"
        intervals.write_bed(output[0])


# ============================================================================
# Tile → score → filter → subsample → extract → encode → upload
# ============================================================================


rule validation_tile:
    """Slice region BED into 255 bp non-overlapping windows.

    Mirrors ``windows.smk:make_windows`` (drops the trailing partial window
    via ``$3-$2 == w``). Placeholder name column is ``.`` —
    ``twoBitToFa -bedPos`` produces ``chrom:start-end`` headers regardless.
    """
    input:
        "results/human/intervals/validation/region/{recipe}.bed.gz",
    output:
        "results/human/intervals/validation/tiled/{recipe}.bed.gz",
    wildcard_constraints:
        recipe=RECIPE_RE,
    conda:
        "../envs/bioinformatics.yaml"
    params:
        w=WINDOW_SIZE,
    shell:
        r"""
        bedtools makewindows -b {input} -w {params.w} -s {params.w} \
            | awk -v w={params.w} 'BEGIN {{OFS="\t"}} \
              $3 - $2 == w {{ print $1, $2, $3, "." }}' \
            | gzip >{output}
        """


rule validation_score:
    """Score 255 bp validation windows against phyloP_447m."""
    input:
        windows="results/human/intervals/validation/tiled/{recipe}.bed.gz",
        bw="results/bigwig/phyloP_447m.bw",
    output:
        "results/human/intervals/validation/scored/{recipe}.parquet",
    wildcard_constraints:
        recipe=RECIPE_RE,
    resources:
        mem_mb=2000,
    params:
        threshold=PHYLOP_447M_THRESHOLD,
    run:
        windows_df = pl.read_csv(
            input.windows,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end", "name"],
            schema_overrides={
                "chrom": pl.Utf8,
                "start": pl.Int64,
                "end": pl.Int64,
                "name": pl.Utf8,
            },
        )
        assert len(windows_df) > 0, f"no windows in {input.windows}"
        assert (windows_df["end"] - windows_df["start"] == WINDOW_SIZE).all()
        scored = _score_windows(input.bw, windows_df, params.threshold)
        assert (scored["proportion_conserved"] >= 0.0).all()
        assert (scored["proportion_conserved"] <= 1.0).all()
        scored.sort(["chrom", "start"]).write_parquet(output[0])


rule validation_filter:
    """Keep only windows with proportion_conserved >= validation_min_p."""
    input:
        "results/human/intervals/validation/scored/{recipe}.parquet",
    output:
        "results/human/intervals/validation/filtered/{recipe}.bed.gz",
    wildcard_constraints:
        recipe=RECIPE_RE,
    params:
        min_p=VALIDATION_MIN_P,
    run:
        df = pl.read_parquet(input[0])
        kept = df.filter(pl.col("proportion_conserved") >= params.min_p)
        assert len(kept) > 0, (
            f"no {wildcards.recipe} windows pass conservation pre-filter "
            f"at min_proportion_conserved >= {params.min_p}"
        )
        with gzip.open(output[0], "wt") as fout:
            for row in kept.iter_rows(named=True):
                fout.write(
                    f"{row['chrom']}\t{row['start']}\t{row['end']}\t{row['name']}\n"
                )


rule validation_subsample:
    """Deterministic subsample to <= validation_max_samples rows."""
    input:
        "results/human/intervals/validation/filtered/{recipe}.bed.gz",
    output:
        "results/human/intervals/validation/sampled/{recipe}.bed.gz",
    wildcard_constraints:
        recipe=RECIPE_RE,
    params:
        max_samples=VALIDATION_MAX_SAMPLES,
        seed=VALIDATION_SEED,
    run:
        from marin_dna_zoonomia_projection.validation import (
            subsample_deterministic,
        )

        df = pl.read_csv(
            input[0],
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end", "name"],
            schema_overrides={
                "chrom": pl.Utf8,
                "start": pl.Int64,
                "end": pl.Int64,
                "name": pl.Utf8,
            },
        )
        out = subsample_deterministic(
            df, max_samples=params.max_samples, seed=params.seed
        )
        assert 0 < len(out) <= params.max_samples
        with gzip.open(output[0], "wt") as fout:
            for row in out.iter_rows(named=True):
                fout.write(
                    f"{row['chrom']}\t{row['start']}\t{row['end']}\t{row['name']}\n"
                )


rule validation_extract_seq:
    """``twoBitToFa -bedPos`` produces FASTA headers ``chrom:start-end``."""
    input:
        bed="results/human/intervals/validation/sampled/{recipe}.bed.gz",
        twobit="results/human/genome.2bit",
    output:
        temp("results/human/intervals/validation/sampled/{recipe}.fa"),
    wildcard_constraints:
        recipe=RECIPE_RE,
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        r"""
        TMP=$(mktemp)
        trap "rm -f $TMP" EXIT
        zcat {input.bed} >$TMP
        twoBitToFa {input.twobit} {output} -bed=$TMP -bedPos
        """


rule validation_dataset:
    """Case-encode sequences using phyloP_447m and write final (id, seq) parquet.

    Each base is uppercase iff the phyloP_447m value at that position is
    >= ``phyloP_447m_threshold``, else lowercase. NaN positions (gaps in
    the bigWig) are lowercase because ``NaN >= t`` is False.
    """
    input:
        fasta="results/human/intervals/validation/sampled/{recipe}.fa",
        bw="results/bigwig/phyloP_447m.bw",
    output:
        "results/human/intervals/validation/dataset/{recipe}.parquet",
    wildcard_constraints:
        recipe=RECIPE_RE,
    resources:
        mem_mb=2000,
    params:
        threshold=PHYLOP_447M_THRESHOLD,
        window_size=WINDOW_SIZE,
    run:
        from marin_dna_zoonomia_projection.validation import (
            case_encode_sequences,
        )

        df = case_encode_sequences(
            input.fasta,
            input.bw,
            threshold=params.threshold,
            window_size=params.window_size,
        )
        assert len(df) > 0, f"empty dataset for {wildcards.recipe}"
        assert (df["seq"].str.len_bytes() == params.window_size).all()
        # Allowed alphabet: ACGT both cases + N both cases (genome may contain
        # N at masked positions; case encodes conservation orthogonally).
        bad = df["seq"].str.contains(r"[^ACGTacgtNn]")
        assert (
            not bad.any()
        ), f"unexpected character in {wildcards.recipe} seq column"
        df.write_parquet(output[0])


rule validation_hf_readme:
    """Generate the per-recipe HuggingFace dataset card (README.md).

    Captures the GitHub permalink (commit-pinned), the recipe-specific
    selection logic, the case-encoding semantics with explicit lowercase
    ambiguity callout, and the conservation pre-filter parameters. Uploaded
    alongside the parquet by ``hf_upload_validation``.
    """
    input:
        parquet="results/human/intervals/validation/dataset/{recipe}.parquet",
    output:
        "results/human/intervals/validation/dataset/{recipe}.README.md",
    wildcard_constraints:
        recipe=RECIPE_RE,
    params:
        commit_sha=GIT_COMMIT_SHA,
        hf_owner=HF_OWNER,
        pipeline_version=PIPELINE_VERSION,
        ensembl_release=int(config["ensembl_release"]),
        threshold=PHYLOP_447M_THRESHOLD,
        min_p=VALIDATION_MIN_P,
        max_samples=VALIDATION_MAX_SAMPLES,
        seed=VALIDATION_SEED,
        ncrna_biotypes=VALIDATION_NCRNA_BIOTYPES,
        canonical_tag=VALIDATION_CANONICAL_TAG,
    run:
        from marin_dna_zoonomia_projection.validation import (
            write_hf_readme,
        )

        write_hf_readme(
            wildcards.recipe,
            output[0],
            commit_sha=params.commit_sha,
            hf_owner=params.hf_owner,
            pipeline_version=params.pipeline_version,
            ensembl_release=params.ensembl_release,
            threshold=params.threshold,
            min_proportion_conserved=params.min_p,
            max_samples=params.max_samples,
            seed=params.seed,
            ncrna_biotypes=params.ncrna_biotypes,
            canonical_tag=params.canonical_tag,
        )


rule hf_upload_validation:
    """Upload validation parquet + README dataset card to a per-recipe HF repo."""
    input:
        parquet="results/human/intervals/validation/dataset/{recipe}.parquet",
        readme="results/human/intervals/validation/dataset/{recipe}.README.md",
    output:
        "results/upload.done/validation/{recipe}",
    wildcard_constraints:
        recipe=RECIPE_RE,
    threads: 4
    params:
        repo=lambda wc: f"{HF_OWNER}/zoonomia-{PIPELINE_VERSION}-{wc.recipe}",
    shell:
        """
        hf upload {params.repo} {input.parquet} {wildcards.recipe}.parquet --repo-type dataset
        hf upload {params.repo} {input.readme} README.md --repo-type dataset
        mkdir -p $(dirname {output})
        touch {output}
        """


rule all_validation:
    """Trigger HF upload for every recipe in ``validation_recipes``."""
    input:
        expand(
            "results/upload.done/validation/{recipe}",
            recipe=VALIDATION_RECIPES,
        ),
