"""Per-anchor region-type annotation.

Labels every conservation-filtered human anchor (255 bp) with exactly one
of ``cds``, ``utr3``, ``ncrna_exon``, ``tss_region_and_utr5``,
``ccre_non_promoter``,
``background``. Used to characterise the composition of the v1 training
dataset and to emit per-label ``subsets_def/v3_*.query_names.txt`` lists
that plug into the existing ``subset_dataset_derived`` machinery.

Library code lives in ``marin_dna_zoonomia_projection.region_labels``;
``run:`` blocks here are thin glue + pipeline-level assertions.

See plan and README "Region-type annotation" section for the priority /
threshold / cCRE-flank semantics.
"""

from marin_dna_zoonomia_projection.region_labels import REGION_LABELS

REGION_LABEL_TSS_RADIUS = int(config["region_label_tss_radius"])
REGION_LABEL_CCRE_FLANK = int(config["region_label_ccre_flank"])
REGION_LABEL_FUNCTIONAL_THRESHOLD = float(config["region_label_functional_threshold"])
REGION_LABEL_PRIORITY = list(config["region_label_priority"])
REGION_LABEL_SUBSETS = list(config["region_label_subsets"])

assert set(REGION_LABEL_PRIORITY) == set(REGION_LABELS), (
    f"config['region_label_priority']={REGION_LABEL_PRIORITY} must be a "
    f"permutation of REGION_LABELS={list(REGION_LABELS)}"
)

# Subset names are v3_<label>; cross-check against the priority list (+ bg).
_EXPECTED_SUBSETS = {f"v3_{lbl}" for lbl in REGION_LABELS} | {"v3_bg"}
assert set(REGION_LABEL_SUBSETS) == _EXPECTED_SUBSETS, (
    f"config['region_label_subsets']={REGION_LABEL_SUBSETS} must equal "
    f"{sorted(_EXPECTED_SUBSETS)}"
)

REGION_LABEL_SUBSET_RE = "|".join(REGION_LABEL_SUBSETS)


rule build_region_labels:
    """Annotate every anchor in min{min_p}.bed.gz with one region label."""
    input:
        anchors="results/human/intervals/filtered/min{min_p}.bed.gz",
        gtf=f"results/annotation/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz",
        cre="results/human/intervals/cre/all.parquet",
        defined="results/human/intervals/defined.bed",
    output:
        labels="results/human/intervals/region_labels/min{min_p}.parquet",
    resources:
        # GTF parse + 5 region BEDs + bf.coverage over 22.9M anchors ×
        # multiple region sets. Peaks around the union build; budget headroom.
        mem_mb=24000,
    params:
        tss_radius=REGION_LABEL_TSS_RADIUS,
        ccre_flank=REGION_LABEL_CCRE_FLANK,
        functional_threshold=REGION_LABEL_FUNCTIONAL_THRESHOLD,
        priority=REGION_LABEL_PRIORITY,
    run:
        from marin_dna.data.intervals import GenomicSet
        from marin_dna_zoonomia_projection.region_labels import (
            build_region_beds,
            label_windows,
        )

        defined = GenomicSet.read_bed(input.defined)
        beds = build_region_beds(
            input.gtf,
            input.cre,
            defined,
            tss_radius=params.tss_radius,
            ccre_flank=params.ccre_flank,
        )
        df = label_windows(
            input.anchors,
            beds,
            functional_threshold=params.functional_threshold,
            priority=params.priority,
        )
        # Partition invariant: every input window receives exactly one label.
        n_in = sum(1 for _ in gzip.open(input.anchors, "rt"))
        assert (
            len(df) == n_in
        ), f"label_windows lost rows: input={n_in}, output={len(df)}"
        valid_labels = set(REGION_LABELS) | {"background"}
        assert set(df["label"].unique().to_list()) <= valid_labels, (
            f"unexpected labels: "
            f"{set(df['label'].unique().to_list()) - valid_labels}"
        )
        df.write_parquet(output.labels)


rule region_label_composition:
    """Composition report: per-label counts + bg subsplit by gene-body coverage."""
    input:
        labels="results/human/intervals/region_labels/min{min_p}.parquet",
    output:
        tsv="results/human/intervals/region_labels/min{min_p}.composition.tsv",
    run:
        df = pl.read_parquet(input.labels)
        n_total = len(df)
        # Per-label counts and total bases.
        by_label = (
            df.group_by("label")
            .agg(
                pl.len().alias("n_windows"),
                pl.col("functional_frac").mean().alias("mean_functional_frac"),
                pl.col("gene_body_frac").mean().alias("mean_gene_body_frac"),
                pl.col("intron_frac").mean().alias("mean_intron_frac"),
                pl.col("intergenic_frac").mean().alias("mean_intergenic_frac"),
            )
            .with_columns(
                (pl.col("n_windows") / n_total).alias("fraction_of_total")
            )
            .sort("label")
        )
        # Background subsplit: intronic (gene_body but not exon) vs intergenic.
        bg = df.filter(pl.col("label") == "background")
        n_bg = len(bg)
        if n_bg > 0:
            n_bg_intronic = bg.filter(pl.col("gene_body_frac") > 0.5).height
            n_bg_intergenic = bg.filter(pl.col("gene_body_frac") <= 0.5).height
        else:
            n_bg_intronic = 0
            n_bg_intergenic = 0
        bg_split = pl.DataFrame(
            {
                "label": ["background_intronic", "background_intergenic"],
                "n_windows": [n_bg_intronic, n_bg_intergenic],
                "mean_functional_frac": [None, None],
                "mean_gene_body_frac": [None, None],
                "mean_intron_frac": [None, None],
                "mean_intergenic_frac": [None, None],
                "fraction_of_total": [
                    n_bg_intronic / n_total if n_total else 0.0,
                    n_bg_intergenic / n_total if n_total else 0.0,
                ],
            }
        )
        out = pl.concat([by_label, bg_split], how="diagonal_relaxed")
        out.write_csv(output.tsv, separator="\t")


def _label_for_subset(subset: str) -> str:
    """Map ``v3_<label>`` → ``<label>`` (e.g. ``v3_bg`` → ``background``)."""
    assert subset.startswith("v3_"), subset
    stripped = subset[len("v3_") :]
    return "background" if stripped == "bg" else stripped


rule derive_subset_v3_region:
    """Emit query_names for one region label."""
    input:
        labels="results/human/intervals/region_labels/min{min_p}.parquet",
    output:
        names="results/projection/min{min_p}/subsets_def/{subset}.query_names.txt",
    wildcard_constraints:
        subset=REGION_LABEL_SUBSET_RE,
    run:
        target = _label_for_subset(wildcards.subset)
        df = pl.read_parquet(input.labels).filter(pl.col("label") == target)
        # Guard against accidental empties (would silently produce a 0-row HF
        # subset later); a label with zero anchors means a config/biology bug.
        assert len(df) > 0, (
            f"subset {wildcards.subset!r} (label {target!r}) is empty — "
            f"is the GTF / cCRE / threshold correct?"
        )
        with open(output.names, "w") as fh:
            for name in df["name"].to_list():
                fh.write(f"{name}\n")


rule all_region_labels:
    """Aggregate target: composition TSV + all v3_<label> query_names lists."""
    input:
        expand(
            "results/human/intervals/region_labels/min{min_p}.composition.tsv",
            min_p=[f"{f['min_proportion_conserved']:.2f}" for f in config["filters"]],
        ),
        expand(
            "results/projection/min{min_p}/subsets_def/{subset}.query_names.txt",
            min_p=[f"{config['project_min_p']}"],
            subset=REGION_LABEL_SUBSETS,
        ),


# ============================================================================
# v4 region labels (issue #227): bp-priority + window-majority labeler,
# PC-only TSS band, ccre_flank=0, tss>ncrna priority. The v3 rules above are
# left byte-for-byte untouched (the v3_* HF datasets are frozen); v4 is a
# parallel derivation off the same anchors / GTF / cCRE inputs.
# ============================================================================


REGION_LABEL_CCRE_FLANK_V4 = int(config["region_label_ccre_flank_v4"])
REGION_LABEL_TSS_PC_ONLY_V4 = bool(config["region_label_tss_pc_only_v4"])
REGION_LABEL_PRIORITY_V4 = list(config["region_label_priority_v4"])
REGION_LABEL_SUBSETS_V4 = list(config["region_label_subsets_v4"])

assert set(REGION_LABEL_PRIORITY_V4) == set(REGION_LABELS), (
    f"config['region_label_priority_v4']={REGION_LABEL_PRIORITY_V4} must be a "
    f"permutation of REGION_LABELS={list(REGION_LABELS)}"
)
_EXPECTED_SUBSETS_V4 = {f"v4_{lbl}" for lbl in REGION_LABELS} | {"v4_bg"}
assert set(REGION_LABEL_SUBSETS_V4) == _EXPECTED_SUBSETS_V4, (
    f"config['region_label_subsets_v4']={REGION_LABEL_SUBSETS_V4} must equal "
    f"{sorted(_EXPECTED_SUBSETS_V4)}"
)
REGION_LABEL_SUBSET_V4_RE = "|".join(REGION_LABEL_SUBSETS_V4)


rule build_region_labels_v4:
    """Annotate every anchor in min{min_p}.bed.gz with one v4 region label.

    Uses the bp-priority + window-majority labeler with a PC-only TSS band and
    ccre_flank=0 (issue #227 / #221). v3's ``build_region_labels`` is unchanged.
    """
    input:
        anchors="results/human/intervals/filtered/min{min_p}.bed.gz",
        gtf=f"results/annotation/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz",
        cre="results/human/intervals/cre/all.parquet",
        defined="results/human/intervals/defined.bed",
    output:
        labels="results/human/intervals/region_labels/v4/min{min_p}.parquet",
    resources:
        mem_mb=24000,
    params:
        tss_radius=REGION_LABEL_TSS_RADIUS,
        ccre_flank=REGION_LABEL_CCRE_FLANK_V4,
        tss_pc_only=REGION_LABEL_TSS_PC_ONLY_V4,
        functional_threshold=REGION_LABEL_FUNCTIONAL_THRESHOLD,
        priority=REGION_LABEL_PRIORITY_V4,
    run:
        from marin_dna.data.intervals import GenomicSet
        from marin_dna_zoonomia_projection.region_labels import (
            build_region_beds,
            label_windows_bp_majority,
        )

        defined = GenomicSet.read_bed(input.defined)
        beds = build_region_beds(
            input.gtf,
            input.cre,
            defined,
            tss_radius=params.tss_radius,
            ccre_flank=params.ccre_flank,
            tss_pc_only=params.tss_pc_only,
        )
        df = label_windows_bp_majority(
            input.anchors,
            beds,
            functional_threshold=params.functional_threshold,
            priority=params.priority,
        )
        # Partition invariant: every input window receives exactly one label.
        n_in = sum(1 for _ in gzip.open(input.anchors, "rt"))
        assert (
            len(df) == n_in
        ), f"label_windows_bp_majority lost rows: input={n_in}, output={len(df)}"
        valid_labels = set(REGION_LABELS) | {"background"}
        assert set(df["label"].unique().to_list()) <= valid_labels, (
            f"unexpected labels: "
            f"{set(df['label'].unique().to_list()) - valid_labels}"
        )
        df.write_parquet(output.labels)


rule region_label_composition_v4:
    """Composition report for the v4 partition (same schema as v3)."""
    input:
        labels="results/human/intervals/region_labels/v4/min{min_p}.parquet",
    output:
        tsv="results/human/intervals/region_labels/v4/min{min_p}.composition.tsv",
    run:
        from marin_dna_zoonomia_projection.region_labels import (
            region_label_composition_table,
        )

        df = pl.read_parquet(input.labels)
        region_label_composition_table(df).write_csv(output.tsv, separator="\t")


def _label_for_v4_subset(subset: str) -> str:
    """Map ``v4_<label>`` → ``<label>`` (e.g. ``v4_bg`` → ``background``)."""
    assert subset.startswith("v4_"), subset
    stripped = subset[len("v4_") :]
    return "background" if stripped == "bg" else stripped


rule derive_subset_v4_region:
    """Emit query_names for one v4 region label."""
    input:
        labels="results/human/intervals/region_labels/v4/min{min_p}.parquet",
    output:
        names="results/projection/min{min_p}/subsets_def/{subset}.query_names.txt",
    wildcard_constraints:
        subset=REGION_LABEL_SUBSET_V4_RE,
    run:
        target = _label_for_v4_subset(wildcards.subset)
        df = pl.read_parquet(input.labels).filter(pl.col("label") == target)
        # Guard against accidental empties (would silently produce a 0-row HF
        # subset later); a label with zero anchors means a config/biology bug.
        assert len(df) > 0, (
            f"subset {wildcards.subset!r} (label {target!r}) is empty — "
            f"is the GTF / cCRE / threshold correct?"
        )
        with open(output.names, "w") as fh:
            for name in df["name"].to_list():
                fh.write(f"{name}\n")


rule all_region_labels_v4:
    """Aggregate target: v4 composition TSV + all v4_<label> query_names lists."""
    input:
        expand(
            "results/human/intervals/region_labels/v4/min{min_p}.composition.tsv",
            min_p=[f"{f['min_proportion_conserved']:.2f}" for f in config["filters"]],
        ),
        expand(
            "results/projection/min{min_p}/subsets_def/{subset}.query_names.txt",
            min_p=[f"{config['project_min_p']}"],
            subset=REGION_LABEL_SUBSETS_V4,
        ),
