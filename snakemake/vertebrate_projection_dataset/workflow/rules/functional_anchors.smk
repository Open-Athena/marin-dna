"""Ensembl-first functional anchor construction and conservation gating."""

from marin_dna_vertebrate_projection.conservation.scoring import score_windows
from marin_dna_vertebrate_projection.functional_pipeline import (
    build_functional_anchor_artifacts,
    write_conservation_catalogs,
    write_human_anchor_audit,
)
from marin_dna_vertebrate_projection.functional_review import (
    write_preprojection_review,
)
from marin_dna_vertebrate_projection.tracks import CONSERVATION_TRACKS

PHYLOP_THRESHOLD = float(config["phyloP_447m_threshold"])
PROJECTION_MIN = float(config["projection_min_proportion_conserved"])
TRAINING_MIN = float(config["training_min_proportion_conserved"])
assert 0.0 <= PROJECTION_MIN < TRAINING_MIN <= 1.0


rule human_undefined_regions:
    input:
        f"{RESULTS}/reference/hg38.2bit",
    output:
        f"{RESULTS}/anchors/undefined.ucsc.bed",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} /dev/stdout -nBed > {output}"


rule human_functional_chrom_sizes:
    input:
        f"{RESULTS}/reference/hg38.chrom.sizes",
    output:
        f"{RESULTS}/anchors/chrom.sizes",
    run:
        sizes = pl.read_csv(
            input[0],
            separator="\t",
            has_header=False,
            new_columns=["chrom", "size"],
        ).filter(pl.col("chrom").is_in(CHROMS))
        assert set(sizes["chrom"].to_list()) == set(CHROMS)
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        sizes.write_csv(output[0], separator="\t", include_header=False)


rule human_defined_regions_bare:
    input:
        sizes=f"{RESULTS}/anchors/chrom.sizes",
        undefined=f"{RESULTS}/anchors/undefined.ucsc.bed",
    output:
        f"{RESULTS}/anchors/defined.bare.bed",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        r"""
        awk 'BEGIN {{OFS="\t"}} {{print $1, 0, $2}}' {input.sizes} \
            | bedtools subtract -a stdin -b {input.undefined} \
            | sed 's/^chr//' >{output}
        """


rule download_functional_ensembl_gtf:
    output:
        f"{RESULTS}/anchors/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz",
    params:
        url=(
            f"https://ftp.ensembl.org/pub/release-{config['ensembl_release']}/gtf/"
            f"homo_sapiens/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz"
        ),
    shell:
        "wget -q -O {output} {params.url}"


rule download_functional_ccre_v4:
    output:
        f"{RESULTS}/anchors/GRCh38-cCREs.Registry-V4.bed",
    params:
        url=str(config["ccre_url"]),
    shell:
        "wget -q -O {output} {params.url}"


rule download_functional_phylop_447m:
    output:
        f"{RESULTS}/anchors/phyloP_447m.bw",
    params:
        url=CONSERVATION_TRACKS["phyloP_447m"],
    shell:
        "wget -q -O {output} {params.url}"


rule build_functional_candidates:
    input:
        gtf=f"{RESULTS}/anchors/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz",
        ccre=f"{RESULTS}/anchors/GRCh38-cCREs.Registry-V4.bed",
        sizes=f"{RESULTS}/anchors/chrom.sizes",
        defined=f"{RESULTS}/anchors/defined.bare.bed",
    output:
        retained=f"{RESULTS}/anchors/candidates.parquet",
        provenance=f"{RESULTS}/anchors/audit/source_provenance.parquet",
        construction_drops=f"{RESULTS}/anchors/audit/construction_drops.parquet",
        ownership=f"{RESULTS}/anchors/audit/window_ownership.parquet",
        feature_summary=f"{RESULTS}/anchors/audit/feature_summary.tsv",
        overlap=f"{RESULTS}/anchors/audit/raw_overlap.tsv",
        summary=f"{RESULTS}/anchors/audit/construction_summary.json",
    resources:
        mem_mb=24000,
    run:
        build_functional_anchor_artifacts(
            input.gtf,
            input.ccre,
            input.sizes,
            input.defined,
            retained_path=output.retained,
            provenance_path=output.provenance,
            construction_drops_path=output.construction_drops,
            ownership_audit_path=output.ownership,
            feature_audit_path=output.feature_summary,
            overlap_audit_path=output.overlap,
            summary_path=output.summary,
            standard_chroms=CHROMS,
            ncrna_biotypes=list(config["ncrna_biotypes"]),
            priority=list(config["functional_priority"]),
            tss_radius=int(config["tss_radius"]),
            window_size=WINDOW_SIZE,
            step_size=int(config["step_size"]),
            feature_flank=int(config["feature_flank"]),
            min_feature_size=int(config["min_feature_size"]),
            max_feature_size=int(config["max_feature_size"]),
        )


rule functional_candidates_by_chrom:
    input:
        f"{RESULTS}/anchors/candidates.parquet",
    output:
        f"{RESULTS}/anchors/by_chrom/{{chrom}}.parquet",
    wildcard_constraints:
        chrom=CHROM_RE,
    run:
        frame = (
            pl.scan_parquet(input[0])
            .filter(pl.col("chrom") == wildcards.chrom.removeprefix("chr"))
            .collect(engine="streaming")
        )
        assert frame.height > 0
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(output[0])


rule score_functional_candidates:
    input:
        candidates=f"{RESULTS}/anchors/by_chrom/{{chrom}}.parquet",
        bigwig=f"{RESULTS}/anchors/phyloP_447m.bw",
    output:
        f"{RESULTS}/anchors/scored/{{chrom}}.parquet",
    wildcard_constraints:
        chrom=CHROM_RE,
    resources:
        mem_mb=3000,
    run:
        candidates = pl.read_parquet(input.candidates)
        scored = score_windows(input.bigwig, candidates, PHYLOP_THRESHOLD)
        assert scored.height == candidates.height
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        scored.write_parquet(output[0])


rule functional_conservation_catalogs:
    input:
        expand(f"{RESULTS}/anchors/scored/{{chrom}}.parquet", chrom=CHROMS),
    output:
        projection=ANCHOR_CATALOG,
        training=TRAINING_CATALOG,
        deferred=DEFERRED_CATALOG,
        summary=f"{RESULTS}/anchors/audit/conservation_summary.json",
    run:
        write_conservation_catalogs(
            list(input),
            output.projection,
            output.training,
            output.deferred,
            output.summary,
            projection_min=PROJECTION_MIN,
            training_min=TRAINING_MIN,
            smoke_training_per_arm=(
                int(config["smoke_training_anchors_per_arm"])
                if TIER == "smoke"
                else None
            ),
            smoke_deferred_per_arm=(
                int(config["smoke_deferred_anchors_per_arm"])
                if TIER == "smoke"
                else None
            ),
            seed=int(config["inspection_seed"]),
        )


rule functional_preprojection_review:
    input:
        projection=ANCHOR_CATALOG_INPUT,
        training=TRAINING_CATALOG,
        deferred=DEFERRED_CATALOG,
    output:
        sample=f"{RESULTS}/anchors/audit/preprojection_sample.tsv",
        report=f"{RESULTS}/anchors/audit/preprojection_review.md",
    run:
        write_preprojection_review(
            input.projection,
            input.training,
            input.deferred,
            output.sample,
            output.report,
            seed=int(config["inspection_seed"]),
        )


rule functional_human_anchor_audit:
    input:
        anchors=ANCHOR_CATALOG_INPUT,
        sequences=HUMAN_SEQUENCES,
    output:
        f"{RESULTS}/anchors/audit/human_sequence.parquet",
    run:
        write_human_anchor_audit(input.anchors, input.sequences, output[0])
