"""Pinned GPN-Star-P scoring and exhaustive six-arm grid assignment."""

import shutil
import urllib.request

from marin_dna_vertebrate_projection.gpn_star_anchors import (
    validate_gpn_entropy_file,
    score_gpn_entropy_windows,
    write_gpn_anchor_catalog,
    write_gpn_selection_outputs,
)
from marin_dna_vertebrate_projection.region_labels import REGION_LABELS


rule gpn_human_undefined_regions:
    input:
        f"{RESULTS}/reference/hg38.2bit",
    output:
        f"{RESULTS}/anchors/undefined.ucsc.bed",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} /dev/stdout -nBed > {output}"


rule gpn_human_chrom_sizes:
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


rule gpn_human_defined_regions_bare:
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


rule gpn_make_anchor_windows:
    input:
        sizes=f"{RESULTS}/anchors/chrom.sizes",
        undefined=f"{RESULTS}/anchors/undefined.ucsc.bed",
    output:
        temp(local(f"{RESULTS}/anchors/windows/{{chrom}}.bed.gz")),
    wildcard_constraints:
        chrom=CHROM_RE,
    conda:
        "../envs/bioinformatics.yaml"
    params:
        window=WINDOW_SIZE,
        step=STEP_SIZE,
    shell:
        r"""
        TMPSIZES=$(mktemp)
        trap "rm -f $TMPSIZES" EXIT
        awk -v c={wildcards.chrom} '$1 == c' {input.sizes} >$TMPSIZES
        bedtools makewindows -g $TMPSIZES -w {params.window} -s {params.step} \
            | awk -v w={params.window} -v c={wildcards.chrom} \
                'BEGIN {{OFS="\t"}} $3-$2 == w \
              {{printf "%s\t%d\t%d\twin_%s_%09d\n", $1,$2,$3,c,NR}}' \
            | bedtools intersect -a stdin -b {input.undefined} -v \
            | gzip >{output}
        """


rule stage_gpn_star_p_entropy:
    input:
        GPN_ENTROPY_MANIFEST_INPUT,
    output:
        temp(local(f"{RESULTS}/anchors/gpn_source/{{chrom}}.parquet")),
    wildcard_constraints:
        chrom=CHROM_RE,
    resources:
        mem_mb=1000,
    retries: 3
    params:
        shard=lambda wc: gpn_entropy_shards[wc.chrom],
        url=lambda wc: (
            f"https://huggingface.co/datasets/{GPN_DATASET_REPO}/resolve/"
            f"{GPN_DATASET_REVISION}/{gpn_entropy_shards[wc.chrom].path}"
        ),
    run:
        target = Path(output[0])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".parquet.download")
        with urllib.request.urlopen(params.url, timeout=300) as source, temporary.open(
            "wb"
        ) as out:
            shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
        validate_gpn_entropy_file(temporary, params.shard)
        temporary.replace(target)


rule score_gpn_star_p_windows:
    input:
        windows=f"{RESULTS}/anchors/windows/{{chrom}}.bed.gz",
        entropy=local(f"{RESULTS}/anchors/gpn_source/{{chrom}}.parquet"),
    output:
        scored=temp(local(f"{RESULTS}/anchors/scored/{{chrom}}.parquet")),
        stats=temp(local(f"{RESULTS}/anchors/scored/{{chrom}}.json")),
    wildcard_constraints:
        chrom=CHROM_RE,
    resources:
        mem_mb=4000,
    run:
        shard = gpn_entropy_shards[wildcards.chrom]
        score_gpn_entropy_windows(
            input.windows,
            input.entropy,
            output.scored,
            output.stats,
            chrom=wildcards.chrom,
            entropy_cutoff=GPN_ENTROPY_CUTOFF,
            expected_rows=shard.rows,
            expected_min_pos=shard.min_pos,
            expected_max_pos=shard.max_pos,
            step_size=STEP_SIZE,
            window_size=WINDOW_SIZE,
        )


rule select_gpn_star_p_windows:
    input:
        scored=expand(
            f"{RESULTS}/anchors/scored/{{chrom}}.parquet", chrom=CHROMS
        ),
        stats=expand(f"{RESULTS}/anchors/scored/{{chrom}}.json", chrom=CHROMS),
    output:
        selected=temp(local(f"{RESULTS}/anchors/selected.parquet")),
        bed=temp(local(f"{RESULTS}/anchors/selected.bare.bed.gz")),
        summary=f"{RESULTS}/anchors/audit/gpn_threshold_summary.json",
    resources:
        mem_mb=8000,
    run:
        write_gpn_selection_outputs(
            list(input.scored),
            list(input.stats),
            output.selected,
            output.bed,
            output.summary,
            min_selected_bases=GPN_MIN_SELECTED_BASES,
            expected_uniform_windows=(
                int(config["expected_uniform_windows"]) if TIER == "full" else None
            ),
            expected_selected_source_positions=(
                int(config["expected_selected_source_positions"])
                if TIER == "full"
                else None
            ),
            expected_windows_ge_10pct=(
                int(config["expected_windows_ge_10pct"])
                if TIER == "full"
                else None
            ),
            expected_windows_ge_20pct=(
                int(config["expected_windows_ge_20pct"])
                if TIER == "full"
                else None
            ),
        )


rule download_gpn_ensembl_gtf:
    output:
        f"{RESULTS}/anchors/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz",
    params:
        url=(
            f"https://ftp.ensembl.org/pub/release-{config['ensembl_release']}/gtf/"
            f"homo_sapiens/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz"
        ),
    shell:
        "wget -q -O {output} {params.url}"


rule download_gpn_ccre:
    output:
        temp(local(f"{RESULTS}/anchors/ccre.ucsc.bed")),
    params:
        url=str(config["ccre_url"]),
    shell:
        "wget -q -O {output} {params.url}"


rule process_gpn_ccre:
    input:
        f"{RESULTS}/anchors/ccre.ucsc.bed",
    output:
        temp(local(f"{RESULTS}/anchors/ccre.bare.parquet")),
    run:
        frame = (
            pl.read_csv(
                input[0],
                separator="\t",
                has_header=False,
                columns=[0, 1, 2, 5],
                new_columns=["chrom", "start", "end", "cre_class"],
            )
            .with_columns(pl.col("chrom").str.strip_prefix("chr"))
            .filter(
                pl.col("chrom").is_in(
                    [chrom.removeprefix("chr") for chrom in CHROMS]
                )
            )
        )
        assert frame.height > 1_000
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(output[0])


rule label_gpn_star_p_windows:
    input:
        anchors=f"{RESULTS}/anchors/selected.bare.bed.gz",
        gtf=f"{RESULTS}/anchors/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz",
        ccre=f"{RESULTS}/anchors/ccre.bare.parquet",
        defined=f"{RESULTS}/anchors/defined.bare.bed",
    output:
        temp(local(f"{RESULTS}/anchors/labels.parquet")),
    resources:
        mem_mb=24000,
    run:
        from marin_dna.data.intervals import GenomicSet
        from marin_dna_vertebrate_projection.region_labels import (
            build_region_beds,
            label_windows_bp_majority,
        )

        beds = build_region_beds(
            input.gtf,
            input.ccre,
            GenomicSet.read_bed(input.defined),
            tss_radius=int(config["region_label_tss_radius"]),
            ccre_flank=int(config["region_label_ccre_flank"]),
            tss_pc_only=bool(config["region_label_tss_pc_only"]),
        )
        labels = label_windows_bp_majority(
            input.anchors,
            beds,
            functional_threshold=float(config["region_label_functional_threshold"]),
            priority=list(config["region_label_priority"]),
        )
        assert set(labels["label"].to_list()) <= set(REGION_LABELS) | {
            "background"
        }
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        labels.write_parquet(output[0])


rule gpn_star_p_anchor_catalog:
    input:
        labels=f"{RESULTS}/anchors/labels.parquet",
        selected=f"{RESULTS}/anchors/selected.parquet",
    output:
        catalog=ANCHOR_CATALOG,
        assignments=ASSIGNMENTS,
        summary=f"{RESULTS}/anchors/audit/assignment_summary.json",
    resources:
        mem_mb=8000,
    run:
        write_gpn_anchor_catalog(
            input.labels,
            input.selected,
            output.catalog,
            output.assignments,
            output.summary,
            score_set=GPN_SCORE_SET,
            dataset_revision=GPN_DATASET_REVISION,
            min_selected_bases=GPN_MIN_SELECTED_BASES,
            expected_full_count=(
                int(config["expected_windows_ge_20pct"])
                if TIER == "full"
                else None
            ),
            smoke_anchors_per_region=(
                int(config["smoke_anchors_per_arm"])
                if TIER == "smoke"
                else None
            ),
            required_arms=list(config["assignment_arms"]),
        )
