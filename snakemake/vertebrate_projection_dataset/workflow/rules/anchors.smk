"""Independent reproduction of the 255 bp conservation-filtered anchors."""

from marin_dna_zoonomia_projection.conservation.scoring import score_windows
from marin_dna_zoonomia_projection.tracks import CONSERVATION_TRACKS
from marin_dna_vertebrate_projection.pipeline_io import (
    write_filtered_anchor_bed,
)
from marin_dna_zoonomia_projection.region_labels import (
    REGION_LABELS,
)

FULL_CHROMS = list(config["standard_chroms"])
FULL_CHROM_RE = "|".join(FULL_CHROMS)
PHYLOP_THRESHOLD = float(config["phyloP_447m_threshold"])
MIN_PROPORTION = float(config["min_proportion_conserved"])


rule human_undefined_regions:
    input:
        f"{RESULTS}/reference/hg38.2bit",
    output:
        f"{RESULTS}/anchors/undefined.ucsc.bed",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} /dev/stdout -nBed > {output}"


rule human_standard_chrom_sizes:
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
        ).filter(pl.col("chrom").is_in(FULL_CHROMS))
        assert set(sizes["chrom"].to_list()) == set(FULL_CHROMS)
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


rule make_anchor_windows:
    input:
        sizes=f"{RESULTS}/anchors/chrom.sizes",
        undefined=f"{RESULTS}/anchors/undefined.ucsc.bed",
    output:
        f"{RESULTS}/anchors/windows/{{chrom}}.bed.gz",
    wildcard_constraints:
        chrom=FULL_CHROM_RE,
    conda:
        "../envs/bioinformatics.yaml"
    params:
        window=WINDOW_SIZE,
        step=int(config["step_size"]),
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


rule download_phylop_447m:
    output:
        f"{RESULTS}/anchors/phyloP_447m.bw",
    params:
        url=CONSERVATION_TRACKS["phyloP_447m"],
    shell:
        "wget -q -O {output} {params.url}"


rule score_anchor_windows:
    input:
        windows=f"{RESULTS}/anchors/windows/{{chrom}}.bed.gz",
        bigwig=f"{RESULTS}/anchors/phyloP_447m.bw",
    output:
        f"{RESULTS}/anchors/scored/{{chrom}}.parquet",
    wildcard_constraints:
        chrom=FULL_CHROM_RE,
    resources:
        mem_mb=1500,
    run:
        windows = pl.read_csv(
            input.windows,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end", "name"],
        )
        assert (windows["end"] - windows["start"] == WINDOW_SIZE).all()
        scored = score_windows(input.bigwig, windows, PHYLOP_THRESHOLD)
        assert scored.height == windows.height
        assert scored["proportion_conserved"].is_between(0.0, 1.0).all()
        scored.write_parquet(output[0])


rule filter_anchor_windows:
    input:
        expand(f"{RESULTS}/anchors/scored/{{chrom}}.parquet", chrom=FULL_CHROMS),
    output:
        f"{RESULTS}/anchors/filtered.bare.bed.gz",
    run:
        write_filtered_anchor_bed(
            list(input),
            output[0],
            min_proportion_conserved=MIN_PROPORTION,
        )


rule download_ensembl_gtf:
    output:
        f"{RESULTS}/anchors/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz",
    params:
        url=(
            f"https://ftp.ensembl.org/pub/release-{config['ensembl_release']}/gtf/"
            f"homo_sapiens/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz"
        ),
    shell:
        "wget -q -O {output} {params.url}"


rule download_ccre:
    output:
        temp(f"{RESULTS}/anchors/ccre.ucsc.bed"),
    params:
        url=str(config["ccre_url"]),
    shell:
        "wget -q -O {output} {params.url}"


rule process_ccre:
    input:
        f"{RESULTS}/anchors/ccre.ucsc.bed",
    output:
        f"{RESULTS}/anchors/ccre.bare.parquet",
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
                    [chrom.removeprefix("chr") for chrom in FULL_CHROMS]
                )
            )
        )
        assert frame.height > 100_000
        frame.write_parquet(output[0])


rule label_full_anchors:
    input:
        anchors=f"{RESULTS}/anchors/filtered.bare.bed.gz",
        gtf=f"{RESULTS}/anchors/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz",
        ccre=f"{RESULTS}/anchors/ccre.bare.parquet",
        defined=f"{RESULTS}/anchors/defined.bare.bed",
    output:
        f"{RESULTS}/anchors/labels.parquet",
    resources:
        mem_mb=24000,
    run:
        from marin_dna.data.intervals import GenomicSet
        from marin_dna_zoonomia_projection.region_labels import (
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
        assert set(labels["label"].to_list()) <= set(REGION_LABELS) | {"background"}
        labels.write_parquet(output[0])


rule full_anchor_catalog:
    input:
        f"{RESULTS}/anchors/labels.parquet",
    output:
        f"{RESULTS}/anchors/catalog.parquet",
    run:
        labels = pl.read_parquet(input[0])
        catalog = labels.select(
            pl.col("name").alias("query_name"),
            (pl.lit("chr") + pl.col("chrom")).alias("source_chrom"),
            pl.col("start").alias("source_start"),
            pl.col("end").alias("source_end"),
            pl.col("label").alias("region_label"),
        )
        assert catalog["query_name"].n_unique() == catalog.height
        assert (catalog["source_end"] - catalog["source_start"] == 255).all()
        catalog.write_parquet(output[0])
