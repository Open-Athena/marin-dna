# Exon extraction + "scannable" regions (defined minus exons) — shared utilities
# for the cCRE interval recipes (v30-v33 in intervals.smk). Split out of the
# former enhancer_prediction.smk when the predicted-enhancer arms (v19/v20) were
# removed (#332); the `g` wildcard constraint is preserved here.

wildcard_constraints:
    g="[^/]+",


rule extract_exons:
    input:
        "results/annotation/{g}.gtf.gz",
    output:
        "results/intervals/exons/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        exons = get_exons_for_masking(ann)
        exons.write_parquet(output[0])


rule scannable_regions:
    input:
        defined="results/intervals/defined/{g}.bed.gz",
        exons="results/intervals/exons/{g}.parquet",
    output:
        "results/intervals/scannable/{g}.bed.gz",
    run:
        defined = GenomicSet.read_bed(input.defined)
        exons = GenomicSet.read_parquet(input.exons)
        scannable = defined - exons
        scannable.write_bed(output[0])
