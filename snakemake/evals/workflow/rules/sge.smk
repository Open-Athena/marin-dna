from marin_dna.pipelines.evals.sge import (
    annotate_sge_variants,
    read_brca1_findlay,
)

# SGE (saturation genome editing) variant-effect dataset (issue #289).
# Phase 1: BRCA1 via the Evo2-bundled Findlay 2018 supplementary xlsx (hg19,
# lifted to GRCh38). Per-SNV experimental function scores with every author column
# preserved (author_ prefix); no matching / no subsampling; HIGH-impact
# `exclude_consequences` dropped. `split_dataset_by_chrom` (common.smk) then makes
# the train/test parquets and `hf_upload` uploads to bolinas-dna/evals_sge.


rule sge_download_brca1:
    """Findlay 2018 BRCA1 SGE supplementary xlsx — hg19 genomic coords for all SNVs
incl. intronic (the cleanest BRCA1 source; MaveDB drops intronic coords)."""
    output:
        "results/sge/brca1_findlay.xlsx",
    params:
        url=config["sge"]["brca1"]["url"],
    shell:
        'curl -fL "{params.url}" -o {output}'


rule sge_dataset_unsplit:
    """Build the evals_sge unsplit parquet (Phase 1: BRCA1, chr17 only): load the
xlsx, lift hg19->GRCh38, annotate consequence/distance, drop HIGH-impact."""
    input:
        xlsx="results/sge/brca1_findlay.xlsx",
        # BRCA1 is chr17, so only chr17 consequences are needed — avoids staging the
        # full multi-GB per-chrom set. Phase 2 widens this to the genes' chromosomes.
        consequences="results/consequences/17.parquet",
        exon_pc="results/intervals/exon_pc.parquet",
        exon_nc="results/intervals/exon_nc.parquet",
        tss_pc="results/intervals/tss_pc.parquet",
        tss_nc="results/intervals/tss_nc.parquet",
        # Reuse the locally-staged GRCh38 (dart_eval_stage_genome). fai/gzi are
        # pyfaidx sibling indexes; local() marks them on-disk, matching the producer.
        genome=local("results/genome_staged/GRCh38.fa.gz"),
        genome_fai=local("results/genome_staged/GRCh38.fa.gz.fai"),
        genome_gzi=local("results/genome_staged/GRCh38.fa.gz.gzi"),
    output:
        "results/dataset_unsplit/sge.parquet",
    run:
        V = read_brca1_findlay(input.xlsx, mavedb_urn=config["sge"]["brca1"]["mavedb_urn"])
        annotate_sge_variants(
            V,
            genome=Genome(input.genome),
            consequence_paths=[input.consequences],
            chroms=["17"],
            exon_pc=pl.read_parquet(input.exon_pc),
            exon_nc=pl.read_parquet(input.exon_nc),
            tss_pc=pl.read_parquet(input.tss_pc),
            tss_nc=pl.read_parquet(input.tss_nc),
            exon_proximal_dist=config["exon_proximal_dist"],
            tss_proximal_dist=config["tss_proximal_dist"],
            exclude_consequences=config["exclude_consequences"],
            lift=config["sge"]["brca1"]["lift"],
            name="brca1",
        ).write_parquet(output[0])
