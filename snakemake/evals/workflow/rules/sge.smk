from pathlib import Path

from marin_dna.pipelines.evals.sge import (
    annotate_sge_variants,
    load_mavedb_genomic_scoreset,
    read_brca1_findlay,
)

# SGE (saturation genome editing) variant-effect dataset (issue #289).
# Phase 1: BRCA1 (Findlay xlsx, hg19->GRCh38). Phase 2a: 6 genome-targeted MaveDB
# score-sets (NC_…:g. hgvs_nt -> parsed directly, intronic kept, GRCh38-native).
# Per-SNV experimental function scores; every author column preserved (author_
# prefix); no matching / no subsampling; HIGH-impact `exclude_consequences` dropped.
# Each gene is loaded + annotated (BRCA1 lifted, genome-targeted native) then
# diagonal-concatenated; `split_dataset_by_chrom` (common.smk) makes the train/test
# parquets and `hf_upload` uploads to bolinas-dna/evals_sge.

_SGE_GENOMIC = {g["gene"]: g for g in config["sge"]["mavedb_genomic"]}
# All chromosomes touched by the SGE genes — picks the per-chrom consequence parquets.
_SGE_CHROMS = sorted(
    {config["sge"]["brca1"]["chrom"]} | {g["chrom"] for g in _SGE_GENOMIC.values()}
)


rule sge_download_brca1:
    """Findlay 2018 BRCA1 SGE supplementary xlsx — hg19 genomic coords for all SNVs
incl. intronic (the cleanest BRCA1 source; MaveDB drops intronic coords)."""
    output:
        "results/sge/brca1_findlay.xlsx",
    params:
        url=config["sge"]["brca1"]["url"],
    shell:
        'curl -fL "{params.url}" -o {output}'


rule sge_download_mavedb:
    """Download a genome-targeted MaveDB SGE score-set CSV (NC_…:g. hgvs_nt)."""
    output:
        "results/sge/mavedb/{gene}.csv",
    wildcard_constraints:
        gene="|".join(_SGE_GENOMIC),
    params:
        urn=lambda wc: _SGE_GENOMIC[wc.gene]["mavedb_urn"],
    shell:
        'curl -fL "https://api.mavedb.org/api/v1/score-sets/{params.urn}/scores" -o {output}'


rule sge_dataset_unsplit:
    """Build the evals_sge unsplit parquet: load + annotate each gene (BRCA1 lifted
hg19->GRCh38; genome-targeted GRCh38-native), drop HIGH-impact, diagonal-concat."""
    input:
        brca1_xlsx="results/sge/brca1_findlay.xlsx",
        mavedb=expand("results/sge/mavedb/{gene}.csv", gene=list(_SGE_GENOMIC)),
        consequences=expand("results/consequences/{chrom}.parquet", chrom=_SGE_CHROMS),
        exon_pc="results/intervals/exon_pc.parquet",
        exon_nc="results/intervals/exon_nc.parquet",
        tss_pc="results/intervals/tss_pc.parquet",
        tss_nc="results/intervals/tss_nc.parquet",
        # Reuse the locally-staged GRCh38 (dart_eval_stage_genome).
        genome=local("results/genome_staged/GRCh38.fa.gz"),
        genome_fai=local("results/genome_staged/GRCh38.fa.gz.fai"),
        genome_gzi=local("results/genome_staged/GRCh38.fa.gz.gzi"),
    output:
        "results/dataset_unsplit/sge.parquet",
    run:
        genome = Genome(input.genome)
        cons = dict(
            zip(_SGE_CHROMS, input.consequences)
        )  # chrom -> consequence parquet
        exon_pc = pl.read_parquet(input.exon_pc)
        exon_nc = pl.read_parquet(input.exon_nc)
        tss_pc = pl.read_parquet(input.tss_pc)
        tss_nc = pl.read_parquet(input.tss_nc)

        def annot(V, chrom, lift, name):
            return annotate_sge_variants(
                V,
                genome=genome,
                consequence_paths=[cons[chrom]],
                chroms=[chrom],
                exon_pc=exon_pc,
                exon_nc=exon_nc,
                tss_pc=tss_pc,
                tss_nc=tss_nc,
                exon_proximal_dist=config["exon_proximal_dist"],
                tss_proximal_dist=config["tss_proximal_dist"],
                exclude_consequences=config["exclude_consequences"],
                lift=lift,
                name=name,
            )

        frames = []
        brca1 = read_brca1_findlay(
            input.brca1_xlsx, mavedb_urn=config["sge"]["brca1"]["mavedb_urn"]
        )
        frames.append(annot(brca1, config["sge"]["brca1"]["chrom"], True, "brca1"))
        for csv in input.mavedb:
            gene = Path(csv).stem
            g = _SGE_GENOMIC[gene]
            V = load_mavedb_genomic_scoreset(
                csv, gene=gene, mavedb_urn=g["mavedb_urn"]
            )
            frames.append(annot(V, g["chrom"], False, gene.lower()))
        # diagonal_relaxed: each study contributes its own author_ columns (sparse
        # union, null-filled) at a common supertype.
        pl.concat(frames, how="diagonal_relaxed").write_parquet(output[0])
