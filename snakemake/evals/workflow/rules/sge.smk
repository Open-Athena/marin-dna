from pathlib import Path

from marin_dna.pipelines.evals.sge import (
    SNV_HGVS_RE,
    annotate_sge_variants,
    build_mavedb_metadata,
    load_mavedb_genomic_scoreset,
    load_mavedb_transcript_scoreset,
    pyhgvs_cdot_mapper,
    read_brca1_findlay,
    recode_hgvs_c_to_genomic,
)

# SGE (saturation genome editing) variant-effect dataset (issue #289). Per-SNV
# experimental function scores; every author column preserved (author_ prefix); no
# matching / no subsampling; HIGH-impact `exclude_consequences` dropped. Three loader
# paths feed one diagonal-concatenated parquet:
#   - BRCA1: Evo2-bundled Findlay xlsx, hg19 -> GRCh38 (phase 1).
#   - genome-targeted MaveDB (NC_…:g.): parse directly, intronic free (phase 2a).
#   - transcript-targeted MaveDB (ENST…:c.): recode c.->g. with pyhgvs+cdot so the
#     intronic variants MaveDB drops are recovered (phase 2b; `hgvs` dep group).
# `split_dataset_by_chrom` (common.smk) makes the train/test parquets.

_SGE_GENOMIC = {g["gene"]: g for g in config["sge"]["mavedb_genomic"]}
_SGE_TRANSCRIPT = {g["gene"]: g for g in config["sge"]["mavedb_transcript"]}
_SGE_MAVEDB = {**_SGE_GENOMIC, **_SGE_TRANSCRIPT}  # all MaveDB score-sets (downloaded)
# All chromosomes touched by the SGE genes — picks the per-chrom consequence parquets.
_SGE_CHROMS = sorted(
    {config["sge"]["brca1"]["chrom"]} | {g["chrom"] for g in _SGE_MAVEDB.values()}
)
# (gene, mavedb_urn) for every study, for the MaveDB study-metadata fetch. BRCA1
# lives under its own config key; the rest are the genomic + transcript score-sets.
_SGE_GENE_URNS = [("BRCA1", config["sge"]["brca1"]["mavedb_urn"])] + [
    (gene, meta["mavedb_urn"]) for gene, meta in _SGE_MAVEDB.items()
]


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
    """Download a MaveDB SGE score-set CSV (genome- or transcript-targeted)."""
    output:
        "results/sge/mavedb/{gene}.csv",
    wildcard_constraints:
        gene="|".join(_SGE_MAVEDB),
    params:
        urn=lambda wc: _SGE_MAVEDB[wc.gene]["mavedb_urn"],
    shell:
        'curl -fL "https://api.mavedb.org/api/v1/score-sets/{params.urn}/scores" -o {output}'


rule sge_mavedb_metadata:
    """Fetch each SGE study's MaveDB record once for study-level metadata:
  - assay_facts.parquet: experiment 'assay facts' (controlled keywords — assay
    readout, molecular mechanism, model system, library mechanism). One row per
    (gene, mavedb_urn); joined onto the dataset as constant-per-gene assay_ columns.
  - calibrations.parquet: scoreCalibrations (investigator + ClinGen/ExCALIBR ACMG
    thresholds, prior-prob, evidence strength) flattened to a tidy long companion
    table (one row per gene x calibration x functional class).
Network fetch from the MaveDB REST API (no input files)."""
    output:
        assay_facts="results/sge/assay_facts.parquet",
        calibrations="results/sge/calibrations.parquet",
    run:
        facts, calibrations = build_mavedb_metadata(_SGE_GENE_URNS)
        facts.write_parquet(output.assay_facts)
        calibrations.write_parquet(output.calibrations)


rule sge_recode_mavedb:
    """Recover genomic coords for a transcript-targeted (c.) MaveDB score-set via
pyhgvs + cdot — keeps the intronic variants MaveDB's own map drops. Caches the
c.->g. mapping. Needs the `hgvs` dependency group."""
    input:
        csv="results/sge/mavedb/{gene}.csv",
        genome=local("results/genome_staged/GRCh38.fa.gz"),
        genome_fai=local("results/genome_staged/GRCh38.fa.gz.fai"),
        genome_gzi=local("results/genome_staged/GRCh38.fa.gz.gzi"),
    output:
        "results/sge/recoded/{gene}.parquet",
    wildcard_constraints:
        gene="|".join(_SGE_TRANSCRIPT),
    run:
        raw = pl.read_csv(input.csv, infer_schema_length=None)
        snv = [h for h in raw["hgvs_nt"].to_list() if SNV_HGVS_RE.search(h)]
        mapper = pyhgvs_cdot_mapper(input.genome)
        recode_hgvs_c_to_genomic(snv, mapper=mapper).write_parquet(output[0])


rule sge_dataset_unsplit:
    """Build the evals_sge unsplit parquet: load + annotate each gene (BRCA1 lifted;
genome-targeted parsed; transcript-targeted joined to its recoded coords), drop
HIGH-impact, diagonal-concat."""
    input:
        brca1_xlsx="results/sge/brca1_findlay.xlsx",
        mavedb=expand("results/sge/mavedb/{gene}.csv", gene=list(_SGE_MAVEDB)),
        recoded=expand("results/sge/recoded/{gene}.parquet", gene=list(_SGE_TRANSCRIPT)),
        assay_facts="results/sge/assay_facts.parquet",
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
        csv_by_gene = {Path(p).stem: p for p in input.mavedb}
        recoded_by_gene = {Path(p).stem: p for p in input.recoded}
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
        for gene, meta in _SGE_GENOMIC.items():
            V = load_mavedb_genomic_scoreset(
                csv_by_gene[gene], gene=gene, mavedb_urn=meta["mavedb_urn"]
            )
            frames.append(annot(V, meta["chrom"], False, gene.lower()))
        for gene, meta in _SGE_TRANSCRIPT.items():
            V = load_mavedb_transcript_scoreset(
                csv_by_gene[gene],
                pl.read_parquet(recoded_by_gene[gene]),
                gene=gene,
                mavedb_urn=meta["mavedb_urn"],
            )
            frames.append(annot(V, meta["chrom"], False, gene.lower()))
        # diagonal_relaxed: each study contributes its own author_ columns (sparse
        # union, null-filled) at a common supertype.
        data = pl.concat(frames, how="diagonal_relaxed")
        # Attach the per-study MaveDB 'assay facts' (constant-per-gene assay_
        # keyword columns) by accession, so the assay characteristics are queryable
        # alongside every variant. (Score calibrations are a separate companion
        # table — wrong grain to join per-variant.)
        facts = pl.read_parquet(input.assay_facts)
        assert set(data["mavedb_urn"].unique()) <= set(facts["mavedb_urn"]), (
            "sge: dataset has mavedb_urn(s) absent from assay_facts"
        )
        data = data.join(
            facts.select("mavedb_urn", pl.col("^assay_.*$")),
            on="mavedb_urn",
            how="left",
        )
        data.write_parquet(output[0])
