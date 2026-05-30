from marin_dna.pipelines.evals.dart_eval import (
    annotate_variants,
    parse_caqtl,
    parse_dsqtl,
)


# DART-Eval Task-5 chromatin-accessibility QTL datasets. caQTL is native
# GRCh38; dsQTL is hg19 and lifted. Positives = significant QTLs in peaks,
# negatives = control variants in peaks; kept at natural ratio (no matching /
# no subsampling). See snakemake/evals/README.md and the dart_eval library
# module for provenance.
DART_EVAL = {
    "caqtl": {
        "synapse_id": config["dart_eval"]["caqtl_synapse_id"],
        "parse": parse_caqtl,
        "lift": False,
    },
    "dsqtl": {
        "synapse_id": config["dart_eval"]["dsqtl_synapse_id"],
        "parse": parse_dsqtl,
        "lift": True,
    },
}


rule dart_eval_download:
    """Download a DART-Eval input TSV from Synapse over plain HTTP using a
    Personal Access Token (no synapseclient dependency). Requires a free
    Synapse account; export the PAT as SYNAPSE_AUTH_TOKEN before running."""
    output:
        "results/dart_eval/{ds}.tsv",
    wildcard_constraints:
        ds="caqtl|dsqtl",
    params:
        syn_id=lambda wc: DART_EVAL[wc.ds]["synapse_id"],
    shell:
        'test -n "$SYNAPSE_AUTH_TOKEN" || {{ echo "ERROR: set SYNAPSE_AUTH_TOKEN (a Synapse Personal Access Token) to download DART-Eval data" >&2; exit 1; }}; '
        'curl -fL -H "Authorization: Bearer $SYNAPSE_AUTH_TOKEN" '
        '"https://repo-prod.prod.sagebase.org/repo/v1/entity/{params.syn_id}/file" '
        "-o {output}"


rule dart_eval_dataset_unsplit:
    """Parse + annotate a DART-Eval QTL TSV into the standard variant schema
    with consequence + distance annotations. No matching, no subsampling —
    `split_dataset_by_chrom` then produces the train/test parquets."""
    input:
        tsv="results/dart_eval/{ds}.tsv",
        consequences=expand("results/consequences/{chrom}.parquet", chrom=CHROMS),
        exon_pc="results/intervals/exon_pc.parquet",
        exon_nc="results/intervals/exon_nc.parquet",
        tss_pc="results/intervals/tss_pc.parquet",
        tss_nc="results/intervals/tss_nc.parquet",
    output:
        "results/dataset_unsplit/{ds}.parquet",
    wildcard_constraints:
        ds="caqtl|dsqtl",
    params:
        # Canonical bgzipped + indexed GRCh38 read directly from S3 via pyfaidx
        # (same reference the materialize rule uses). check_ref_alt does a
        # single-base lookup per variant.
        genome_path=config["canonical_genome_path"],
        lift=lambda wc: DART_EVAL[wc.ds]["lift"],
    run:
        # Select the parser inside the run block — a function object in `params`
        # has an unstable repr (memory address) that would churn snakemake's
        # params-hash and trigger spurious reruns.
        parse = DART_EVAL[wildcards.ds]["parse"]
        raw = pl.read_csv(input.tsv, separator="\t", infer_schema_length=10000)
        V = parse(raw)
        annotate_variants(
            V,
            genome=Genome(params.genome_path),
            consequence_paths=list(input.consequences),
            chroms=CHROMS,
            exon_pc=pl.read_parquet(input.exon_pc),
            exon_nc=pl.read_parquet(input.exon_nc),
            tss_pc=pl.read_parquet(input.tss_pc),
            tss_nc=pl.read_parquet(input.tss_nc),
            exon_proximal_dist=config["exon_proximal_dist"],
            tss_proximal_dist=config["tss_proximal_dist"],
            lift=params.lift,
            name=wildcards.ds,
        ).write_parquet(output[0])
