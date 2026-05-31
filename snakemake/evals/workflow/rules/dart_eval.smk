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


rule dart_eval_stage_genome:
    """Stage the canonical bgzipped GRCh38 reference (+ .fai/.gzi indexes)
    onto local disk so check_ref_alt reads from disk instead of doing a
    per-variant S3 round-trip (~100x faster on ~110k variants). Downloaded
    once via boto3 (the s3 storage plugin's dependency — no s3fs needed) and
    kept local via local() so snakemake doesn't round-trip it back to storage.
    pyfaidx needs the .fai (and, for BGZF, .gzi) as same-named siblings."""
    output:
        fa=local("results/genome_staged/GRCh38.fa.gz"),
        fai=local("results/genome_staged/GRCh38.fa.gz.fai"),
        gzi=local("results/genome_staged/GRCh38.fa.gz.gzi"),
    params:
        src=config["canonical_genome_path"],
    run:
        import boto3
        from urllib.parse import urlparse

        u = urlparse(params.src)
        bucket, key = u.netloc, u.path.lstrip("/")
        s3 = boto3.client("s3")
        s3.download_file(bucket, key, output.fa)
        s3.download_file(bucket, key + ".fai", output.fai)
        s3.download_file(bucket, key + ".gzi", output.gzi)


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
        # Locally-staged GRCh38 (see dart_eval_stage_genome) — fai/gzi are
        # pyfaidx sibling indexes, declared so snakemake stages them too.
        # local() marks them as on-disk (not storage) to match the producer.
        genome=local("results/genome_staged/GRCh38.fa.gz"),
        genome_fai=local("results/genome_staged/GRCh38.fa.gz.fai"),
        genome_gzi=local("results/genome_staged/GRCh38.fa.gz.gzi"),
    output:
        "results/dataset_unsplit/{ds}.parquet",
    wildcard_constraints:
        ds="caqtl|dsqtl",
    params:
        lift=lambda wc: DART_EVAL[wc.ds]["lift"],
    run:
        # Select the parser inside the run block — a function object in `params`
        # has an unstable repr (memory address) that would churn snakemake's
        # params-hash and trigger spurious reruns.
        parse = DART_EVAL[wildcards.ds]["parse"]
        # infer_schema_length=None: scan the whole TSV for dtype inference
        # (the files are bounded; avoids mis-inferring a flag column from a
        # non-representative first chunk).
        raw = pl.read_csv(input.tsv, separator="\t", infer_schema_length=None)
        V = parse(raw)
        annotate_variants(
            V,
            genome=Genome(input.genome),
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
