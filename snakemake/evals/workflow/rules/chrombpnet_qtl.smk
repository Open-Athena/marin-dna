from marin_dna.pipelines.evals.chrombpnet_qtl import (
    STANDARDIZED_QTL,
    build_qtl_dataset,
)


# Canonical caQTL/dsQTL accessibility-QTL datasets from the **standardized**
# ChromBPNet benchmark files (Synapse syn64126763 children; issues #309/#310/#262).
# One all-chroms variant set per dataset — [chrom, pos, ref, alt, label, effect] plus
# carried ChromBPNet/Enformer baseline scores — split odd/even by the generic
# split_dataset_by_chrom and uploaded by the generic hf_upload. caQTL is native hg38;
# dsQTL is hg19 and lifted. See snakemake/evals/README.md + the chrombpnet_qtl library
# module for provenance and the supervised official-metrics protocol.

# Guard against config/spec drift: the canonical Synapse IDs live in the library spec;
# config repoints the download. They must agree.
for _ds, _spec in STANDARDIZED_QTL.items():
    assert config["chrombpnet_qtl"][f"{_ds}_synapse_id"] == _spec.synapse_id, (
        f"config chrombpnet_qtl.{_ds}_synapse_id != STANDARDIZED_QTL[{_ds!r}].synapse_id"
    )


rule chrombpnet_qtl_download:
    """Download a standardized caQTL/dsQTL benchmark file from Synapse over plain HTTP
    using a Personal Access Token (no synapseclient). Requires a free Synapse account;
    export the PAT as SYNAPSE_AUTH_TOKEN before running. Unlike the plain-TSV DART-Eval
    entities, each standardized FileEntity is a zip wrapping a single .tsv."""
    output:
        "results/chrombpnet_qtl/{ds}.tsv",
    wildcard_constraints:
        ds="caqtl|dsqtl",
    params:
        syn_id=lambda wc: config["chrombpnet_qtl"][f"{wc.ds}_synapse_id"],
    run:
        import io
        import os
        import zipfile

        import requests

        token = os.environ.get("SYNAPSE_AUTH_TOKEN")
        assert token, (
            "set SYNAPSE_AUTH_TOKEN (a Synapse Personal Access Token) to download the "
            "standardized caQTL/dsQTL data"
        )
        url = (
            "https://repo-prod.prod.sagebase.org/repo/v1/entity/"
            f"{params.syn_id}/file"
        )
        resp = requests.get(
            url, headers={"Authorization": f"Bearer {token}"}, timeout=600
        )
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = [n for n in zf.namelist() if n.endswith(".tsv")]
            assert len(names) == 1, (
                f"{params.syn_id}: expected 1 .tsv in the zip, got {names}"
            )
            with open(output[0], "wb") as fh:
                fh.write(zf.read(names[0]))


rule chrombpnet_qtl_dataset_unsplit:
    """Build the canonical genome-oriented caQTL/dsQTL variant set:
    [chrom, pos, ref, alt, label, effect] + carried ChromBPNet/Enformer baseline
    scores. dsQTL is lifted hg19->hg38; check_ref_alt orients ref/alt, flipping
    `effect` and every signed score column on swap. No matching / no subsampling.
    The generic split_dataset_by_chrom then writes the odd/even train/test parquets."""
    input:
        tsv="results/chrombpnet_qtl/{ds}.tsv",
        # Locally-staged GRCh38 (see stage_genome in common.smk) — fai/gzi are pyfaidx
        # sibling indexes, declared so snakemake stages them too. local() marks them
        # on-disk (not storage) to match the producer.
        genome=local("results/genome_staged/GRCh38.fa.gz"),
        genome_fai=local("results/genome_staged/GRCh38.fa.gz.fai"),
        genome_gzi=local("results/genome_staged/GRCh38.fa.gz.gzi"),
    output:
        "results/dataset_unsplit/{ds}.parquet",
    wildcard_constraints:
        ds="caqtl|dsqtl",
    run:
        build_qtl_dataset(
            wildcards.ds, input.tsv, Genome(input.genome)
        ).write_parquet(output[0])
