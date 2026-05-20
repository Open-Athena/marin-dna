from marin_dna.data.utils import load_annotation


rule annotation_download:
    output:
        "results/annotation.gtf.gz",
    params:
        url=config["annotation_url"],
    shell:
        "wget -O {output} {params.url}"


rule extract_tss_pc:
    input:
        "results/annotation.gtf.gz",
    output:
        "results/intervals/tss_pc.parquet",
    run:
        load_annotation(input[0]).pipe(get_tss).write_parquet(output[0])


rule extract_tss_nc:
    input:
        "results/annotation.gtf.gz",
    output:
        "results/intervals/tss_nc.parquet",
    run:
        # `nc` = transcript_biotype != "protein_coding" — matches VEP's structural test for non_coding_transcript_exon_variant (any transcript without a translation).
        nc_filter = pl.col("transcript_biotype") != "protein_coding"
        load_annotation(input[0]).pipe(get_tss, biotype_filter=nc_filter).write_parquet(
            output[0]
        )


rule extract_exon_pc:
    input:
        "results/annotation.gtf.gz",
    output:
        "results/intervals/exon_pc.parquet",
    run:
        load_annotation(input[0]).pipe(get_exon).write_parquet(output[0])


rule extract_exon_nc:
    input:
        "results/annotation.gtf.gz",
    output:
        "results/intervals/exon_nc.parquet",
    run:
        nc_filter = pl.col("transcript_biotype") != "protein_coding"
        load_annotation(input[0]).pipe(
            get_exon, biotype_filter=nc_filter
        ).write_parquet(output[0])


rule extract_gene_biotype:
    input:
        "results/annotation.gtf.gz",
    output:
        "results/intervals/gene_biotype.parquet",
    run:
        # Per-gene biotype lookup for the eqtl pipeline.
        gene_id_re = r'gene_id "([^;]*)";'
        biotype_re = r'gene_biotype "([^;]*)";'
        (
            load_annotation(input[0])
            .filter(pl.col("feature") == "gene")
            .with_columns(
                pl.col("attribute").str.extract(gene_id_re).alias("gene_id"),
                pl.col("attribute").str.extract(biotype_re).alias("gene_biotype"),
            )
            .select(["gene_id", "gene_biotype"])
            .unique()
            .write_parquet(output[0])
        )
