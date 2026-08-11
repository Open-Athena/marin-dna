"""Subset derivation for the zoonomia projection dataset."""

TSS_FLANK = int(config.get("tss_flank", 256))


rule derive_subset_v2_tss_mrna:
    """Generate v2 query_names: windows overlapping any mRNA TSS ± tss_flank."""
    input:
        gtf=f"results/annotation/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz",
        bed="results/human/intervals/filtered/min{min_p}.bed.gz",
    output:
        band="results/projection/min{min_p}/subsets_def/v2_band.bed",
        names="results/projection/min{min_p}/subsets_def/v2.query_names.txt",
    conda:
        "../envs/bioinformatics.yaml"
    params:
        flank=TSS_FLANK,
    run:
        from marin_dna_zoonomia_projection.projection.tss import (
            write_mrna_tss_band_bed,
        )

        n_band = write_mrna_tss_band_bed(input.gtf, params.flank, output.band)
        assert n_band > 30_000, f"unexpectedly few mRNA TSS bands: {n_band}"
        shell(
            "bedtools intersect -u -a {input.bed} -b {output.band} "
            "| cut -f4 > {output.names}"
        )
        with open(output.names) as fh:
            kept = sum(1 for _ in fh)
        assert kept > 0, "v2 subset is empty"


ruleorder: subset_dataset_derived > subset_dataset


rule subset_dataset_derived:
    """Lazy-filter all_species_with_sequence to a derived subset."""
    input:
        all_species="results/projection/min{min_p}/all_species_with_sequence.parquet",
        names="results/projection/min{min_p}/subsets_def/{subset}.query_names.txt",
    output:
        "results/projection/min{min_p}/subsets/{subset}.parquet",
    threads: 1
    resources:
        # filter_to_subset eagerly decodes the ~10 GB Parquet to ~25 GB in RAM
        # (lazy sink_parquet path was buggy; see src/marin_dna/projection/subset.py).
        mem_mb=32000,
    run:
        filter_to_subset(input.all_species, input.names, output[0])
