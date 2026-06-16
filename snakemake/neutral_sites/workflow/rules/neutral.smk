"""Neutral set = ancestral repeats ∩ low-constraint sites → per-base parquet."""


rule neutral_intervals:
    """Clip ancestral repeats to the low-constraint runs. Both inputs are sets
of non-overlapping intervals, so the intersection is non-overlapping too."""
    input:
        ancestral="results/ancestral/ancestral_repeats.bed",
        conserved="results/conserved/conserved_sites.bed",
    output:
        "results/neutral/neutral_intervals.bed",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "bedtools intersect -a {input.ancestral} -b {input.conserved} "
        "| cut -f1-3 | bedtools sort -i - > {output}"


rule neutral_sites_parquet:
    """Enumerate every base in the neutral intervals → (chrom, pos, ref)."""
    input:
        "results/neutral/neutral_intervals.bed",
    output:
        "results/neutral_sites.parquet",
    run:
        intervals = pd.read_csv(
            input[0],
            sep="\t",
            header=None,
            names=["chrom", "start", "end"],
            dtype={"chrom": str},
        )
        # enumerate_positions handles the chr→bare boundary, 1-based pos, ACGT
        # filtering, and (chrom,pos) dedup — all unit-tested in the library.
        df = enumerate_positions(
            intervals, Genome(config["genome_path"]), set(CHROMS)
        )
        assert len(df) > 0, "empty neutral set"
        df.to_parquet(output[0], index=False)
        print(
            f"[neutral_sites] {len(df):,} neutral sites across "
            f"{df['chrom'].nunique()} chromosomes -> {output[0]}"
        )
