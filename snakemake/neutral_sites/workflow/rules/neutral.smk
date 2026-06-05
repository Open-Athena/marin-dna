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
        genome = Genome(config["genome_path"])
        df = enumerate_positions(intervals, genome, set(CHROMS))
        # rmsk/conserved intervals are non-overlapping so this is normally a
        # no-op, but a duplicate (chrom,pos) would double-weight a calibration
        # bin — drop loudly if it ever happens.
        before = len(df)
        df = df.drop_duplicates(["chrom", "pos"]).reset_index(drop=True)
        if len(df) < before:
            print(
                f"[neutral_sites] dropped {before - len(df):,} duplicate positions"
            )
        assert len(df) > 0, "empty neutral set"
        assert df["ref"].isin(set("ACGT")).all()
        assert (df["pos"] >= 1).all()
        df.to_parquet(output[0], index=False)
        print(
            f"[neutral_sites] {len(df):,} neutral sites across "
            f"{df['chrom'].nunique()} chromosomes -> {output[0]}"
        )
