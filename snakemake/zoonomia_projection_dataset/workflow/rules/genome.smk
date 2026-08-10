"""Genome download + 2bit + chrom_sizes + N-region BED for hg38.

Pipeline is human-only by design (no ``{species}`` wildcard); outputs sit
under ``results/human/`` to keep the species explicit and to separate
genome-sequence files from interval-style outputs (which live under
``results/human/intervals/``).
"""


rule download_genome:
    output:
        "results/human/genome.fa.gz",
    params:
        url=config["genome_url"],
    shell:
        "wget -q -O {output} {params.url}"


rule genome_to_2bit:
    input:
        "results/human/genome.fa.gz",
    output:
        "results/human/genome.2bit",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "zcat {input} | faToTwoBit stdin {output}"


rule chrom_sizes:
    input:
        "results/human/genome.2bit",
    output:
        "results/human/chrom.sizes",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} {output}"


rule chrom_sizes_filtered:
    """Restrict to ``standard_chroms`` (autosomes + X + Y)."""
    input:
        "results/human/chrom.sizes",
    output:
        "results/human/chrom.sizes.filtered",
    run:
        df = pd.read_csv(input[0], sep="\t", header=None, names=["chrom", "size"])
        df = df[df["chrom"].isin(STANDARD_CHROMS)]
        assert len(df) == len(
            STANDARD_CHROMS
        ), f"missing chroms in 2bit: {set(STANDARD_CHROMS) - set(df['chrom'])}"
        df.to_csv(output[0], sep="\t", header=False, index=False)


rule undefined_regions:
    """N-region BED. ``twoBitInfo -nBed`` reports stretches of undefined bases."""
    input:
        "results/human/genome.2bit",
    output:
        "results/human/intervals/undefined.bed",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} /dev/stdout -nBed > {output}"
