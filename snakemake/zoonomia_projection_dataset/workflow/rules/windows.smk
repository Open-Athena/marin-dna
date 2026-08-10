"""Tile + N-filter the genome into fixed-length windows, per chromosome.

Per-chrom outputs (rather than one whole-genome BED) so each
score_windows_chrom worker reads only its ~30 MB BED instead of ~700 MB.
Window names embed the chrom (``win_<chrom>_<NNNNNNNNN>``) so they stay
unique after merging across chroms.
"""


rule make_windows:
    """255 bp windows on a single chromosome, dropping any that overlap N regions."""
    input:
        sizes="results/human/chrom.sizes.filtered",
        undefined="results/human/intervals/undefined.bed",
    output:
        "results/human/intervals/windows/{chrom}.bed.gz",
    wildcard_constraints:
        chrom="|".join(STANDARD_CHROMS),
    conda:
        "../envs/bioinformatics.yaml"
    params:
        w=WINDOW_SIZE,
        s=STEP_SIZE,
    shell:
        r"""
        TMPSIZES=$(mktemp)
        trap "rm -f $TMPSIZES" EXIT
        awk -v c={wildcards.chrom} '$1 == c' {input.sizes} >$TMPSIZES
        bedtools makewindows -g $TMPSIZES -w {params.w} -s {params.s} \
            | awk -v w={params.w} -v c={wildcards.chrom} 'BEGIN {{OFS="\t"}}
              $3 - $2 == w {{ printf "%s\t%d\t%d\twin_%s_%09d\n", $1, $2, $3, c, NR }}' \
            | bedtools intersect -a stdin -b {input.undefined} -v \
            | gzip >{output}
        """
