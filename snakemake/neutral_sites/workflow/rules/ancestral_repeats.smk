"""Ancestral repeats = hg38 repeats overlapping mm39 repeats lifted to hg38.

Mirrors GPN-Star's ``get_ancestral_repeats`` (Simple_repeat / Low_complexity
excluded inside ``parse_rmsk`` — see its docstring for why that differs from
the reference's no-op awk filter).
"""


rule rmsk_target_bed:
    input:
        "results/downloads/rmsk_target.txt.gz",
    output:
        "results/ancestral/target.rmsk.bed",
    run:
        parse_rmsk(input[0]).to_csv(output[0], sep="\t", header=False, index=False)


rule rmsk_outgroup_bed:
    input:
        "results/downloads/rmsk_outgroup.txt.gz",
    output:
        "results/ancestral/outgroup.rmsk.bed",
    run:
        parse_rmsk(input[0]).to_csv(output[0], sep="\t", header=False, index=False)


rule liftover_outgroup:
    """Lift mm39 repeats into hg38 coordinates via the reciprocal-best chain."""
    input:
        bed="results/ancestral/outgroup.rmsk.bed",
        chain="results/downloads/mm39.hg38.rbest.chain",
    output:
        mapped="results/ancestral/outgroup_in_target.bed",
        unmapped="results/ancestral/outgroup_unmapped.bed",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "liftOver {input.bed} {input.chain} {output.mapped} {output.unmapped}"


rule ancestral_repeats:
    """hg38 repeats overlapping any lifted mm39 repeat (`-u` = whole A feature)."""
    input:
        target="results/ancestral/target.rmsk.bed",
        lifted="results/ancestral/outgroup_in_target.bed",
    output:
        "results/ancestral/ancestral_repeats.bed",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "bedtools intersect -u -a {input.target} -b {input.lifted} > {output}"
