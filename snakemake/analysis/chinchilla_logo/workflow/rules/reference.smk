"""Download the browser-authoritative UCSC sequence inventory."""


rule download_chrom_sizes:
    output:
        CHROM_SIZES,
    params:
        url=config["assembly"]["chrom_sizes_url"],
    shell:
        "curl --fail --location --retry 3 {params.url:q} --output {output:q}"


rule download_two_bit:
    output:
        TWO_BIT,
    params:
        url=config["assembly"]["two_bit_url"],
    shell:
        "curl --fail --location --retry 3 {params.url:q} --output {output:q}"


rule convert_reference_fasta:
    input:
        TWO_BIT,
    output:
        fasta=FASTA,
        fai=FASTA_INDEX,
    conda:
        "../envs/ucsc.yaml"
    shell:
        "twoBitToFa {input:q} {output.fasta:q} && samtools faidx {output.fasta:q}"
