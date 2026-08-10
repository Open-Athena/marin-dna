rule download_genome_and_annotation:
    output:
        "results/genome/{g}.2bit",
        "results/annotation/{g}.gtf.gz",
    conda:
        "../envs/bioinformatics.yaml"
    threads: 4
    params:
        tmp_dir=directory("tmp/{g}"),
        genome_path=lambda wildcards: genomes.loc[wildcards.g, "genome_path"],
        annotation_path=lambda wildcards: genomes.loc[wildcards.g, "annotation_path"],
    shell:
        """
        mkdir -p {params.tmp_dir} && cd {params.tmp_dir} \
            && datasets download genome accession {wildcards.g} --include genome,gtf \
            && unzip ncbi_dataset.zip && cd - && faToTwoBit {params.genome_path} {output[0]} \
            && gzip -c {params.annotation_path} >{output[1]} \
            && rm -r {params.tmp_dir}
        """
