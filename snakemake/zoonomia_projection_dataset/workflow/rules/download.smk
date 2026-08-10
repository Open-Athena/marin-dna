"""Download phyloP_447m bigWig.

URL comes from ``marin_dna.pipelines.evals.conservation.CONSERVATION_TRACKS`` — the
single source of truth for conservation-track URLs.
"""


rule download_bigwig:
    output:
        "results/bigwig/{track}.bw",
    wildcard_constraints:
        track="|".join(CONSERVATION_TRACKS),
    params:
        url=lambda wc: CONSERVATION_TRACKS[wc.track],
    shell:
        "wget -q {params.url} -O {output}"


rule download_annotation:
    """Ensembl human GTF; release pinned by ``ensembl_release`` in config."""
    output:
        f"results/annotation/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz",
    params:
        url=(
            f"https://ftp.ensembl.org/pub/release-{config['ensembl_release']}/gtf/"
            f"homo_sapiens/Homo_sapiens.GRCh38.{config['ensembl_release']}.gtf.gz"
        ),
    shell:
        "wget -q -O {output} {params.url}"
