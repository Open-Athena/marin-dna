"""Fetch the GPN-Star calibration inputs (UCSC). Plain ``wget``, no conda."""


rule download_rmsk_target:
    output:
        "results/downloads/rmsk_target.txt.gz",
    params:
        url=config["rmsk_target_url"],
    shell:
        "wget -q -O {output} {params.url}"


rule download_rmsk_outgroup:
    output:
        "results/downloads/rmsk_outgroup.txt.gz",
    params:
        url=config["rmsk_outgroup_url"],
    shell:
        "wget -q -O {output} {params.url}"


rule download_chain:
    """mm39->hg38 reciprocal-best chain, gunzipped (liftOver needs it plain)."""
    output:
        "results/downloads/mm39.hg38.rbest.chain",
    params:
        url=config["chain_url"],
    shell:
        "wget -q -O {output}.gz {params.url} && gunzip -f {output}.gz"


rule download_phylop:
    output:
        "results/downloads/phylop.bw",
    params:
        url=PHYLOP_URL,
    shell:
        "wget -q -O {output} {params.url}"


rule download_phastcons:
    output:
        "results/downloads/phastcons.bw",
    params:
        url=PHASTCONS_URL,
    shell:
        "wget -q -O {output} {params.url}"
