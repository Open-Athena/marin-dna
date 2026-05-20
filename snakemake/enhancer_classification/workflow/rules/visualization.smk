VIS_CONFIG = config.get("visualization", {})
VIS_REGIONS = VIS_CONFIG.get("regions", [])


def get_region(name: str) -> dict:
    """Look up a visualization region config by name."""
    for region in VIS_REGIONS:
        if region["name"] == name:
            return region
    raise ValueError(f"Unknown visualization region: {name}")


rule predict_region:
    input:
        genome=lambda wc: f"results/genome/{get_region(wc.name)['genome']}.fa.gz",
        checkpoint="results/model/{model}/{dataset}/best.ckpt",
    output:
        "results/visualization/{model}/{dataset}/{name}.bedgraph",
    threads: workflow.cores
    params:
        chrom=lambda wc: get_region(wc.name)["chrom"],
        start=lambda wc: get_region(wc.name)["start"],
        end=lambda wc: get_region(wc.name)["end"],
        window_size=VIS_CONFIG.get("window_size", 255),
        step_size=VIS_CONFIG.get("step_size", 32),
    shell:
        """
        uv run python -m marin_dna.pipelines.enhancer_classification.predict \
            --genome {input.genome} \
            --checkpoint {input.checkpoint} \
            --chrom {params.chrom} \
            --start {params.start} \
            --end {params.end} \
            --window-size {params.window_size} \
            --step-size {params.step_size} \
            --output {output} \
            --name "Enhancer probability ({wildcards.model}/{wildcards.dataset})"
        """
