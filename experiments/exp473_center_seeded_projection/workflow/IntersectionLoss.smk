"""Isolated paired intersection-loss workflow for issue #473.

This workflow is additive and runs in the pinned official ``evals_v2``
environment.  It imports the unchanged official causal-LM scorer but includes
none of the maintained VEP workflow rules.
"""

import re
from pathlib import Path

from exp473_center_seeded_projection.intersection_loss import (
    analyze_loss_scores,
    score_intersection,
)

MODELS = [dict(model) for model in config["models"]]
MODEL_BY_NAME = {model["name"]: model for model in MODELS}
assert MODELS and len(MODEL_BY_NAME) == len(MODELS)
MODEL_NAMES = list(MODEL_BY_NAME)
MODEL_RE = "|".join(re.escape(name) for name in MODEL_NAMES)
SOURCES = {key: dict(value) for key, value in config["sources"].items()}
RESULTS_ROOT = str(config["results_root"]).rstrip("/")
assert config["vep_held_out_access"] is False
assert config["split"] == "chromosome_18_paired_intersection"


def model_config(model_name):
    return MODEL_BY_NAME[model_name]


def source_for_model(wildcards):
    model = model_config(wildcards.model)
    key = f"{model['region']}_{model['policy']}"
    source = SOURCES[key]
    assert source["region"] == model["region"]
    assert source["policy"] == model["policy"]
    uri = source["uri"]
    return uri if uri.startswith("s3://") else local(uri)


SCORE_OUTPUTS = expand(
    f"{RESULTS_ROOT}/scores/{{model}}.parquet",
    model=MODEL_NAMES,
)
ANALYSIS_OUTPUTS = {
    "points": f"{RESULTS_ROOT}/analysis/paired_loss_metrics.parquet",
    "samples": f"{RESULTS_ROOT}/analysis/paired_loss_bootstrap_samples.parquet",
    "deltas": f"{RESULTS_ROOT}/analysis/paired_loss_deltas.parquet",
    "summary": f"{RESULTS_ROOT}/analysis/summary.md",
    "manifest": f"{RESULTS_ROOT}/analysis/manifest.json",
}


rule issue_473_intersection_download_model:
    input:
        lambda wildcards: [],
    output:
        temp(directory(local("work/issue473_intersection_loss/checkpoints/{model}"))),
    wildcard_constraints:
        model=MODEL_RE,
    params:
        gcs_path=lambda wc: model_config(wc.model)["gcs_path"],
    shell:
        "mkdir -p {output} && " "gcloud storage cp -r '{params.gcs_path}/*' {output}/"


rule issue_473_intersection_score:
    input:
        checkpoint=local("work/issue473_intersection_loss/checkpoints/{model}"),
        sequences=source_for_model,
    output:
        f"{RESULTS_ROOT}/scores/{{model}}.parquet",
    wildcard_constraints:
        model=MODEL_RE,
    threads: 4
    resources:
        gpu=1,
        mem_mb=12000,
    params:
        arm=lambda wc: model_config(wc.model)["arm"],
        step=lambda wc: int(model_config(wc.model)["step"]),
    run:
        score_intersection(
            input.checkpoint,
            input.sequences,
            output[0],
            arm=params.arm,
            step=params.step,
            batch_size=int(config["inference"]["batch_size"]),
            num_workers=int(config["inference"]["num_workers"]),
            torch_compile=bool(config["inference"]["torch_compile"]),
        )


rule issue_473_intersection_analyze:
    input:
        SCORE_OUTPUTS,
    output:
        **ANALYSIS_OUTPUTS,
    threads: 4
    resources:
        mem_mb=12000,
    run:
        output_dir = Path(output.points).parent
        analyze_loss_scores(
            list(input),
            output_dir,
            n_bootstrap=int(config["analysis"]["n_bootstrap"]),
            seed=int(config["analysis"]["seed"]),
        )
        assert Path(output.points).is_file()
        assert Path(output.samples).is_file()
        assert Path(output.deltas).is_file()
        assert Path(output.summary).is_file()
        assert Path(output.manifest).is_file()


rule issue_473_intersection_loss:
    """Score and analyze every paired loss checkpoint without VEP data."""
    input:
        list(ANALYSIS_OUTPUTS.values()),
