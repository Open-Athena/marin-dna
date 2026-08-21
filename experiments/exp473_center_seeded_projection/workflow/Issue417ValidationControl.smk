"""Additive exact-validation control for the two existing issue #417 models."""

import re
from pathlib import Path

MODELS = [dict(model) for model in config["models"]]
MODEL_BY_NAME = {model["name"]: model for model in MODELS}
assert len(MODELS) == len(MODEL_BY_NAME) == 19
MODEL_NAMES = list(MODEL_BY_NAME)
MODEL_RE = "|".join(re.escape(name) for name in MODEL_NAMES)
SOURCES = {arm: dict(source) for arm, source in config["sources"].items()}
RESULTS_ROOT = str(config["results_root"]).rstrip("/")
assert config["purpose"] == "damage_control_issue417_validation_control"
assert config["interpretation_allowed"] is False
assert config["vep_held_out_access"] is False


def model_config(name):
    return MODEL_BY_NAME[name]


def source_config_for_model(wildcards):
    return SOURCES[model_config(wildcards.model)["arm"]]


def validation_input_for_model(wildcards):
    arm = model_config(wildcards.model)["arm"]
    return local(f"work/issue417_validation_control/validation/{arm}.jsonl.zst")


SCORE_OUTPUTS = [f"{RESULTS_ROOT}/scores/{name}.parquet" for name in MODEL_NAMES]
CELL_MANIFESTS = [f"{RESULTS_ROOT}/scores/{name}.manifest.json" for name in MODEL_NAMES]
ANALYSIS_OUTPUTS = {
    "points": f"{RESULTS_ROOT}/analysis/loss_points.parquet",
    "curves": f"{RESULTS_ROOT}/analysis/curve_summary.parquet",
    "summary": f"{RESULTS_ROOT}/analysis/summary.md",
    "manifest": f"{RESULTS_ROOT}/analysis/manifest.json",
}


rule issue_473_issue417_control_download_source:
    output:
        local("work/issue417_validation_control/validation/{arm}.jsonl.zst"),
    wildcard_constraints:
        arm="|".join(re.escape(arm) for arm in SOURCES),
    params:
        repo=lambda wc: SOURCES[wc.arm]["repo"],
        revision=lambda wc: SOURCES[wc.arm]["revision"],
        filename=lambda wc: SOURCES[wc.arm]["filename"],
    shell:
        "python -m exp473_center_seeded_projection.native_validation_replay download "
        "--repo '{params.repo}' --revision '{params.revision}' "
        "--filename '{params.filename}' --output '{output}'"


rule issue_473_issue417_control_download_model:
    output:
        temp(directory(local("work/issue417_validation_control/checkpoints/{model}"))),
    wildcard_constraints:
        model=MODEL_RE,
    params:
        gcs_path=lambda wc: model_config(wc.model)["gcs_path"],
    shell:
        "mkdir -p {output} && gcloud storage cp -r '{params.gcs_path}/*' {output}/"


rule issue_473_issue417_control_score:
    input:
        checkpoint=local("work/issue417_validation_control/checkpoints/{model}"),
        validation=validation_input_for_model,
    output:
        score=f"{RESULTS_ROOT}/scores/{{model}}.parquet",
        manifest=f"{RESULTS_ROOT}/scores/{{model}}.manifest.json",
    wildcard_constraints:
        model=MODEL_RE,
    threads: 4
    resources:
        gpu=1,
        mem_mb=16000,
    params:
        arm=lambda wc: model_config(wc.model)["arm"],
        step=lambda wc: int(model_config(wc.model)["step"]),
        checkpoint_uri=lambda wc: model_config(wc.model)["gcs_path"],
        validation_repo=lambda wc: source_config_for_model(wc)["repo"],
        validation_revision=lambda wc: source_config_for_model(wc)["revision"],
        validation_filename=lambda wc: source_config_for_model(wc)["filename"],
        expected_rows=lambda wc: int(source_config_for_model(wc)["expected_rows"]),
        compile_flag=("--torch-compile" if config["inference"]["torch_compile"] else ""),
    shell:
        "python -m exp473_center_seeded_projection.issue417_validation_control score "
        "--checkpoint '{input.checkpoint}' --validation '{input.validation}' "
        "--output '{output.score}' --manifest '{output.manifest}' "
        "--arm '{params.arm}' --step {params.step} "
        "--expected-rows {params.expected_rows} "
        "--checkpoint-uri '{params.checkpoint_uri}' "
        "--validation-repo '{params.validation_repo}' "
        "--validation-revision '{params.validation_revision}' "
        "--validation-filename '{params.validation_filename}' "
        "--batch-size {config[inference][batch_size]} "
        "--num-workers {config[inference][num_workers]} {params.compile_flag}"


rule issue_473_issue417_control_analyze:
    input:
        scores=SCORE_OUTPUTS,
        manifests=CELL_MANIFESTS,
    output:
        **ANALYSIS_OUTPUTS,
    threads: 2
    resources:
        mem_mb=8000,
    params:
        score_args=lambda wc, input: " ".join(str(path) for path in input.scores),
        manifest_args=lambda wc, input: " ".join(str(path) for path in input.manifests),
        output_dir=lambda wc, output: str(Path(output.points).parent),
    shell:
        "python -m exp473_center_seeded_projection.issue417_validation_control analyze "
        "--scores {params.score_args} --cell-manifests {params.manifest_args} "
        "--output-dir '{params.output_dir}'"


rule issue_473_issue417_validation_control:
    """Reconstruct both missing #417 validation curves without VEP access."""
    input:
        list(ANALYSIS_OUTPUTS.values()),
