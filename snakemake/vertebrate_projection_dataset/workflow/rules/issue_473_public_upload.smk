"""Additive public-only upload rules for validated issue #473 artifacts."""

import re

from marin_dna_vertebrate_projection.issue_473.public_upload import (
    upload_public_validated_dataset,
)
from marin_dna_vertebrate_projection.issue_473.publication import (
    parse_publication_datasets,
)
from marin_dna_vertebrate_projection.provenance import resolve_pipeline_commit

ISSUE_473_PUBLIC_UPLOAD_COMMIT = resolve_pipeline_commit()
ISSUE_473_PUBLIC_UPLOAD_DATASETS = parse_publication_datasets(
    config["publication_datasets"]
)
ISSUE_473_PUBLIC_UPLOAD_BY_KEY = {
    dataset.key: dataset for dataset in ISSUE_473_PUBLIC_UPLOAD_DATASETS
}
ISSUE_473_PUBLIC_UPLOAD_KEYS = list(ISSUE_473_PUBLIC_UPLOAD_BY_KEY)
ISSUE_473_PUBLIC_UPLOAD_KEY_RE = "|".join(
    re.escape(key) for key in ISSUE_473_PUBLIC_UPLOAD_KEYS
)
ISSUE_473_PUBLIC_UPLOAD_ARTIFACT_COMMIT = str(config["publication_artifact_commit"])
ISSUE_473_PUBLIC_UPLOAD_ARTIFACT_CONFIG = str(
    config["publication_artifact_config_sha256"]
)
ISSUE_473_PUBLIC_UPLOAD_ARTIFACT_GENERATION = str(
    config["publication_artifact_generation"]
)
ISSUE_473_PUBLIC_UPLOAD_ARTIFACT_ROOT = (
    f"results/{config['pipeline_version']}/"
    f"{ISSUE_473_PUBLIC_UPLOAD_ARTIFACT_COMMIT}/"
    f"{ISSUE_473_PUBLIC_UPLOAD_ARTIFACT_CONFIG}/"
    f"{ISSUE_473_PUBLIC_UPLOAD_ARTIFACT_GENERATION}"
)
ISSUE_473_PUBLIC_UPLOAD_HF_ROOT = f"{ISSUE_473_PUBLIC_UPLOAD_ARTIFACT_ROOT}/hf"
ISSUE_473_PUBLIC_UPLOAD_MANIFEST = (
    f"{ISSUE_473_PUBLIC_UPLOAD_ARTIFACT_ROOT}/validation/"
    "hf_publication_manifest.json"
)
ISSUE_473_PUBLIC_UPLOAD_TRAIN_SHARDS = [
    f"shard_{index:04d}" for index in range(int(config["publication_train_shards"]))
]
ISSUE_473_PUBLIC_UPLOAD_VALIDATION_SHARDS = [
    f"shard_{index:04d}"
    for index in range(int(config["publication_validation_shards"]))
]


rule issue_473_public_hf_upload_dataset:
    input:
        manifest=local(ISSUE_473_PUBLIC_UPLOAD_MANIFEST),
        train=lambda wc: [
            local(
                f"{ISSUE_473_PUBLIC_UPLOAD_HF_ROOT}/{wc.dataset}/data/train/"
                f"{shard}.jsonl.zst"
            )
            for shard in ISSUE_473_PUBLIC_UPLOAD_TRAIN_SHARDS
        ],
        validation=lambda wc: [
            local(
                f"{ISSUE_473_PUBLIC_UPLOAD_HF_ROOT}/{wc.dataset}/data/validation/"
                f"{shard}.jsonl.zst"
            )
            for shard in ISSUE_473_PUBLIC_UPLOAD_VALIDATION_SHARDS
        ],
        card=local(f"{ISSUE_473_PUBLIC_UPLOAD_HF_ROOT}/{{dataset}}/README.md"),
    output:
        local(
            f"{ISSUE_473_PUBLIC_UPLOAD_ARTIFACT_ROOT}/public_upload.done/" "{dataset}"
        ),
    wildcard_constraints:
        dataset=ISSUE_473_PUBLIC_UPLOAD_KEY_RE,
    resources:
        hf_uploads=1,
    params:
        repo=lambda wc: ISSUE_473_PUBLIC_UPLOAD_BY_KEY[wc.dataset].hf_repo,
        workers=int(config["hf_upload_workers"]),
    run:
        upload_public_validated_dataset(
            ISSUE_473_PUBLIC_UPLOAD_HF_ROOT,
            input.manifest,
            output[0],
            cohort=wildcards.dataset,
            repo_id=params.repo,
            workers=params.workers,
        )


rule issue_473_all_public_hf:
    input:
        local(
            expand(
                f"{ISSUE_473_PUBLIC_UPLOAD_ARTIFACT_ROOT}/public_upload.done/"
                "{dataset}",
                dataset=ISSUE_473_PUBLIC_UPLOAD_KEYS,
            )
        ),
