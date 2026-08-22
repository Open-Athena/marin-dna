"""Audit fresh-process Mendelian parity for the retained standard-rate BICO adapter."""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import torch
import wandb
from peft import PeftModel
from sklearn.metrics import average_precision_score

from exp479_mntp.bico_lora_gate_audit import _attach_adapter, _source_bundle
from exp479_mntp.bico_lora_mntp import BICO_LORA_STANDARD_EVALUATION_ARTIFACT
from exp479_mntp.bico_vep import score_bico_vep
from exp479_mntp.config import BUDGET_USD, EXPERIMENT_TAGS, MODEL_ID, MODEL_REVISION, WANDB_PROJECT
from exp479_mntp.issue_storage import download_verified_issue_object
from exp479_mntp.lora_reload_audit import (
    assert_reloaded_adapter_contract,
    assert_source_tokenizer_contract,
    configure_training_evaluation_numerics,
)
from exp479_mntp.publishing import assert_budget_reserve, write_cost_estimate
from exp479_mntp.vep import _protocol_scores

WANDB_ENTITY = "gonzalobenegas"
FINAL_EVALUATION_ARTIFACT = (
    f"{WANDB_ENTITY}/{WANDB_PROJECT}/{BICO_LORA_STANDARD_EVALUATION_ARTIFACT}:v0"
)
AUDIT_RUN_NAME = "dna-exp479-bico-lora-standard-rate-mendelian-reload-audit"
AUDIT_ARTIFACT = AUDIT_RUN_NAME
FINAL_READOUT = "step_1000"
CHECKPOINT_PREFIX = "s3://oa-bolinas/issues/479/bico-lora-standard-rate/v1"
GROUPS_PER_SUBSET = 8
EXPECTED_SUBSETS = 8
EXPECTED_ROWS = 640
SCORE_TOLERANCE = 4e-3
MAXIMUM_INSTANCE_HOURS = 0.75

REFERENCE_BUCKET = "oa-bolinas"
REFERENCE_PREFIX = "data/genomes/homo_sapiens/GRCh38/ensembl-release-115"
REFERENCE_NAME = "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa"
REFERENCE_KEY = f"{REFERENCE_PREFIX}/{REFERENCE_NAME}"
REFERENCE_FAI_KEY = f"{REFERENCE_KEY}.fai"
REFERENCE_SIZE_BYTES = 3_151_426_433
REFERENCE_ETAG = '"ee056697b0c4d008ab312fae45f5d67b-376"'
WINDOW_LENGTH = 255
VARIANT_INDEX = WINDOW_LENGTH // 2


@dataclass(frozen=True)
class FaiRecord:
    """One uncompressed FASTA index record."""

    length: int
    offset: int
    line_bases: int
    line_width: int


def parse_fai(payload: str) -> dict[str, FaiRecord]:
    """Parse the four byte-layout fields needed for exact FASTA range reads."""

    records: dict[str, FaiRecord] = {}
    for line in payload.splitlines():
        fields = line.split("\t")
        if len(fields) < 5:
            raise ValueError("FAI row lacks five required fields")
        name = fields[0]
        if name in records:
            raise ValueError(f"duplicate FAI sequence {name}")
        record = FaiRecord(*(int(value) for value in fields[1:5]))
        if min(record.length, record.line_bases, record.line_width) <= 0:
            raise ValueError(f"invalid FAI geometry for {name}")
        if record.offset < 0 or record.line_width < record.line_bases:
            raise ValueError(f"invalid FAI byte layout for {name}")
        records[name] = record
    if not records:
        raise ValueError("FAI contains no sequences")
    return records


def _fasta_byte_offset(position: int, record: FaiRecord) -> int:
    return record.offset + (position // record.line_bases) * record.line_width + (
        position % record.line_bases
    )


def fetch_s3_reference_sequence(
    client: Any,
    records: dict[str, FaiRecord],
    *,
    chrom: str,
    start: int,
    end: int,
) -> str:
    """Fetch one 0-based half-open interval from the uncompressed S3 FASTA."""

    if chrom not in records:
        raise ValueError(f"reference lacks Ensembl sequence {chrom}")
    record = records[chrom]
    if start < 0 or end <= start or end > record.length:
        raise ValueError(f"reference interval outside {chrom}:0-{record.length}")
    first_byte = _fasta_byte_offset(start, record)
    last_byte = _fasta_byte_offset(end - 1, record)
    response = client.get_object(
        Bucket=REFERENCE_BUCKET,
        Key=REFERENCE_KEY,
        Range=f"bytes={first_byte}-{last_byte}",
    )
    sequence = response["Body"].read().replace(b"\n", b"").replace(b"\r", b"")
    if len(sequence) != end - start:
        raise RuntimeError(f"short S3 reference range at {chrom}:{start}-{end}")
    return sequence.decode("ascii").upper()


def select_mendelian_reload_rows(
    scores: pd.DataFrame,
    *,
    groups_per_subset: int = GROUPS_PER_SUBSET,
) -> pd.DataFrame:
    """Select fixed matched groups from every subset used by the Mendelian macro."""

    required = {
        "chrom",
        "pos",
        "ref",
        "alt",
        "label",
        "subset",
        "match_group",
        FINAL_READOUT,
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Mendelian reload frame lacks columns {sorted(missing)}")
    if groups_per_subset <= 0:
        raise ValueError("groups per subset must be positive")
    selected: list[pd.DataFrame] = []
    for _, cell in scores.groupby("subset", sort=True):
        group_counts = cell.groupby("match_group", sort=True).size()
        if len(group_counts) < 30:
            continue
        groups = group_counts.index[:groups_per_subset]
        sample = cell[cell["match_group"].isin(groups)].copy()
        if sample["match_group"].nunique() != groups_per_subset:
            raise RuntimeError("Mendelian reload sample omitted a selected match group")
        if not sample.groupby("match_group")["label"].nunique().eq(2).all():
            raise RuntimeError("Mendelian reload sample lacks both labels in a match group")
        selected.append(sample)
    if not selected:
        raise RuntimeError("Mendelian reload sample contains no qualifying subsets")
    return pd.concat(selected, ignore_index=True).sort_values(
        ["subset", "match_group", "label", "chrom", "pos", "ref", "alt"],
        kind="stable",
        ignore_index=True,
    )


def attach_s3_reference_windows(frame: pd.DataFrame, *, client: Any) -> pd.DataFrame:
    """Attach canonical 255-bp windows through bounded S3 byte-range requests."""

    head = client.head_object(Bucket=REFERENCE_BUCKET, Key=REFERENCE_KEY)
    if (
        int(head.get("ContentLength", -1)) != REFERENCE_SIZE_BYTES
        or head.get("ETag") != REFERENCE_ETAG
        or head.get("AcceptRanges") != "bytes"
    ):
        raise RuntimeError("canonical uncompressed S3 reference object changed")
    fai_payload = client.get_object(Bucket=REFERENCE_BUCKET, Key=REFERENCE_FAI_KEY)[
        "Body"
    ].read()
    records = parse_fai(fai_payload.decode("ascii"))
    sequences: list[str] = []
    for row in frame.itertuples(index=False):
        center = int(row.pos) - 1
        start = center - VARIANT_INDEX
        end = start + WINDOW_LENGTH
        sequence = fetch_s3_reference_sequence(
            client,
            records,
            chrom=str(row.chrom),
            start=start,
            end=end,
        )
        if sequence[VARIANT_INDEX] != str(row.ref):
            raise RuntimeError(
                f"GRCh38 reference mismatch at {row.chrom}:{row.pos}: "
                f"dataset={row.ref}, fasta={sequence[VARIANT_INDEX]}"
            )
        sequences.append(sequence)
    result = frame.copy()
    result["sequence"] = sequences
    return result


def mendelian_macro_auprc(frame: pd.DataFrame, score_column: str) -> float:
    """Compute the unweighted macro AUPRC over retained Mendelian subsets."""

    values = [
        average_precision_score(cell["label"], cell[score_column])
        for _, cell in frame.groupby("subset", sort=True)
    ]
    return float(np.mean(values))


def mendelian_score_parity(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    tolerance: float = SCORE_TOLERANCE,
) -> dict[str, float | int | bool]:
    """Compare paired protocol scores produced in training and after reload."""

    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("Mendelian parity scores must be paired one-dimensional arrays")
    if len(reference) == 0 or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Mendelian parity scores must be nonempty and finite")
    delta = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return {
        "passed": bool(float(delta.max()) <= tolerance),
        "n_variants": len(reference),
        "score_tolerance": tolerance,
        "maximum_absolute_delta": float(delta.max()),
        "mean_absolute_delta": float(delta.mean()),
    }


def _download_final_adapter(
    retention: dict[str, Any],
    *,
    destination: Path,
    client: Any,
) -> list[dict[str, object]]:
    if (
        retention.get("destination_prefix") != CHECKPOINT_PREFIX
        or retention.get("base_model") != MODEL_ID
        or retention.get("base_revision") != MODEL_REVISION
        or retention.get("configuration", {}).get("learning_rate") != 5e-5
    ):
        raise RuntimeError("standard-rate S3 retention manifest changed")
    records = [
        record
        for record in retention.get("objects", [])
        if record.get("kind") == "peft_adapter" and int(record.get("step", -1)) == 1_000
    ]
    expected_names = {"README.md", "adapter_config.json", "adapter_model.safetensors"}
    if {Path(str(record["s3_uri"])).name for record in records} != expected_names:
        raise RuntimeError("retention manifest lacks the complete step-1000 PEFT adapter")
    verified = []
    for record in records:
        name = Path(str(record["s3_uri"])).name
        verified.append(
            download_verified_issue_object(
                s3_uri=str(record["s3_uri"]),
                destination=destination / name,
                expected_size_bytes=int(record["size_bytes"]),
                expected_sha256=str(record["sha256"]),
                client=client,
            )
        )
    return verified


def run_mendelian_reload_audit(
    *,
    artifact_dir: Path,
    output_dir: Path,
    batch_size: int,
) -> None:
    """Verify retained bytes and reproduce final Mendelian scores in a fresh process."""

    numeric_controls = configure_training_evaluation_numerics()
    if not torch.cuda.is_available():
        raise RuntimeError("Mendelian reload audit requires one CUDA GPU")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    price = float(os.getenv("EXP479_INSTANCE_PRICE_PER_HOUR_USD", "1.006"))
    if prior_cost + MAXIMUM_INSTANCE_HOURS * price >= BUDGET_USD:
        raise RuntimeError("Mendelian reload audit projection reaches the issue budget cap")

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=WANDB_PROJECT,
        group="dna-exp479-bico-lora-standard-rate",
        name=AUDIT_RUN_NAME,
        tags=[*EXPERIMENT_TAGS, "bico", "lora", "mendelian-only", "reload-parity"],
        config={
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "evaluation_artifact": FINAL_EVALUATION_ARTIFACT,
            "checkpoint_prefix": CHECKPOINT_PREFIX,
            "batch_size": batch_size,
            "score_tolerance": SCORE_TOLERANCE,
            "sample_design": "8 matched groups from each of 8 macro subsets",
            "maximum_instance_hours": MAXIMUM_INSTANCE_HOURS,
            "numeric_controls": numeric_controls,
        },
    )
    if run is None:
        raise RuntimeError("W&B did not create the Mendelian reload-audit run")

    try:
        assert_budget_reserve()
        evaluation_artifact = run.use_artifact(FINAL_EVALUATION_ARTIFACT, type="evaluation")
        evaluation_root = Path(
            evaluation_artifact.download(root=artifact_dir / "standard-rate-evaluation")
        )
        evaluation_manifest = json.loads(
            (evaluation_root / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            evaluation_manifest.get("status") != "completed"
            or evaluation_manifest.get("base_model") != MODEL_ID
            or evaluation_manifest.get("base_revision") != MODEL_REVISION
            or evaluation_manifest.get("base_frozen") is not True
            or evaluation_manifest.get("physical_batch_size") != 94
            or evaluation_manifest.get("accumulation_steps") != 1
            or evaluation_manifest.get("checkpoint_retention") != CHECKPOINT_PREFIX
            or evaluation_manifest.get("configuration", {}).get("learning_rate") != 5e-5
        ):
            raise RuntimeError("standard-rate evaluation manifest changed")
        retention = json.loads(
            (evaluation_root / "retention-manifest.json").read_text(encoding="utf-8")
        )
        s3 = boto3.client("s3", region_name="us-east-2")
        adapter_dir = artifact_dir / "retained-step-1000-adapter"
        verified_objects = _download_final_adapter(
            retention,
            destination=adapter_dir,
            client=s3,
        )

        stored = pd.read_parquet(evaluation_root / "mendelian_traits.scores.parquet")
        sample = select_mendelian_reload_rows(stored)
        if len(sample) != EXPECTED_ROWS or sample["subset"].nunique() != EXPECTED_SUBSETS:
            raise RuntimeError("fixed Mendelian reload sample changed")
        sample = attach_s3_reference_windows(sample, client=s3)

        source = _source_bundle()
        bundle = _attach_adapter(source, adapter_dir)
        if not isinstance(bundle.model, PeftModel):
            raise TypeError("reloaded standard-rate adapter did not produce a PEFT model")
        bundle.model.to(device="cuda").eval()
        tokenizer_contract = assert_source_tokenizer_contract(bundle.tokenizer)
        adapter_contract = assert_reloaded_adapter_contract(bundle.model)
        fresh_llr = score_bico_vep(bundle, sample, batch_size=batch_size)
        fresh_scores = _protocol_scores(fresh_llr, "minus_llr")
        stored_scores = sample[FINAL_READOUT].to_numpy(dtype=np.float32)
        parity = mendelian_score_parity(stored_scores, fresh_scores)
        sample["fresh_reload_step_1000"] = fresh_scores
        stored_auprc = mendelian_macro_auprc(sample, FINAL_READOUT)
        fresh_auprc = mendelian_macro_auprc(sample, "fresh_reload_step_1000")
        parity["stored_subset_macro_auprc"] = stored_auprc
        parity["fresh_reload_subset_macro_auprc"] = fresh_auprc
        parity["subset_macro_auprc_delta"] = fresh_auprc - stored_auprc
        if not bool(parity["passed"]):
            raise RuntimeError(f"fresh Mendelian reload parity failed: {parity}")

        public_scores = sample.drop(columns="sequence")
        scores_path = output_dir / "mendelian-reload-scores.parquet"
        checks_path = output_dir / "mendelian-reload-checks.json"
        manifest_path = output_dir / "manifest.json"
        public_scores.to_parquet(scores_path, index=False)
        checks_path.write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
        cost_path = write_cost_estimate(artifact_dir=artifact_dir)
        manifest = {
            "status": "completed",
            "audit_passed": True,
            "scope": "Mendelian development variants only",
            "evaluation_artifact": FINAL_EVALUATION_ARTIFACT,
            "evaluation_artifact_id": evaluation_artifact.id,
            "evaluation_artifact_digest": evaluation_artifact.digest,
            "checkpoint_prefix": CHECKPOINT_PREFIX,
            "verified_checkpoint_objects": verified_objects,
            "checks": parity,
            "sample": {
                "rows": len(sample),
                "subsets": int(sample["subset"].nunique()),
                "matched_groups_per_subset": GROUPS_PER_SUBSET,
                "split": "public train labels on odd-numbered autosomes and chromosome X",
            },
            "reference": {
                "s3_uri": f"s3://{REFERENCE_BUCKET}/{REFERENCE_KEY}",
                "etag": REFERENCE_ETAG,
                "size_bytes": REFERENCE_SIZE_BYTES,
                "assembly": "Ensembl release 115 GRCh38 soft-masked primary assembly",
                "coordinates": "0-based half-open internally; input POS converted from 1-based",
                "access": "byte-range queries; full reference not downloaded",
            },
            "source_tokenizer_contract": tokenizer_contract,
            "reloaded_adapter_contract": adapter_contract,
            "elapsed_seconds": time.time() - started,
            "checkpoint_deletion": "not performed",
            "hugging_face_upload": "not performed",
            "other_vep_datasets": "not evaluated",
            "nucleotide_dependency": "not performed",
            "knowledge_base_update": "not performed",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for key, value in parity.items():
            run.summary[f"mendelian_reload/{key}"] = value
        result = wandb.Artifact(AUDIT_ARTIFACT, type="evaluation")
        for path in (scores_path, checks_path, manifest_path, cost_path):
            result.add_file(str(path))
        logged = run.log_artifact(result, aliases=["step-1000", "reload-parity"])
        logged.wait()
        run.finish(exit_code=0)
        del bundle, source
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException:
        run.finish(exit_code=1)
        raise
