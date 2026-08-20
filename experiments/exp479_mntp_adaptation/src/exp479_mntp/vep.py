"""Odd-autosome/X VEP diagnostics for the exp479 arms and controls."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from pyfaidx import Fasta
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from exp479_mntp.config import MODEL_ID, MODEL_REVISION, NUCLEOTIDE_LENGTH
from exp479_mntp.modeling import canonical_token_ids, load_model_bundle, model_logits
from exp479_mntp.probes import context_dependence
from exp479_mntp.publishing import assert_budget_reserve
from exp479_mntp.vep_metrics import GLOBAL, matched_metrics, paired_ap_delta, sge_metrics

REFERENCE_REPO = "marin-dna/human-genome"
REFERENCE_REVISION = "11b9433582981bb929af333bc6422f10a8fd71b4"
REFERENCE_FASTA = "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
REFERENCE_ASSETS = (
    REFERENCE_FASTA,
    f"{REFERENCE_FASTA}.fai",
    f"{REFERENCE_FASTA}.gzi",
    "SHA256SUMS",
)
DEVELOPMENT_CHROMS = frozenset({*(str(value) for value in range(1, 23, 2)), "X"})
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str
    revision: str
    protocol: Literal["minus_llr", "abs_llr"]
    evaluation: Literal["matched", "sge"]


DATASETS = (
    DatasetSpec(
        name="mendelian_traits",
        repo_id="bolinas-dna/evals_mendelian_traits",
        revision="4aed58e50c5dea0b878a665007af2ef9e5108e9f",
        protocol="minus_llr",
        evaluation="matched",
    ),
    DatasetSpec(
        name="complex_traits",
        repo_id="bolinas-dna/evals_complex_traits",
        revision="22f86a89c65cb8f3007ac3cc2739f40efefa4340",
        protocol="abs_llr",
        evaluation="matched",
    ),
    DatasetSpec(
        name="sge",
        repo_id="bolinas-dna/evals_sge",
        revision="225d3d1ea32a4af547891b13c33b5e92a5aae849",
        protocol="minus_llr",
        evaluation="sge",
    ),
)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    objective: Literal["mntp", "clm"]
    source_control: bool = False
    no_adaptation_control: bool = False


ARMS = (
    ArmSpec("source_clm", "clm", source_control=True),
    ArmSpec("full_attention_no_adaptation", "mntp", no_adaptation_control=True),
    ArmSpec("transferred_mntp", "mntp"),
    ArmSpec("scratch_mntp", "mntp"),
    ArmSpec("clm_continuation", "clm"),
)


@dataclass
class LoadedArm:
    model: PreTrainedModel
    tokenizer: Any
    canonical_ids: tuple[int, ...]
    mask_token_id: int | None


def reverse_complement(sequence: str) -> str:
    """Reverse-complement one DNA sequence without changing its length."""

    return sequence.translate(COMPLEMENT)[::-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_reference(destination: Path) -> Path:
    """Download and checksum the pinned Ensembl-115 GRCh38 BGZF bundle."""

    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename in REFERENCE_ASSETS:
        paths[filename] = Path(
            hf_hub_download(
                repo_id=REFERENCE_REPO,
                filename=filename,
                repo_type="dataset",
                revision=REFERENCE_REVISION,
                local_dir=destination,
            )
        )
    expected: dict[str, str] = {}
    for line in paths["SHA256SUMS"].read_text(encoding="utf-8").splitlines():
        if line.strip():
            checksum, filename = line.split(maxsplit=1)
            expected[filename.lstrip("* ")] = checksum
    for filename in REFERENCE_ASSETS[:-1]:
        if filename not in expected:
            raise ValueError(f"reference checksum manifest omits {filename}")
        observed = _sha256(paths[filename])
        if observed != expected[filename]:
            raise ValueError(f"reference checksum mismatch for {filename}: {observed}")
    return paths[REFERENCE_FASTA]


def assert_development_split(frame: pd.DataFrame, dataset_name: str) -> None:
    """Fail before scoring if a labeled frame crosses the odd/X boundary."""

    if "chrom" not in frame or "label" not in frame:
        raise ValueError(f"{dataset_name} lacks chrom/label")
    observed = {str(value) for value in frame["chrom"].unique()}
    forbidden = observed - DEVELOPMENT_CHROMS
    if forbidden:
        raise RuntimeError(f"{dataset_name} train split contains held-out chromosomes {forbidden}")
    if not observed:
        raise ValueError(f"{dataset_name} train split is empty")


def load_variant_frame(spec: DatasetSpec) -> pd.DataFrame:
    """Load only the pinned public train split, which is odd autosomes plus X."""

    dataset = load_dataset(spec.repo_id, split="train", revision=spec.revision)
    frame = dataset.to_pandas()
    frame["chrom"] = frame["chrom"].astype(str)
    frame["ref"] = frame["ref"].str.upper()
    frame["alt"] = frame["alt"].str.upper()
    assert_development_split(frame, spec.name)
    if not frame["ref"].isin(list("ACGT")).all() or not frame["alt"].isin(list("ACGT")).all():
        raise ValueError(f"{spec.name} contains non-SNV alleles")
    return frame


def attach_reference_windows(frame: pd.DataFrame, fasta_path: Path) -> pd.DataFrame:
    """Attach 255-bp uppercase reference windows using 0-based half-open slices."""

    center = NUCLEOTIDE_LENGTH // 2
    sequences: list[str] = []
    with Fasta(fasta_path, as_raw=True, rebuild=False) as genome:
        sequence_names = genome.keys()
        chrom_sizes = {name: len(genome[name]) for name in sequence_names}
        for row in frame.itertuples(index=False):
            chrom = str(row.chrom)
            center_zero_based = int(row.pos) - 1
            start = center_zero_based - center
            end = start + NUCLEOTIDE_LENGTH
            if chrom not in chrom_sizes or start < 0 or end > chrom_sizes[chrom]:
                raise ValueError(f"variant window outside {chrom}:0-{chrom_sizes.get(chrom)}")
            sequence = str(genome[chrom][start:end]).upper()
            if len(sequence) != NUCLEOTIDE_LENGTH:
                raise ValueError(f"short reference window at {chrom}:{start}-{end}")
            if sequence[center] != row.ref:
                raise ValueError(
                    f"GRCh38 reference mismatch at {chrom}:{row.pos}: "
                    f"dataset={row.ref}, fasta={sequence[center]}"
                )
            sequences.append(sequence)
    result = frame.copy()
    result["sequence"] = sequences
    return result


def _replace_center(sequence: str, allele: str) -> str:
    center = NUCLEOTIDE_LENGTH // 2
    return sequence[:center] + allele + sequence[center + 1 :]


def _alleles(frame: pd.DataFrame, strand: Literal["fwd", "rc"]) -> tuple[list[str], list[str]]:
    if strand == "fwd":
        return frame["ref"].tolist(), frame["alt"].tolist()
    return (
        [base.translate(COMPLEMENT) for base in frame["ref"]],
        [base.translate(COMPLEMENT) for base in frame["alt"]],
    )


def _strand_sequences(frame: pd.DataFrame, strand: Literal["fwd", "rc"]) -> list[str]:
    sequences = frame["sequence"].tolist()
    return sequences if strand == "fwd" else [reverse_complement(value) for value in sequences]


def _autocast(device: torch.device) -> Any:
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


@torch.inference_mode()
def score_strand(
    arm: LoadedArm,
    frame: pd.DataFrame,
    *,
    objective: Literal["mntp", "clm"],
    strand: Literal["fwd", "rc"],
    batch_size: int,
) -> np.ndarray:
    """Score one orientation with central-mask MNTP or full-sequence CLM LLR."""

    if batch_size <= 0:
        raise ValueError("VEP batch size must be positive")
    model = arm.model
    device = next(model.parameters()).device
    sequences = _strand_sequences(frame, strand)
    ref_alleles, alt_alleles = _alleles(frame, strand)
    scores = np.empty(len(frame), dtype=np.float32)
    center_token = 1 + NUCLEOTIDE_LENGTH // 2
    nucleotide_lookup = {
        base: token_id for base, token_id in zip("ACGT", arm.canonical_ids, strict=True)
    }

    for start in range(0, len(frame), batch_size):
        assert_budget_reserve()
        stop = min(len(frame), start + batch_size)
        batch_sequences = sequences[start:stop]
        batch_ref = ref_alleles[start:stop]
        batch_alt = alt_alleles[start:stop]
        if objective == "mntp":
            if arm.mask_token_id is None:
                raise RuntimeError("MNTP scorer requires a mask token")
            encoded = arm.tokenizer(
                batch_sequences,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            if input_ids.shape[1] != NUCLEOTIDE_LENGTH + 1:
                raise ValueError(f"unexpected MNTP tokenized length {input_ids.shape[1]}")
            input_ids[:, center_token] = arm.mask_token_id
            with _autocast(device):
                logits = model_logits(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    attention_mode="full",
                )[:, center_token - 1, list(arm.canonical_ids)]
            log_probabilities = torch.log_softmax(logits.float(), dim=-1)
            canonical_index = {token_id: index for index, token_id in enumerate(arm.canonical_ids)}
            ref_index = torch.tensor(
                [canonical_index[nucleotide_lookup[value]] for value in batch_ref], device=device
            )
            alt_index = torch.tensor(
                [canonical_index[nucleotide_lookup[value]] for value in batch_alt], device=device
            )
            rows = torch.arange(stop - start, device=device)
            values = log_probabilities[rows, alt_index] - log_probabilities[rows, ref_index]
        else:
            alt_sequences = [
                _replace_center(sequence, allele)
                for sequence, allele in zip(batch_sequences, batch_alt, strict=True)
            ]
            encoded = arm.tokenizer(
                [*batch_sequences, *alt_sequences],
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            if input_ids.shape[1] != NUCLEOTIDE_LENGTH + 1:
                raise ValueError(f"unexpected CLM tokenized length {input_ids.shape[1]}")
            with _autocast(device):
                logits = model_logits(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    attention_mode="causal",
                )
            token_log_probabilities = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            token_log_probabilities = token_log_probabilities.gather(
                2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            sequence_log_probabilities = token_log_probabilities.sum(dim=-1)
            ref_log_probability, alt_log_probability = sequence_log_probabilities.chunk(2)
            values = alt_log_probability - ref_log_probability
        scores[start:stop] = values.float().cpu().numpy()
    if not np.isfinite(scores).all():
        raise RuntimeError(f"non-finite {objective} {strand} VEP scores")
    return scores


def _trained_checkpoint(artifact_dir: Path, hf_repo_id: str, arm: str) -> Path:
    local = artifact_dir / arm / "hf" / "step-1000"
    if (local / "config.json").exists():
        return local
    cache = artifact_dir / "evaluation-models"
    snapshot_download(
        repo_id=hf_repo_id,
        repo_type="model",
        allow_patterns=[f"hf/{arm}/step-1000/*"],
        local_dir=cache,
    )
    downloaded = cache / "hf" / arm / "step-1000"
    if not (downloaded / "config.json").exists():
        raise FileNotFoundError(f"missing final export for {arm}")
    return downloaded


def load_arm(spec: ArmSpec, artifact_dir: Path, hf_repo_id: str) -> LoadedArm:
    """Load one control or trained arm in bf16 for single-GH200 inference."""

    if spec.source_control or spec.no_adaptation_control:
        bundle = load_model_bundle(
            initialization="transferred",
            add_mask=spec.no_adaptation_control,
            attention_implementation="sdpa",
            dtype=None if spec.no_adaptation_control else torch.bfloat16,
        )
        return LoadedArm(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            canonical_ids=bundle.canonical_token_ids,
            mask_token_id=bundle.mask_token_id,
        )
    checkpoint = _trained_checkpoint(artifact_dir, hf_repo_id, spec.name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    mask_token_id = int(tokenizer.mask_token_id) if spec.objective == "mntp" else None
    return LoadedArm(
        model=model,
        tokenizer=tokenizer,
        canonical_ids=canonical_token_ids(tokenizer),
        mask_token_id=mask_token_id,
    )


def _protocol_scores(llr: np.ndarray, protocol: str) -> np.ndarray:
    if protocol == "minus_llr":
        return -llr
    if protocol == "abs_llr":
        return np.abs(llr)
    raise ValueError(f"unknown VEP protocol {protocol}")


def _paired_matched_table(
    variants: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> pd.DataFrame:
    comparisons = {
        "source_clm_avg": "source_clm_avg",
        "scratch_mntp_fwd": "scratch_mntp_fwd",
        "clm_continuation_avg": "clm_continuation_avg",
        "transferred_mntp_avg": "transferred_mntp_avg",
    }
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.Series]] = [(GLOBAL, pd.Series(True, index=variants.index))]
    scopes.extend(
        (str(subset), variants["subset"] == subset) for subset in variants["subset"].unique()
    )
    for scope, selected in scopes:
        cell = variants[selected]
        for comparison, baseline_column in comparisons.items():
            result = paired_ap_delta(
                cell["label"],
                scores.loc[selected, "transferred_mntp_fwd"],
                scores.loc[selected, baseline_column],
                cell["match_group"],
                n_bootstrap=n_bootstrap,
                seed=0,
            )
            rows.append({"subset": scope, "comparison": comparison, **result})
    return pd.DataFrame(rows)


def _paired_sge_table(
    variants: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> pd.DataFrame:
    comparisons = {
        "source_clm_avg": "source_clm_avg",
        "scratch_mntp_fwd": "scratch_mntp_fwd",
        "clm_continuation_avg": "clm_continuation_avg",
        "transferred_mntp_avg": "transferred_mntp_avg",
    }
    rows: list[dict[str, object]] = []
    subsets = sorted(str(value) for value in variants["subset"].unique())
    for comparison, baseline_column in comparisons.items():
        leaf_rows: list[dict[str, object]] = []
        for accession, accession_frame in variants.groupby("mavedb_urn", sort=False):
            scopes = {
                subset: accession_frame[accession_frame["subset"] == subset] for subset in subsets
            }
            scopes["both"] = accession_frame
            for scope, cell in scopes.items():
                n_pos = int(cell["label"].astype(bool).sum())
                if n_pos < 30 or len(cell) - n_pos < 30:
                    continue
                result = paired_ap_delta(
                    cell["label"],
                    scores.loc[cell.index, "transferred_mntp_fwd"],
                    scores.loc[cell.index, baseline_column],
                    np.arange(len(cell)),
                    n_bootstrap=n_bootstrap,
                    seed=0,
                )
                row = {
                    "comparison": comparison,
                    "subset": scope,
                    "accession": str(accession),
                    **result,
                }
                rows.append(row)
                leaf_rows.append(row)
        for scope in [*subsets, "both"]:
            children = [row for row in leaf_rows if row["subset"] == scope]
            if children:
                count = len(children)
                rows.append(
                    {
                        "comparison": comparison,
                        "subset": scope,
                        "accession": "_macro_avg_",
                        "delta": float(sum(float(child["delta"]) for child in children) / count),
                        "se": float(
                            np.sqrt(sum(float(child["se"]) ** 2 for child in children)) / count
                        ),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "n_groups": count,
                        "n_rows": sum(int(child["n_rows"]) for child in children),
                    }
                )
    return pd.DataFrame(rows)


def _probe_sequences(frame: pd.DataFrame, count: int = 32) -> list[str]:
    left = NUCLEOTIDE_LENGTH // 2 - 16
    right = NUCLEOTIDE_LENGTH // 2 + 16
    selected = [
        sequence
        for sequence in frame["sequence"]
        if sequence[left] in "ACGT" and sequence[right] in "ACGT"
    ][:count]
    if len(selected) != count:
        raise ValueError(f"VEP context probe found only {len(selected)} canonical sequences")
    return selected


def run_vep_evaluation(
    *,
    artifact_dir: Path,
    output_dir: Path,
    hf_repo_id: str,
    batch_size: int,
    n_bootstrap: int = 1_000,
) -> None:
    """Score every registered development VEP task and publish compact artifacts."""

    if not torch.cuda.is_available():
        raise RuntimeError("full exp479 VEP evaluation requires the Lambda GH200")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = download_reference(artifact_dir / "reference")
    frames = {
        spec.name: attach_reference_windows(load_variant_frame(spec), reference)
        for spec in DATASETS
    }
    score_frames = {spec.name: pd.DataFrame(index=frames[spec.name].index) for spec in DATASETS}
    runtime_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    probe_sequences = _probe_sequences(frames["mendelian_traits"])

    for arm_spec in ARMS:
        assert_budget_reserve()
        arm = load_arm(arm_spec, artifact_dir, hf_repo_id)
        arm.model.to(device="cuda", dtype=torch.bfloat16).eval()
        torch.cuda.reset_peak_memory_stats()
        probe_encoded = arm.tokenizer(
            probe_sequences,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        probe_values = context_dependence(
            arm.model,
            probe_encoded["input_ids"].to("cuda"),
            probe_encoded["attention_mask"].to("cuda"),
            target_input_position=1 + NUCLEOTIDE_LENGTH // 2,
            mask_token_id=arm.mask_token_id,
            canonical_ids=arm.canonical_ids,
            attention_mode="full" if arm_spec.objective == "mntp" else "causal",
        )
        context_rows.append({"arm": arm_spec.name, **probe_values})
        for dataset_spec in DATASETS:
            frame = frames[dataset_spec.name]
            strand_scores: dict[str, np.ndarray] = {}
            strand_seconds: dict[str, float] = {}
            strand_peaks: dict[str, int] = {}
            for strand in ("fwd", "rc"):
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                strand_scores[strand] = score_strand(
                    arm,
                    frame,
                    objective=arm_spec.objective,
                    strand=strand,
                    batch_size=batch_size,
                )
                elapsed = time.perf_counter() - started
                peak = int(torch.cuda.max_memory_allocated())
                strand_seconds[strand] = elapsed
                strand_peaks[strand] = peak
                runtime_rows.append(
                    {
                        "arm": arm_spec.name,
                        "dataset": dataset_spec.name,
                        "strand": strand,
                        "rows": len(frame),
                        "seconds": elapsed,
                        "variants_per_second": len(frame) / elapsed,
                        "peak_cuda_allocated_bytes": peak,
                    }
                )
            combined_seconds = strand_seconds["fwd"] + strand_seconds["rc"]
            runtime_rows.append(
                {
                    "arm": arm_spec.name,
                    "dataset": dataset_spec.name,
                    "strand": "fwd_rc",
                    "rows": len(frame),
                    "seconds": combined_seconds,
                    "variants_per_second": len(frame) / combined_seconds,
                    "peak_cuda_allocated_bytes": max(strand_peaks.values()),
                }
            )
            raw_average = (strand_scores["fwd"] + strand_scores["rc"]) / 2
            score_frames[dataset_spec.name][f"{arm_spec.name}_fwd"] = _protocol_scores(
                strand_scores["fwd"], dataset_spec.protocol
            )
            score_frames[dataset_spec.name][f"{arm_spec.name}_rc"] = _protocol_scores(
                strand_scores["rc"], dataset_spec.protocol
            )
            score_frames[dataset_spec.name][f"{arm_spec.name}_avg"] = _protocol_scores(
                raw_average, dataset_spec.protocol
            )
        del arm
        gc.collect()
        torch.cuda.empty_cache()

    for dataset_spec in DATASETS:
        frame = frames[dataset_spec.name]
        scores = score_frames[dataset_spec.name]
        public_columns = [column for column in frame.columns if column != "sequence"]
        pd.concat([frame[public_columns], scores], axis=1).to_parquet(
            output_dir / f"{dataset_spec.name}.scores.parquet", index=False
        )
        if dataset_spec.evaluation == "matched":
            metrics = matched_metrics(
                frame,
                scores,
                n_bootstrap=n_bootstrap,
                seed=0,
            )
            paired = _paired_matched_table(frame, scores, n_bootstrap=n_bootstrap)
        else:
            metrics = sge_metrics(
                frame,
                scores,
                n_bootstrap=n_bootstrap,
                seed=0,
            )
            paired = _paired_sge_table(frame, scores, n_bootstrap=n_bootstrap)
        metrics.to_parquet(output_dir / f"{dataset_spec.name}.metrics.parquet", index=False)
        paired.to_parquet(output_dir / f"{dataset_spec.name}.paired.parquet", index=False)

    pd.DataFrame(runtime_rows).to_parquet(output_dir / "runtime.parquet", index=False)
    pd.DataFrame(context_rows).to_parquet(output_dir / "context-probes.parquet", index=False)
    manifest = {
        "source_model": f"{MODEL_ID}@{MODEL_REVISION}",
        "reference": f"{REFERENCE_REPO}@{REFERENCE_REVISION}/{REFERENCE_FASTA}",
        "split": "train",
        "allowed_chromosomes": sorted(DEVELOPMENT_CHROMS),
        "datasets": {spec.name: f"{spec.repo_id}@{spec.revision}" for spec in DATASETS},
        "arms": [spec.name for spec in ARMS],
        "batch_size": batch_size,
        "n_bootstrap": n_bootstrap,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    HfApi().upload_folder(
        folder_path=output_dir,
        path_in_repo="evaluation/vep",
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Upload exp479 odd/X VEP diagnostics",
    )
