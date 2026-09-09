"""Joint development scoring of human-last, fixed-shape RAG documents.

The producer owns projection and species permutations. This consumer verifies
canonical cohort membership, changes only the human SNV, and routes predictions
back to the unchanged benchmark rows. Coordinate fields are 0-based, half-open;
canonical ``pos`` is converted from 1-based at the validation boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch import Tensor

from marin_dna_evals.hf_compat import load_hf_causal_lm_and_tokenizer
from marin_dna_evals.inference import fwd_rc_average_f16
from marin_dna_evals.model.runner import run_inference
from marin_dna_evals.model.scoring import _token_id_to_nuc_idx

BENCHMARKS = frozenset({"mendelian_traits", "complex_traits", "sge"})
DEVELOPMENT_CHROMS = {str(number) for number in range(1, 23, 2)} | {"X"}
WINDOW_BP = 255
MODEL_TOKENS = 10_240
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _digest(*parts: str | int) -> str:
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_harness(
    dataset: Dataset,
    canonical: Mapping[str, Sequence[Mapping[str, Any]]],
    revisions: Mapping[str, str],
) -> dict[str, list[int]]:
    """Return canonical-order row indices after exact, label-independent matching."""
    if set(canonical) != BENCHMARKS or set(revisions) != BENCHMARKS:
        raise ValueError("joint RAG scoring requires all three canonical benchmarks")
    routing = {name: [-1] * len(rows) for name, rows in canonical.items()}
    for index, row in enumerate(dataset):
        name = row["benchmark"]
        source_index = row["source_row_index"]
        if name not in canonical or type(source_index) is not int:
            raise ValueError("unregistered benchmark or invalid source index")
        if not 0 <= source_index < len(canonical[name]):
            raise ValueError("source row outside canonical cohort")
        if routing[name][source_index] != -1:
            raise ValueError("duplicate canonical source row")
        metadata = dict(canonical[name][source_index])
        chrom = str(metadata["chrom"]).removeprefix("chr")
        if row["split"] != "train" or chrom not in DEVELOPMENT_CHROMS:
            raise ValueError("held-out rows are not authorized for RAG development")
        if json.loads(row["metadata_json"]) != metadata:
            raise ValueError("harness metadata differs from pinned canonical source")
        if row["source_row_id"] != _digest(
            name, revisions[name], "train", source_index
        ):
            raise ValueError("source identity does not match the pinned revision")
        if (
            row["source_chrom"] != "chr" + chrom
            or row["source_start"] != int(metadata["pos"]) - 128
            or row["source_end"] != int(metadata["pos"]) + 127
        ):
            raise ValueError("variant-centered source coordinates changed")
        if row["query_name"] != "rag_" + _digest(
            "hg38", row["source_chrom"], row["source_start"], row["source_end"]
        ):
            raise ValueError("projection request identity changed")
        if (row["ref"], row["alt"]) != (metadata["ref"], metadata["alt"]):
            raise ValueError("canonical alleles changed")
        for field, source_field, domain in (
            ("group_id", "match_group", "match_group"),
            ("accession_id", "mavedb_urn", "accession"),
        ):
            expected = (
                _digest(name, domain, str(metadata[source_field]))
                if source_field in metadata
                else None
            )
            if row[field] != expected:
                raise ValueError("benchmark group namespace changed")
        validate_document(row)
        routing[name][source_index] = index
    if any(index < 0 for indices in routing.values() for index in indices):
        raise ValueError("harness is missing canonical benchmark rows")
    return routing


def validate_document(row: Mapping[str, Any]) -> list[str]:
    segments = row["sequence"].split("[SEQ]")
    species = row["species_order"]
    if (
        not 1 <= len(segments) <= 40
        or len(segments) != len(species)
        or len(set(species)) != len(species)
        or species[-1] != "hg38"
    ):
        raise ValueError("RAG document requires unique species with human last")
    if any(
        len(segment) != WINDOW_BP or set(segment.upper()) - set("ACGTN")
        for segment in segments
    ):
        raise ValueError("RAG segments must contain 255 ACGTN bases")
    ref, alt = row["ref"], row["alt"]
    if (
        ref not in "ACGT"
        or alt not in "ACGT"
        or len(ref) != 1
        or len(alt) != 1
        or ref == alt
    ):
        raise ValueError("RAG scoring requires distinct canonical SNV alleles")
    if segments[-1][127].upper() != ref:
        raise ValueError("human REF mismatch at the projected center")
    # Preserve the maintained 4-nucleotide LLR contract; never reinterpret N as A.
    if "N" in segments[-1].upper():
        raise ValueError(
            "the canonical 4-nucleotide score requires an ACGT human window"
        )
    return segments


def transform_rag(
    row: Mapping[str, Any], *, tokenizer: Any, strand: str
) -> dict[str, Any]:
    segments = validate_document(row)
    if strand not in {"fwd", "rc"}:
        raise ValueError("unknown strand")
    alt = row["alt"]
    if strand == "rc":
        segments = [segment.translate(_COMPLEMENT)[::-1] for segment in segments]
        alt = alt.translate(_COMPLEMENT)
    ids = tokenizer.encode("[SEQ]".join(segments), add_special_tokens=True)
    end = 256 * len(segments)
    if len(ids) != end or ids[0] != tokenizer.bos_token_id:
        raise ValueError("tokenizer violated atomic-separator and single-BOS geometry")
    alt_id = tokenizer.encode(alt, add_special_tokens=False)
    if len(alt_id) != 1:
        raise ValueError("alternate base must have one token")
    return {
        "input_ids": np.asarray(
            ids + [tokenizer.pad_token_id] * (MODEL_TOKENS - end), dtype=np.int64
        ),
        "alt_token_id": alt_id[0],
        "human_start": end - WINDOW_BP,
    }


def compute_rag_bundle(
    model: Any,
    input_ids: Tensor,
    alt_token_id: Tensor,
    human_start: Tensor,
    *,
    nuc_token_ids: Tensor,
    return_embeddings: bool,
) -> Tensor:
    """Full fixed-shape forwards, with per-row human-only score/pooling bounds.

    Variable retrieval coverage gives each row a different human position. Two
    alleles share the same padding and all context tokens; causal attention makes
    later padding invisible to real tokens. Full forwards avoid bucketing by length
    and reuse one compiled shape across benchmarks and coverage counts.
    The LLR/JSD reductions match ``compute_variant_score_bundle`` on ACGT windows.
    """
    batch, length = input_ids.shape
    var_pos = human_start + 127
    alternate = input_ids.scatter(1, var_pos[:, None], alt_token_id[:, None])
    alleles = torch.stack((input_ids, alternate), dim=1).reshape(2 * batch, length)
    captured: list[Tensor] = []
    hook = (
        model.base_model.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output.last_hidden_state)
        )
        if return_embeddings
        else None
    )
    try:
        logits = model(alleles, use_cache=False).logits.reshape(batch, 2, length, -1)
    finally:
        if hook is not None:
            hook.remove()
    # Exactly 128 targets: the SNV and the remaining 127 human bases.
    positions = var_pos[:, None] + torch.arange(128, device=input_ids.device)
    logit_positions = (positions - 1)[:, None, :, None].expand(
        -1, 2, -1, logits.shape[-1]
    )
    selected = logits.gather(2, logit_positions)
    nuc = nuc_token_ids.to(logits.device)
    log_probs = F.log_softmax(selected[..., nuc].float(), dim=-1)
    targets = alleles.reshape(batch, 2, length).gather(
        2, positions[:, None].expand(-1, 2, -1)
    )
    target_indices = _token_id_to_nuc_idx(targets, nuc)
    likelihoods = log_probs.gather(-1, target_indices[..., None]).squeeze(-1)
    llr = (likelihoods[:, 1] - likelihoods[:, 0]).sum(-1)
    ref, alt = log_probs[:, 0, 1:], log_probs[:, 1, 1:]
    mean = torch.logaddexp(ref, alt) - math.log(2)
    jsd = (
        0.5 * ((ref.exp() * (ref - mean)).sum(-1) + (alt.exp() * (alt - mean)).sum(-1))
    ).mean(-1)
    scores = torch.stack((llr, jsd), dim=1)
    if not return_embeddings:
        return scores
    if len(captured) != 1:
        raise ValueError("expected one final hidden-state capture")
    hidden = captured[0].reshape(batch, 2, length, -1)
    human_positions = human_start[:, None] + torch.arange(
        WINDOW_BP, device=input_ids.device
    )
    human = hidden.gather(
        2, human_positions[:, None, :, None].expand(-1, 2, -1, hidden.shape[-1])
    )
    pooled = human.mean(dim=2, dtype=torch.float32)
    return torch.cat((scores, pooled[:, 0], pooled[:, 1]), dim=1)


def score_rag_dataset(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    *,
    inference_kwargs: Mapping[str, Any],
    return_embeddings: bool = True,
) -> dict[str, np.ndarray]:
    """Use the established Trainer loop for both strands on the same loaded model."""
    nuc_ids = [tokenizer.encode(base, add_special_tokens=False) for base in "ACGT"]
    special_ids = [
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
        *tokenizer.encode("[SEQ]", add_special_tokens=False),
    ]
    if (
        any(len(ids) != 1 for ids in nuc_ids)
        or len(special_ids) != 3
        or len(set(special_ids + [ids[0] for ids in nuc_ids])) != 7
        or None in special_ids
    ):
        raise ValueError(
            "RAG tokenizer must have atomic bases and distinct BOS, PAD, and SEQ"
        )
    result = {}
    for strand in ("fwd", "rc"):
        result[strand] = np.asarray(
            run_inference(
                model,
                tokenizer,
                dataset,
                compute_fn=partial(
                    compute_rag_bundle,
                    nuc_token_ids=torch.tensor([ids[0] for ids in nuc_ids]),
                    return_embeddings=return_embeddings,
                ),
                data_transform_fn=partial(transform_rag, strand=strand),
                data_transform_on_the_fly=True,
                inference_kwargs=dict(inference_kwargs),
            )
        )
    return result


def compute_combined_rag_scores(
    checkpoint_path: str | Path,
    harness_path: str | Path,
    canonical: Mapping[str, Sequence[Mapping[str, Any]]],
    revisions: Mapping[str, str],
    *,
    inference_kwargs: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    dataset = Dataset.from_parquet(str(harness_path))
    routing = validate_harness(dataset, canonical, revisions)
    tokenizer, model = load_hf_causal_lm_and_tokenizer(checkpoint_path)
    if model.config.max_position_embeddings < MODEL_TOKENS:
        raise ValueError("checkpoint does not support the RAG context length")
    arrays = score_rag_dataset(
        model, tokenizer, dataset, inference_kwargs=inference_kwargs
    )
    dim = model.config.hidden_size
    if any(
        array.shape != (len(dataset), 2 + 2 * dim) or not np.isfinite(array).all()
        for array in arrays.values()
    ):
        raise ValueError("joint inference emitted invalid score/embedding arrays")
    columns = {
        f"{score}_{strand}": array[:, index]
        for strand, array in arrays.items()
        for index, score in enumerate(("llr", "jsd"))
    }
    columns["emb_ref"] = list(
        fwd_rc_average_f16([array[:, 2 : 2 + dim] for array in arrays.values()])
    )
    columns["emb_alt"] = list(
        fwd_rc_average_f16([array[:, 2 + dim :] for array in arrays.values()])
    )
    scores = pd.DataFrame(columns)
    return {
        name: pd.concat(
            (
                pd.DataFrame(canonical[name]).reset_index(drop=True),
                scores.iloc[indices].reset_index(drop=True),
            ),
            axis=1,
        )
        for name, indices in routing.items()
    }
