"""Interpret the block-19 dsQTL direction lead with bounded mutagenesis.

This post-hoc mechanism pass selects contexts only by feature activation, then
saturates the causal 16-bp prefix ending at the focal allele. It also subtracts
the decoded feature contribution from the final residual stream to measure the
feature's direct four-nucleotide next-token readout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl
import torch
from huggingface_hub import snapshot_download
from marin_dna.data.dna import reverse_complement
from marin_dna.model.sae import M51_HIDDEN_SIZE, load_frozen_m51
from sae_lens.load_model import HookedProxyLM
from sae_lens.saes.sae import SAE
from scipy.stats import pearsonr, spearmanr

from build_panel import FOCAL_INDEX, WINDOW_BP
from extract_focal import MODEL_ID, MODEL_REVISION, validate_panel

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ISSUE = 434
FEATURE_ID = 1_829
REPORT_BLOCK = 19
ARM = "block19-25m"
HOOK_NAME = "model.layers.18"
ORIENTATIONS = ("forward", "reverse_complement")
ALLELES = ("ref", "alt")
NUCLEOTIDES = "ACGT"
DEFAULT_CONTEXTS = 32
DEFAULT_RADIUS = 15
DEFAULT_BATCH_SIZE = 256
NOOP_ABSOLUTE_TOLERANCE = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def assert_commit(value: str) -> None:
    assert len(value) == 40
    assert all(character in "0123456789abcdef" for character in value)


def verify_extraction(extraction_root: Path) -> dict[str, Any]:
    manifest_path = extraction_root / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE
    assert manifest["model"]["reported_blocks"] == [1, 10, 19]
    assert manifest["protocol"]["window_bp"] == WINDOW_BP
    for relative, expected in manifest["artifacts"].items():
        path = extraction_root / relative
        assert path.is_file(), path
        assert path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]
    return manifest


def dense_feature(
    panel: pl.DataFrame, sparse: pl.DataFrame, *, feature_id: int
) -> pl.DataFrame:
    selected = sparse.filter(pl.col("feature_id") == feature_id).select(
        "panel_row", "ref_activation", "alt_activation", "delta"
    )
    assert selected["panel_row"].n_unique() == selected.height
    dense = panel.join(selected, on="panel_row", how="left").with_columns(
        pl.col(column).fill_null(0.0)
        for column in ("ref_activation", "alt_activation", "delta")
    )
    assert dense.height == panel.height
    assert dense.filter(
        ~pl.col("ref_activation").is_finite()
        | ~pl.col("alt_activation").is_finite()
        | ~pl.col("delta").is_finite()
    ).is_empty()
    return dense


def select_contexts(
    panel: pl.DataFrame,
    sparse_by_orientation: dict[str, pl.DataFrame],
    *,
    feature_id: int,
    contexts_per_orientation: int,
) -> pl.DataFrame:
    """Select one maximally active allele per variant, then the top contexts."""

    assert contexts_per_orientation > 0
    output: list[pl.DataFrame] = []
    for orientation in ORIENTATIONS:
        dense = dense_feature(
            panel, sparse_by_orientation[orientation], feature_id=feature_id
        )
        allele_frames: list[pl.DataFrame] = []
        for allele in ALLELES:
            frame = dense.select(
                "panel_row",
                "chrom",
                "pos",
                "ref",
                "alt",
                "effect",
                "official_split",
                pl.lit(orientation).alias("orientation"),
                pl.lit(allele).alias("allele"),
                pl.col(f"{allele}_activation").alias("recorded_activation"),
                pl.col(f"{allele}_sequence").alias("input_sequence"),
            )
            if orientation == "reverse_complement":
                frame = frame.with_columns(
                    pl.col("input_sequence")
                    .map_elements(reverse_complement, return_dtype=pl.String)
                    .alias("input_sequence")
                )
            allele_frames.append(frame)
        candidates = pl.concat(allele_frames).sort(
            "recorded_activation",
            "panel_row",
            "allele",
            descending=[True, False, False],
        )
        selected = candidates.unique("panel_row", keep="first", maintain_order=True)
        selected = selected.filter(pl.col("recorded_activation") > 0).head(
            contexts_per_orientation
        )
        assert selected.height == contexts_per_orientation
        selected = selected.with_row_index("selection_rank", offset=1).with_columns(
            (
                pl.col("orientation")
                + pl.lit(":")
                + pl.col("selection_rank").cast(pl.String)
                + pl.lit(":")
                + pl.col("panel_row").cast(pl.String)
                + pl.lit(":")
                + pl.col("allele")
            ).alias("context_id")
        )
        output.append(selected)
    contexts = pl.concat(output)
    assert contexts.height == contexts_per_orientation * len(ORIENTATIONS)
    assert contexts["context_id"].n_unique() == contexts.height
    assert contexts.filter(
        pl.col("input_sequence").str.len_chars() != WINDOW_BP
    ).is_empty()
    return contexts


def counterfactual_sequences(sequence: str, *, radius: int) -> list[dict[str, Any]]:
    assert len(sequence) == WINDOW_BP
    assert set(sequence) <= set(NUCLEOTIDES)
    assert 0 <= radius <= FOCAL_INDEX
    rows: list[dict[str, Any]] = []
    for relative_position in range(-radius, 1):
        position = FOCAL_INDEX + relative_position
        reference_base = sequence[position]
        for target_base in NUCLEOTIDES:
            rows.append(
                {
                    "relative_position_input": relative_position,
                    "reference_base": reference_base,
                    "target_base": target_base,
                    "changed": target_base != reference_base,
                    "input_sequence": (
                        sequence[:position] + target_base + sequence[position + 1 :]
                    ),
                }
            )
    assert len(rows) == (radius + 1) * len(NUCLEOTIDES)
    return rows


def mutation_frame(contexts: pl.DataFrame, *, radius: int) -> pl.DataFrame:
    records: list[dict[str, Any]] = []
    metadata = [column for column in contexts.columns if column != "input_sequence"]
    for context in contexts.to_dicts():
        for mutation in counterfactual_sequences(
            context["input_sequence"], radius=radius
        ):
            records.append(
                {
                    **{column: context[column] for column in metadata},
                    **mutation,
                }
            )
    frame = pl.DataFrame(records)
    expected = contexts.height * (radius + 1) * len(NUCLEOTIDES)
    assert frame.height == expected
    assert frame.filter(
        pl.col("input_sequence").str.len_chars() != WINDOW_BP
    ).is_empty()
    return frame


def _correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    assert x.shape == y.shape and x.ndim == 1 and x.size >= 3
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "n": int(x.size),
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def robustness_summary(
    panel: pl.DataFrame, dense_by_orientation: dict[str, pl.DataFrame]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for orientation in ORIENTATIONS:
        frame = dense_by_orientation[orientation]
        delta = frame["delta"].to_numpy()
        effect = frame["effect"].to_numpy()
        order = np.argsort(np.abs(delta))[::-1]
        groups = np.asarray(
            [
                f"{ref}>{alt}"
                for ref, alt in zip(
                    frame["ref"].to_list(), frame["alt"].to_list(), strict=True
                )
            ]
        )
        delta_residual = delta.copy()
        effect_residual = effect.copy()
        for group in np.unique(groups):
            mask = groups == group
            delta_residual[mask] -= delta_residual[mask].mean()
            effect_residual[mask] -= effect_residual[mask].mean()
        split = {
            key[0]: _correlations(value["delta"].to_numpy(), value["effect"].to_numpy())
            for key, value in frame.group_by("official_split")
        }
        leave_chromosome = []
        chromosome = frame["chrom"].to_numpy()
        for held_out in frame["chrom"].unique().to_list():
            keep = chromosome != held_out
            leave_chromosome.append(_correlations(delta[keep], effect[keep]))
        predictors = {}
        for column in (
            "chrombpnet_atac_logfc",
            "chrombpnet_dnase_logfc",
            "enformer_dnase_local_logfc",
            "chrombpnet_atac_ips",
            "chrombpnet_dnase_ips",
        ):
            predictors[column] = _correlations(delta, frame[column].to_numpy())
        output[orientation] = {
            "raw": _correlations(delta, effect),
            "nonzero": _correlations(delta[delta != 0], effect[delta != 0]),
            "trim_largest_1pct": _correlations(
                np.delete(delta, order[:6]), np.delete(effect, order[:6])
            ),
            "trim_largest_5pct": _correlations(
                np.delete(delta, order[:28]), np.delete(effect, order[:28])
            ),
            "within_ref_alt_pair": _correlations(delta_residual, effect_residual),
            "official_split": split,
            "leave_one_chromosome": {
                "pearson_min": min(item["pearson"] for item in leave_chromosome),
                "pearson_max": max(item["pearson"] for item in leave_chromosome),
                "spearman_min": min(item["spearman"] for item in leave_chromosome),
                "spearman_max": max(item["spearman"] for item in leave_chromosome),
            },
            "official_predictors": predictors,
        }
    forward = dense_by_orientation["forward"]["delta"].to_numpy()
    reverse = dense_by_orientation["reverse_complement"]["delta"].to_numpy()
    both = (forward != 0) & (reverse != 0)
    output["cross_orientation"] = {
        "all_rows": _correlations(forward, reverse),
        "both_nonzero": _correlations(forward[both], reverse[both]),
        "same_sign_fraction_both_nonzero": float(
            np.mean(np.sign(forward[both]) == np.sign(reverse[both]))
        ),
        "support_intersection": int(both.sum()),
        "support_union": int(np.sum((forward != 0) | (reverse != 0))),
    }
    return output


@torch.inference_mode()
def extract_feature_and_raw(
    sequences: list[str],
    *,
    tokenizer: Any,
    model: HookedProxyLM,
    sae: SAE,
    batch_size: int,
) -> tuple[np.ndarray, torch.Tensor]:
    activations: list[np.ndarray] = []
    raw_rows: list[torch.Tensor] = []
    for offset in range(0, len(sequences), batch_size):
        batch = sequences[offset : offset + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_tensors="pt",
        )
        tokens = encoded["input_ids"].to("cuda")
        assert tokens.shape == (len(batch), WINDOW_BP + 1)
        output, cache = model.run_with_cache(
            tokens, names_filter=[HOOK_NAME], stop_at_layer=REPORT_BLOCK
        )
        assert output is None and set(cache) == {HOOK_NAME}
        raw = cache[HOOK_NAME][:, FOCAL_INDEX + 1, :].float()
        assert raw.shape == (len(batch), M51_HIDDEN_SIZE)
        feature = sae.encode(raw)[:, FEATURE_ID]
        assert torch.isfinite(feature).all() and torch.all(feature >= 0)
        activations.append(feature.cpu().numpy())
        raw_rows.append(raw.cpu())
    return np.concatenate(activations), torch.cat(raw_rows)


@torch.inference_mode()
def next_base_readout(
    contexts: pl.DataFrame,
    raw: torch.Tensor,
    feature_activation: np.ndarray,
    *,
    frozen_model: Any,
    tokenizer: Any,
    sae: SAE,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    assert raw.shape == (contexts.height, M51_HIDDEN_SIZE)
    assert feature_activation.shape == (contexts.height,)
    code = torch.zeros(
        (contexts.height, sae.cfg.d_sae), dtype=torch.float32, device="cuda"
    )
    code[:, FEATURE_ID] = torch.from_numpy(feature_activation).to("cuda")
    contribution = sae.decode(code) - sae.decode(torch.zeros_like(code))
    assert contribution.shape == (contexts.height, M51_HIDDEN_SIZE)
    raw_cuda = raw.to("cuda")
    ablated = raw_cuda - contribution
    base_model = frozen_model.model
    assert hasattr(base_model, "model") and hasattr(base_model, "lm_head")
    full_logits = base_model.lm_head(
        base_model.model.norm(raw_cuda.to(torch.bfloat16))
    ).float()
    ablated_logits = base_model.lm_head(
        base_model.model.norm(ablated.to(torch.bfloat16))
    ).float()
    token_ids = []
    for base in NUCLEOTIDES:
        encoded = tokenizer(base, add_special_tokens=True)["input_ids"]
        assert len(encoded) == 2
        token_ids.append(encoded[1])
    full_nuc = full_logits[:, token_ids]
    ablated_nuc = ablated_logits[:, token_ids]
    full_probability = torch.softmax(full_nuc, dim=-1)
    ablated_probability = torch.softmax(ablated_nuc, dim=-1)
    records = []
    context_rows = contexts.to_dicts()
    for row_index, context in enumerate(context_rows):
        for base_index, base in enumerate(NUCLEOTIDES):
            records.append(
                {
                    "context_id": context["context_id"],
                    "orientation": context["orientation"],
                    "next_base": base,
                    "full_logit": float(full_nuc[row_index, base_index]),
                    "ablated_logit": float(ablated_nuc[row_index, base_index]),
                    "logit_effect": float(
                        full_nuc[row_index, base_index]
                        - ablated_nuc[row_index, base_index]
                    ),
                    "full_probability_4nuc": float(
                        full_probability[row_index, base_index]
                    ),
                    "ablated_probability_4nuc": float(
                        ablated_probability[row_index, base_index]
                    ),
                    "probability_effect_4nuc": float(
                        full_probability[row_index, base_index]
                        - ablated_probability[row_index, base_index]
                    ),
                }
            )
    frame = pl.DataFrame(records)
    summary = (
        frame.group_by("orientation", "next_base")
        .agg(
            pl.col("logit_effect").mean().alias("mean_logit_effect"),
            pl.col("logit_effect").median().alias("median_logit_effect"),
            pl.col("probability_effect_4nuc")
            .mean()
            .alias("mean_probability_effect_4nuc"),
            pl.col("context_id").n_unique().alias("contexts"),
        )
        .sort("orientation", "next_base")
    )
    assert summary.height == len(ORIENTATIONS) * len(NUCLEOTIDES)
    return frame, summary


def summarize_mutations(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.group_by("orientation", "relative_position_input", "target_base")
        .agg(
            pl.col("activation_delta").mean().alias("mean_activation_delta"),
            pl.col("activation_delta").median().alias("median_activation_delta"),
            (pl.col("activation_delta") < 0).mean().alias("fraction_contexts_reduced"),
            pl.col("context_id").n_unique().alias("contexts"),
            pl.col("changed").sum().alias("changed_contexts"),
        )
        .sort("orientation", "relative_position_input", "target_base")
    )


def sequence_enrichment(contexts: pl.DataFrame, *, radius: int) -> pl.DataFrame:
    records = []
    for orientation in ORIENTATIONS:
        frame = contexts.filter(pl.col("orientation") == orientation)
        sequences = frame["input_sequence"].to_list()
        weights = frame["recorded_activation"].to_numpy()
        total_weight = weights.sum()
        for relative_position in range(-radius, 1):
            position = FOCAL_INDEX + relative_position
            for base in NUCLEOTIDES:
                mask = np.asarray(
                    [sequence[position] == base for sequence in sequences]
                )
                frequency = float(mask.mean())
                mass = float(weights[mask].sum() / total_weight)
                records.append(
                    {
                        "orientation": orientation,
                        "relative_position_input": relative_position,
                        "base": base,
                        "context_frequency": frequency,
                        "activation_mass_fraction": mass,
                        "log2_mass_enrichment": float(
                            np.log2((mass + 1e-12) / (frequency + 1e-12))
                        ),
                    }
                )
    return pl.DataFrame(records).sort("orientation", "relative_position_input", "base")


def plot_summary(
    mutation_summary: pl.DataFrame,
    readout_summary: pl.DataFrame,
    output_dir: Path,
    *,
    radius: int,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for row_index, orientation in enumerate(ORIENTATIONS):
        frame = mutation_summary.filter(pl.col("orientation") == orientation)
        positions = list(range(-radius, 1))
        matrix = np.empty((len(NUCLEOTIDES), len(positions)))
        for base_index, target_base in enumerate(NUCLEOTIDES):
            values = (
                frame.filter(pl.col("target_base") == target_base)
                .sort("relative_position_input")["mean_activation_delta"]
                .to_numpy()
            )
            assert values.shape == (len(positions),)
            matrix[base_index] = values
        limit = max(float(np.quantile(np.abs(matrix), 0.98)), 1e-6)
        axis = axes[row_index, 0]
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            extent=(-radius - 0.5, 0.5, 3.5, -0.5),
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(range(len(NUCLEOTIDES)), list(NUCLEOTIDES))
        axis.set_xlabel("position relative to focal allele in model input")
        axis.set_ylabel("substitute target base")
        axis.set_title(f"{orientation}: mean f1829 activation change")
        figure.colorbar(image, ax=axis, shrink=0.8)

        readout = readout_summary.filter(pl.col("orientation") == orientation).sort(
            "next_base"
        )
        axis = axes[row_index, 1]
        axis.bar(
            readout["next_base"].to_list(),
            readout["mean_logit_effect"].to_numpy(),
            color=colors,
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel("next nucleotide")
        axis.set_ylabel("mean full - f1829-ablated logit")
        axis.set_title(f"{orientation}: direct final-head readout")
    figure.suptitle(
        "Block-19 feature 1829: activation-only dsQTL context interpretation"
    )
    figure.savefig(output_dir / "feature1829_interpretation.png", dpi=180)
    figure.savefig(output_dir / "feature1829_interpretation.svg")
    plt.close(figure)


@torch.inference_mode()
def run(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    extraction_root: Path,
    sae_path: Path,
    output_dir: Path,
    contexts_per_orientation: int,
    radius: int,
    batch_size: int,
) -> dict[str, Any]:
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert not output_dir.exists()
    assert panel_path.is_file() and panel_manifest_path.is_file()
    assert sae_path.is_dir()
    assert_commit(os.environ.get("EXPERIMENT_COMMIT", ""))
    run_id = os.environ.get("RUN_ID", "")
    assert run_id
    started = time.monotonic()

    panel_manifest = json.loads(panel_manifest_path.read_text())
    panel = pl.read_parquet(panel_path)
    validate_panel(panel, panel_manifest, panel_path)
    extraction_manifest = verify_extraction(extraction_root)
    assert extraction_manifest["panel"]["sha256"] == sha256_file(panel_path)
    sparse_by_orientation = {
        orientation: pl.read_parquet(
            extraction_root / ARM / f"sae_focal_{orientation}.parquet"
        )
        for orientation in ORIENTATIONS
    }
    dense_by_orientation = {
        orientation: dense_feature(
            panel, sparse_by_orientation[orientation], feature_id=FEATURE_ID
        )
        for orientation in ORIENTATIONS
    }
    contexts = select_contexts(
        panel,
        sparse_by_orientation,
        feature_id=FEATURE_ID,
        contexts_per_orientation=contexts_per_orientation,
    )
    mutations = mutation_frame(contexts, radius=radius)

    sae = SAE.load_from_disk(sae_path, device="cuda", dtype="float32")
    sae.requires_grad_(False)
    sae.eval()
    assert sae.cfg.architecture() == "jumprelu"
    assert sae.cfg.d_in == M51_HIDDEN_SIZE and sae.cfg.d_sae == 15_360
    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    frozen.model.config.use_cache = False
    model = HookedProxyLM(frozen.model, frozen.tokenizer, hook_names=[HOOK_NAME])
    torch.cuda.reset_peak_memory_stats()

    base_activation, base_raw = extract_feature_and_raw(
        contexts["input_sequence"].to_list(),
        tokenizer=frozen.tokenizer,
        model=model,
        sae=sae,
        batch_size=batch_size,
    )
    np.testing.assert_allclose(
        base_activation,
        contexts["recorded_activation"].to_numpy(),
        rtol=2e-3,
        atol=NOOP_ABSOLUTE_TOLERANCE,
    )
    mutated_activation, _ = extract_feature_and_raw(
        mutations["input_sequence"].to_list(),
        tokenizer=frozen.tokenizer,
        model=model,
        sae=sae,
        batch_size=batch_size,
    )
    base_by_context = dict(
        zip(contexts["context_id"].to_list(), base_activation, strict=True)
    )
    mutations = mutations.with_columns(
        pl.Series("mutated_activation", mutated_activation),
        pl.col("context_id")
        .replace_strict(base_by_context, return_dtype=pl.Float32)
        .alias("base_activation"),
    ).with_columns(
        (pl.col("mutated_activation") - pl.col("base_activation")).alias(
            "activation_delta"
        )
    )
    noops = mutations.filter(~pl.col("changed"))
    max_abs_noop = float(noops["activation_delta"].abs().max())
    assert max_abs_noop <= NOOP_ABSOLUTE_TOLERANCE
    mutation_summary = summarize_mutations(mutations)
    assert mutation_summary.height == (
        len(ORIENTATIONS) * (radius + 1) * len(NUCLEOTIDES)
    )
    readout, readout_summary = next_base_readout(
        contexts,
        base_raw,
        base_activation,
        frozen_model=frozen,
        tokenizer=frozen.tokenizer,
        sae=sae,
    )
    enrichment = sequence_enrichment(contexts, radius=radius)
    robustness = robustness_summary(panel, dense_by_orientation)

    output_dir.mkdir(parents=True)
    contexts.write_parquet(output_dir / "contexts.parquet")
    mutations.write_parquet(output_dir / "mutations.parquet")
    mutation_summary.write_parquet(output_dir / "mutation_summary.parquet")
    readout.write_parquet(output_dir / "next_base_readout.parquet")
    readout_summary.write_parquet(output_dir / "next_base_readout_summary.parquet")
    enrichment.write_parquet(output_dir / "sequence_enrichment.parquet")
    write_json(output_dir / "robustness.json", robustness)
    plot_summary(mutation_summary, readout_summary, output_dir, radius=radius)

    artifact_names = (
        "contexts.parquet",
        "mutations.parquet",
        "mutation_summary.parquet",
        "next_base_readout.parquet",
        "next_base_readout_summary.parquet",
        "sequence_enrichment.parquet",
        "robustness.json",
        "feature1829_interpretation.png",
        "feature1829_interpretation.svg",
    )
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": run_id,
        "experiment_commit": os.environ["EXPERIMENT_COMMIT"],
        "feature": {
            "id": FEATURE_ID,
            "arm": ARM,
            "report_block": REPORT_BLOCK,
            "sae_weights_sha256": sha256_file(sae_path / "sae_weights.safetensors"),
        },
        "inputs": {
            "panel_sha256": sha256_file(panel_path),
            "panel_manifest_sha256": sha256_file(panel_manifest_path),
            "extraction_manifest_sha256": sha256_file(
                extraction_root / "manifest.json"
            ),
            "extraction_commit": extraction_manifest["experiment_commit"],
        },
        "design": {
            "selection": (
                "top activation only; one most-active allele per variant; "
                "QTL effect not used"
            ),
            "contexts_per_orientation": contexts_per_orientation,
            "orientations": list(ORIENTATIONS),
            "radius": radius,
            "relative_positions_input": [-radius, 0],
            "target_bases": list(NUCLEOTIDES),
            "counterfactual_sequences": mutations.height,
            "noop_absolute_tolerance": NOOP_ABSOLUTE_TOLERANCE,
            "final_head_intervention": (
                "subtract decoded f1829-only contribution from block-19 focal "
                "residual, then apply the frozen final norm and LM head"
            ),
        },
        "numerical_checks": {"max_abs_noop_activation_delta": max_abs_noop},
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "gpu": torch.cuda.get_device_name(0),
            "batch_size": batch_size,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "artifacts": {
            name: {
                "bytes": (output_dir / name).stat().st_size,
                "sha256": sha256_file(output_dir / name),
            }
            for name in artifact_names
        },
    }
    write_json(output_dir / "results.json", result)
    result["artifacts"]["results.json"] = {
        "bytes": (output_dir / "results.json").stat().st_size,
        "sha256": sha256_file(output_dir / "results.json"),
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--contexts-per-orientation", type=int, default=DEFAULT_CONTEXTS
    )
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    result = run(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        extraction_root=args.extraction_root,
        sae_path=args.sae,
        output_dir=args.output_dir,
        contexts_per_orientation=args.contexts_per_orientation,
        radius=args.radius,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
