#!/usr/bin/env python3
"""Independently audit issue #402's accepted large-batch sanity bundle."""

from __future__ import annotations

import json
import subprocess

import polars as pl

from marin_dna.pipelines.rag_glm.model_sanity import rag_target_position_metadata


ROOT = (
    "gs://marin-us-east5/evals/dna-exp402-rag-h640-p46m-b2m-30k/"
    "2026.07.26.5/sanity-step5000-7d9a7c9"
)
MODEL_SOURCE = (
    "gs://marin-us-east5/checkpoints/dna-exp402-rag-h640-p46m-b2m-30k/"
    "2026.07.26.5/hf/step-5000"
)
CODE_REVISION = "7d9a7c9f5a2f8040af3daadb8c2be10804c211fc"
TRAINING_DATASET_REVISION = "5e6b30cf878b61c99e6432ad8ab7865b18cbe0e7"
LM_MODES = ("full", "all_n", "roll", "unrelated", "bos_to_pad", "seq_to_unk")
VEP_MODES = ("full", "all_n", "human_only", "bos_to_pad", "seq_to_unk")
TOKEN_ABLATIONS = ("bos_to_pad", "seq_to_unk")
MODE_GEOMETRIES = {
    "full": "fixed_2048_full_context",
    "all_n": "fixed_2048_all_ortholog_slots_N",
    "human_only": "literal_256_BOS_plus_human_out_of_distribution",
    "bos_to_pad": "fixed_2048_BOS_replaced_by_PAD",
    "seq_to_unk": "fixed_2048_SEQ_replaced_by_UNK",
}
BENCHMARKS = {
    "mendelian_traits": {
        "repo": "marin-dna/evals_mendelian_traits_rag_harness_255_v1",
        "revision": "9acedb683463477f34745af30a63a289873008a4",
        "n_documents": 18_980,
        "n_variants": 9_490,
        "score_column": "minus_llr_avg",
    },
    "complex_traits": {
        "repo": "marin-dna/evals_complex_traits_rag_harness_255_v1",
        "revision": "0252a883f650819a8e1fa22062027daafe956540",
        "n_documents": 20_000,
        "n_variants": 10_000,
        "score_column": "abs_llr_avg",
    },
    "sge": {
        "repo": "marin-dna/evals_sge_rag_harness_255_v1",
        "revision": "c20cc58fceb9bc053a55152a89d160f1b070f75d",
        "n_documents": 29_776,
        "n_variants": 14_888,
        "score_column": "minus_llr_avg",
    },
}


def read_manifest(uri: str) -> dict[str, object]:
    """Read one immutable JSON manifest from cloud storage."""
    completed = subprocess.run(
        ["gcloud", "storage", "cat", uri],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(completed.stdout)
    assert isinstance(manifest, dict)
    return manifest


def assert_finite(frame: pl.DataFrame, *, label: str) -> None:
    """Assert that every floating-point column is complete and finite."""
    float_columns = [
        column
        for column, dtype in frame.schema.items()
        if dtype in (pl.Float32, pl.Float64)
    ]
    assert float_columns, label
    for column in float_columns:
        assert frame[column].null_count() == 0, (label, column)
        assert frame.select(pl.col(column).is_finite().all()).item(), (
            label,
            column,
        )


def max_abs(frame: pl.DataFrame, expression: pl.Expr) -> float:
    """Return one finite maximum absolute error."""
    value = frame.select(expression.abs().max()).item()
    assert value is not None
    assert float(value) >= 0
    return float(value)


def audit_root_manifest() -> None:
    """Assert exact checkpoint, data, code, sampling, and mode provenance."""
    expected = {
        "ablation_rows": 512,
        "attention_rows": 4,
        "code_revision": CODE_REVISION,
        "lm_ablation_modes": list(LM_MODES),
        "model": "/home/ubuntu/issue402-checkpoint",
        "model_label": "46M",
        "model_source": MODEL_SOURCE,
        "n_bootstrap": 1_000,
        "training_dataset": "bolinas-dna/zoonomia-rag-v1-v1",
        "training_dataset_revision": TRAINING_DATASET_REVISION,
        "validation_rows": 2_048,
        "vep_context_modes": list(VEP_MODES),
    }
    assert read_manifest(f"{ROOT}/manifest.json") == expected


def audit_validation_loss() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Audit per-position decay and every frozen context/token ablation."""
    position = pl.read_parquet(f"{ROOT}/validation_position_loss.parquet")
    assert position.height == 2_047
    assert position["position"].to_list() == list(range(1, 2_048))
    assert position["model"].unique().to_list() == ["46M"]
    assert position["n_documents"].unique().to_list() == [2_048]
    expected_metadata = rag_target_position_metadata()
    assert position.select(expected_metadata.columns).equals(expected_metadata)
    assert_finite(position, label="validation position loss")
    assert position.filter(
        (pl.col("mean_loss") <= 0) | (pl.col("se_loss") < 0)
    ).is_empty()

    bases = position.filter(
        pl.col("layout_token_type").is_in(["ortholog_base", "human_base"])
    )
    assert bases.height == 8 * 255
    quarters = (
        bases.group_by("segment_index", "segment")
        .agg(
            pl.col("mean_loss")
            .filter(pl.col("within_segment_offset") < 64)
            .mean()
            .alias("first_quarter"),
            pl.col("mean_loss")
            .filter(pl.col("within_segment_offset") >= 191)
            .mean()
            .alias("last_quarter"),
        )
        .sort("segment_index")
    )
    assert quarters.height == 8
    assert quarters["segment_index"].to_list() == list(range(8))
    assert quarters.filter(
        pl.col("last_quarter") >= pl.col("first_quarter")
    ).is_empty(), quarters

    context = pl.read_parquet(f"{ROOT}/validation_context_ablation.parquet")
    assert context.height == len(LM_MODES)
    assert set(context["mode"]) == set(LM_MODES)
    assert context["model"].unique().to_list() == ["46M"]
    assert context["n_documents"].unique().to_list() == [512]
    assert_finite(context, label="validation context ablation")
    assert context.filter(
        (pl.col("mean_human_loss") <= 0) | (pl.col("se_human_loss") < 0)
    ).is_empty()
    full_loss = context.filter(pl.col("mode") == "full")["mean_human_loss"].item()
    effects = (
        context.filter(pl.col("mode") != "full")
        .with_columns(
            (pl.col("mean_human_loss") - full_loss).alias("loss_increase_vs_full")
        )
        .sort("mode")
    )
    assert effects.filter(pl.col("loss_increase_vs_full") <= 0).is_empty(), effects
    assert (
        effects.filter(pl.col("mode") == "roll")["loss_increase_vs_full"].item() > 0.05
    )
    assert (
        effects.filter(pl.col("mode").is_in(["all_n", "unrelated"]))[
            "loss_increase_vs_full"
        ].min()
        > 0.9
    )
    assert (
        effects.filter(pl.col("mode") == "seq_to_unk")["loss_increase_vs_full"].item()
        > 0.02
    )
    return quarters, effects


def audit_attention() -> pl.DataFrame:
    """Audit causal masking and aligned-position attention with an N control."""
    diagnostics = pl.read_parquet(f"{ROOT}/attention_diagnostics.parquet")
    assert diagnostics.height == 4 * 7
    assert diagnostics["document_index"].n_unique() == 4
    assert diagnostics["anchor_id"].n_unique() == 4
    assert diagnostics["layer"].unique().sort().to_list() == list(range(7))
    assert diagnostics["model"].unique().to_list() == ["46M"]
    assert_finite(diagnostics, label="attention diagnostics")
    assert diagnostics["future_attention_max_abs"].max() == 0
    assert diagnostics["row_sum_max_abs_error"].max() < 0.01

    alignment = pl.read_parquet(f"{ROOT}/attention_alignment.parquet")
    assert alignment.height == 12_740
    assert alignment["model"].unique().to_list() == ["46M"]
    assert alignment["document_index"].n_unique() == 4
    assert alignment["layer"].unique().sort().to_list() == list(range(7))
    assert alignment["offset"].min() == -32
    assert alignment["offset"].max() == 32
    assert set(alignment["availability"]) == {"available", "missing"}
    assert_finite(alignment, label="attention alignment")
    assert alignment.filter(
        (pl.col("mean_attention") <= 0)
        | (pl.col("n_documents") <= 0)
        | (pl.col("n_query_offsets") <= 0)
    ).is_empty()
    pooled = (
        alignment.with_columns(
            (pl.col("n_documents") * pl.col("n_query_offsets")).alias("weight")
        )
        .group_by("availability", "offset")
        .agg(
            (
                (pl.col("mean_attention") * pl.col("weight")).sum()
                / pl.col("weight").sum()
            ).alias("mean_attention")
        )
        .sort("availability", "offset")
    )
    assert pooled.height == 2 * 65
    peaks = (
        pooled.sort(
            "availability",
            "mean_attention",
            descending=[False, True],
        )
        .group_by("availability", maintain_order=True)
        .head(1)
        .sort("availability")
    )
    available_peak = peaks.filter(pl.col("availability") == "available")
    missing_peak = peaks.filter(pl.col("availability") == "missing")
    assert available_peak["offset"].item() == 1
    assert missing_peak["offset"].item() == 1
    assert (
        available_peak["mean_attention"].item() / missing_peak["mean_attention"].item()
        > 10
    )

    regions = pl.read_parquet(f"{ROOT}/attention_regions.parquet")
    assert regions.height == 252
    assert regions["model"].unique().to_list() == ["46M"]
    assert regions["document_index"].n_unique() == 4
    assert regions["layer"].unique().sort().to_list() == list(range(7))
    assert_finite(regions, label="attention regions")
    assert regions.filter(
        (pl.col("mean_attention_mass") < 0)
        | (pl.col("n_documents") <= 0)
        | (pl.col("n_query_offsets") <= 0)
    ).is_empty()
    return peaks


def headline_row(metrics: pl.DataFrame, benchmark: str) -> pl.DataFrame:
    """Select the one frozen headline AUPRC row."""
    if benchmark == "sge":
        selected = metrics.filter(
            (pl.col("metric") == "AUPRC")
            & (pl.col("subset") == "_macro_avg_")
            & (pl.col("accession") == "_macro_avg_")
            & (pl.col("gene") == "_macro_avg_")
        )
    else:
        selected = metrics.filter(pl.col("subset") == "_global_")
    assert selected.height == 1, (benchmark, selected)
    return selected


def audit_vep() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Audit all VEP modes, RC aggregation, score transforms, and sensitivity."""
    headline_rows: list[dict[str, object]] = []
    for benchmark, spec in BENCHMARKS.items():
        for mode in VEP_MODES:
            root = f"{ROOT}/vep/{benchmark}/{mode}"
            expected_manifest = {
                "benchmark": benchmark,
                "benchmark_repo": spec["repo"],
                "benchmark_revision": spec["revision"],
                "context_mode": mode,
                "geometry": MODE_GEOMETRIES[mode],
                "n_document_rows": spec["n_documents"],
                "n_variants": spec["n_variants"],
            }
            assert read_manifest(f"{root}/manifest.json") == expected_manifest

            variants = pl.read_parquet(f"{root}/variants.parquet")
            assert variants.height == spec["n_variants"]
            assert variants["variant_id"].n_unique() == spec["n_variants"]
            score_column = str(spec["score_column"])
            required = {
                "llr_fwd",
                "llr_rc",
                "llr_avg",
                "ref_loglikelihood_fwd",
                "ref_loglikelihood_rc",
                "ref_loglikelihood_avg",
                "alt_loglikelihood_fwd",
                "alt_loglikelihood_rc",
                "alt_loglikelihood_avg",
                score_column,
            }
            assert required <= set(variants.columns)
            assert_finite(variants, label=f"{benchmark} {mode} variants")
            checked = variants.with_columns(
                ((pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("expected_llr_avg"),
                (
                    (pl.col("ref_loglikelihood_fwd") + pl.col("ref_loglikelihood_rc"))
                    / 2
                ).alias("expected_ref_avg"),
                (
                    (pl.col("alt_loglikelihood_fwd") + pl.col("alt_loglikelihood_rc"))
                    / 2
                ).alias("expected_alt_avg"),
            )
            expected_score = (
                pl.col("llr_avg").abs()
                if score_column == "abs_llr_avg"
                else -pl.col("llr_avg")
            )
            exact_errors = {
                "llr_average": max_abs(
                    checked,
                    pl.col("llr_avg") - pl.col("expected_llr_avg"),
                ),
                "ref_average": max_abs(
                    checked,
                    pl.col("ref_loglikelihood_avg") - pl.col("expected_ref_avg"),
                ),
                "alt_average": max_abs(
                    checked,
                    pl.col("alt_loglikelihood_avg") - pl.col("expected_alt_avg"),
                ),
                "score_transform": max_abs(
                    checked,
                    pl.col(score_column) - expected_score,
                ),
            }
            assert all(error == 0 for error in exact_errors.values()), (
                benchmark,
                mode,
                exact_errors,
            )

            metrics = pl.read_parquet(f"{root}/metrics.parquet")
            assert_finite(metrics, label=f"{benchmark} {mode} metrics")
            headline = headline_row(metrics, benchmark)
            assert headline["score_type"].item() == score_column
            assert headline["context_mode"].item() == mode
            headline_rows.append(
                {
                    "benchmark": benchmark,
                    "mode": mode,
                    "auprc": headline["value"].item(),
                    "se": headline["se"].item(),
                    "score_column": score_column,
                    "max_exact_error": max(exact_errors.values()),
                }
            )

    headlines = pl.DataFrame(headline_rows).sort("benchmark", "mode")
    assert headlines.height == len(BENCHMARKS) * len(VEP_MODES)
    assert headlines.filter(
        ~pl.col("auprc").is_finite()
        | ~pl.col("se").is_finite()
        | (pl.col("auprc") < 0)
        | (pl.col("auprc") > 1)
        | (pl.col("se") < 0)
        | (pl.col("max_exact_error") != 0)
    ).is_empty()
    context_comparison = (
        headlines.filter(pl.col("mode").is_in(["full", "all_n", "human_only"]))
        .pivot(index="benchmark", on="mode", values="auprc")
        .sort("benchmark")
    )
    assert context_comparison.filter(
        (pl.col("full") <= pl.col("all_n")) | (pl.col("full") <= pl.col("human_only"))
    ).is_empty(), context_comparison

    token_diagnostics = pl.read_parquet(
        f"{ROOT}/vep_special_token_diagnostics.parquet"
    ).sort("benchmark", "ablation")
    assert token_diagnostics.height == len(BENCHMARKS) * len(TOKEN_ABLATIONS)
    assert set(token_diagnostics["benchmark"]) == set(BENCHMARKS)
    assert set(token_diagnostics["ablation"]) == set(TOKEN_ABLATIONS)
    assert_finite(token_diagnostics, label="VEP special-token diagnostics")
    assert token_diagnostics.filter(
        (pl.col("n_documents") <= 0)
        | (pl.col("mean_abs_llr_delta") <= 0.01)
        | (pl.col("max_abs_llr_delta") <= pl.col("mean_abs_llr_delta"))
        | (pl.col("fraction_llr_changed_gt_1e_6") < 0.999)
        | (pl.col("llr_pearson") <= 0.9)
        | (pl.col("llr_pearson") > 1)
    ).is_empty(), token_diagnostics
    return headlines, token_diagnostics


def main() -> None:
    audit_root_manifest()
    quarters, loss_effects = audit_validation_loss()
    attention_peaks = audit_attention()
    headlines, token_diagnostics = audit_vep()
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=180):
        print("POSITIONAL LOSS QUARTERS")
        print(quarters)
        print("\nHUMAN LOSS INCREASE VS FULL")
        print(
            loss_effects.select(
                "mode",
                "mean_human_loss",
                "se_human_loss",
                "loss_increase_vs_full",
            )
        )
        print("\nATTENTION PEAKS")
        print(attention_peaks)
        print("\nVEP HEADLINES")
        print(headlines)
        print("\nSPECIAL-TOKEN LLR DIAGNOSTICS")
        print(token_diagnostics)
        print("\nLARGE_BATCH_SANITY_AUDIT accepted")


if __name__ == "__main__":
    main()
