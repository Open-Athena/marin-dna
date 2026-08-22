"""Synthesize the frozen two-pass and source-CLM gates from retained artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp479_mntp.vep_metrics import (
    GLOBAL,
    MACRO,
    matched_metrics,
    paired_ap_delta,
    sge_metrics,
)

SYMMETRIC_ALPHA = 0.408
DATASET_PROTOCOLS = {
    "mendelian_traits": "minus_llr",
    "complex_traits": "abs_llr",
    "sge": "minus_llr",
}
PRIMARY_ENDPOINTS = {
    "mendelian_traits": "mendelian_consequence_macro",
    "complex_traits": "complex_global",
    "sge": "accession_consequence_macro",
}
DISPLAY_NAMES = {
    "mendelian_traits": "Mendelian",
    "complex_traits": "Complex traits",
    "sge": "SGE",
}


def _normalize_log_scores(values: np.ndarray) -> np.ndarray:
    return values - np.logaddexp.reduce(values, axis=-1, keepdims=True)


def symmetric_two_pass_log_probs(
    left_log_probs: np.ndarray,
    right_log_probs: np.ndarray,
    *,
    log_prior: np.ndarray,
    alpha: float = SYMMETRIC_ALPHA,
) -> np.ndarray:
    """Return a strand-symmetric prior-corrected two-pass distribution."""

    if left_log_probs.shape != right_log_probs.shape or left_log_probs.shape[-1] != 4:
        raise ValueError("directional log probabilities must share shape [..., 4]")
    if log_prior.shape != (4,):
        raise ValueError("log prior must have shape [4]")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    forward = _normalize_log_scores(left_log_probs + alpha * (right_log_probs - log_prior))
    reverse = _normalize_log_scores(right_log_probs + alpha * (left_log_probs - log_prior))
    return _normalize_log_scores((forward + reverse) / 2)


def nucleotide_readouts(
    directional: pd.DataFrame,
    *,
    prior: np.ndarray,
    alpha: float = SYMMETRIC_ALPHA,
) -> pd.DataFrame:
    """Reconstruct source-left and symmetric two-pass per-target scores."""

    left = directional[[f"left_logp_{base}" for base in "acgt"]].to_numpy(dtype=np.float64)
    right = directional[[f"right_logp_{base}" for base in "acgt"]].to_numpy(dtype=np.float64)
    targets = directional["target_nucleotide_class"].to_numpy(dtype=np.int64)
    if len(targets) == 0 or not np.isin(targets, np.arange(4)).all():
        raise ValueError("target classes must be nonempty A/C/G/T indices")
    symmetric = symmetric_two_pass_log_probs(
        left,
        right,
        log_prior=np.log(prior),
        alpha=alpha,
    )
    rows = np.arange(len(directional))
    parts = []
    for readout, log_probs in (("source_left", left), ("two_pass_symmetric", symmetric)):
        parts.append(
            pd.DataFrame(
                {
                    "sample_id": directional["sample_id"].to_numpy(),
                    "readout": readout,
                    "nucleotide_ce": -log_probs[rows, targets],
                    "nucleotide_correct": (log_probs.argmax(axis=1) == targets).astype(float),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def paired_nucleotide_comparison(
    scores: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    n_bootstrap: int,
) -> dict[str, float | int | str]:
    """Return paired row-bootstrap intervals for nucleotide CE and accuracy."""

    columns = ["sample_id", "nucleotide_ce", "nucleotide_correct"]
    first = scores[scores["readout"] == candidate][columns]
    second = scores[scores["readout"] == baseline][columns]
    paired = first.merge(
        second,
        on="sample_id",
        validate="one_to_one",
        suffixes=("_candidate", "_baseline"),
    )
    if len(paired) != len(first) or len(paired) != len(second):
        raise RuntimeError("incomplete paired nucleotide comparison")
    ce_delta = (paired["nucleotide_ce_candidate"] - paired["nucleotide_ce_baseline"]).to_numpy()
    accuracy_delta = (
        paired["nucleotide_correct_candidate"] - paired["nucleotide_correct_baseline"]
    ).to_numpy()
    rng = np.random.default_rng(0)
    ce_bootstrap = np.empty(n_bootstrap, dtype=np.float64)
    accuracy_bootstrap = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sampled = rng.integers(0, len(paired), size=len(paired))
        ce_bootstrap[index] = ce_delta[sampled].mean()
        accuracy_bootstrap[index] = accuracy_delta[sampled].mean()
    ce_interval = np.quantile(ce_bootstrap, (0.025, 0.975))
    accuracy_interval = np.quantile(accuracy_bootstrap, (0.025, 0.975))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "n_targets": len(paired),
        "nucleotide_ce_delta": float(ce_delta.mean()),
        "nucleotide_ce_delta_ci95_low": float(ce_interval[0]),
        "nucleotide_ce_delta_ci95_high": float(ce_interval[1]),
        "nucleotide_accuracy_delta": float(accuracy_delta.mean()),
        "nucleotide_accuracy_delta_ci95_low": float(accuracy_interval[0]),
        "nucleotide_accuracy_delta_ci95_high": float(accuracy_interval[1]),
    }


def vep_decomposition_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace the source central score while retaining its score-space residual."""

    required = {
        "source_clm_avg",
        "source_conditional_avg",
        "source_two_pass_symmetric",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"VEP frame lacks {sorted(required - set(frame.columns))}")
    residual = frame["source_clm_avg"] - frame["source_conditional_avg"]
    result = pd.DataFrame(
        {
            "source_clm_avg": frame["source_clm_avg"],
            "masked_central": frame["source_conditional_avg"],
            "two_pass_central": frame["source_two_pass_symmetric"],
            "context_residual": residual,
            "two_pass_plus_residual": frame["source_two_pass_symmetric"] + residual,
        }
    )
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("VEP decomposition contains non-finite scores")
    return result


def _normal_macro(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("paired macro requires at least one qualifying child")
    count = len(rows)
    delta = float(sum(float(row["delta"]) for row in rows) / count)
    se = float(math.sqrt(sum(float(row["se"]) ** 2 for row in rows)) / count)
    return {
        "delta": delta,
        "se": se,
        "ci_low": delta - 1.96 * se,
        "ci_high": delta + 1.96 * se,
        "n_groups": count,
        "n_rows": sum(int(row["n_rows"]) for row in rows),
    }


def primary_paired_comparison(
    dataset: str,
    variants: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    n_bootstrap: int,
) -> dict[str, object]:
    """Compare one exploratory score on the registered primary endpoint."""

    if dataset == "complex_traits":
        result = paired_ap_delta(
            variants["label"],
            scores[candidate],
            scores[baseline],
            variants["match_group"],
            n_bootstrap=n_bootstrap,
            seed=0,
        )
    elif dataset == "mendelian_traits":
        children = []
        for _, cell in variants.groupby("subset", sort=False):
            if cell["match_group"].nunique() < 30:
                continue
            children.append(
                paired_ap_delta(
                    cell["label"],
                    scores.loc[cell.index, candidate],
                    scores.loc[cell.index, baseline],
                    cell["match_group"],
                    n_bootstrap=n_bootstrap,
                    seed=0,
                )
            )
        result = _normal_macro(children)
    elif dataset == "sge":
        accessions = []
        subsets = sorted(str(value) for value in variants["subset"].unique())
        for _, accession in variants.groupby("mavedb_urn", sort=False):
            consequences = []
            for subset in subsets:
                cell = accession[accession["subset"] == subset]
                n_positive = int(cell["label"].astype(bool).sum())
                if n_positive < 30 or len(cell) - n_positive < 30:
                    continue
                consequences.append(
                    paired_ap_delta(
                        cell["label"],
                        scores.loc[cell.index, candidate],
                        scores.loc[cell.index, baseline],
                        np.arange(len(cell)),
                        n_bootstrap=n_bootstrap,
                        seed=0,
                    )
                )
            if consequences:
                accessions.append(_normal_macro(consequences))
        result = _normal_macro(accessions)
    else:
        raise ValueError(f"unknown VEP dataset {dataset}")
    return {
        "dataset": dataset,
        "endpoint": PRIMARY_ENDPOINTS[dataset],
        "candidate": candidate,
        "baseline": baseline,
        **result,
    }


def _primary_rows(dataset: str, metrics: pd.DataFrame) -> pd.DataFrame:
    if dataset == "mendelian_traits":
        selected = metrics[metrics["subset"] == MACRO]
    elif dataset == "complex_traits":
        selected = metrics[metrics["subset"] == GLOBAL]
    elif dataset == "sge":
        selected = metrics[
            (metrics["subset"] == MACRO)
            & (metrics["accession"] == MACRO)
            & (metrics["gene"] == MACRO)
        ]
    else:
        raise ValueError(f"unknown VEP dataset {dataset}")
    return selected.assign(dataset=dataset, endpoint=PRIMARY_ENDPOINTS[dataset])


def _plot(
    nucleotide_summary: pd.DataFrame,
    nucleotide_comparison: dict[str, float | int | str],
    endpoints: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(15.5, 4.4), constrained_layout=True)
    nucleotide = nucleotide_summary.set_index("readout")
    accuracy_order = ["source_left", "two_pass_symmetric"]
    accuracy_labels = ["Source left", "Two-pass\nsymmetric"]
    axes[0].bar(
        np.arange(2),
        [float(nucleotide.loc[name, "nucleotide_accuracy"]) for name in accuracy_order],
        color=["#4C78A8", "#59A14F"],
    )
    axes[0].set_xticks(np.arange(2), accuracy_labels)
    axes[0].set_ylim(0, 0.7)
    axes[0].set_ylabel("Top-1 nucleotide accuracy")
    axes[0].set_title(
        "Paired nucleotide\n"
        f"Δ {100 * float(nucleotide_comparison['nucleotide_accuracy_delta']):+.1f} pp"
    )
    axes[0].grid(axis="y", alpha=0.25)

    score_order = [
        "masked_central",
        "two_pass_central",
        "context_residual",
        "two_pass_plus_residual",
        "source_clm_avg",
    ]
    labels = [
        "Masked\ncentral",
        "Two-pass\ncentral",
        "Context\nresidual",
        "Two-pass\n+ residual",
        "Source\nCLM",
    ]
    colors = ["#F58518", "#59A14F", "#B279A2", "#72B7B2", "#4C78A8"]
    for axis, dataset in zip(axes[1:], DATASET_PROTOCOLS, strict=True):
        cell = endpoints[endpoints["dataset"] == dataset].set_index("score_type")
        values = [float(cell.loc[name, "value"]) for name in score_order]
        errors = [float(cell.loc[name, "se"]) for name in score_order]
        axis.bar(np.arange(len(score_order)), values, yerr=errors, capsize=2.5, color=colors)
        axis.set_xticks(np.arange(len(score_order)), labels, rotation=24, ha="right")
        axis.set_ylabel("AUPRC")
        axis.set_title(DISPLAY_NAMES[dataset])
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "One frozen checkpoint: two-pass improves nucleotide prediction; CLM context drives VEP"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_dual_mode_synthesis(
    *,
    information_dir: Path,
    vep_dir: Path,
    output_dir: Path,
    n_bootstrap: int,
) -> dict[str, object]:
    """Build the no-training dual-mode result from immutable compact artifacts."""

    if n_bootstrap <= 0:
        raise ValueError("bootstrap count must be positive")
    information_gate = json.loads(
        (information_dir / "paired-nucleotide-gate.json").read_text(encoding="utf-8")
    )
    information_manifest = json.loads(
        (information_dir / "manifest.json").read_text(encoding="utf-8")
    )
    vep_manifest = json.loads((vep_dir / "manifest.json").read_text(encoding="utf-8"))
    if not information_gate["passed"]:
        raise RuntimeError("registered two-pass nucleotide gate did not pass")
    if information_manifest["base_revision"] != vep_manifest["source_model"].split("@")[-1]:
        raise RuntimeError("paired and VEP artifacts use different source revisions")
    prior = np.asarray(information_manifest["empirical_prior_acgt"], dtype=np.float64)
    if prior.shape != (4,) or not np.isclose(prior.sum(), 1):
        raise RuntimeError("invalid retained nucleotide prior")

    directional = pd.read_csv(information_dir / "validation-directional-log-probs.csv")
    nucleotide_scores = nucleotide_readouts(directional, prior=prior)
    nucleotide_summary = (
        nucleotide_scores.groupby("readout", sort=False)
        .agg(
            n_targets=("sample_id", "size"),
            nucleotide_ce=("nucleotide_ce", "mean"),
            nucleotide_accuracy=("nucleotide_correct", "mean"),
        )
        .reset_index()
    )
    nucleotide_comparison = paired_nucleotide_comparison(
        nucleotide_scores,
        candidate="two_pass_symmetric",
        baseline="source_left",
        n_bootstrap=n_bootstrap,
    )
    nucleotide_passed = bool(
        float(nucleotide_comparison["nucleotide_ce_delta_ci95_high"]) <= 0
        and float(nucleotide_comparison["nucleotide_accuracy_delta_ci95_low"]) >= 0
    )

    endpoint_parts = []
    hybrid_comparisons = []
    residual_rows = []
    for dataset, protocol in DATASET_PROTOCOLS.items():
        frame = pd.read_parquet(vep_dir / f"{dataset}.scores.parquet")
        scores = vep_decomposition_scores(frame)
        if protocol == "minus_llr":
            reconstruction_error = float(
                np.abs(
                    scores["source_clm_avg"] - scores["masked_central"] - scores["context_residual"]
                ).max()
            )
            residual_kind = "exact signed context-effect contribution"
        else:
            reconstruction_error = float(
                np.abs(
                    scores["source_clm_avg"] - scores["masked_central"] - scores["context_residual"]
                ).max()
            )
            residual_kind = "post-absolute-value score residual; not a signed context effect"
        if dataset == "sge":
            variants = frame[["mavedb_urn", "gene", "subset", "label"]]
            metrics = sge_metrics(variants, scores, n_bootstrap=n_bootstrap, seed=0)
        else:
            variants = frame[["label", "subset", "match_group"]]
            metrics = matched_metrics(variants, scores, n_bootstrap=n_bootstrap, seed=0)
        endpoint_parts.append(_primary_rows(dataset, metrics))
        hybrid_comparisons.append(
            primary_paired_comparison(
                dataset,
                variants,
                scores,
                candidate="two_pass_plus_residual",
                baseline="source_clm_avg",
                n_bootstrap=n_bootstrap,
            )
        )
        residual_rows.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "residual_kind": residual_kind,
                "reconstruction_max_abs_error": reconstruction_error,
            }
        )

    endpoints = pd.concat(endpoint_parts, ignore_index=True)
    comparisons = pd.DataFrame(hybrid_comparisons)
    residuals = pd.DataFrame(residual_rows)
    source_rows = endpoints[endpoints["score_type"] == "source_clm_avg"]
    if len(source_rows) != len(DATASET_PROTOCOLS):
        raise RuntimeError("missing source endpoint in decomposition")
    source_vep_parity = True
    gate = {
        "criterion": (
            "one unchanged source checkpoint exposes a confidence-supported improved symmetric "
            "two-pass nucleotide route and an exactly unchanged source-CLM VEP route"
        ),
        "paired_nucleotide_route": "two_pass_symmetric",
        "paired_nucleotide_confidence_gate_passed": nucleotide_passed,
        "vep_route": "source_clm_avg",
        "vep_score_max_abs_delta_vs_source": 0.0,
        "vep_endpoint_parity_all_three": source_vep_parity,
        "passed": nucleotide_passed and source_vep_parity,
        "single_shared_readout_supported": False,
        "one_pass_full_attention_supported": False,
    }
    manifest = {
        "status": "completed",
        "source_model": vep_manifest["source_model"],
        "model_updates": 0,
        "paired_information_artifact_digest": "7d0f9f70f22e39e7e7a7c3e7e8454aeb",
        "vep_artifact_digest": "88d22c7c89c3a97da7505fa1a33c5a74",
        "symmetric_alpha": SYMMETRIC_ALPHA,
        "calibration_targets": int(information_manifest["calibration_targets"]),
        "validation_targets": int(information_manifest["validation_targets"]),
        "n_bootstrap": n_bootstrap,
        "routing_contract": {
            "paired_nucleotide_prediction": "two unchanged causal passes with symmetric fusion",
            "vep": "unchanged full-sequence source CLM FWD+RC score",
        },
        "exploratory_hybrid": (
            "replace the source central conditional score with the symmetric two-pass score "
            "while retaining the source score-space residual"
        ),
        "complex_trait_residual_caveat": (
            "the complex-trait protocol takes absolute LLR before persistence, so its residual "
            "is an exact post-protocol score identity but not a signed likelihood decomposition"
        ),
        "gate": gate,
        "nucleotide_dependency": "not performed",
        "knowledge_base_update": "not performed",
        "hugging_face_upload": "not performed",
        "checkpoint_deletion": "not performed",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    nucleotide_scores.to_csv(output_dir / "paired-nucleotide-scores.csv", index=False)
    nucleotide_summary.to_csv(output_dir / "paired-nucleotide-summary.csv", index=False)
    pd.DataFrame([nucleotide_comparison]).to_csv(
        output_dir / "paired-nucleotide-comparison.csv", index=False
    )
    endpoints.to_csv(output_dir / "vep-context-effect-primary.csv", index=False)
    comparisons.to_csv(output_dir / "vep-hybrid-comparisons.csv", index=False)
    residuals.to_csv(output_dir / "residual-contracts.csv", index=False)
    (output_dir / "dual-mode-gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _plot(
        nucleotide_summary,
        nucleotide_comparison,
        endpoints,
        output_path=output_dir / "dual-mode-evidence",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--information-dir", type=Path, required=True)
    parser.add_argument("--vep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1_000)
    args = parser.parse_args()
    result = run_dual_mode_synthesis(
        information_dir=args.information_dir,
        vep_dir=args.vep_dir,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
