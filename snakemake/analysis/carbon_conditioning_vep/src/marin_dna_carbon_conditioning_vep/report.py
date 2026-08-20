"""Compact Markdown summary rendering for the Carbon conditioning analysis."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

TICK = chr(96)


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _bytes_gib(value: float) -> str:
    return f"{float(value) / (1024**3):.2f} GiB"


def _code(value: Any) -> str:
    return f"{TICK}{value}{TICK}"


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def render_summary(
    *,
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    absolute_metrics: pd.DataFrame,
    paired_deltas: pd.DataFrame,
    exclusions: pd.DataFrame,
    runtimes: list[Mapping[str, Any]],
) -> str:
    """Render all required result and provenance fields without a winner gate."""
    conditions = [str(value) for value in config["analysis"]["conditions"]]
    scope = config["analysis"].get("subset") or config["analysis"]["subset_label"]
    config_path = str(config["config_path"])
    metric_subsets = [
        subset
        for subset in absolute_metrics["subset"].drop_duplicates()
        if subset != "_macro_avg_"
    ] + ["_macro_avg_"]
    absolute_rows: list[dict[str, str]] = []
    for subset in metric_subsets:
        row: dict[str, str] = {"subset": subset}
        subset_metrics = absolute_metrics.loc[absolute_metrics["subset"] == subset]
        for condition in conditions:
            cell = subset_metrics.loc[subset_metrics["condition"] == condition].iloc[0]
            row[condition] = _format_float(float(cell["auprc"]))
        sample = subset_metrics.iloc[0]
        row["rows"] = str(int(sample["n_rows"]))
        row["groups"] = str(int(sample["n_groups"]))
        row["macro"] = "yes" if bool(sample["macro_eligible"]) else "no"
        row["warning"] = "low sample" if bool(sample["low_sample"]) else ""
        absolute_rows.append(row)
    absolute_table = pd.DataFrame(absolute_rows)

    delta_table = paired_deltas[
        ["comparison", "subset", "delta", "ci_low", "ci_high", "n_groups", "low_sample"]
    ].copy()
    for column in ("delta", "ci_low", "ci_high"):
        delta_table[column] = delta_table[column].map(_format_float)
    delta_table["low_sample"] = delta_table["low_sample"].map(
        lambda value: "yes" if value else ""
    )

    lines = [
        "# Carbon species conditioning on Mendelian VEP",
        "",
        "## TL;DR",
        "",
        f"This exploratory pilot reports {len(conditions)} prompt conditions on the {_code(scope)} scope.",
        "No pass/fail outcome or testing hierarchy is assigned.",
        "",
        "## Fixed contract",
        "",
        f"- Model: {_code(config['model']['repo'])} at {_code(config['model']['revision'])}.",
        f"- Dataset: {_code(config['dataset']['repo'])} {_code(config['dataset']['split'])} at {_code(config['dataset']['revision'])}.",
        f"- Reference: {config['reference']['assembly']} Ensembl release {config['reference']['ensembl_release']} {config['reference']['masking']} primary assembly.",
        f"- Window: {config['inference']['window_size_bp']:,} bp, truncated to {(config['inference']['window_size_bp'] // config['inference']['kmer_size']) * config['inference']['kmer_size']:,} bp at Carbon's 6-mer boundary.",
        f"- Score: mean causal token log likelihood in bf16; FWD/RC LLR average; {_code('score = -llr')}.",
        f"- Bootstrap: {config['metrics']['n_bootstrap']:,} seeded match-group draws.",
        "",
        "## Prompt preflight",
        "",
        f"- Selected grammar: {_code(preflight['selected_grammar'])} with template {_code(preflight['grammar_templates'][preflight['selected_grammar']])}.",
        f"- Rejected grammar: {_code(preflight['rejected_grammar'])} with template {_code(preflight['grammar_templates'][preflight['rejected_grammar']])}.",
        f"- Tokenizer revision: {_code(preflight['tokenizer_revision'])}.",
        "",
    ]
    for condition in conditions:
        lines.append(
            f"- {_code(condition)}: {_code(preflight['selected_prefixes'][condition])}; "
            f"prefix IDs {_code(preflight['prefix_ids'][condition])}."
        )
    lines.extend(
        [
            "",
            "## Absolute AUPRC",
            "",
            *_markdown_table(absolute_table),
            "",
            f"Subsets with fewer than {config['metrics']['min_groups_for_macro']} match groups are reported and excluded from the macro average.",
            "",
            "## Paired AUPRC differences",
            "",
            *_markdown_table(delta_table),
            "",
            "Intervals are paired bootstrap intervals over identical rows and shared match-group draws.",
            "An interval crossing zero is not evidence that two prompt conditions are equivalent.",
            "",
            "## Exclusions and deviations",
            "",
        ]
    )
    if exclusions.empty:
        lines.append("- No match groups were excluded during window validation.")
    else:
        for row in exclusions.sort_values(["match_group", "variant_id"]).itertuples():
            lines.append(
                f"- Match group {_code(row.match_group)}, variant {_code(row.variant_id)}: "
                f"{_code(row.reason)} ({row.detail})."
            )
    lines.extend(
        [
            "- No scorer-contract deviations were recorded; dataset rows that failed the window contract are listed above.",
            "",
            "## Runtime",
            "",
        ]
    )
    runtime_table = pd.DataFrame(
        [
            {
                "condition": runtime["condition"],
                "rows": runtime["rows"],
                "devices": ", ".join(runtime["devices"]),
                "elapsed": f"{float(runtime['elapsed_seconds']):.1f} s",
                "peak GPU": _bytes_gib(runtime["peak_gpu_memory_bytes"]),
                "peak RSS": _bytes_gib(runtime["peak_rss_bytes"]),
            }
            for runtime in runtimes
        ]
    )
    lines.extend(_markdown_table(runtime_table))
    lines.extend(
        [
            "",
            "## Exact commands",
            "",
            TICK * 3 + "bash",
            "cd snakemake/analysis/carbon_conditioning_vep",
            f"uv run --locked --group genome-s3 snakemake --configfile {config_path} --profile workflow/profiles/default smoke",
            f"uv run --locked --group genome-s3 snakemake -n --configfile {config_path} --profile workflow/profiles/default",
            f"uv run --locked --group genome-s3 snakemake --configfile {config_path} --profile workflow/profiles/default",
            TICK * 3,
            "",
        ]
    )
    return "\n".join(lines)
