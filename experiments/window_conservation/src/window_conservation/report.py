"""Collect resources, scaling summaries, and inspectable scientific plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from window_conservation.scaling import resource


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    out = root / "report"
    resources = []
    for k in [17, 21, 25]:
        build, score = (
            resource(root / "logs" / f"k{k}-build"),
            resource(root / "logs" / f"k{k}-score"),
        )
        resources.append(
            {
                "k": k,
                "build_seconds": build["wall_seconds"],
                "score_seconds": score["wall_seconds"],
                "total_seconds": build["wall_seconds"] + score["wall_seconds"],
                "peak_rss_kib": max(build["max_rss_kib"], score["max_rss_kib"]),
                "index_bytes": build["measurements"][0]["index_bytes"],
                "bases": build["measurements"][0]["bases"],
                "windows": sum(row["windows"] for row in score["measurements"]),
                "unique_words": build["measurements"][0]["unique_keys"],
                "lookup_operations": sum(
                    row["lookups"] for row in score["measurements"]
                ),
            }
        )
    (out / "real_resources.json").write_text(json.dumps(resources, indent=2) + "\n")
    raw = json.loads((root / "scaling/measurements.json").read_text())
    shapes = sorted({(r["species"], r["windows_per_species"]) for r in raw})
    summaries = []
    for species, windows in shapes:
        rows = [
            r
            for r in raw
            if (r["species"], r["windows_per_species"]) == (species, windows)
        ]
        times = [r["total_seconds"] for r in rows]
        summaries.append(
            {
                "species": species,
                "windows_per_species": windows,
                "windows": species * windows,
                "median_seconds": float(np.median(times)),
                "min_seconds": min(times),
                "max_seconds": max(times),
                "median_peak_rss_kib": float(
                    np.median([r["max_rss_kib"] for r in rows])
                ),
                "index_bytes": rows[0]["build"]["measurements"][0]["index_bytes"],
                "selected_occurrences": rows[0]["build"]["measurements"][0][
                    "selected_occurrences"
                ],
                "score_lookups": sum(
                    v["lookups"] for v in rows[0]["score"]["measurements"]
                ),
                "repetitions": len(rows),
            }
        )
    with (out / "scaling.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    fitted = {}
    for ax, axis in zip(axes, ["species", "windows_per_species"], strict=True):
        selected = [
            r
            for r in summaries
            if (
                r["windows_per_species"] == 2048
                if axis == "species"
                else r["species"] == 250
            )
        ]
        selected.sort(key=lambda r: r[axis])
        x = np.array([r[axis] for r in selected])
        y = np.array([r["median_seconds"] for r in selected])
        errors = np.array(
            [
                [r["median_seconds"] - r["min_seconds"] for r in selected],
                [r["max_seconds"] - r["median_seconds"] for r in selected],
            ]
        )
        ax.errorbar(x, y, yerr=errors, fmt="o-", capsize=0, label="Measured median")
        ax.plot(
            x, y[0] * x / x[0], linestyle="--", color="0.5", label="Linear reference"
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(x, labels=[str(int(v)) for v in x])
        ax.set_ylabel("Construction + scoring (s)")
        ax.set_xlabel(
            "Synthetic species" if axis == "species" else "Windows per species"
        )
        ax.set_box_aspect(1)
        ax.legend(title="Time", loc="upper left")
        fitted[axis] = {
            "log_log_slope": float(np.polyfit(np.log(x), np.log(y), 1)[0]),
            "input_growth": float(x[-1] / x[0]),
            "time_growth": float(y[-1] / y[0]),
            "peak_rss_growth": selected[-1]["median_peak_rss_kib"]
            / selected[0]["median_peak_rss_kib"],
        }
    fig.savefig(out / "scaling.svg")
    fig.savefig(out / "scaling.png", dpi=160)
    plt.close(fig)
    held = json.loads((out / "heldout.json").read_text())
    primary = held["budgets"]["0.05"]
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    names = ["Versus random", "Versus matched"]
    fields = ["random_enrichment", "matched_enrichment"]
    heights = np.array([primary[field] for field in fields])
    ci = np.array([held["primary_ci"][field] for field in fields])
    ax.errorbar(
        [0, 1],
        heights,
        yerr=np.stack([heights - ci[:, 0], ci[:, 1] - heights]),
        fmt="o",
        capsize=0,
    )
    ax.axhline(1, color="0.5", linestyle="--")
    ax.set_xticks([0, 1], labels=names)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Conserved-base enrichment")
    ax.set_ylim(bottom=0)
    fig.savefig(out / "conservation.svg")
    fig.savefig(out / "conservation.png", dpi=160)
    plt.close(fig)
    selection = held["selection"]
    chosen = next(r for r in resources if r["k"] == selection["k"])
    # The real run uses a query-restricted index: do not scale its query-only
    # memory or score time as if it indexed/scored every genome.
    largest = max(summaries, key=lambda row: row["windows"])
    extrapolations = []
    for target_bases in [100_000_000_000, 3_000_000_000_000]:
        scale = target_bases / (largest["windows"] * 100)
        extrapolations.append(
            {
                "target_bases": target_bases,
                "target_100bp_intervals": target_bases // 100,
                "linear_single_process_hours": largest["median_seconds"] * scale / 3600,
                "linear_index_bytes": largest["index_bytes"] * scale,
                "linear_peak_rss_bytes": largest["median_peak_rss_kib"] * 1024 * scale,
            }
        )
    stretches = []
    for line in (out / "stretches-0.05.bed").read_text().splitlines():
        _, start, end, _, _ = line.split("\t")
        stretches.append(int(end) - int(start))
    assert sum(stretches) == primary["selected_bases"]
    summary = {
        "real_resources": resources,
        "scaling_fits": fitted,
        "real_scope": "Query-restricted chr1/chr2 scores, full-genome support counts across three species",
        "selected_real_setting": chosen,
        "global_extrapolation_not_measurement": {
            "basis": largest,
            "scenarios": extrapolations,
            "window_bases": 100,
            "hash_sampling": "1/4",
            "limits": "Synthetic uniform-DNA, constant-throughput and distinct-word-rate extrapolation. Real genome repetition/divergence, RAM/cache/I/O and species diversity change both. The full 1,000-genome workload was not run.",
        },
        "primary_stretches": {
            "count": len(stretches),
            "total_bases": sum(stretches),
            "median_bases": float(np.median(stretches)),
            "max_bases": max(stretches),
        },
        "biological_gate_passed": held["biological_gate_passed"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
