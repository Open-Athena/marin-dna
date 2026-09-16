"""Figures and compact summaries for the frozen extension and diagnostics."""

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


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=160)
    target = path.with_suffix(".svg")
    target.write_text(
        "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root / "extension"
    report = root / "report"
    validation = json.loads((report / "validation.json").read_text())
    cells = validation["cells"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    names = {
        "rate4": "Fixed rate 1/4",
        "rate8": "Fixed rate 1/8",
        "rate16": "Fixed rate 1/16",
        "bottom16": "Bottom 16",
        "bottom32": "Bottom 32",
    }
    table = []
    for i, (sample, label) in enumerate(names.items()):
        rows = sorted(
            [row for row in cells if row["sample"] == sample], key=lambda r: r["panel"]
        )
        x = np.arange(3) + (i - 2) * 0.045
        y = np.array(
            [
                row["validation_metrics"]["selected_annotated_fraction"] * 100
                for row in rows
            ]
        )
        ci = (
            np.array(
                [
                    row["validation_intervals"]["selected_annotated_fraction"]
                    for row in rows
                ]
            )
            * 100
        )
        ax.vlines(x, ci[:, 0], ci[:, 1], color=f"C{i}")
        ax.plot(x, y, "o-", color=f"C{i}", label=label)
        for row in rows:
            table.append(
                {
                    "panel": row["panel"],
                    "sample": sample,
                    "score": row["score"],
                    **row["validation_metrics"],
                }
            )
    ax.set_xticks([0, 1, 2], labels=["3", "6", "10"])
    ax.set_xlabel("Species in panel")
    ax.set_ylabel("Conserved bases among selected bases (%)")
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)
    ax.legend(
        title="Word sample", loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2
    )
    save(fig, report / "panel-sampling")
    with (report / "validation-cells.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    measurements = json.loads((root / "memory/measurements.json").read_text())
    original = json.loads((args.root / "scaling/measurements.json").read_text())
    summaries = []
    shapes = sorted(
        {(row["species"], row["windows_per_species"]) for row in measurements}
    )
    for species, windows in shapes:
        for mode, bits in [
            ("baseline", 2),
            ("compact", 2),
            ("partition", 2),
            ("compact", 4),
            ("partition", 4),
        ]:
            if mode == "baseline":
                rows = [
                    r
                    for r in original
                    if r["species"] == species and r["windows_per_species"] == windows
                ]
                times = [r["total_seconds"] for r in rows]
            else:
                rows = [
                    r
                    for r in measurements
                    if r["species"] == species
                    and r["windows_per_species"] == windows
                    and r["mode"] == mode
                    and r["bits"] == bits
                ]
                times = [r["wall_seconds"] for r in rows]
            assert len(rows) == 3
            memory = [r["max_rss_kib"] / 2**20 for r in rows]
            summaries.append(
                {
                    "species": species,
                    "windows_per_species": windows,
                    "mode": mode,
                    "sample_rate": f"1/{2**bits}",
                    "median_seconds": float(np.median(times)),
                    "min_seconds": min(times),
                    "max_seconds": max(times),
                    "median_gib": float(np.median(memory)),
                    "min_gib": min(memory),
                    "max_gib": max(memory),
                }
            )
    (report / "memory-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    chosen = [
        r
        for r in summaries
        if r["species"] == 1000 and r["windows_per_species"] == 2048
    ]
    labels = [f"{r['mode'].capitalize()}\n{r['sample_rate']}" for r in chosen]
    for ax, field, low, high, label in [
        (
            axes[0],
            "median_seconds",
            "min_seconds",
            "max_seconds",
            "Construction + scoring (s)",
        ),
        (axes[1], "median_gib", "min_gib", "max_gib", "Peak RSS (GiB)"),
    ]:
        y = [r[field] for r in chosen]
        ax.vlines(
            range(len(chosen)),
            [r[low] for r in chosen],
            [r[high] for r in chosen],
            color="C0",
        )
        ax.plot(range(len(chosen)), y, "o", color="C0")
        ax.set_xticks(range(len(chosen)), labels=labels)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.set_box_aspect(1)
    save(fig, report / "global-memory")

    biology = json.loads((report / "biology-chr3.json").read_text())
    features = [
        "CDS",
        "exon",
        "TSS_plus_minus_1kb",
        "cCRE_dELS",
        "cCRE_pELS",
        "repeat_LINE",
        "repeat_SINE",
        "repeat_LTR",
    ]
    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")
    for offset, choice, label in [
        (-0.12, "baseline", "Three-species baseline"),
        (0.12, "primary", "Selected extension"),
    ]:
        lookup = {
            r["feature"]: r
            for r in biology["feature_overlap"]
            if r["choice"] == choice and r["budget"] == 0.05
        }
        y = [
            lookup.get(name, {}).get("coverage_enrichment", np.nan) for name in features
        ]
        ax.plot(y, np.arange(len(features)) + offset, "o", label=label)
    ax.set_yticks(
        range(len(features)),
        labels=[
            "CDS",
            "Exon",
            "TSS ±1 kb",
            "Distal cCRE",
            "Proximal cCRE",
            "LINE",
            "SINE",
            "LTR",
        ],
    )
    ax.axvline(1, color="0.5", linestyle="--")
    ax.set_xlabel("Annotation coverage enrichment")
    ax.set_xlim(left=0)
    ax.set_box_aspect(1)
    ax.legend(title="Selection", loc="lower center", bbox_to_anchor=(0.5, 1.01))
    save(fig, report / "biology")

    spatial = json.loads((report / "spatial.json").read_text())
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    for i, (sample, label) in enumerate(names.items()):
        rows = sorted(
            [
                r
                for r in spatial["summary"]
                if r["sample"] == sample
                and r["score"] == "any_copy4"
                and r["tract_length"] == 150
                and r["control"] == "unique"
                and r["indel"] == 0.01
            ],
            key=lambda r: r["substitution"],
        )
        y = np.array([r["positive_detection"] for r in rows])
        # Wilson intervals describe replicate detection, not genome-level uncertainty.
        n, z = 20, 1.96
        center = (y + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(y * (1 - y) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        x = np.array([r["substitution"] * 100 for r in rows]) + (i - 2) * 0.15
        ax.vlines(x, (center - half) * 100, (center + half) * 100, color=f"C{i}")
        ax.plot(x, y * 100, "o-", color=f"C{i}", label=label)
    ax.set_xlabel("Substitutions per support tract (%)")
    ax.set_ylabel("Tracts with local shared-word evidence (%)")
    ax.set_ylim(0, 103)
    ax.set_box_aspect(1)
    ax.legend(
        title="Word sample", loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2
    )
    save(fig, report / "spatial")
    (report / "resources.json").write_text(
        json.dumps(
            {
                "compact_pilot_parity": resource(root / "logs/compact-parity"),
                "rate_panels": resource(root / "logs/rate-panels"),
                "bottom_panels": resource(root / "logs/bottom-panels"),
                "development": resource(root / "logs/development"),
                "validation": resource(root / "logs/validation"),
            },
            indent=2,
        )
        + "\n"
    )
    print("extension figures and tables ready", flush=True)


if __name__ == "__main__":
    main()
