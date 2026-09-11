"""Build auditable tables and figures from final physical-locus predictions."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def group_counts(rows: list[dict], budget: int = 10) -> dict[str, tuple[int, int]]:
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = row.get("split_component", row["query"])
        groups[key][0] += sum(
            rank is not None and rank <= budget for rank in row["ranks"].values()
        )
        groups[key][1] += len(row["ranks"])
    return {k: (v[0], v[1]) for k, v in groups.items()}


def interval(rows: list[dict], budget: int = 10) -> tuple[float, float]:
    values = np.array(list(group_counts(rows, budget).values()))
    rng = np.random.default_rng(568)
    sampled = values[rng.integers(0, len(values), (2000, len(values)))].sum(axis=1)
    return tuple(
        float(v) for v in np.quantile(sampled[:, 0] / sampled[:, 1], [0.025, 0.975])
    )


def paired_difference(a: list[dict], b: list[dict], budget: int = 10) -> dict:
    ga, gb = group_counts(a, budget), group_counts(b, budget)
    assert ga.keys() == gb.keys()
    names = sorted(ga)
    assert all(ga[k][1] == gb[k][1] for k in names)
    data = np.array([(ga[k][0] - gb[k][0], ga[k][1]) for k in names])
    rng = np.random.default_rng(568)
    draws = data[rng.integers(0, len(data), (2000, len(data)))].sum(axis=1)
    return {
        "difference": float(data[:, 0].sum() / data[:, 1].sum()),
        "ci95": [
            float(v) for v in np.quantile(draws[:, 0] / draws[:, 1], [0.025, 0.975])
        ],
        "bootstrap_groups": len(names),
        "budget": budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    contexts = read_rows(args.root / "data/contexts.jsonl.gz")
    anchors: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in contexts:
        if record["kind"] == "anchor":
            anchors[record["species"], record["component"]].append(record)
    table, details = [], {}
    for path in sorted((args.root / "results").glob("*.summary.json")):
        result = json.loads(path.read_text())
        if "verify" in path.name:
            continue
        prediction_path = Path(
            str(path).replace(".summary.json", ".predictions.jsonl.gz")
        )
        rows = read_rows(prediction_path)
        assert sum(len(r["ranks"]) for r in rows) == result["n"]
        assert len({(r["query"], r["source"], r["target"]) for r in rows}) == len(rows)
        method = result.get("method", "union" if "union" in path.name else "exact")
        lower, upper = interval(rows)
        resources = result.get("resources", {})
        if method in ["scan", "lsh"]:
            # Sum measured named stages; do not mistake cached wall time for cold cost.
            preparation = result["feature_seconds"] + result["sketch_seconds"]
            index_seconds = sum(r["index_seconds"] for r in resources.values())
            index_bytes = sum(
                r["index_bytes"] + r["signature_bytes"] for r in resources.values()
            )
        elif method == "linclust":
            preparation = result.get("feature_seconds", 0) + result.get(
                "cluster_load_seconds", 0
            )
            index_seconds = sum(s["seconds"] for s in result["clustering_stages"])
            index_bytes = None
        elif method == "union":
            preparation = result["build_windows_seconds"]
            index_seconds, index_bytes = result["index_seconds"], result["index_bytes"]
        else:
            preparation = result["build_windows_seconds"]
            index_seconds = sum(r["index_seconds"] for r in resources.values())
            index_bytes = sum(r["index_bytes"] for r in resources.values())
        record = {
            "name": path.name.removesuffix(".summary.json"),
            "split": path.name.split("-")[0],
            "method": method,
            "W": result.get("width"),
            "k": result.get("k"),
            "stride_divisor": result.get("divisor"),
            "mask": result.get("mask", False),
            "hashes": result.get("hashes"),
            "rows_per_band": result.get("rows"),
            "known_pairs": result["n"],
            "query_loci": len(rows),
            "recall1": result["recall_at_1"],
            "recall10": result["recall_at_10"],
            "recall100": result["recall_at_100"],
            "recall10_ci_low": lower,
            "recall10_ci_high": upper,
            "decoy_fraction10": result["injected_decoy_fraction_at_10"],
            "preparation_seconds": preparation,
            "index_seconds": index_seconds,
            "query_seconds": result["query_seconds"],
            "target_index_bytes": index_bytes,
            "cold_stage_seconds": preparation + index_seconds + result["query_seconds"]
            if preparation is not None
            else None,
            "max_rss_kib": result.get("max_rss_kib"),
            "raw_windows": result.get("raw_windows"),
            "candidate_loci_mean": float(np.mean([r["candidate_loci"] for r in rows])),
            "candidate_loci_p95": float(
                np.quantile([r["candidate_loci"] for r in rows], 0.95)
            ),
            "raw_work": result.get("work", result.get("posting_work")),
            "work_unit": "signature entries compared"
            if method == "scan"
            else "band entries"
            if method == "lsh"
            else "kmer postings"
            if method == "exact"
            else "cluster window pairs"
            if method == "linclust"
            else "list entries",
        }
        table.append(record)
        strata = {}
        for pair in sorted({(r["source"], r["target"]) for r in rows}):
            subset = [r for r in rows if (r["source"], r["target"]) == pair]
            count = sum(len(r["ranks"]) for r in subset)
            strata["-".join(pair)] = {
                "pairs": count,
                "recall10": sum(
                    v is not None and v <= 10
                    for r in subset
                    for v in r["ranks"].values()
                )
                / count,
                "ci95": interval(subset),
            }
        bins: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            metadata = anchors[row["source"], row["query"]]
            for feature, cuts in [
                ("gc", [0.4, 0.6]),
                ("repeat", [0.1, 0.5]),
                ("complexity", [0.9, 1.0]),
            ]:
                value = float(np.mean([r[feature] for r in metadata]))
                label = f"{feature}:" + (
                    f"<{cuts[0]}"
                    if value < cuts[0]
                    else f"{cuts[0]}-{cuts[1]}"
                    if value < cuts[1]
                    else f">={cuts[1]}"
                )
                bins[label].append(row)
            if "ancestral_span" in metadata[0]:
                for feature in ["ancestral_span", "divergence", "indel_rate"]:
                    bins[f"{feature}:{metadata[0][feature]}"].append(row)
                bins[
                    f"span_divergence:{metadata[0]['ancestral_span']}:{metadata[0]['divergence']}:"
                    + row["source"]
                    + "-"
                    + row["target"]
                ].append(row)
        binned = {}
        for label, subset in bins.items():
            count = sum(len(r["ranks"]) for r in subset)
            binned[label] = {
                "pairs": count,
                "recall1": sum(
                    v is not None and v <= 1
                    for r in subset
                    for v in r["ranks"].values()
                )
                / count,
                "recall10": sum(
                    v is not None and v <= 10
                    for r in subset
                    for v in r["ranks"].values()
                )
                / count,
                "ci95": interval(subset),
            }
        candidate_kinds: dict[str, int] = defaultdict(int)
        for row in rows:
            for hit in row["hits"][:10]:
                candidate_kinds[hit["kind"]] += 1
        details[record["name"]] = {
            "species_pairs": strata,
            "query_context_strata": binned,
            "top10_candidate_kind_counts": dict(candidate_kinds),
        }
    with (args.out / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (args.out / "strata.json").write_text(json.dumps(details, indent=2) + "\n")
    (args.out / "metrics.json").write_text(json.dumps(table, indent=2) + "\n")
    if args.figures:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"svg.fonttype": "none", "font.size": 11})
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        for k in [9, 13, 17, 21]:
            selected = sorted(
                [
                    r
                    for r in table
                    if r["split"] == "dev"
                    and r["method"] == "exact"
                    and r["stride_divisor"] == 2
                    and not r["mask"]
                    and r["k"] == k
                    and r["W"] <= 1024
                ],
                key=lambda r: r["W"],
            )
            y = np.array([r["recall10"] for r in selected]) * 100
            errors = (
                np.array(
                    [
                        [r["recall10"] - r["recall10_ci_low"] for r in selected],
                        [r["recall10_ci_high"] - r["recall10"] for r in selected],
                    ]
                )
                * 100
            )
            ax.errorbar(
                [r["W"] for r in selected], y, yerr=errors, label=str(k), capsize=0
            )
        ax.set(
            xscale="log",
            xlabel="Window length (bp)",
            ylabel="Known-pair recall at C=10 (%)",
            ylim=(0, 101),
            title="Full k-mer sets on development loci",
        )
        ax.set_xticks([64, 128, 255, 511, 1024], ["64", "128", "255", "511", "1024"])
        ax.set_box_aspect(1)
        ax.legend(title="k-mer length", loc="lower right")
        fig.savefig(args.out / "window-screen.svg")
        fig.savefig(args.out / "window-screen.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
        for axis, width in zip(axes, [255, 1024], strict=True):
            for method, color in [("exact", "C0"), ("scan", "C1"), ("lsh", "C2")]:
                rows = [
                    r
                    for r in table
                    if r["split"] == "dev"
                    and r["method"] == method
                    and r["W"] == width
                    and r["stride_divisor"] == 2
                    and not r["mask"]
                    and r["k"] == (9 if width == 255 else 13)
                ]
                axis.errorbar(
                    [r["query_seconds"] for r in rows],
                    [r["recall10"] * 100 for r in rows],
                    yerr=np.array(
                        [
                            [r["recall10"] - r["recall10_ci_low"] for r in rows],
                            [r["recall10_ci_high"] - r["recall10"] for r in rows],
                        ]
                    )
                    * 100,
                    label=method,
                    color=color,
                    fmt="o",
                    capsize=0,
                )
            axis.set(
                xscale="log",
                xlabel="Query wall time (s)",
                ylabel="Known-pair recall at C=10 (%)",
                ylim=(0, 101),
                title=f"W={width}, k={9 if width == 255 else 13}",
            )
            axis.set_box_aspect(1)
        axes[1].legend(title="Method", loc="lower right")
        fig.savefig(args.out / "index-frontier.svg")
        fig.savefig(args.out / "index-frontier.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
