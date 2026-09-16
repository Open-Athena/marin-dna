"""Window-level annotated-conserved-base enrichment with matched controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from window_conservation.labels import window_labels
from window_conservation.prepare import sha256


def load_chromosome(
    root: Path, k: int, chrom: str, protocol: dict, score_path: Path | None = None
) -> dict[str, np.ndarray]:
    path = score_path or root / "scores" / f"k{k}-human.tsv"
    with path.open() as handle:
        fields = handle.readline().strip().split("\t")[1:]
        table = np.loadtxt(
            (
                line.split("\t", 1)[1]
                for line in handle
                if line.startswith(chrom + "\t")
            ),
            delimiter="\t",
            ndmin=2,
        )
    valid = (
        table[:, fields.index("valid")] / protocol["window_bases"]
        >= protocol["minimum_valid_fraction"]
    )
    if "maximum_repeat_fraction" in protocol:
        valid &= table[:, fields.index("repeat")] <= protocol["maximum_repeat_fraction"]
    table = table[valid]
    assert len(table)
    values = {field: table[:, i].copy() for i, field in enumerate(fields)}
    del table
    starts, ends = values["start"].astype(np.int64), values["end"].astype(np.int64)
    assert np.all(ends - starts == protocol["window_bases"]) and np.all(
        starts[1:] >= ends[:-1]
    )
    cache = root / "data" / f"labels-{chrom}.npz"
    mask_policy = protocol.get("exclude_lowercase_twobit", "")
    if cache.exists():
        labels = np.load(cache)
        if "mask_policy" in labels:
            assert str(labels["mask_policy"]) == mask_policy
        else:
            assert not mask_policy, (
                "Unmasked label cache cannot serve masked evaluation"
            )
        assert np.array_equal(labels["starts"], starts) and np.array_equal(
            labels["ends"], ends
        )
        assert float(labels["threshold"]) == protocol["conservation_threshold"]
        assert str(labels["input_manifest_sha256"]) == sha256(
            root / "data/manifest.json"
        )
        values.update(
            {
                key: labels[key]
                for key in ["conserved_bases", "label_covered_bases", "mean_phylop"]
            }
        )
    else:
        labels = window_labels(
            root / "data/phyloP_447m.bw",
            chrom,
            starts,
            ends,
            protocol["conservation_threshold"],
            Path(protocol["exclude_lowercase_twobit"])
            if "exclude_lowercase_twobit" in protocol
            else None,
        )
        np.savez_compressed(
            cache,
            starts=starts,
            ends=ends,
            threshold=protocol["conservation_threshold"],
            mask_policy=mask_policy,
            input_manifest_sha256=sha256(root / "data/manifest.json"),
            **labels,
        )
        values.update(labels)
    values["label"] = values["conserved_bases"] / (ends - starts)
    values["label_coverage"] = values["label_covered_bases"] / (ends - starts)
    raw_strata = np.stack(
        [
            np.floor(values[field] / step).astype(int)
            for field, step in [("gc", 0.05), ("repeat", 0.1), ("entropy", 0.1)]
        ],
        axis=1,
    )
    _, values["strata"] = np.unique(raw_strata, axis=0, return_inverse=True)
    values["blocks"] = starts // protocol["bootstrap_block_bases"]
    values["tie"] = np.array(
        [
            int.from_bytes(
                hashlib.sha256(f"577:{chrom}:{start}".encode()).digest()[:8], "big"
            )
            for start in starts
        ],
        dtype=np.uint64,
    )
    return values


def metrics(
    values: dict[str, np.ndarray],
    score: np.ndarray,
    fraction: float,
    weights: np.ndarray | None = None,
    order: np.ndarray | None = None,
) -> dict:
    unweighted = weights is None
    if weights is None:
        weights = np.ones(len(score), dtype=np.float64)
    assert len(score) == len(weights) and np.all(np.isfinite(score))
    budget = max(1, int(np.floor(fraction * weights.sum())))
    selected = np.zeros(len(score), dtype=np.float64)
    if unweighted:
        selected[select_indices(score, values["tie"], budget)] = 1
    else:
        if order is None:
            order = np.lexsort((values["tie"], -score))
        remaining = np.clip(budget - np.r_[0, np.cumsum(weights[order])[:-1]], 0, None)
        selected[order] = np.minimum(weights[order], remaining)
    assert np.isclose(selected.sum(), budget)
    y, strata = values["label"], values["strata"]
    denominators = np.bincount(strata, weights=weights)
    numerator = np.bincount(strata, weights=weights * y)
    bin_mean = np.divide(
        numerator, denominators, out=np.zeros_like(numerator), where=denominators > 0
    )
    mean = float(np.dot(selected, y) / budget)
    matched = float(np.dot(selected, bin_mean[strata]) / budget)
    baseline = float(np.dot(weights, y) / weights.sum())
    return {
        "windows": int(weights.sum()),
        "selected_windows": budget,
        "selected_bases": budget * int(values["end"][0] - values["start"][0]),
        "selected_annotated_fraction": mean,
        "random_annotated_fraction": baseline,
        "matched_annotated_fraction": matched,
        "random_enrichment": mean / baseline if baseline else None,
        "matched_enrichment": mean / matched if matched else None,
        "annotated_base_recall": float(np.dot(selected, y) / np.dot(weights, y))
        if np.dot(weights, y)
        else None,
        "selected_gc": float(np.dot(selected, values["gc"]) / budget),
        "selected_repeat": float(np.dot(selected, values["repeat"]) / budget),
        "selected_entropy": float(np.dot(selected, values["entropy"]) / budget),
        "selected_seeds": float(np.dot(selected, values["seeds"]) / budget),
        "selected_label_coverage": float(
            np.dot(selected, values.get("label_coverage", np.ones(len(score)))) / budget
        ),
        "population_label_coverage": float(
            np.dot(weights, values.get("label_coverage", np.ones(len(score))))
            / weights.sum()
        ),
        "selected_positive_window_fraction": float(np.dot(selected, y >= 0.2) / budget),
        "population_positive_window_fraction": float(
            np.dot(weights, y >= 0.2) / weights.sum()
        ),
    }


def intervals(
    values: dict[str, np.ndarray], score: np.ndarray, fraction: float, repeats: int
) -> dict:
    unique, inverse = np.unique(values["blocks"], return_inverse=True)
    rng = np.random.default_rng(577)
    draws: dict[str, list[float]] = {
        "selected_annotated_fraction": [],
        "random_enrichment": [],
        "matched_enrichment": [],
        "annotated_base_recall": [],
    }
    order = np.lexsort((values["tie"], -score))
    for _ in range(repeats):
        sampled = rng.integers(0, len(unique), size=len(unique))
        weights = np.bincount(sampled, minlength=len(unique))[inverse].astype(float)
        result = metrics(values, score, fraction, weights, order)
        for key, samples in draws.items():
            if result[key] is not None:
                samples.append(result[key])
    return {
        key: np.quantile(value, [0.025, 0.975]).tolist() for key, value in draws.items()
    }


def select_indices(score: np.ndarray, tie: np.ndarray, count: int) -> np.ndarray:
    """Fixed-budget selection without a global sort; return genomic-order indices."""
    assert 1 <= count <= len(score)
    threshold = np.partition(score, len(score) - count)[len(score) - count]
    chosen = score > threshold
    remaining = count - int(chosen.sum())
    candidates = np.flatnonzero(score == threshold)
    picks = np.argpartition(tie[candidates], remaining - 1)[:remaining]
    chosen[candidates[picks]] = True
    return np.flatnonzero(chosen)


def write_stretches(
    path: Path,
    chrom: str,
    values: dict[str, np.ndarray],
    score: np.ndarray,
    fraction: float,
) -> None:
    """Merge only directly adjacent selected bins; retain 0-based half-open bounds."""
    count = max(1, int(np.floor(fraction * len(score))))
    chosen = select_indices(score, values["tie"], count)
    with path.open("w") as handle:
        start, end, total, bins, ordinal = 0, 0, 0.0, 0, 0
        for i in chosen:
            left, right = int(values["start"][i]), int(values["end"][i])
            if bins and left != end:
                ordinal += 1
                handle.write(
                    f"{chrom}\t{start}\t{end}\tcandidate_{ordinal}\t{round(1000 * total / bins)}\t.\t{total / bins:.8g}\t{bins}\n"
                )
                bins = 0
            if not bins:
                start = left
                total = 0.0
            end = right
            total += float(score[i])
            bins += 1
        if bins:
            ordinal += 1
            handle.write(
                f"{chrom}\t{start}\t{end}\tcandidate_{ordinal}\t{round(1000 * total / bins)}\t.\t{total / bins:.8g}\t{bins}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--freeze-sha")
    args = parser.parse_args()
    root = args.root
    report = root / "report"
    report.mkdir(exist_ok=True)
    protocol = json.loads(Path("config/protocol.json").read_text())
    if args.split == "dev":
        matrix = []
        for k in protocol["k"]:
            values = load_chromosome(
                root, k, protocol["development_chromosome"], protocol
            )
            for field in protocol["scores"]:
                result = metrics(
                    values, values[field], protocol["primary_selected_fraction"]
                )
                matrix.append({"k": k, "score": field, **result})
        chosen = min(
            matrix,
            key=lambda x: (
                -x["matched_enrichment"],
                -x["random_enrichment"],
                x["k"],
                x["score"],
            ),
        )
        (report / "development.json").write_text(json.dumps(matrix, indent=2) + "\n")
        selection = {
            "k": chosen["k"],
            "score": chosen["score"],
            "protocol_sha256": sha256(Path("config/protocol.json")),
            "development": chosen,
            "input_manifest_sha256": sha256(root / "data/manifest.json"),
            "density_alternative": min(
                matrix,
                key=lambda x: (-x["selected_annotated_fraction"], x["k"], x["score"]),
            ),
        }
        (report / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        print(json.dumps(selection, indent=2))
    else:
        assert args.freeze_sha and len(args.freeze_sha) == 40
        assert not (report / "heldout.json").exists(), "held-out report already exists"
        selection = json.loads((report / "selection.json").read_text())
        assert selection["protocol_sha256"] == sha256(Path("config/protocol.json"))
        assert selection["input_manifest_sha256"] == sha256(root / "data/manifest.json")
        values = load_chromosome(
            root, selection["k"], protocol["heldout_chromosome"], protocol
        )
        output: dict = {
            "freeze_sha": args.freeze_sha,
            "selection": selection,
            "budgets": {},
        }
        for fraction in [
            protocol["primary_selected_fraction"],
            *protocol["secondary_selected_fractions"],
        ]:
            output["budgets"][str(fraction)] = metrics(
                values, values[selection["score"]], fraction
            )
        for fraction in [
            protocol["primary_selected_fraction"],
            *protocol["secondary_selected_fractions"],
        ]:
            write_stretches(
                report / f"stretches-{fraction}.bed",
                protocol["heldout_chromosome"],
                values,
                values[selection["score"]],
                fraction,
            )
        output["primary_ci"] = intervals(
            values,
            values[selection["score"]],
            protocol["primary_selected_fraction"],
            protocol["bootstrap_replicates"],
        )
        output["simple_baselines"] = {
            field: metrics(values, score, protocol["primary_selected_fraction"])
            for field, score in {
                "low_repeat": -values["repeat"],
                "high_entropy": values["entropy"],
                "high_gc": values["gc"],
            }.items()
        }
        alternative = selection["density_alternative"]
        alternate_values = (
            values
            if alternative["k"] == selection["k"]
            else load_chromosome(
                root, alternative["k"], protocol["heldout_chromosome"], protocol
            )
        )
        output["density_alternative"] = {"selection": alternative, "budgets": {}}
        for fraction in [
            protocol["primary_selected_fraction"],
            *protocol["secondary_selected_fractions"],
        ]:
            output["density_alternative"]["budgets"][str(fraction)] = metrics(
                alternate_values, alternate_values[alternative["score"]], fraction
            )
            write_stretches(
                report / f"stretches-density-{fraction}.bed",
                protocol["heldout_chromosome"],
                alternate_values,
                alternate_values[alternative["score"]],
                fraction,
            )
        output["density_alternative"]["primary_ci"] = intervals(
            alternate_values,
            alternate_values[alternative["score"]],
            protocol["primary_selected_fraction"],
            protocol["bootstrap_replicates"],
        )
        primary = output["budgets"][str(protocol["primary_selected_fraction"])]
        gate = protocol["advance_gate"]
        output["biological_gate_passed"] = bool(
            primary["random_enrichment"] >= gate["random_enrichment"]
            and primary["matched_enrichment"] >= gate["matched_enrichment"]
            and output["primary_ci"]["matched_enrichment"][0]
            > gate["matched_bootstrap_lower"]
        )
        (report / "heldout.json").write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
