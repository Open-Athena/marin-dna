"""Descriptive annotation overlap for fixed selections, without retuning."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from window_conservation.evaluate import load_chromosome, select_indices


def merge(intervals: list[tuple[int, int]], length: int) -> np.ndarray:
    """Return a disjoint, bounded union in zero-based half-open coordinates."""
    result: list[list[int]] = []
    for start, end in sorted(intervals):
        assert 0 <= start <= end <= length
        if start == end:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return np.asarray(result, dtype=np.int64).reshape(-1, 2)


def coverage(intervals: np.ndarray, mask: np.ndarray, width: int = 100) -> int:
    """Count union bases inside selected grid bins, including partial overlaps."""
    prefix = np.r_[0, np.cumsum(mask, dtype=np.int64) * width]
    padded = np.r_[mask, False]
    points = np.minimum(intervals, len(mask) * width)
    integral = prefix[points // width] + (points % width) * padded[points // width]
    return int(np.sum(integral[:, 1] - integral[:, 0]))


def fractions(intervals: np.ndarray, bins: int, width: int = 100) -> np.ndarray:
    """Exact union overlap fraction for each full grid bin."""
    result = np.zeros(bins + 1, dtype=np.float64)
    delta = np.zeros(bins + 1, dtype=np.int32)
    for start, end in intervals:
        start, end = min(int(start), bins * width), min(int(end), bins * width)
        left, right = start // width, end // width
        if left == right:
            result[left] += end - start
        else:
            result[left] += width - start % width
            result[right] += end % width
            delta[left + 1] += 1
            delta[right] -= 1
    result += np.cumsum(delta) * width
    assert np.all(result >= 0) and np.all(result <= width)
    return result[:bins] / width


def annotations(data: Path, chrom: str, length: int) -> tuple[dict, dict]:
    features: dict[str, list[tuple[int, int]]] = defaultdict(list)
    families: dict[str, list[tuple[int, int]]] = defaultdict(list)
    bare = chrom.removeprefix("chr")
    with gzip.open(data / "Homo_sapiens.GRCh38.115.gtf.gz", "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if fields[0] != bare:
                continue
            start, end = int(fields[3]) - 1, int(fields[4])
            assert 0 <= start < end <= length
            feature = fields[2]
            if feature in {"CDS", "exon", "five_prime_utr", "three_prime_utr", "gene"}:
                features["gene_body" if feature == "gene" else feature].append(
                    (start, end)
                )
            if feature == "transcript":
                tss = start if fields[6] == "+" else end - 1
                features["TSS_plus_minus_1kb"].append(
                    (max(0, tss - 1000), min(length, tss + 1001))
                )
    table = pq.read_table(
        data / "ccre.bare.parquet", filters=[("chrom", "=", bare)], use_threads=False
    )
    assert features["CDS"] and table.num_rows > 0, "annotation chromosome mismatch"
    cols = table.to_pydict()
    for start, end, category in zip(
        cols["start"], cols["end"], cols["cre_class"], strict=True
    ):
        features["cCRE_" + category].append((int(start), int(end)))
    with gzip.open(data / "rmsk.txt.gz", "rt") as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if fields[5] != chrom:
                continue
            start, end = int(fields[6]), int(fields[7])
            features["repeat_" + fields[11]].append((start, end))
            families[fields[11] + ":" + fields[12]].append((start, end))
    return (
        {name: merge(rows, length) for name, rows in features.items()},
        {name: merge(rows, length) for name, rows in families.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=["pilot", "extension"], required=True)
    args = parser.parse_args()
    root = args.root
    extension = root / "extension"
    protocol = json.loads(Path("config/protocol.json").read_text())
    if args.split == "pilot":
        chrom, source = "chr2", root
        choices = [
            {"name": "primary", "score": "any", "path": root / "scores/k25-human.tsv"},
            {
                "name": "copy_filtered",
                "score": "any_copy4",
                "path": root / "scores/k25-human.tsv",
            },
        ]
    else:
        chrom, source = "chr3", extension
        assert (extension / "report/validation.json").exists()
        selection = json.loads((extension / "report/selection.json").read_text())
        choices = [
            {"name": name, **selection[name]} for name in ["primary", "baseline"]
        ]
    lengths = json.loads((root / "data/manifest.json").read_text())["sources"][0][
        "chromosomes"
    ]
    length = lengths[chrom]
    features, families = annotations(extension / "data", chrom, length)
    output = {
        "chromosome": chrom,
        "coordinate_mapping": "GTF and cCRE bare 1/2/3 map explicitly to matching hg38 chr1/chr2/chr3; GTF start-1.",
        "feature_overlap": [],
        "family_overlap": [],
        "within_feature_bins": [],
        "caveat": "Feature and repeat categories overlap; coverage rows are not an exclusive partition. Within-feature conservation uses whole bins with at least 50% feature coverage, not only annotated sub-bases. Exploratory diagnostics do not tune selection.",
    }
    for choice in choices:
        values = load_chromosome(source, 25, chrom, protocol, Path(choice["path"]))
        ids = (values["start"] // 100).astype(np.int64)
        eligible = np.zeros(length // 100, dtype=bool)
        eligible[ids] = True
        for budget in [0.01, 0.05]:
            selected_ids = select_indices(
                values[choice["score"]], values["tie"], int(len(ids) * budget)
            )
            selected = np.zeros_like(eligible)
            selected[ids[selected_ids]] = True
            selected_eligible = np.zeros(len(ids), dtype=bool)
            selected_eligible[selected_ids] = True
            for group, destination in [
                (features, "feature_overlap"),
                (families, "family_overlap"),
            ]:
                for name, rows in group.items():
                    pop = coverage(rows, eligible)
                    chosen = coverage(rows, selected)
                    result = {
                        "choice": choice["name"],
                        "budget": budget,
                        "feature": name,
                        "eligible_overlap_bases": pop,
                        "selected_overlap_bases": chosen,
                        "eligible_fraction": pop / (100 * eligible.sum()),
                        "selected_fraction": chosen / (100 * selected.sum()),
                    }
                    result["coverage_enrichment"] = (
                        result["selected_fraction"] / result["eligible_fraction"]
                        if pop
                        else None
                    )
                    output[destination].append(result)
                    if destination == "feature_overlap" and budget == 0.05:
                        mask = fractions(rows, len(eligible))[ids] >= 0.5
                        hit = mask & selected_eligible
                        output["within_feature_bins"].append(
                            {
                                "choice": choice["name"],
                                "feature": name,
                                "eligible_bins": int(mask.sum()),
                                "selected_bins": int(hit.sum()),
                                "eligible_conserved_fraction": float(
                                    values["label"][mask].mean()
                                )
                                if mask.any()
                                else None,
                                "selected_conserved_fraction": float(
                                    values["label"][hit].mean()
                                )
                                if hit.any()
                                else None,
                            }
                        )
        print("annotated", chrom, choice["name"], flush=True)
    report = extension / "report"
    report.mkdir(exist_ok=True)
    (report / f"biology-{chrom}.json").write_text(json.dumps(output, indent=2) + "\n")
    for name in ["feature_overlap", "family_overlap", "within_feature_bins"]:
        with (report / f"biology-{chrom}-{name}.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output[name][0]))
            writer.writeheader()
            writer.writerows(output[name])


if __name__ == "__main__":
    main()
