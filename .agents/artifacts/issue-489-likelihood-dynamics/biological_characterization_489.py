"""Characterize globally fitted issue #489 trajectory groups biologically."""

from __future__ import annotations

import gc as garbage_collector
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

import preview_global_threshold_489 as trajectory


OUTPUT_ROOT = Path(__file__).resolve().parent / "biology"
METADATA_ROOT = Path("/tmp/ld489-preview.MRb9Kt/metadata")
GROUP_ORDER = trajectory.GROUP_ORDER
REGIONS = trajectory.REGIONS
BASES = ("A", "C", "G", "T")
N_SAMPLES_PER_CELL = 3
BOOTSTRAP_REPLICATES = 2_000
SEED = 489_10


def base_codes(table: pa.Table, mask: np.ndarray) -> np.ndarray:
    """Return A/C/G/T codes without materializing Python string arrays."""
    encoded = pc.dictionary_encode(table["base"].combine_chunks())
    lookup = np.array([BASES.index(str(value)) for value in encoded.dictionary.to_pylist()])
    indices = np.asarray(encoded.indices.to_numpy(zero_copy_only=False), dtype=np.int8)
    return lookup[indices][mask]


def repeat_context(region: str) -> tuple[np.ndarray, np.ndarray]:
    """Return per-window repeat fraction and nearest-repeat distance by position."""
    table = pq.read_table(
        METADATA_ROOT / f"{region}.parquet",
        columns=["row_index", "is_repeat"],
    )
    row_index = np.asarray(
        table["row_index"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int32,
    )
    assert np.array_equal(row_index, np.arange(len(row_index), dtype=np.int32))
    repeat_array = table["is_repeat"].combine_chunks()
    is_repeat = np.asarray(
        repeat_array.values.to_numpy(zero_copy_only=False), dtype=bool
    ).reshape(len(row_index), 255)
    repeat_fraction = is_repeat.mean(axis=1)
    distance = np.full(is_repeat.shape, 255, dtype=np.uint16)
    running = np.full(len(row_index), 255, dtype=np.uint16)
    for position in range(is_repeat.shape[1]):
        running = np.where(is_repeat[:, position], 0, np.minimum(running + 1, 255))
        distance[:, position] = running
    running.fill(255)
    for position in range(is_repeat.shape[1] - 1, -1, -1):
        running = np.where(is_repeat[:, position], 0, np.minimum(running + 1, 255))
        distance[:, position] = np.minimum(distance[:, position], running)
    return repeat_fraction, distance


def update_samples(
    candidates: dict[tuple[int, int], list[tuple[float, dict[str, object]]]],
    *,
    region_index: int,
    group: np.ndarray,
    priorities: np.ndarray,
    arrays: dict[str, np.ndarray],
    losses: np.ndarray,
) -> None:
    """Retain the lowest-priority reproducible random samples per cell."""
    for group_index in range(len(GROUP_ORDER)):
        indices = np.flatnonzero(group == group_index)
        if not len(indices):
            continue
        keep = min(N_SAMPLES_PER_CELL, len(indices))
        selected = indices[np.argpartition(priorities[indices], keep - 1)[:keep]]
        cell = candidates.setdefault((region_index, group_index), [])
        for index in selected:
            row = {
                name: value[index].item() if hasattr(value[index], "item") else value[index]
                for name, value in arrays.items()
            }
            for checkpoint_index, tokens in enumerate(trajectory.TOKENS):
                row[f"nll_{int(tokens)}"] = float(losses[checkpoint_index, index])
            cell.append((float(priorities[index]), row))
        cell.sort(key=lambda item: item[0])
        del cell[N_SAMPLES_PER_CELL:]


def bootstrap_intervals(
    block_counts: np.ndarray,
    block_conserved: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap group frequency and conservation prevalence over 10 Mb blocks."""
    frequency_low = np.full((len(REGIONS), len(GROUP_ORDER)), np.nan)
    frequency_high = np.full_like(frequency_low, np.nan)
    conserved_low = np.full_like(frequency_low, np.nan)
    conserved_high = np.full_like(frequency_low, np.nan)
    rng = np.random.default_rng(SEED)
    for region_index in range(len(REGIONS)):
        block_slice = slice(region_index * 10_000, (region_index + 1) * 10_000)
        cell_counts = block_counts[:, block_slice]
        cell_conserved = block_conserved[:, block_slice]
        totals = cell_counts.sum(axis=0)
        active = totals > 0
        totals = totals[active]
        cell_counts = cell_counts[:, active]
        cell_conserved = cell_conserved[:, active]
        assert 0 < len(totals) < 5_000
        draws = rng.integers(0, len(totals), size=(BOOTSTRAP_REPLICATES, len(totals)))
        denominators = totals[draws].sum(axis=1)
        for group_index in range(len(GROUP_ORDER)):
            counts = cell_counts[group_index]
            group_draw_counts = counts[draws].sum(axis=1)
            frequency = group_draw_counts / denominators
            active_group = counts > 0
            conserved = cell_conserved[group_index, active_group]
            conserved_counts = counts[active_group]
            conserved_draws = rng.integers(
                0,
                len(conserved_counts),
                size=(BOOTSTRAP_REPLICATES, len(conserved_counts)),
            )
            prevalence = conserved[conserved_draws].sum(axis=1) / conserved_counts[
                conserved_draws
            ].sum(axis=1)
            frequency_low[region_index, group_index], frequency_high[
                region_index, group_index
            ] = np.quantile(frequency, (0.025, 0.975))
            conserved_low[region_index, group_index], conserved_high[
                region_index, group_index
            ] = np.quantile(prevalence, (0.025, 0.975))
    return frequency_low, frequency_high, conserved_low, conserved_high


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    s3 = pafs.S3FileSystem(region="us-east-2")
    early_mean, terminal_mean, checkpoint_means, population = (
        trajectory.global_fitted_thresholds(s3)
    )
    n_groups = len(GROUP_ORDER)
    n_regions = len(REGIONS)
    counts = np.zeros((n_regions, n_groups), dtype=np.int64)
    conserved_counts = np.zeros_like(counts)
    base_counts = np.zeros((n_regions, n_groups, len(BASES)), dtype=np.int64)
    gc_sums = np.zeros((n_regions, n_groups), dtype=np.float64)
    kmer_sums = np.zeros_like(gc_sums)
    repeat_fraction_sums = np.zeros_like(gc_sums)
    repeat_within_10_counts = np.zeros_like(counts)
    repeat_within_50_counts = np.zeros_like(counts)
    chromosome_counts = np.zeros((n_regions, n_groups, 26), dtype=np.int64)
    block_counts = np.zeros((n_groups, trajectory.N_BLOCK_KEYS), dtype=np.int64)
    block_conserved = np.zeros_like(block_counts)
    candidates: dict[tuple[int, int], list[tuple[float, dict[str, object]]]] = {}
    rng = np.random.default_rng(SEED)

    first_columns = (
        "row_index",
        "chrom",
        "target_pos",
        "genomic_pos",
        "base",
        "is_conserved",
        "is_scorable",
        "is_ambiguous",
        "is_repeat",
        "window_gc",
        "kmer7_nll",
        "nll",
    )
    for region_index, region in enumerate(REGIONS):
        window_repeat_fraction, repeat_distance = repeat_context(region)
        files = [
            pq.ParquetFile(s3.open_input_file(trajectory.atom_path(checkpoint, region)))
            for checkpoint in trajectory.CHECKPOINTS
        ]
        assert len({file.num_row_groups for file in files}) == 1
        for row_group in range(files[0].num_row_groups):
            first = files[0].read_row_group(row_group, columns=first_columns)
            tables = [first]
            tables.extend(
                file.read_row_group(row_group, columns=("nll",)) for file in files[1:]
            )
            mask = trajectory.primary_mask(first)
            losses = np.stack(
                [trajectory.as_numpy(table, "nll", np.float32)[mask] for table in tables]
            )
            fitted_start, fitted_end = trajectory.fitted_endpoints(losses)
            early_low = fitted_start <= early_mean
            terminal_low = fitted_end <= terminal_mean
            group = np.zeros(int(mask.sum()), dtype=np.int8)
            group[early_low & ~terminal_low] = 1
            group[~early_low & terminal_low] = 2
            group[early_low & terminal_low] = 3

            conserved = trajectory.as_numpy(first, "is_conserved", bool)[mask]
            window_gc = trajectory.as_numpy(first, "window_gc", np.float32)[mask]
            kmer = trajectory.as_numpy(first, "kmer7_nll", np.float32)[mask]
            bases = base_codes(first, mask)
            chrom = trajectory.chromosome_codes(first)[mask].astype(np.int64)
            genomic_pos = trajectory.as_numpy(first, "genomic_pos", np.int64)[mask]
            row_index = trajectory.as_numpy(first, "row_index", np.int32)[mask]
            target_pos = trajectory.as_numpy(first, "target_pos", np.int16)[mask]
            window_repeat = window_repeat_fraction[row_index]
            nearest_repeat = repeat_distance[row_index, target_pos]
            genomic_bin = genomic_pos // trajectory.BLOCK_BP
            block = region_index * 10_000 + chrom * 100 + genomic_bin
            assert int(block.max()) < trajectory.N_BLOCK_KEYS

            for group_index in range(n_groups):
                selected = group == group_index
                counts[region_index, group_index] += int(selected.sum())
                conserved_counts[region_index, group_index] += int(conserved[selected].sum())
                gc_sums[region_index, group_index] += window_gc[selected].sum(
                    dtype=np.float64
                )
                kmer_sums[region_index, group_index] += kmer[selected].sum(dtype=np.float64)
                repeat_fraction_sums[region_index, group_index] += window_repeat[
                    selected
                ].sum(dtype=np.float64)
                repeat_within_10_counts[region_index, group_index] += int(
                    (nearest_repeat[selected] <= 10).sum()
                )
                repeat_within_50_counts[region_index, group_index] += int(
                    (nearest_repeat[selected] <= 50).sum()
                )
                block_counts[group_index] += np.bincount(
                    block[selected], minlength=trajectory.N_BLOCK_KEYS
                )
                block_conserved[group_index] += np.bincount(
                    block[selected],
                    weights=conserved[selected],
                    minlength=trajectory.N_BLOCK_KEYS,
                ).astype(np.int64)
                chromosome_counts[region_index, group_index] += np.bincount(
                    chrom[selected], minlength=26
                )
                base_counts[region_index, group_index] += np.bincount(
                    bases[selected], minlength=len(BASES)
                )

            arrays = {
                "row_index": row_index,
                "chromosome_code": chrom,
                "window_start": genomic_pos - target_pos,
                "window_end": genomic_pos - target_pos + 255,
                "target_pos": target_pos,
                "genomic_pos": genomic_pos,
                "base_code": bases,
                "is_conserved": conserved,
                "window_gc": window_gc,
                "kmer7_nll": kmer,
                "window_repeat_fraction": window_repeat,
                "nearest_annotated_repeat_bp": nearest_repeat,
            }
            update_samples(
                candidates,
                region_index=region_index,
                group=group,
                priorities=rng.random(len(group)),
                arrays=arrays,
                losses=losses,
            )
            del first, tables, mask, losses, fitted_start, fitted_end
            del early_low, terminal_low, group, conserved, window_gc, kmer, bases
            del chrom, genomic_pos, row_index, target_pos, genomic_bin, block
            del window_repeat, nearest_repeat, selected, arrays
            garbage_collector.collect()
            pa.default_memory_pool().release_unused()
        del window_repeat_fraction, repeat_distance

    assert int(counts.sum()) == population
    frequency_low, frequency_high, conserved_low, conserved_high = bootstrap_intervals(
        block_counts, block_conserved
    )
    region_totals = counts.sum(axis=1)
    group_totals = counts.sum(axis=0)
    global_region_fraction = region_totals / population
    rows: list[dict[str, object]] = []
    for region_index, region in enumerate(REGIONS):
        for group_index, group_name in enumerate(GROUP_ORDER):
            count = int(counts[region_index, group_index])
            rows.append(
                {
                    "region": region,
                    "group": group_name,
                    "n_positions": count,
                    "group_frequency_within_region": count / region_totals[region_index],
                    "group_frequency_ci_low": frequency_low[region_index, group_index],
                    "group_frequency_ci_high": frequency_high[region_index, group_index],
                    "region_fraction_within_group": count / group_totals[group_index],
                    "region_enrichment_log2": np.log2(
                        (count / group_totals[group_index]) / global_region_fraction[region_index]
                    ),
                    "conservation_prevalence": conserved_counts[region_index, group_index]
                    / count,
                    "conservation_ci_low": conserved_low[region_index, group_index],
                    "conservation_ci_high": conserved_high[region_index, group_index],
                    "mean_window_gc": gc_sums[region_index, group_index] / count,
                    "mean_kmer7_nll": kmer_sums[region_index, group_index] / count,
                    "mean_window_repeat_fraction": repeat_fraction_sums[
                        region_index, group_index
                    ]
                    / count,
                    "fraction_with_repeat_within_10bp": repeat_within_10_counts[
                        region_index, group_index
                    ]
                    / count,
                    "fraction_with_repeat_within_50bp": repeat_within_50_counts[
                        region_index, group_index
                    ]
                    / count,
                }
            )
    pd.DataFrame(rows).to_parquet(OUTPUT_ROOT / "region_group_statistics.parquet", index=False)
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "region_group_statistics.csv", index=False)

    base_rows = []
    for region_index, region in enumerate(REGIONS):
        for group_index, group_name in enumerate(GROUP_ORDER):
            for base_index, base in enumerate(BASES):
                base_rows.append(
                    {
                        "region": region,
                        "group": group_name,
                        "base": base,
                        "n_positions": int(base_counts[region_index, group_index, base_index]),
                        "frequency": base_counts[region_index, group_index, base_index]
                        / counts[region_index, group_index],
                    }
                )
    pd.DataFrame(base_rows).to_csv(OUTPUT_ROOT / "base_composition.csv", index=False)

    chromosome_rows = []
    for region_index, region in enumerate(REGIONS):
        for group_index, group_name in enumerate(GROUP_ORDER):
            for chromosome in range(1, 26):
                chromosome_rows.append(
                    {
                        "region": region,
                        "group": group_name,
                        "chromosome_code": chromosome,
                        "n_positions": int(
                            chromosome_counts[region_index, group_index, chromosome]
                        ),
                    }
                )
    pd.DataFrame(chromosome_rows).to_csv(OUTPUT_ROOT / "chromosome_counts.csv", index=False)

    sample_rows = []
    for (region_index, group_index), values in sorted(candidates.items()):
        for priority, row in values:
            row["base"] = BASES[int(row.pop("base_code"))]
            sample_rows.append(
                {
                    "region": REGIONS[region_index],
                    "group": GROUP_ORDER[group_index],
                    "sample_priority": priority,
                    **row,
                }
            )
    pd.DataFrame(sample_rows).to_parquet(OUTPUT_ROOT / "trajectory_group_samples.parquet", index=False)
    pd.DataFrame(sample_rows).to_csv(OUTPUT_ROOT / "trajectory_group_samples.csv", index=False)

    manifest = {
        "population": "target_pos [32, 223), scorable, nonambiguous, nonrepeat",
        "n_positions": population,
        "group_definition": (
            "OLS fitted first and terminal NLL, thresholded by the corresponding "
            "global fitted mean over cumulative training tokens"
        ),
        "early_global_fitted_mean_nll": early_mean,
        "terminal_global_fitted_mean_nll": terminal_mean,
        "global_checkpoint_mean_nll": checkpoint_means.tolist(),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": SEED,
            "unit": "region-specific genomic 10 Mb block",
        },
        "sample": {
            "method": "lowest independent pseudorandom priorities per region/group cell",
            "per_cell": N_SAMPLES_PER_CELL,
            "seed": SEED,
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
