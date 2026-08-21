"""Stream issue #489 atoms and preview globally thresholded loss trajectories."""

from __future__ import annotations

import json
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import seaborn as sns


S3_ROOT = (
    "oa-bolinas/snakemake/analysis/evals_v2/results/"
    "m13_likelihood_dynamics_489/v1/scoring/full/atoms"
)
CHECKPOINTS = (
    "mix-v0.9-p1B-i20-exp135-zoonomia-m1-step-10000",
    "mix-v0.9-p1B-i26-exp135-zoonomia-m1.1-step-30000",
    "mix-v0.9-p1B-i28-exp135-zoonomia-m1.2-step-50000",
    "mix-v0.9-p1B-i30-exp135-zoonomia-m1.3-step-70000",
    "mix-v0.9-p1B-i30-exp135-zoonomia-m1.3-step-82823",
)
TOKENS = np.array((20_971_520_000, 62_914_560_000, 104_857_600_000,
                   146_800_640_000, 173_691_518_976), dtype=np.int64)
REGIONS = ("cds", "upstream", "downstream", "ncrna", "enhancer")
PRIMARY_START = 32
PRIMARY_END = 223
BLOCK_BP = 10_000_000
N_BLOCK_KEYS = 50_000
BOOTSTRAP_REPLICATES = 2_000
SEED = 489
OUTPUT_ROOT = Path("/tmp/ld489-preview.MRb9Kt")
GROUP_ORDER = ("high_to_high", "low_to_high", "high_to_low", "low_to_low")
GROUP_NAMES = {
    "high_to_high": "H\N{RIGHTWARDS ARROW}H",
    "low_to_high": "L\N{RIGHTWARDS ARROW}H",
    "high_to_low": "H\N{RIGHTWARDS ARROW}L",
    "low_to_low": "L\N{RIGHTWARDS ARROW}L",
}
GROUP_COLORS = {
    "high_to_high": "#ff7f00",
    "low_to_high": "#e41a1c",
    "high_to_low": "#377eb8",
    "low_to_low": "#4daf4a",
}


def atom_path(checkpoint: str, region: str) -> str:
    return f"{S3_ROOT}/{checkpoint}/{region}.parquet"


def as_numpy(table: object, column: str, dtype: object) -> np.ndarray:
    array = table[column].combine_chunks()
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=dtype)


def primary_mask(table: object) -> np.ndarray:
    target_pos = as_numpy(table, "target_pos", np.int16)
    return (
        (target_pos >= PRIMARY_START)
        & (target_pos < PRIMARY_END)
        & as_numpy(table, "is_scorable", bool)
        & ~as_numpy(table, "is_ambiguous", bool)
        & ~as_numpy(table, "is_repeat", bool)
    )


def chromosome_codes(table: object) -> np.ndarray:
    encoded = pc.dictionary_encode(table["chrom"].combine_chunks())
    dictionary = encoded.dictionary.to_pylist()

    def chromosome_code(value: object) -> int:
        label = str(value)
        if label.isdigit():
            return int(label)
        if label in {"X", "Y", "MT"}:
            return {"X": 23, "Y": 24, "MT": 25}[label]
        accession = int(label.removeprefix("NC_").split(".", maxsplit=1)[0])
        if 1 <= accession <= 24:
            return accession
        if accession == 12_920:
            return 25
        raise ValueError(f"Unsupported primary-assembly sequence name: {label}")

    lookup = np.array(
        [chromosome_code(value) for value in dictionary],
        dtype=np.int16,
    )
    return lookup[np.asarray(encoded.indices.to_numpy(zero_copy_only=False))]


def fitted_endpoints(losses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return fitted losses at the observed first and terminal checkpoints."""
    x = TOKENS.astype(np.float64)
    centered_x = x - x.mean()
    slopes = centered_x @ losses / np.square(centered_x).sum()
    mean_loss = losses.mean(axis=0)
    return (
        mean_loss + slopes * (x[0] - x.mean()),
        mean_loss + slopes * (x[-1] - x.mean()),
    )


def global_fitted_thresholds(
    s3: pafs.S3FileSystem,
) -> tuple[float, float, np.ndarray, int]:
    sums = np.zeros(len(CHECKPOINTS), dtype=np.float64)
    count = 0
    for region in REGIONS:
        files = [pq.ParquetFile(s3.open_input_file(atom_path(checkpoint, region)))
                 for checkpoint in CHECKPOINTS]
        assert len({file.num_row_groups for file in files}) == 1
        for row_group in range(files[0].num_row_groups):
            first = files[0].read_row_group(
                row_group,
                columns=("target_pos", "is_scorable", "is_ambiguous", "is_repeat", "nll"),
            )
            mask = primary_mask(first)
            sums[0] += as_numpy(first, "nll", np.float32)[mask].sum(dtype=np.float64)
            for checkpoint_index, file in enumerate(files[1:], start=1):
                table = file.read_row_group(row_group, columns=("nll",))
                sums[checkpoint_index] += as_numpy(table, "nll", np.float32)[mask].sum(
                    dtype=np.float64
                )
            count += int(mask.sum())
            del first, table, mask
            gc.collect()
            pa.default_memory_pool().release_unused()
    assert count > 0
    checkpoint_means = sums / count
    fitted_start, fitted_end = fitted_endpoints(checkpoint_means[:, None])
    return float(fitted_start[0]), float(fitted_end[0]), checkpoint_means, count


def stream_summaries(
    s3: pafs.S3FileSystem,
    early_mean: float,
    terminal_mean: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sums = np.zeros((4, len(CHECKPOINTS)), dtype=np.float64)
    counts = np.zeros(4, dtype=np.int64)
    block_sums = np.zeros((4, len(CHECKPOINTS), N_BLOCK_KEYS), dtype=np.float64)
    block_counts = np.zeros((4, N_BLOCK_KEYS), dtype=np.int64)
    for region_index, region in enumerate(REGIONS):
        files = [pq.ParquetFile(s3.open_input_file(atom_path(checkpoint, region)))
                 for checkpoint in CHECKPOINTS]
        assert len({file.num_row_groups for file in files}) == 1
        for row_group in range(files[0].num_row_groups):
            first = files[0].read_row_group(
                row_group,
                columns=("target_pos", "is_scorable", "is_ambiguous",
                         "is_repeat", "chrom", "genomic_pos", "nll"),
            )
            tables = [first]
            tables.extend(
                file.read_row_group(row_group, columns=("nll",))
                for file in files[1:]
            )
            mask = primary_mask(first)
            losses = np.stack([as_numpy(table, "nll", np.float32)[mask] for table in tables])
            fitted_start, fitted_end = fitted_endpoints(losses)
            early_low = fitted_start <= early_mean
            terminal_low = fitted_end <= terminal_mean
            group = np.zeros(mask.sum(), dtype=np.int8)
            group[early_low & ~terminal_low] = 1
            group[~early_low & terminal_low] = 2
            group[early_low & terminal_low] = 3
            chrom = chromosome_codes(first)[mask].astype(np.int64)
            genomic_bin = as_numpy(first, "genomic_pos", np.int64)[mask] // BLOCK_BP
            block = region_index * 10_000 + chrom * 100 + genomic_bin
            assert int(block.max()) < N_BLOCK_KEYS
            for group_index in range(4):
                selected = group == group_index
                n_selected = int(selected.sum())
                counts[group_index] += n_selected
                if not n_selected:
                    continue
                block_counts[group_index] += np.bincount(
                    block[selected], minlength=N_BLOCK_KEYS
                )
                for checkpoint_index in range(len(CHECKPOINTS)):
                    values = losses[checkpoint_index, selected]
                    sums[group_index, checkpoint_index] += values.sum(dtype=np.float64)
                    block_sums[group_index, checkpoint_index] += np.bincount(
                        block[selected], weights=values, minlength=N_BLOCK_KEYS
                    )
            del first, tables, mask, losses, fitted_start, fitted_end
            del early_low, terminal_low, group, chrom, genomic_bin, block, selected, values
            gc.collect()
            pa.default_memory_pool().release_unused()
    return sums, counts, block_sums, block_counts


def confidence_intervals(
    block_sums: np.ndarray,
    block_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    low = np.empty((4, len(CHECKPOINTS)), dtype=np.float64)
    high = np.empty_like(low)
    rng = np.random.default_rng(SEED)
    for group_index in range(4):
        active = block_counts[group_index] > 0
        counts = block_counts[group_index, active]
        sums = block_sums[group_index][:, active]
        assert 0 < len(counts) < 5_000
        draws = rng.integers(0, len(counts), size=(BOOTSTRAP_REPLICATES, len(counts)))
        denominators = counts[draws].sum(axis=1)
        for checkpoint_index in range(len(CHECKPOINTS)):
            means = sums[checkpoint_index, draws].sum(axis=1) / denominators
            low[group_index, checkpoint_index], high[group_index, checkpoint_index] = np.quantile(
                means, (0.025, 0.975)
            )
    return low, high


def plot(
    means: np.ndarray,
    counts: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    output_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(6, 6))
    x = TOKENS / 1e9
    for group_index, group in enumerate(GROUP_ORDER):
        frequency = counts[group_index] / counts.sum()
        axis.plot(
            x,
            means[group_index],
            color=GROUP_COLORS[group],
            marker="o",
            label=f"{GROUP_NAMES[group]} ({frequency:.1%})",
        )
        axis.fill_between(
            x,
            ci_low[group_index],
            ci_high[group_index],
            color=GROUP_COLORS[group],
            alpha=0.15,
            linewidth=0,
        )
    axis.set_xlabel("Training tokens (billions)")
    axis.set_ylabel("Mean loss (nats/base)")
    axis.set_title("Loss by global trajectory type")
    axis.set_ylim(bottom=0)
    axis.set_box_aspect(1)
    axis.legend(
        title="Trajectory group",
        loc="upper left",
        bbox_to_anchor=(0.0, -0.19),
        ncol=2,
        borderaxespad=0,
    )
    figure.subplots_adjust(bottom=0.30)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    s3 = pafs.S3FileSystem(region="us-east-2")
    early_mean, terminal_mean, checkpoint_means, population = global_fitted_thresholds(s3)
    sums, counts, block_sums, block_counts = stream_summaries(
        s3, early_mean, terminal_mean
    )
    assert int(counts.sum()) == population
    np.savez(
        OUTPUT_ROOT / "04-global-fitted-trajectory-accumulators.npz",
        sums=sums,
        counts=counts,
        block_sums=block_sums,
        block_counts=block_counts,
    )
    means = sums / counts[:, None]
    ci_low, ci_high = confidence_intervals(block_sums, block_counts)
    summary = {
        "early_global_fitted_mean_nll": early_mean,
        "terminal_global_fitted_mean_nll": terminal_mean,
        "global_checkpoint_mean_nll": checkpoint_means.tolist(),
        "n_positions": population,
        "counts": {group: int(counts[index]) for index, group in enumerate(GROUP_ORDER)},
        "frequencies": {
            group: float(counts[index] / population) for index, group in enumerate(GROUP_ORDER)
        },
        "means": {group: means[index].tolist() for index, group in enumerate(GROUP_ORDER)},
        "ci_low": {group: ci_low[index].tolist() for index, group in enumerate(GROUP_ORDER)},
        "ci_high": {group: ci_high[index].tolist() for index, group in enumerate(GROUP_ORDER)},
    }
    (OUTPUT_ROOT / "04-global-fitted-trajectory-groups.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    for suffix in ("svg", "png"):
        plot(means, counts, ci_low, ci_high, OUTPUT_ROOT / f"04-global-fitted-trajectory-groups.{suffix}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
