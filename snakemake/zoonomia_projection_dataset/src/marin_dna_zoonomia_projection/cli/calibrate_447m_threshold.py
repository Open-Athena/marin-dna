"""One-off: calibrate ``phyloP_447m`` threshold to match the *proportion* of
non-NaN bases passing in ``phyloP_241m``.

Run once on a SkyPilot box (or anywhere with both bigWigs reachable). Builds
per-base histograms over defined ACGT regions of both ``phyloP_241m`` and
``phyloP_447m``, then finds the ``phyloP_447m`` threshold whose
proportion-of-non-NaN-bases passing matches ``phyloP_241m >= 2.27``.
Prints the threshold and writes a JSON file with the full calibration
record (counts, proportions, totals, relative error, hist meta).

After running, paste the printed threshold into ``config/config.yaml``
under ``phyloP_447m_threshold`` and commit.

Usage::

    uv run --locked --project snakemake/zoonomia_projection_dataset marin-dna-calibrate-447m \\
        --output results/calibration/calibration.json \\
        --bigwig-dir /path/to/local/bigwigs \\
        --genome-2bit /path/to/hg38.2bit
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from marin_dna_zoonomia_projection.conservation.calibration import calibrate_to_match_proportion
from marin_dna_zoonomia_projection.conservation.histogram import (
    PhylopHistogram,
    build_histogram_for_chrom,
)
from marin_dna_zoonomia_projection.tracks import CONSERVATION_TRACKS

REFERENCE_TRACK = "phyloP_241m"
TARGET_TRACK = "phyloP_447m"
REFERENCE_THRESHOLD = 2.27

STANDARD_CHROMS_HUMAN = [str(i) for i in range(1, 23)] + ["X", "Y"]


def _wget(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  exists, skipping: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url} -> {dest}")
    subprocess.run(["wget", "-q", "-O", str(dest), url], check=True)


def _read_undefined_bed(path: Path) -> pd.DataFrame:
    """Read a 3-column BED of undefined (N) regions, no header."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )
    return df


def _build_defined_intervals_per_chrom(
    chrom_sizes: pd.DataFrame, undefined: pd.DataFrame, chroms: list[str]
) -> dict[str, pd.DataFrame]:
    """Per-chrom 0-based half-open intervals covering the defined (ACGT) bases.

    Computed as ``[0, chrom_size)`` minus the union of N-region intervals on
    the same chrom. Pure pandas — no bedtools dependency in this script.
    """
    out: dict[str, pd.DataFrame] = {}
    for chrom in chroms:
        size = int(chrom_sizes.loc[chrom_sizes["chrom"] == chrom, "size"].iloc[0])
        undef = (
            undefined[undefined["chrom"] == chrom]
            .sort_values("start")
            .reset_index(drop=True)
        )
        # Walk through N regions, emitting the gaps as defined intervals.
        rows: list[dict] = []
        cursor = 0
        for _, u in undef.iterrows():
            us, ue = int(u["start"]), int(u["end"])
            if us > cursor:
                rows.append({"chrom": chrom, "start": cursor, "end": us})
            cursor = max(cursor, ue)
        if cursor < size:
            rows.append({"chrom": chrom, "start": cursor, "end": size})
        df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
        if not df.empty:
            assert (df["end"] > df["start"]).all()
            assert (df["end"].shift(-1, fill_value=size) >= df["end"]).all()
        out[chrom] = df
    return out


def _build_genome_histogram(
    bw_path: Path,
    defined_per_chrom: dict[str, pd.DataFrame],
    edges: np.ndarray,
) -> PhylopHistogram:
    """Sum per-chrom histograms into one whole-genome histogram."""
    total: PhylopHistogram | None = None
    for chrom, defined in defined_per_chrom.items():
        if defined.empty:
            continue
        print(
            f"  histogram: {bw_path.name} chrom {chrom} "
            f"({(defined['end'] - defined['start']).sum()} bp)"
        )
        h = build_histogram_for_chrom(bw_path, chrom, defined, edges)
        total = h if total is None else total + h
    assert total is not None
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/calibration/calibration.json"),
        help="Where to write the calibration JSON.",
    )
    parser.add_argument(
        "--bigwig-dir",
        type=Path,
        default=Path("results/calibration/bigwig"),
        help="Where to download / cache the bigWigs.",
    )
    parser.add_argument(
        "--genome-2bit",
        type=Path,
        default=Path("results/human/genome.2bit"),
        help="hg38 .2bit file for chrom-sizes + N-region extraction. "
        "If missing, will be built from the Ensembl FASTA.",
    )
    parser.add_argument(
        "--genome-url",
        default="https://ftp.ensembl.org/pub/release-115/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz",
    )
    parser.add_argument("--hist-min", type=float, default=-20.0)
    parser.add_argument("--hist-max", type=float, default=20.0)
    parser.add_argument("--hist-bins", type=int, default=1000)
    parser.add_argument(
        "--chroms",
        nargs="+",
        default=STANDARD_CHROMS_HUMAN,
        help="Chromosomes to histogram (Ensembl bare names). Defaults to all "
        "autosomes + X + Y. Pass a smaller subset (e.g. `--chroms 1`) to "
        "speed up calibration: a single large autosome already gives "
        "well below 1% relative error in the calibrated threshold.",
    )
    args = parser.parse_args()
    chroms = list(args.chroms)
    unknown = set(chroms) - set(STANDARD_CHROMS_HUMAN)
    assert not unknown, f"unknown chrom names: {unknown}"

    edges = np.linspace(args.hist_min, args.hist_max, args.hist_bins + 1)

    # 1) Stage genome 2bit + chrom sizes + N-regions
    if not args.genome_2bit.exists():
        fa_gz = args.genome_2bit.with_suffix(".fa.gz")
        _wget(args.genome_url, fa_gz)
        args.genome_2bit.parent.mkdir(parents=True, exist_ok=True)
        print(f"  faToTwoBit -> {args.genome_2bit}")
        subprocess.run(
            f"zcat {fa_gz} | faToTwoBit stdin {args.genome_2bit}",
            shell=True,
            check=True,
        )

    chrom_sizes_path = args.genome_2bit.with_suffix(".chrom.sizes")
    undefined_path = args.genome_2bit.with_suffix(".undefined.bed")
    if not chrom_sizes_path.exists():
        subprocess.run(
            ["twoBitInfo", str(args.genome_2bit), str(chrom_sizes_path)], check=True
        )
    if not undefined_path.exists():
        with open(undefined_path, "w") as f:
            subprocess.run(
                ["twoBitInfo", str(args.genome_2bit), "/dev/stdout", "-nBed"],
                check=True,
                stdout=f,
            )

    chrom_sizes = pd.read_csv(
        chrom_sizes_path,
        sep="\t",
        header=None,
        names=["chrom", "size"],
        dtype={"chrom": str},
    )
    chrom_sizes = chrom_sizes[chrom_sizes["chrom"].isin(chroms)]
    assert len(chrom_sizes) == len(chroms), (
        f"missing chroms in 2bit: {set(chroms) - set(chrom_sizes['chrom'])}"
    )
    undefined = _read_undefined_bed(undefined_path)

    defined_per_chrom = _build_defined_intervals_per_chrom(
        chrom_sizes, undefined, chroms
    )
    total_defined = sum(
        int((d["end"] - d["start"]).sum()) for d in defined_per_chrom.values()
    )
    print(f"chroms used for calibration: {chroms}")
    print(f"defined bases (over selected chroms): {total_defined:,}")

    # 2) Stage both bigWigs
    args.bigwig_dir.mkdir(parents=True, exist_ok=True)
    ref_bw = args.bigwig_dir / f"{REFERENCE_TRACK}.bw"
    tgt_bw = args.bigwig_dir / f"{TARGET_TRACK}.bw"
    _wget(CONSERVATION_TRACKS[REFERENCE_TRACK], ref_bw)
    _wget(CONSERVATION_TRACKS[TARGET_TRACK], tgt_bw)

    # 3) Build per-track genome-wide histograms
    print(f"\nbuilding {REFERENCE_TRACK} histogram...")
    ref_hist = _build_genome_histogram(ref_bw, defined_per_chrom, edges)
    print(f"  {REFERENCE_TRACK}: {ref_hist.total():,} non-NaN, {ref_hist.n_nan:,} NaN")

    print(f"\nbuilding {TARGET_TRACK} histogram...")
    tgt_hist = _build_genome_histogram(tgt_bw, defined_per_chrom, edges)
    print(f"  {TARGET_TRACK}: {tgt_hist.total():,} non-NaN, {tgt_hist.n_nan:,} NaN")

    # 4) Calibrate (match proportion of non-NaN bases passing, not raw count)
    print(
        f"\ncalibrating {TARGET_TRACK} to match {REFERENCE_TRACK} >= {REFERENCE_THRESHOLD} "
        f"(proportion of non-NaN bases passing)..."
    )
    out = calibrate_to_match_proportion(
        tgt_hist,
        ref_hist,
        ref_threshold=REFERENCE_THRESHOLD,
        target_name=TARGET_TRACK,
        ref_name=REFERENCE_TRACK,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.output}")
    print(json.dumps(out, indent=2))
    print(
        f"\n>>> Set in config.yaml: phyloP_447m_threshold: {out[TARGET_TRACK]['threshold']:.4f}"
    )


if __name__ == "__main__":
    main()
