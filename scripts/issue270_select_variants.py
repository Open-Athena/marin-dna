"""issue #270 (cLLR subsampling pilot): select neutral variants for two 5-mer contexts.

Picks two pentanucleotide contexts from the pinned neutral-site set — the
most-abundant CpG context (center C, +1 G) and the most-abundant non-CpG context
— samples 1000 neutral sites of each, expands every site to its 3 non-ref alts,
and writes a small variants parquet for GPU LLR scoring.

Each neutral site (chrom, pos, ref) is scored against all 3 alts, so the
calibration cell `pentanuc_mut = 5mer + "_" + alt` draws its observations from
exactly the sites carrying that 5-mer; the convergence study subsamples those.

Centering: the central base (0-based index pos-1, matching
`marin_dna.data.transforms._get_variant_window`) must equal the parquet `ref`;
asserted end-to-end here.

Reads a *plain* (decompressed) GRCh38 FASTA via pyfaidx mmap — the bgzipped
whole-chromosome read through `Genome` was pathologically slow.

Run:
    uv run python scripts/issue270_select_variants.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyfaidx import Fasta

NEUTRAL_PARQUET = "scratch/issue270/neutral_sites.parquet"
PLAIN_FASTA = "scratch/issue270/GRCh38.fa"
OUT_VARIANTS = "scratch/issue270/pilot_variants.parquet"

N_SITES = 1000  # sampled neutral sites per context
SEED = 270
NUCLEOTIDES = ("A", "C", "G", "T")
OFFSETS = np.arange(-2, 3)  # 5 bp window around the variant
CENTER = 2  # variant index within the 5-mer


def extract_pentanucs(df: pd.DataFrame, fasta: Fasta) -> np.ndarray:
    """Uppercase central 5-mer ('S5' bytes) per (chrom, pos) row; assert center == ref.

    Reads each chromosome once (plain-FASTA mmap slice). `df` must have a
    RangeIndex; returns an array aligned with it.
    """
    out = np.empty(len(df), dtype="S5")
    filled = np.zeros(len(df), dtype=bool)
    for chrom, sub in df.groupby("chrom", sort=False):
        seq = fasta[chrom][:].upper()  # whole chromosome, uppercased
        arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        center0 = sub["pos"].to_numpy() - 1  # 1-based -> 0-based variant index
        assert center0.min() - 2 >= 0 and center0.max() + 2 < len(arr), (
            f"{chrom}: 5-mer window runs off the chromosome edge"
        )
        kmer = arr[center0[:, None] + OFFSETS[None, :]]  # [n, 5] uint8
        ref_u8 = np.frombuffer("".join(sub["ref"]).encode("ascii"), dtype=np.uint8)
        mismatch = kmer[:, CENTER] != ref_u8
        assert not mismatch.any(), (
            f"{chrom}: {int(mismatch.sum())} center!=ref mismatches "
            f"(coordinate/centering bug)"
        )
        out[sub.index.to_numpy()] = np.ascontiguousarray(kmer).view("S5").ravel()
        filled[sub.index.to_numpy()] = True
        print(f"  {chrom}: {len(sub):,} sites")
    assert filled.all(), "some rows never filled"
    return out


def expand_alts(sites: pd.DataFrame) -> pd.DataFrame:
    """Each (chrom, pos, ref, pentanuc) site -> 3 rows, one per non-ref alt."""
    rows = []
    for r in sites.itertuples(index=False):
        for alt in NUCLEOTIDES:
            if alt == r.ref:
                continue
            rows.append(
                (
                    r.chrom,
                    r.pos,
                    r.ref,
                    alt,
                    r.pentanuc,
                    f"{r.pentanuc}_{alt}",
                    r.context_label,
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "chrom",
            "pos",
            "ref",
            "alt",
            "pentanuc",
            "pentanuc_mut",
            "context_label",
        ],
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(NEUTRAL_PARQUET).reset_index(drop=True)
    print(f"neutral sites: {len(df):,}")

    fasta = Fasta(PLAIN_FASTA, as_raw=True)
    df["pentanuc"] = pd.Index(extract_pentanucs(df, fasta)).str.decode("ascii")

    valid = df["pentanuc"].str.fullmatch("[ACGT]{5}")
    print(f"degenerate (non-ACGT flank) 5-mers dropped: {int((~valid).sum()):,}")
    df = df.loc[valid].reset_index(drop=True)

    counts = df["pentanuc"].value_counts()
    is_cpg_c = counts.index.str.slice(CENTER, CENTER + 2) == "CG"
    has_cg = counts.index.str.contains("CG")
    n_distinct = counts.index.map(lambda k: len(set(k)))
    cpg_5mer = counts[is_cpg_c].index[0]  # most-abundant center-C CpG (hypermutable)
    # "typical" = most-abundant CpG-free 5-mer with >=3 distinct bases: a
    # representative mixed context, avoiding the homopolymer (e.g. AAAAA) whose
    # LLR variance is atypically low and would over-state how few sites suffice.
    typ_5mer = counts[~has_cg & (n_distinct >= 3)].index[0]
    print("\nchosen contexts:")
    print(
        f"  CpG (center C, +1 G):    {cpg_5mer}  ({counts[cpg_5mer]:,} sites available)"
    )
    print(
        f"  typical (no CpG, >=3 nt): {typ_5mer}  ({counts[typ_5mer]:,} sites available)"
    )

    parts = []
    for label, kmer in (("cpg", cpg_5mer), ("typical", typ_5mer)):
        pool = df[df["pentanuc"] == kmer]
        assert len(pool) >= N_SITES, (
            f"{label} 5-mer {kmer} has only {len(pool)} sites (< {N_SITES})"
        )
        take = pool.iloc[rng.choice(len(pool), size=N_SITES, replace=False)].copy()
        take["context_label"] = label
        parts.append(take[["chrom", "pos", "ref", "pentanuc", "context_label"]])
    sites = pd.concat(parts, ignore_index=True)

    variants = expand_alts(sites)
    # Each context: N_SITES sites x 3 alts.
    assert len(variants) == 2 * N_SITES * 3, f"unexpected row count {len(variants)}"
    assert (variants["ref"] != variants["alt"]).all()
    assert variants["pentanuc"].str[CENTER].equals(variants["ref"]), "center != ref"

    variants.to_parquet(OUT_VARIANTS, index=False)
    print(f"\nwrote {len(variants):,} variants -> {OUT_VARIANTS}")
    print("\nper (context, alt) cell counts (each = N_SITES):")
    print(variants.groupby(["context_label", "pentanuc_mut"]).size().to_string())
    print("\nhead:")
    print(variants.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
