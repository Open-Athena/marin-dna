"""Missense variant loading, window tokenization, and chromosome-grouped folds (#369).

The variant windows (ref/alt token ids on the FWD and RC strands) are **model-weight
independent** — they depend only on the genome, the tokenizer, and the window size — so
they are built once and cached, then reused across every fine-tuning run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from marin_dna.data.genome import Genome
from marin_dna.data.transforms import (
    _get_special_token_counts,
    in_seq_var_pos,
    transform_llr_clm,
)

MENDELIAN_HF = "bolinas-dna/evals_mendelian_traits"
MENDELIAN_REVISION = "4aed58e50c5dea0b878a665007af2ef9e5108e9f"  # PR #194 k=9 rebuild
MISSENSE_SUBSET = "missense_variant"
# Canonical GRCh38 reference used across evals_v2 (config.yaml genome_path).
GENOME_PATH = (
    "s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115/"
    "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
)


def load_missense_train(
    revision: str = MENDELIAN_REVISION, split: str = "train"
) -> pd.DataFrame:
    """Load the Mendelian ``missense_variant`` rows of the given split (default train).

    Columns include ``chrom, pos, ref, alt, label, subset, match_group``. Kept as a
    thin loader so tests can patch it; the heavy lifting is in ``build_variant_windows``.
    """
    from datasets import load_dataset

    df = (
        load_dataset(MENDELIAN_HF, split=split, revision=revision)
        .to_pandas()
        .query("subset == @MISSENSE_SUBSET")
        .reset_index(drop=True)
    )
    for col in ("chrom", "pos", "ref", "alt", "label", "subset", "match_group"):
        assert col in df.columns, f"missing column {col!r}"
    df["chrom"] = df["chrom"].astype(str)
    assert (df["ref"].str.len() == 1).all() and (df["alt"].str.len() == 1).all(), (
        "non-SNV rows present — the transform is SNV-only"
    )
    return df


@dataclass
class VariantWindows:
    """Precomputed, model-independent tokenized windows for a set of variants.

    ``ref_fwd``/``alt_fwd``/``ref_rc``/``alt_rc`` are ``[N, L]`` long tensors (L =
    ``window_size`` + ``n_prefix``). Pooling covers ``[pool_lo, pool_hi)`` — the
    ``window_size`` DNA tokens, BOS excluded (matches the #341 ``entire_window`` pool).
    ``chrom``/``label``/``match_group`` are ``[N]`` metadata arrays row-aligned to the
    token tensors.
    """

    ref_fwd: torch.Tensor
    alt_fwd: torch.Tensor
    ref_rc: torch.Tensor
    alt_rc: torch.Tensor
    pool_lo: int
    pool_hi: int
    chrom: np.ndarray
    label: np.ndarray
    match_group: np.ndarray

    def __len__(self) -> int:
        return self.ref_fwd.shape[0]


def build_variant_windows(
    df: pd.DataFrame, tokenizer, genome, window_size: int
) -> VariantWindows:
    """Tokenize FWD+RC ref/alt windows for every variant via ``transform_llr_clm``.

    The alt sequence is the ref window with a single token swapped at the variant
    position ``n_prefix + in_seq_var_pos(window_size, strand)`` — we assert exactly that
    (ref/alt differ in one position, and it is the variant token) so a window/offset bug
    fails loudly rather than silently scoring the wrong site.
    """
    n_prefix, _ = _get_special_token_counts(tokenizer)
    banks: dict[str, list[torch.Tensor]] = {k: [] for k in ("rf", "af", "rr", "ar")}
    for row in df.itertuples(index=False):
        example = {
            "chrom": str(row.chrom),
            "pos": int(row.pos),
            "ref": row.ref,
            "alt": row.alt,
        }
        for strand, ref_key, alt_key in (("+", "rf", "af"), ("-", "rr", "ar")):
            d = transform_llr_clm(example, tokenizer, genome, window_size, strand)
            ref_ids = d["input_ids"]
            var_pos = n_prefix + in_seq_var_pos(window_size, strand)
            assert 0 < var_pos < len(ref_ids) - 1, (var_pos, len(ref_ids))
            alt_ids = ref_ids.clone()
            alt_ids[var_pos] = int(d["alt_token_id"])
            assert int((alt_ids != ref_ids).sum()) == 1, "alt must differ in one token"
            assert alt_ids[var_pos] != ref_ids[var_pos], "variant token unchanged"
            banks[ref_key].append(ref_ids)
            banks[alt_key].append(alt_ids)

    ref_fwd = torch.stack(banks["rf"])
    assert ref_fwd.shape[1] == window_size + n_prefix, ref_fwd.shape
    return VariantWindows(
        ref_fwd=ref_fwd,
        alt_fwd=torch.stack(banks["af"]),
        ref_rc=torch.stack(banks["rr"]),
        alt_rc=torch.stack(banks["ar"]),
        pool_lo=n_prefix,
        pool_hi=n_prefix + window_size,
        chrom=df["chrom"].astype(str).to_numpy(),
        label=df["label"].to_numpy().astype(np.int64),
        match_group=df["match_group"].to_numpy(),
    )


def _variant_signature(df: pd.DataFrame, window_size: int, tokenizer) -> str:
    """Cache key: variant identity + window + this tokenizer's nucleotide token ids.

    Keys on the tokenizer's A/C/G/T ids (not just its name) so a genuinely different
    tokenization can never silently reuse another model's cached windows.
    """
    from marin_dna.data.transforms import _get_nucleotide_token_ids

    ids = _get_nucleotide_token_ids(tokenizer)
    idsig = ",".join(f"{k}{ids[k]}" for k in sorted(ids))
    key = (
        df[["chrom", "pos", "ref", "alt"]]
        .astype(str)
        .agg(":".join, axis=1)
        .str.cat(sep="|")
    )
    h = hashlib.sha1(f"{window_size}|{idsig}|{key}".encode()).hexdigest()[:16]
    return f"missense_w{window_size}_n{len(df)}_{h}"


def build_or_load_windows(
    df: pd.DataFrame,
    tokenizer,
    window_size: int,
    cache_dir: str | Path,
    genome: Genome | None = None,
) -> VariantWindows:
    """Build the windows, or load them from ``cache_dir`` if a matching cache exists."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_variant_signature(df, window_size, tokenizer)}.pt"
    if path.exists():
        blob = torch.load(path, weights_only=False)
        return VariantWindows(**blob)
    if genome is None:
        genome = Genome(GENOME_PATH)
    vw = build_variant_windows(df, tokenizer, genome, window_size)
    torch.save(vw.__dict__, path)
    return vw


def chrom_fold_masks(
    chrom: np.ndarray, test_chrom: str, val_chrom: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean (train, val, test) masks for a leave-out-``test_chrom`` fold.

    ``val_chrom`` (held out of training) drives early stopping / selection. Asserts the
    three chromosome sets are disjoint — the leak-proofing invariant.
    """
    chrom = np.asarray(chrom, dtype=str)
    assert test_chrom != val_chrom, (test_chrom, val_chrom)
    test = chrom == test_chrom
    val = chrom == val_chrom
    train = ~(test | val)
    assert test.any() and val.any() and train.any(), "empty split partition"
    tr_c, va_c, te_c = set(chrom[train]), set(chrom[val]), set(chrom[test])
    assert not (tr_c & va_c) and not (tr_c & te_c) and not (va_c & te_c), (
        "chromosome leakage across train/val/test"
    )
    return train, val, test


def nested_loco_folds(
    chrom: np.ndarray,
) -> Iterator[tuple[str, str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Yield ``(test_chrom, val_chrom, (train, val, test) masks)`` for every chromosome.

    ``val_chrom`` is the deterministic next chromosome in sorted order (wrapping), so the
    inner early-stopping split is reproducible and never equals the outer test chromosome.
    """
    uniq = sorted(set(np.asarray(chrom, dtype=str).tolist()))
    assert len(uniq) >= 3, f"need >=3 chromosomes for nested LOCO, got {len(uniq)}"
    for i, test_chrom in enumerate(uniq):
        val_chrom = uniq[(i + 1) % len(uniq)]
        yield test_chrom, val_chrom, chrom_fold_masks(chrom, test_chrom, val_chrom)


def iter_minibatches(
    n: int, batch_size: int, generator: torch.Generator, shuffle: bool = True
) -> Iterator[torch.Tensor]:
    """Yield index tensors of size ``batch_size`` (last batch may be smaller)."""
    order = (
        torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    )
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]
