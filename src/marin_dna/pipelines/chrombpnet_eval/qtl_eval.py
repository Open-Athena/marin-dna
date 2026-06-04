"""Online caQTL/dsQTL variant-effect metric for ChromBPNet training (#241).

A **light, continuous** live-validation signal: score each QTL **positive**
variant by the model's predicted ``log2`` fold-change of accessibility counts
(alt vs ref 2114 bp windows) and correlate — signed **Pearson** —
against the observed study ``effect``. Positives-only and correlation-only by
design, so it runs every validation cheaply (caqtl ~3,173 + dsqtl ~309
positives). The binary AUROC/AUPRC need the full negative set (~10–50× larger)
and belong to the heavier final/test eval, not this callback.

This tracks the **eval target** (does the model rank QTL effects?), which is
distinct from ``val_count_pearson`` — that's the accessibility-count fit on the
held-out chromosome split, a training-health proxy, not a QTL metric.

The score is oriented to ``alt`` (``pred(alt) - pred(ref)``) and the parquet
``effect`` is signed to ``alt``, so the signed correlation is meaningful — the
same convention as the M1a supervised metric.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import lightning as L
import numpy as np
import torch
from scipy.stats import pearsonr

from marin_dna.pipelines.chrombpnet_eval._vendor.chrombpnet.data_utils import (
    dna_to_one_hot,
)

_NUC = frozenset("ACGT")

# caqtl/dsqtl variant parquets (HF), pinned to the same revisions M1a uses.
# (name, hf_repo, revision, flip_effect). dsQTL's study effect (``obs.estimate``,
# our parquet ``effect``) is sign-flipped vs alt-oriented accessibility log2FC —
# the same convention M1a handles with ``flip_logfc=True`` (ARSENAL's notebook
# correlates ``-obs.estimate`` against ChromBPNet's logfc). Flip the effect so a
# model that agrees with the study reads as a POSITIVE correlation. caQTL: no flip.
QTL_DATASETS: list[tuple[str, str, str, bool]] = [
    (
        "caqtl",
        "bolinas-dna/evals_caqtl",
        "9d004a21812c067b9ba1ebfe72f51b9095a5d0f8",
        False,
    ),
    (
        "dsqtl",
        "bolinas-dna/evals_dsqtl",
        "b7e02a07beb831c7047286aacd3ddfd299d6f88f",
        True,
    ),
]


@dataclass
class QTLSpec:
    """Pre-extracted positives for one QTL dataset (ready for repeated scoring).

    ``ref_oh`` / ``alt_oh`` are ``[N, 4, window]`` float32 one-hots; ``effect``
    is the ``[N]`` signed study effect oriented to ``alt`` (positives only).
    """

    name: str
    ref_oh: np.ndarray
    alt_oh: np.ndarray
    effect: np.ndarray


def extract_ref_alt_onehot(
    chroms: Sequence[str],
    positions: Sequence[int],
    refs: Sequence[str],
    alts: Sequence[str],
    genome: Callable[[str, int, int, str], str],
    *,
    window: int = 2114,
    chrom_prefix: str = "",
    min_ref_match: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ref/alt one-hot windows centered on each (SNP) variant.

    Args:
        chroms / positions / refs / alts: parallel arrays; ``positions`` are
            **1-based** (parquet convention), ``refs``/``alts`` single ACGT bases.
        genome: callable ``(chrom, start, end, strand) -> seq`` over 0-based
            half-open coords (the ``marin_dna.data.genome.Genome`` interface).
        window: window length (2114 for ChromBPNet); the variant sits at the
            center index ``window // 2``.
        chrom_prefix: prepended to ``chrom`` before the genome lookup — pass
            ``"chr"`` to read a chr-prefixed fasta with our non-prefixed
            ("1".."22") variant chroms.
        min_ref_match: assert at least this fraction of variants have the genome
            base at the center equal to ``ref`` — a loud guard against a
            reference-build / chrom-prefix mismatch (the parquets were built with
            ``ref`` oriented to the genome, so a match near 1.0 is expected).

    Returns:
        ``(ref_oh, alt_oh)`` each ``[N, 4, window]`` float32 (channels A,C,G,T).
    """
    half = window // 2
    ref_seqs: list[str] = []
    alt_seqs: list[str] = []
    mismatches: list[tuple[str, int, str, str]] = []
    for chrom, pos, ref, alt in zip(chroms, positions, refs, alts, strict=True):
        ref, alt = ref.upper(), alt.upper()
        assert ref in _NUC and alt in _NUC, (
            f"SNP-only expected; got ref={ref!r} alt={alt!r} at {chrom}:{pos}"
        )
        center0 = int(pos) - 1  # 1-based parquet pos -> 0-based genome coord
        start = center0 - half
        seq = genome(f"{chrom_prefix}{chrom}", start, start + window, "+").upper()
        assert len(seq) == window, f"{chrom}:{pos} got {len(seq)} bp, want {window}"
        if seq[half] != ref:
            mismatches.append((chrom, int(pos), seq[half], ref))
        ref_seqs.append(seq)
        alt_seqs.append(seq[:half] + alt + seq[half + 1 :])

    n = len(ref_seqs)
    assert n > 0, "no variants to extract"
    match = 1.0 - len(mismatches) / n
    assert match >= min_ref_match, (
        f"genome base == ref for only {match:.3f} of {n} variants (< "
        f"{min_ref_match}); suspect a reference-build / chrom-prefix mismatch "
        f"(chrom_prefix={chrom_prefix!r}). e.g. {mismatches[:3]}"
    )
    ref_oh = dna_to_one_hot(ref_seqs).astype(np.float32).transpose(0, 2, 1)
    alt_oh = dna_to_one_hot(alt_seqs).astype(np.float32).transpose(0, 2, 1)
    return np.ascontiguousarray(ref_oh), np.ascontiguousarray(alt_oh)


@torch.no_grad()
def score_log2fc(
    model: torch.nn.Module,
    ref_oh: np.ndarray,
    alt_oh: np.ndarray,
    *,
    batch_size: int = 256,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Predicted ``log2(counts(alt) / counts(ref))`` per variant.

    The model's count head outputs (natural-)log counts, so the alt−ref
    difference divided by ``ln 2`` is the log2 fold-change (and the constant is
    irrelevant to the rank/Pearson correlation anyway). Runs in eval mode and
    restores the prior train/eval state.
    """
    assert len(ref_oh) == len(alt_oh)
    if len(ref_oh) == 0:
        return np.zeros(0, dtype=np.float32)
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    out: list[np.ndarray] = []
    for i in range(0, len(ref_oh), batch_size):
        r = torch.from_numpy(ref_oh[i : i + batch_size]).to(device)
        a = torch.from_numpy(alt_oh[i : i + batch_size]).to(device)
        _, ref_counts = model(r)
        _, alt_counts = model(a)
        fc = (alt_counts.squeeze(-1) - ref_counts.squeeze(-1)) / math.log(2)
        out.append(fc.float().cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(out)


def signed_pearson(scores: np.ndarray, effect: np.ndarray) -> float:
    """Signed Pearson of ``scores`` vs ``effect``; ``0.0`` if degenerate — either
    input constant, fewer than 2 points (an early smoke pass), or a **non-finite**
    result (e.g. a diverged model emitting overflowing scores, so ``pearsonr``
    returns NaN). Never emits NaN. (Spearman dropped, #259 — not used in the field.)
    """
    if len(scores) > 1 and np.std(scores) > 0 and np.std(effect) > 0:
        pearson = float(pearsonr(scores, effect).statistic)
        if np.isfinite(pearson):
            return pearson
    return 0.0


class QTLEvalCallback(L.Callback):
    """Log signed Pearson of predicted log2FC vs observed ``effect`` over the QTL
    positives — per dataset plus their mean ``qtl_avg_pearson`` — once per
    validation.

    Single-device exact (we train on 1 GPU); each rank would recompute the same
    rank-local scalar, so it's logged with ``sync_dist=False``.
    """

    def __init__(self, specs: Sequence[QTLSpec], *, batch_size: int = 256) -> None:
        super().__init__()
        self.specs = list(specs)
        self.batch_size = batch_size

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        pearsons: list[float] = []
        for spec in self.specs:
            scores = score_log2fc(
                cast(torch.nn.Module, pl_module.model),
                spec.ref_oh,
                spec.alt_oh,
                batch_size=self.batch_size,
                device=pl_module.device,
            )
            pearson = signed_pearson(scores, spec.effect)
            pl_module.log(
                f"qtl_{spec.name}_pearson", pearson, prog_bar=True, sync_dist=False
            )
            pearsons.append(pearson)
        # Mean across datasets — the headline insight curve (#259); logged for
        # monitoring only (fixed-budget runs do not select on it).
        if pearsons:
            pl_module.log(
                "qtl_avg_pearson",
                float(np.mean(pearsons)),
                prog_bar=True,
                sync_dist=False,
            )


def build_qtl_specs(
    genome: Callable[[str, int, int, str], str],
    datasets: Sequence[tuple[str, str, str, bool]] = tuple(QTL_DATASETS),
    *,
    split: str = "train",
    window: int = 2114,
    chrom_prefix: str = "",
) -> list[QTLSpec]:
    """Load the QTL **positive** variants (one split) and pre-extract their
    ref/alt one-hot windows — done once before training; the callback then just
    re-scores the cached arrays each validation.

    ``datasets`` is ``[(name, hf_repo, revision, flip_effect), ...]`` (defaults to
    :data:`QTL_DATASETS`). Develop on ``split="train"`` (test held out).
    """
    import polars as pl
    from huggingface_hub import hf_hub_download

    specs: list[QTLSpec] = []
    for name, repo, rev, flip_effect in datasets:
        path = hf_hub_download(
            repo, f"{split}.parquet", revision=rev, repo_type="dataset"
        )
        df = pl.read_parquet(path).filter(
            pl.col("label") & pl.col("effect").is_not_null()
        )
        assert df.height > 0, f"{name}: no positive variants with an effect in {split}"
        ref_oh, alt_oh = extract_ref_alt_onehot(
            df["chrom"].to_list(),
            df["pos"].to_list(),
            df["ref"].to_list(),
            df["alt"].to_list(),
            genome,
            window=window,
            chrom_prefix=chrom_prefix,
        )
        effect = df["effect"].to_numpy().astype(np.float64)
        if flip_effect:
            effect = -effect  # align dsQTL obs.estimate to alt-oriented log2FC
        print(
            f"[qtl] {name}: {df.height} positives ({split} split)"
            + (" [effect sign-flipped]" if flip_effect else "")
        )
        specs.append(QTLSpec(name=name, ref_oh=ref_oh, alt_oh=alt_oh, effect=effect))
    return specs
