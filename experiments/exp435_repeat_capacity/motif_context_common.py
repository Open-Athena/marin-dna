"""Frozen helpers for the post-hoc repeat motif/context pass."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import fisher_exact

MOTIF_CONTEXT_RUN_ID = "dna-exp435-repeat-motif-context-r1"
REFERENCE_PANEL_ARCHIVE_SHA256 = (
    "420237266f074f154fea189281ecdcbb5893afc48d37c2db761e38dbed6d22f7"
)
REFERENCE_ACTIVATION_ARCHIVE_SHA256 = (
    "7a02652172eb42efb5228a0e45a3a495a59fe58f20fdd60b787d01ac000649f6"
)
REFERENCE_ASSOCIATION_ARCHIVE_SHA256 = (
    "cc72fbb0033290af54d2c6dcb0a7521e9b23f5f84b6906bfd9fee1f69206ece0"
)
VARIANT_PANEL_ARCHIVE_SHA256 = (
    "1617638081e099af2e7b370b221c5cb964e8eb85ff8cbbcc2ede8643764bdf60"
)
PAIRED_ACTIVATION_MANIFEST_SHA256 = (
    "0b2e77abf4967a6c9bf7e07bbebe11b0585e21fc4290fcb571cbb16117da10c5"
)
PAIRED_ANALYSIS_ARCHIVE_SHA256 = (
    "2460efef6dbd28e4c71c95d081276d15140db41d1919e50bcf99025f786cf9d6"
)

TOP_CONTEXTS = 256
TOP_VARIANTS = 64
MIN_CONTEXTS = 32
MOTIF_RADIUS = 31
KMER_LENGTHS = (3, 4, 5, 6)
MIN_KMER_SUPPORT = 8
NUCLEOTIDES = ("A", "C", "G", "T")


@dataclass(frozen=True)
class SelectedFeature:
    block: int
    feature_id: int
    reason: str


SELECTED_FEATURES = (
    SelectedFeature(1, 10488, "shallow A/GA-rich and simple-repeat anchor"),
    SelectedFeature(10, 7168, "paired SINE lead"),
    SelectedFeature(10, 11265, "paired Alu and reference SINE lead"),
    SelectedFeature(10, 6903, "paired simple-repeat lead"),
    SelectedFeature(10, 9767, "paired low-complexity lead"),
    SelectedFeature(10, 14271, "paired LINE/L2 lead"),
    SelectedFeature(10, 13092, "reference SATR1 lead"),
    SelectedFeature(10, 12341, "reference SVA lead"),
    SelectedFeature(19, 13894, "paired SINE and reference MIR lead"),
    SelectedFeature(19, 1132, "paired Alu lead"),
    SelectedFeature(19, 13311, "paired simple-repeat and low-complexity lead"),
    SelectedFeature(19, 219, "paired low-complexity lead"),
    SelectedFeature(19, 7778, "paired LTR lead"),
    SelectedFeature(19, 7762, "paired L1 lead"),
    SelectedFeature(19, 1255, "paired DNA-transposon lead"),
    SelectedFeature(19, 12244, "reference forward SATR1 lead"),
    SelectedFeature(19, 2767, "reference RC SATR1 and SVA lead"),
    SelectedFeature(19, 7307, "reference SVA and simple-repeat lead"),
)

assert len({(item.block, item.feature_id) for item in SELECTED_FEATURES}) == len(
    SELECTED_FEATURES
)
assert {item.block for item in SELECTED_FEATURES} == {1, 10, 19}


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of an uppercase A/C/G/T sequence."""

    assert sequence and set(sequence) <= set(NUCLEOTIDES)
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def stable_rank(namespace: str, value: int) -> str:
    """Return a stable hexadecimal rank for deterministic sampling."""

    return hashlib.sha256(f"{namespace}|{value}".encode()).hexdigest()


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment that preserves input order."""

    values = np.asarray(p_values, dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all()
    assert ((values >= 0) & (values <= 1)).all()
    if not values.size:
        return values.copy()
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def select_top_contexts(
    activations: pl.DataFrame,
    *,
    feature_id: int,
    limit: int = TOP_CONTEXTS,
) -> pl.DataFrame:
    """Select deterministic highest nonzero activations for one feature."""

    assert {"context_id", "feature_id", "activation"} <= set(activations.columns)
    selected = (
        activations.filter(
            (pl.col("feature_id") == feature_id) & (pl.col("activation") > 0)
        )
        .sort("activation", "context_id", descending=[True, False])
        .head(limit)
    )
    assert selected["context_id"].n_unique() == selected.height
    return selected


def _candidate_mask(
    candidates: pl.DataFrame,
    row: dict[str, object],
    level: str,
) -> pl.Expr:
    repeat_status = bool(row["is_repeat"])
    expression = pl.col("is_repeat") == repeat_status
    if level in {"chrom_class_gc", "class_gc"}:
        expression &= pl.col("gc_bin") == int(row["gc_bin"])
        if repeat_status:
            expression &= pl.col("repeat_class") == str(row["repeat_class"])
        else:
            expression &= pl.col("repeat_class").is_null()
    elif level == "status_gc":
        expression &= pl.col("gc_bin") == int(row["gc_bin"])
    else:
        assert level == "status"
    if level == "chrom_class_gc":
        expression &= pl.col("chrom") == str(row["chrom"])
    return expression


def match_controls(
    contexts: pl.DataFrame,
    top_context_ids: list[int],
    *,
    namespace: str,
) -> pl.DataFrame:
    """Match unique controls with declared deterministic relaxation levels."""

    required = {"context_id", "chrom", "is_repeat", "repeat_class", "gc_bin"}
    assert required <= set(contexts.columns)
    assert len(top_context_ids) == len(set(top_context_ids))
    top_set = set(top_context_ids)
    rows = {
        int(row["context_id"]): row
        for row in contexts.filter(
            pl.col("context_id").is_in(top_context_ids)
        ).iter_rows(named=True)
    }
    assert set(rows) == top_set
    available = contexts.filter(~pl.col("context_id").is_in(top_context_ids))
    used: set[int] = set()
    matched: list[dict[str, object]] = []
    levels = ("chrom_class_gc", "class_gc", "status_gc", "status")
    for top_id in top_context_ids:
        row = rows[top_id]
        chosen: int | None = None
        chosen_level = ""
        for level in levels:
            candidates = available.filter(_candidate_mask(available, row, level))
            candidate_ids = [
                int(value)
                for value in candidates["context_id"].to_list()
                if int(value) not in used
            ]
            if candidate_ids:
                chosen = min(
                    candidate_ids,
                    key=lambda value: stable_rank(
                        f"{namespace}|{level}|{top_id}", value
                    ),
                )
                chosen_level = level
                break
        assert chosen is not None, (namespace, top_id)
        used.add(chosen)
        matched.append(
            {
                "top_context_id": top_id,
                "control_context_id": chosen,
                "match_level": chosen_level,
            }
        )
    result = pl.DataFrame(matched)
    assert result.height == len(top_context_ids)
    assert result["control_context_id"].n_unique() == result.height
    assert not (set(result["control_context_id"].to_list()) & top_set)
    return result


def _fisher(
    top_with: int,
    top_without: int,
    control_with: int,
    control_without: int,
) -> tuple[float, float]:
    odds_ratio, p_value = fisher_exact(
        [[top_with, top_without], [control_with, control_without]],
        alternative="two-sided",
    )
    return float(odds_ratio), float(p_value)


def positional_enrichment(
    top_sequences: list[str],
    control_sequences: list[str],
    *,
    radius: int = MOTIF_RADIUS,
) -> pl.DataFrame:
    """Test nucleotide enrichment over a focal centered model-input window."""

    assert len(top_sequences) == len(control_sequences) >= 2
    lengths = {len(sequence) for sequence in top_sequences + control_sequences}
    assert len(lengths) == 1
    sequence_length = lengths.pop()
    focal = sequence_length // 2
    assert sequence_length % 2 == 1 and focal - radius >= 0
    assert focal + radius < sequence_length
    rows: list[dict[str, object]] = []
    pseudocount = 0.5
    n = len(top_sequences)
    for offset in range(-radius, radius + 1):
        index = focal + offset
        for base in NUCLEOTIDES:
            top_count = sum(sequence[index] == base for sequence in top_sequences)
            control_count = sum(
                sequence[index] == base for sequence in control_sequences
            )
            odds_ratio, p_value = _fisher(
                top_count,
                n - top_count,
                control_count,
                n - control_count,
            )
            top_frequency = top_count / n
            control_frequency = control_count / n
            log2_odds = math.log2(
                ((top_count + pseudocount) / (n - top_count + pseudocount))
                / ((control_count + pseudocount) / (n - control_count + pseudocount))
            )
            rows.append(
                {
                    "offset": offset,
                    "base": base,
                    "top_count": top_count,
                    "control_count": control_count,
                    "top_frequency": top_frequency,
                    "control_frequency": control_frequency,
                    "odds_ratio": odds_ratio,
                    "log2_odds": log2_odds,
                    "p_value": p_value,
                }
            )
    result = pl.DataFrame(rows)
    result = result.with_columns(
        pl.Series("q_value", bh_adjust(result["p_value"].to_numpy()))
    )
    assert result.height == (2 * radius + 1) * len(NUCLEOTIDES)
    return result


def sequence_consensus(position: pl.DataFrame) -> str:
    """Return the declared q/effect-thresholded positional consensus."""

    symbols: list[str] = []
    for current in position.sort("offset", "base").partition_by(
        "offset", maintain_order=True
    ):
        eligible = current.filter(
            (pl.col("q_value") < 0.05) & (pl.col("log2_odds") >= 1.0)
        ).sort("log2_odds", descending=True)
        symbols.append(str(eligible["base"][0]) if eligible.height else ".")
    return "".join(symbols)


def _presence_counts(sequences: list[str], k: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sequence in sequences:
        counts.update(
            {sequence[index : index + k] for index in range(len(sequence) - k + 1)}
        )
    return counts


def kmer_enrichment(
    top_sequences: list[str],
    control_sequences: list[str],
    *,
    minimum_support: int = MIN_KMER_SUPPORT,
) -> pl.DataFrame:
    """Test presence enrichment for all observed 3--6-mers in a fixed window."""

    assert len(top_sequences) == len(control_sequences) >= 2
    assert {len(item) for item in top_sequences + control_sequences} == {
        len(top_sequences[0])
    }
    n = len(top_sequences)
    rows: list[dict[str, object]] = []
    for k in KMER_LENGTHS:
        top_counts = _presence_counts(top_sequences, k)
        control_counts = _presence_counts(control_sequences, k)
        for kmer in sorted(set(top_counts) | set(control_counts)):
            top_count = top_counts[kmer]
            control_count = control_counts[kmer]
            if top_count + control_count < minimum_support:
                continue
            odds_ratio, p_value = _fisher(
                top_count,
                n - top_count,
                control_count,
                n - control_count,
            )
            log2_odds = math.log2(
                ((top_count + 0.5) / (n - top_count + 0.5))
                / ((control_count + 0.5) / (n - control_count + 0.5))
            )
            rows.append(
                {
                    "k": k,
                    "kmer": kmer,
                    "top_count": top_count,
                    "control_count": control_count,
                    "top_frequency": top_count / n,
                    "control_frequency": control_count / n,
                    "odds_ratio": odds_ratio,
                    "log2_odds": log2_odds,
                    "p_value": p_value,
                }
            )
    assert rows
    result = pl.DataFrame(rows)
    return result.with_columns(
        pl.Series("q_value", bh_adjust(result["p_value"].to_numpy()))
    )
