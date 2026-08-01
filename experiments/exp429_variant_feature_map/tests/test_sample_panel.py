from __future__ import annotations

import polars as pl

from sample_panel import (
    FOCAL_INDEX,
    filter_valid_candidate_contexts,
    sample_balanced_panel,
    split_for_position,
)


def test_split_for_position_uses_one_based_blocks() -> None:
    assert split_for_position(1) == (0, "discovery")
    assert split_for_position(1_000_000) == (0, "discovery")
    assert split_for_position(1_000_001) == (1, "discovery")
    assert split_for_position(3_000_001) == (3, "validation")
    assert split_for_position(4_000_001) == (4, "test")
    assert split_for_position(5_000_001) == (5, "discovery")


def _fixture() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    bases = ["A", "C", "G", "T"]
    for class_index, consequence in enumerate(["class_a", "class_b", "too_small"]):
        for fold in range(5):
            n = 6 if consequence != "too_small" else (1 if fold == 4 else 6)
            block = 10 + fold
            for index in range(n):
                ref = bases[(class_index + index) % 4]
                alt = bases[(class_index + index + 1) % 4]
                rows.append(
                    {
                        "chrom": "21",
                        "pos": block * 1_000_000 + 10 + index,
                        "ref": ref,
                        "alt": alt,
                        "consequence": f"raw_{consequence}",
                        "consequence_cre": consequence,
                    }
                )
    return pl.DataFrame(rows)


def test_balanced_sampler_is_exact_deterministic_and_order_independent() -> None:
    frame = _fixture()
    quotas = {"discovery": 4, "validation": 3, "test": 2}
    first, _, retained, excluded, first_audit = sample_balanced_panel(
        frame.lazy(), quotas=quotas, oversample_factor=8, target_classes=None
    )
    second, _, retained_second, excluded_second, second_audit = sample_balanced_panel(
        frame.reverse().lazy(), quotas=quotas, oversample_factor=8, target_classes=None
    )

    assert retained == retained_second == ["class_a", "class_b"]
    assert not first_audit["checked"] and not second_audit["checked"]
    assert excluded == excluded_second == ["too_small"]
    assert first.equals(second)
    assert first.height == 2 * sum(quotas.values())
    assert (
        first.select(pl.struct(["chrom", "pos", "ref", "alt"]).n_unique()).item()
        == first.height
    )

    observed = {
        (row["consequence_cre"], row["split"]): row["len"]
        for row in first.group_by(["consequence_cre", "split"])
        .len()
        .iter_rows(named=True)
    }
    for consequence in retained:
        for split, quota in quotas.items():
            assert observed[(consequence, split)] == quota


def test_all_alts_at_one_position_share_a_split() -> None:
    frame = _fixture()
    panel, _, _, _, _ = sample_balanced_panel(
        frame.lazy(),
        quotas={"discovery": 2, "validation": 2, "test": 1},
        oversample_factor=8,
        target_classes=None,
    )
    split_counts = panel.group_by("pos").agg(pl.col("split").n_unique().alias("n"))
    assert split_counts["n"].max() == 1


class _FakeGenome:
    def __init__(self, sequences: dict[int, str]) -> None:
        self.sequences = sequences

    def __call__(self, chrom: str, start: int, end: int, strand: str) -> str:
        assert chrom == "21" and strand == "+"
        assert end - start == 2 * FOCAL_INDEX + 1
        return self.sequences[start + FOCAL_INDEX]


def test_filter_valid_candidate_contexts_rejects_n_and_ref_mismatch() -> None:
    candidates = pl.DataFrame(
        {
            "chrom": ["21", "21", "21"],
            "pos": [1001, 2001, 3001],
            "ref": ["C", "T", "G"],
            "alt": ["A", "A", "A"],
            "consequence_cre": ["a", "a", "a"],
            "split": ["discovery", "discovery", "discovery"],
            "sample_hash": [1, 2, 3],
        }
    )
    valid = "A" * FOCAL_INDEX + "C" + "G" * FOCAL_INDEX
    has_n = "A" * (FOCAL_INDEX - 1) + "N" + "T" + "G" * FOCAL_INDEX
    ref_mismatch = "A" * FOCAL_INDEX + "C" + "G" * FOCAL_INDEX
    genome = _FakeGenome({1000: valid, 2000: has_n, 3000: ref_mismatch})

    filtered, invalid = filter_valid_candidate_contexts(candidates, genome)

    assert filtered["pos"].to_list() == [1001]
    assert len(invalid) == 2
    assert invalid[0]["reasons"] == ["non_acgt=N"]
    assert invalid[1]["reasons"] == ["center_ref=C expected=G"]
