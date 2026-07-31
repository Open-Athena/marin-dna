from __future__ import annotations

import polars as pl

from sample_panel import sample_balanced_panel, split_for_position


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
    first, _, retained, excluded = sample_balanced_panel(
        frame.lazy(), quotas=quotas, oversample_factor=8
    )
    second, _, retained_second, excluded_second = sample_balanced_panel(
        frame.reverse().lazy(), quotas=quotas, oversample_factor=8
    )

    assert retained == retained_second == ["class_a", "class_b"]
    assert excluded == excluded_second == ["too_small"]
    assert first.equals(second)
    assert first.height == 2 * sum(quotas.values())
    assert first.select(pl.struct(["chrom", "pos", "ref", "alt"]).n_unique()).item() == first.height

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
    panel, _, _, _ = sample_balanced_panel(
        frame.lazy(),
        quotas={"discovery": 2, "validation": 2, "test": 1},
        oversample_factor=8,
    )
    split_counts = panel.group_by("pos").agg(pl.col("split").n_unique().alias("n"))
    assert split_counts["n"].max() == 1
