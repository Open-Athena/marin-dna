from __future__ import annotations

import polars as pl

from mutagenesis import (
    FOCAL_INDEX,
    NUCLEOTIDES,
    TASKS,
    _plot,
    counterfactual_sequences,
)


def test_counterfactual_sequences_include_unchanged_target() -> None:
    sequence = "A" * 255
    variants = counterfactual_sequences(sequence, radius=2)

    assert len(variants) == 5 * len(NUCLEOTIDES)
    for relative_position in range(-2, 3):
        position_variants = [
            variant
            for variant in variants
            if variant["relative_position"] == relative_position
        ]
        assert {variant["target_base"] for variant in position_variants} == set(
            NUCLEOTIDES
        )
        unchanged = [variant for variant in position_variants if not variant["changed"]]
        assert len(unchanged) == 1
        assert unchanged[0]["target_base"] == "A"
        assert unchanged[0]["sequence"] == sequence
        for variant in position_variants:
            changed_positions = [
                index
                for index, (left, right) in enumerate(
                    zip(sequence, variant["sequence"], strict=True)
                )
                if left != right
            ]
            if variant["changed"]:
                assert changed_positions == [FOCAL_INDEX + relative_position]
            else:
                assert changed_positions == []


def test_dependency_plot_writes_png_and_svg(tmp_path) -> None:
    rows = []
    for task_index, task in enumerate(TASKS):
        for position in range(-1, 2):
            for base_index, target_base in enumerate(NUCLEOTIDES):
                rows.append(
                    {
                        "task": task,
                        "relative_position": position,
                        "target_base": target_base,
                        "mean_sae_delta": task_index + position + base_index / 10,
                        "mean_raw_delta": task_index - position - base_index / 10,
                    }
                )

    _plot(pl.DataFrame(rows), tmp_path, radius=1)

    assert (tmp_path / "mutagenesis.png").stat().st_size > 0
    assert (tmp_path / "mutagenesis.svg").stat().st_size > 0
