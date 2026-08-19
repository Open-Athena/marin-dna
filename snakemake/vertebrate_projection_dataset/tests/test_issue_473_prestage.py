from __future__ import annotations

from pathlib import Path

from marin_dna_vertebrate_projection.issue_473.prestage import (
    required_inventory_paths,
)


def test_required_inventory_paths_selects_only_consumed_baseline_files(
    tmp_path: Path,
) -> None:
    paths = [
        "anchors/scored/chr1.parquet",
        "hal/rejected/Mus_musculus.parquet",
        "hal/sequence_rejected/Mus_musculus.parquet",
        "multiz/rejected/danRer10.parquet",
        "multiz/sequence_rejected/danRer10.parquet",
        "datasets/cds/train.parquet",
    ]
    inventory = tmp_path / "inventory.tsv"
    inventory.write_text(
        "".join(f"{path}\t{index + 1}\n" for index, path in enumerate(paths))
    )
    species = tmp_path / "species.tsv"
    species.write_text(
        "alignment_name\tbackend\tselected\n"
        "Mus_musculus\tzoonomia_cactus\ttrue\n"
        "danRer10\tucsc_multiz100way\ttrue\n"
        "unused\tzoonomia_cactus\tfalse\n"
    )

    selected = required_inventory_paths(inventory, species, ["chr1"])
    assert list(selected) == sorted(paths[:5])
    assert "datasets/cds/train.parquet" not in selected
