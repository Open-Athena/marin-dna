from __future__ import annotations

from marin_dna_vertebrate_projection.gpn_star_publication import (
    rewrite_gpn_star_dataset_card,
)


def test_rewrite_gpn_star_dataset_card_replaces_generic_phyloP_claim(tmp_path) -> None:
    path = tmp_path / "README.md"
    path.write_text(
        "Anchor eligibility uses the pipeline's pinned phyloP conservation filter.\n"
    )

    rewrite_gpn_star_dataset_card(path)

    text = path.read_text()
    assert "phyloP" not in text
    assert "gpn-star-hg38-p243-200m" in text
    assert "at least 51 of 255" in text
    assert "entropy_calibrated < 0.081001" in text
