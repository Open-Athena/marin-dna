"""Publication-specific card checks for the GPN-Star-P uniform datasets."""

from __future__ import annotations

from pathlib import Path


_PHYLOP_SENTENCE = (
    "Anchor eligibility uses the pipeline's pinned phyloP conservation filter."
)


def rewrite_gpn_star_dataset_card(path: str | Path) -> None:
    """Replace the generic phyloP wording with the exact GPN selection contract."""
    card = Path(path)
    text = card.read_text()
    assert text.count(_PHYLOP_SENTENCE) == 1
    replacement = (
        "Anchor eligibility uses calibrated entropy from the primate "
        "`gpn-star-hg38-p243-200m` score set. "
        "A human window is eligible when at least 51 of 255 positions satisfy "
        "the strict rule `entropy_calibrated < 0.081001`."
    )
    rewritten = text.replace(_PHYLOP_SENTENCE, replacement)
    assert "phyloP conservation filter" not in rewritten
    assert "gpn-star-hg38-p243-200m" in rewritten
    card.write_text(rewritten)
