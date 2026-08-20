import pandas as pd
import pytest
from marin_dna_carbon_conditioning_vep.preflight import (
    PromptPreflightBlocked,
    _truncate_recovery_context,
    choose_prompt_grammar,
    continuation_accuracy,
    select_fixed_preflight_slice,
)


def test_recovery_truncation_preserves_released_ambiguous_bases() -> None:
    assert _truncate_recovery_context("AACCGGNN", 8) == "CCGGNN"


def test_fixed_preflight_slice_is_balanced_and_deterministic() -> None:
    frame = pd.DataFrame(
        [
            {
                "sequence": "ACGTAC" * (index + 1),
                "label": "A" * 30,
                "type": species,
            }
            for species in ("fungi", "protozoa", "invertebrate")
            for index in range(4)
        ]
    )
    first = select_fixed_preflight_slice(
        frame, target_species=["fungi", "protozoa", "invertebrate"], rows_per_species=2
    )
    second = select_fixed_preflight_slice(
        frame, target_species=["fungi", "protozoa", "invertebrate"], rows_per_species=2
    )
    assert first["sample_key"].tolist() == second["sample_key"].tolist()
    assert first.groupby("species").size().eq(2).all()


def test_preflight_selects_positive_stronger_grammar() -> None:
    rows = []
    for grammar, deltas in (("model_card", [0.1, 0.0]), ("corpus_card", [0.2, 0.3])):
        for delta in deltas:
            rows.append(
                {
                    "grammar": grammar,
                    "correct_accuracy": 0.4 + delta,
                    "untagged_accuracy": 0.4,
                    "delta": delta,
                }
            )
    selected, rejected, summaries = choose_prompt_grammar(
        pd.DataFrame(rows),
        grammar_names=["model_card", "corpus_card"],
        tolerance=1e-12,
    )
    assert (selected, rejected) == ("corpus_card", "model_card")
    assert summaries[selected]["delta"] == pytest.approx(0.25)


def test_preflight_blocks_without_metadata_sensitivity() -> None:
    frame = pd.DataFrame(
        [
            {
                "grammar": grammar,
                "correct_accuracy": 0.4,
                "untagged_accuracy": 0.4,
                "delta": 0.0,
            }
            for grammar in ("model_card", "corpus_card")
        ]
    )
    with pytest.raises(PromptPreflightBlocked, match="neither"):
        choose_prompt_grammar(
            frame,
            grammar_names=["model_card", "corpus_card"],
            tolerance=1e-12,
        )


def test_continuation_accuracy_uses_fixed_30_base_denominator() -> None:
    assert continuation_accuracy("A" * 15, "A" * 30) == 0.5
