"""Frozen Carbon-3B tokenizer IDs for every configured VEP prefix."""

from typing import Any

from marin_dna_carbon_conditioning_vep.prompts import (
    CONDITION_ORDER,
    PromptGrammar,
    assert_prefix_outside_dna_mode,
)

FROZEN_PREFIX_TOKEN_IDS: dict[str, dict[str, list[int]]] = {
    "model_card": {
        "untagged": [151669],
        "correct": [27, 64832, 64116, 717, 8666, 10480, 29, 151669],
        "near_wrong": [27, 64832, 64116, 30456, 29, 151669],
        "far_wrong": [63895, 81490, 29, 151669],
    },
    "corpus_card": {
        "untagged": [151669],
        "correct": [27, 42490, 29, 64832, 64116, 717, 8666, 10480, 151669],
        "near_wrong": [27, 42490, 29, 64832, 64116, 30456, 151669],
        "far_wrong": [27, 42490, 29, 78606, 72, 151669],
    },
}


def assert_frozen_prefix_token_ids(
    tokenizer: Any,
    grammar_templates: dict[str, str],
    conditions: dict[str, str | None],
) -> dict[str, dict[str, list[int]]]:
    """Fail before inference if the pinned tokenizer changes any prompt IDs."""
    assert tuple(conditions) == CONDITION_ORDER, (
        f"condition order must be {CONDITION_ORDER}, got {tuple(conditions)}"
    )
    assert set(grammar_templates) == set(FROZEN_PREFIX_TOKEN_IDS), (
        "candidate grammar names differ from the frozen tokenizer snapshot"
    )
    observed: dict[str, dict[str, list[int]]] = {}
    for grammar_name, template in grammar_templates.items():
        grammar = PromptGrammar(grammar_name, template)
        observed[grammar_name] = {}
        for condition, species in conditions.items():
            prefix = "<dna>" if species is None else grammar.render(species)
            ids = assert_prefix_outside_dna_mode(tokenizer, prefix)
            expected = FROZEN_PREFIX_TOKEN_IDS[grammar_name][condition]
            assert ids == expected, (
                f"pinned tokenizer IDs changed for {grammar_name}/{condition}: "
                f"expected {expected}, got {ids}"
            )
            observed[grammar_name][condition] = ids
    return observed
