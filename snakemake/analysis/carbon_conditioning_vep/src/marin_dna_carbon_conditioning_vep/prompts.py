"""Carbon metadata prompt rendering and tokenizer-boundary validation."""

from dataclasses import dataclass
from typing import Any

CONDITION_ORDER = ("untagged", "correct", "near_wrong", "far_wrong")


@dataclass(frozen=True)
class PromptGrammar:
    """Named metadata scaffold whose format field accepts ``species``."""

    name: str
    template: str

    def render(self, species: str) -> str:
        """Render one tagged prefix, including the DNA-mode opener."""
        assert species and species.strip() == species, (
            f"invalid species value {species!r}"
        )
        prefix = self.template.format(species=species)
        assert prefix.endswith("<dna>"), f"grammar {self.name!r} must end with <dna>"
        assert prefix.count("<dna>") == 1, (
            f"grammar {self.name!r} has multiple <dna> tags"
        )
        return prefix


def render_condition_prefix(
    condition: str,
    conditions: dict[str, str | None],
    grammar: PromptGrammar,
) -> str:
    """Render the frozen prefix for one VEP prompt condition."""
    assert condition in CONDITION_ORDER, f"unknown prompt condition {condition!r}"
    assert tuple(conditions) == CONDITION_ORDER, (
        f"condition order must be {CONDITION_ORDER}, got {tuple(conditions)}"
    )
    species = conditions[condition]
    if species is None:
        assert condition == "untagged", "only the untagged condition may omit species"
        return "<dna>"
    return grammar.render(species)


def encode_prefix(tokenizer: Any, prefix: str) -> tuple[list[int], list[int]]:
    """Encode a prefix and return IDs plus Carbon's token-mode mask."""
    encoded = tokenizer(
        prefix,
        add_special_tokens=False,
        return_token_mask=True,
    )
    ids = list(encoded["input_ids"])
    mask = list(encoded["token_mask"])
    assert len(ids) == len(mask) and ids, "empty or malformed encoded prefix"
    return ids, mask


def assert_prefix_outside_dna_mode(tokenizer: Any, prefix: str) -> list[int]:
    """Assert metadata tokens are text tokens and only ``<dna>`` opens DNA mode."""
    ids, mask = encode_prefix(tokenizer, prefix)
    dna_begin_id = int(tokenizer.dna_begin_token_id)
    assert ids.count(dna_begin_id) == 1, (
        f"prefix must contain one DNA begin token {dna_begin_id}, got IDs {ids}"
    )
    dna_index = ids.index(dna_begin_id)
    assert dna_index == len(ids) - 1, f"<dna> must be the final prefix token: {ids}"
    assert mask[dna_index] == 0, f"<dna> token mask must be 0, got {mask}"
    assert all(value == -1 for value in mask[:dna_index]), (
        f"metadata escaped text mode before <dna>: IDs={ids}, mask={mask}"
    )
    return ids


def truncate_dna_to_kmer_boundary(sequence: str, kmer_size: int = 6) -> str:
    """Apply Carbon's official deterministic right truncation before tokenization."""
    assert kmer_size > 0
    n = (len(sequence) // kmer_size) * kmer_size
    return sequence[:n]


def render_prompt(prefix: str, sequence: str, kmer_size: int = 6) -> str:
    """Render one complete Carbon input after canonical-base validation."""
    normalized = sequence.upper()
    assert normalized and set(normalized) <= set("ACGT"), (
        "Carbon token scoring requires non-empty canonical A/C/G/T sequence"
    )
    return prefix + truncate_dna_to_kmer_boundary(normalized, kmer_size)


def assert_allele_token_count_parity(
    tokenizer: Any,
    prefix: str,
    ref_sequence: str,
    alt_sequence: str,
    kmer_size: int = 6,
) -> int:
    """Assert REF and ALT prompts within a condition have identical token counts."""
    ref_ids = tokenizer.encode(
        render_prompt(prefix, ref_sequence, kmer_size), add_special_tokens=False
    )
    alt_ids = tokenizer.encode(
        render_prompt(prefix, alt_sequence, kmer_size), add_special_tokens=False
    )
    assert len(ref_ids) == len(alt_ids), (
        f"REF/ALT token-count mismatch under {prefix!r}: {len(ref_ids)} != {len(alt_ids)}"
    )
    return len(ref_ids)
