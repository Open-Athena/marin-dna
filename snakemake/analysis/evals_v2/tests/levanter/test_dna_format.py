# Copyright The MarinDNA Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``marin_dna_evals.levanter`` glue (format registration + seq-len helper).

These cover behaviors that are marin-dna-specific (i.e. not just ports of
upstream marin tests).
"""

import pytest

pytest.importorskip(
    "levanter",
    reason="requires the marin runtime (not a core dep); see the marin-experiment skill",
)

from levanter.data.text.formats import LmDatasetFormatBase

from marin_dna_evals.levanter import (
    formats as _formats_module,
)
from marin_dna_evals.levanter.batch_tokenizer import DNABatchTokenizer
from marin_dna_evals.levanter.defaults import dna_effective_seq_len
from marin_dna_evals.levanter.formats import DNALmDatasetFormat

BOS_EOS_TOKENIZER = "bolinas-dna/tokenizer-char"
NO_BOS_EOS_TOKENIZER = "songlab/tokenizer-dna-clm"


def _tokenizer_available(name: str) -> bool:
    from levanter.tokenizers import load_tokenizer

    try:
        load_tokenizer(name)
    except Exception:  # noqa: BLE001 - any download/load failure means unavailable
        return False
    return True


def test_dna_format_registered_on_import():
    """Importing ``marin_dna_evals.levanter.formats`` activates ``register_subclass('dna')``."""
    choices = LmDatasetFormatBase.get_known_choices()
    assert "dna" in choices, f"'dna' not registered; got {sorted(choices)}"


def test_dna_format_round_trips_via_choice_registry():
    """The registered class must round-trip through ChoiceRegistry by key."""
    cls = LmDatasetFormatBase.get_choice_class("dna")
    assert cls is DNALmDatasetFormat


@pytest.mark.skipif(
    not _tokenizer_available(BOS_EOS_TOKENIZER),
    reason=f"Tokenizer {BOS_EOS_TOKENIZER} not accessible",
)
def test_dna_effective_seq_len_with_bos_eos():
    """255 bp base + BOS + EOS = 257 tokens."""
    assert dna_effective_seq_len(255, BOS_EOS_TOKENIZER) == 257


@pytest.mark.skipif(
    not _tokenizer_available(NO_BOS_EOS_TOKENIZER),
    reason=f"Tokenizer {NO_BOS_EOS_TOKENIZER} not accessible",
)
def test_dna_effective_seq_len_with_no_special_tokens():
    """No BOS/EOS → effective length matches base."""
    assert dna_effective_seq_len(255, NO_BOS_EOS_TOKENIZER) == 255


def test_dna_format_build_preprocessor_returns_dna_batch_tokenizer():
    """Format's ``build_preprocessor`` must return a ``DNABatchTokenizer`` instance.

    Uses a stub tokenizer so the test doesn't hit HF.
    """

    class _StubTokenizer:
        bos_token_id = None
        eos_token_id = None
        name_or_path = "stub"
        vocab_size = 0

        def as_hf_tokenizer(self):
            return None

    fmt = DNALmDatasetFormat(text_key="sequence", lowercase_weight=0.01)
    proc = fmt.build_preprocessor(_StubTokenizer())
    assert isinstance(proc, DNABatchTokenizer)
    assert proc.text_field == "sequence"
    assert proc.lowercase_weight == 0.01
    assert proc.uppercase_weight == 1.0


# ---------------------------------------------------------------------------
# dataset_for_component monkey-patch (the train-time dispatch fix)
# ---------------------------------------------------------------------------


def test_dataset_for_component_patch_installed_on_import():
    """Importing the formats module patches ``dataset_for_component``.

    The released ``marin-levanter`` doesn't know ``DNALmDatasetFormat`` at
    train-time dispatch; ``_install_dataset_for_component_patch`` (run on
    import) adds the branch. Marker attribute confirms it's our wrapper.
    """
    from levanter.data.text import datasets

    assert getattr(datasets.dataset_for_component, "_marin_dna_patched", False), (
        "dataset_for_component is not the marin_dna-patched wrapper"
    )


def test_dataset_for_component_patch_is_idempotent():
    """Re-running the installer must not double-wrap (wraps the *original*)."""
    from levanter.data.text import datasets

    before = datasets.dataset_for_component
    _formats_module._install_dataset_for_component_patch()
    after = datasets.dataset_for_component
    assert before is after, "installer re-wrapped an already-patched function"


def test_dataset_for_component_routes_dna_format(monkeypatch):
    """A ``DNALmDatasetFormat`` component routes through ``TokenSeqDataset``
    (with ``loss_weights_key='loss_weight'``) wrapped in ``CausalLmDataset``.

    Stubs ``TokenSeqDataset`` / ``CausalLmDataset`` so the test asserts the
    dispatch wiring without building a real cache.
    """
    from levanter.data.text import datasets

    captured: dict = {}

    class _StubTokenSeq:
        def __init__(self, cache, seq_len, loss_weights_key=None):
            captured["loss_weights_key"] = loss_weights_key
            captured["seq_len"] = seq_len

    class _StubCausal:
        def __init__(self, inner, Pos, *, eos_id, block_cross_document_attention):
            captured["inner"] = inner
            captured["eos_id"] = eos_id

    monkeypatch.setattr(datasets, "TokenSeqDataset", _StubTokenSeq)
    monkeypatch.setattr(datasets, "CausalLmDataset", _StubCausal)

    class _Pos:
        size = 256

    class _Component:
        format = DNALmDatasetFormat(text_key="sequence")

    out = datasets.dataset_for_component(
        _Component(),
        _Pos(),
        cache=object(),
        eos_id=7,
        block_cross_document_attention=True,
    )
    assert isinstance(out, _StubCausal)
    assert captured["loss_weights_key"] == "loss_weight", (
        "DNA branch must read the per-token 'loss_weight' key written by "
        "DNABatchTokenizer"
    )
    assert captured["seq_len"] == 256
    assert captured["eos_id"] == 7
    assert isinstance(captured["inner"], _StubTokenSeq)


def test_dataset_for_component_passes_through_non_dna(monkeypatch):
    """Non-DNA formats must delegate to the original implementation, not the
    DNA branch.

    We make the DNA-branch constructors explode; a non-DNA component must
    therefore reach ``original(...)`` instead. The original raises on a bare
    ``object()`` format (it's not one of the four upstream-known classes),
    which is exactly the delegation we want to prove — the error comes from
    upstream, not our ``_boom`` stubs.
    """
    from levanter.data.text import datasets

    def _boom(*a, **k):
        raise AssertionError("non-DNA format wrongly routed through DNA branch")

    monkeypatch.setattr(datasets, "TokenSeqDataset", _boom)
    monkeypatch.setattr(datasets, "CausalLmDataset", _boom)

    class _NonDnaComponent:
        format = object()  # not a DNALmDatasetFormat

    # If the wrapper wrongly took the DNA branch, _boom raises AssertionError.
    # Correct delegation reaches the original, which rejects the unknown format
    # with a non-AssertionError exception.
    with pytest.raises(Exception) as excinfo:
        datasets.dataset_for_component(
            _NonDnaComponent(),
            object(),
            cache=object(),
            eos_id=0,
            block_cross_document_attention=False,
        )
    assert not isinstance(excinfo.value, AssertionError), (
        "non-DNA format was routed through the DNA branch instead of delegating"
    )


def test_patch_canary_does_not_false_fire_on_current_upstream():
    """The 'upstream added a DNA branch' canary must not trip on the pinned
    marin-levanter (which has no DNA branch). If this fails, either upstream
    changed (good — drop the patch) or the canary logic regressed."""
    import inspect

    from levanter.data.text import datasets

    # The currently-installed fn is our wrapper; the canary inspects the
    # *original*. Re-running the installer is a no-op (idempotent) and must not
    # raise — which it would if the canary saw 'DNALmDatasetFormat' in upstream.
    _formats_module._install_dataset_for_component_patch()  # must not raise
    # Belt-and-suspenders: confirm our marker is the only DNA reference path.
    assert getattr(datasets.dataset_for_component, "_marin_dna_patched", False)
    # Sanity that inspect.getsource works in this env (the canary depends on it).
    assert inspect.getsource(datasets.dataset_for_component)
