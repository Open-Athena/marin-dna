# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""DNA dataset format for levanter, registered with the ``"dna"`` choice key.

Ported from ``marin-community/marin@dna-dev``:
``lib/levanter/src/levanter/data/text/formats.py`` (the ``DNALmDatasetFormat``
block only — other formats stay in upstream marin).

Importing this module triggers the ``@LmDatasetFormatBase.register_subclass("dna")``
decorator, making ``"dna"`` a valid choice for ``LmDatasetFormatBase`` consumers.
``marin_dna.levanter.__init__`` imports this module so any code that does
``import marin_dna.levanter`` activates the registration.
"""

from dataclasses import dataclass

from levanter.data._preprocessor import BatchProcessor
from levanter.data.text.formats import LmDatasetFormatBase
from levanter.tokenizers import MarinTokenizer

from marin_dna.levanter.batch_tokenizer import DNABatchTokenizer


@LmDatasetFormatBase.register_subclass("dna")
@dataclass(frozen=True)
class DNALmDatasetFormat(LmDatasetFormatBase):
    """Dataset configuration for DNA sequences with case-based loss weighting.

    Supports position-wise loss weighting based on character case:
    - Uppercase nucleotides (ACGT): weight = uppercase_weight
    - Lowercase nucleotides (acgt): weight = lowercase_weight

    Common use cases:
    - Repeat masking: lowercase_weight=0.01 (down-weight repetitive elements)
    - Functional positions only: uppercase_weight=1.0, lowercase_weight=0.0
    - Nonfunctional positions only: uppercase_weight=0.0, lowercase_weight=1.0

    Attributes:
        text_key: Field name containing the DNA sequence.
        uppercase_weight: Loss weight for uppercase positions.
        lowercase_weight: Loss weight for lowercase positions.
    """

    text_key: str = "seq"
    uppercase_weight: float = 1.0
    lowercase_weight: float = 1.0

    def build_preprocessor(
        self,
        tokenizer: MarinTokenizer,
        *,
        enforce_eos: bool = True,
        enforce_bos: bool = True,
    ) -> BatchProcessor[dict, dict]:
        del enforce_eos, enforce_bos
        return DNABatchTokenizer(
            tokenizer,
            text_field=self.text_key,
            uppercase_weight=self.uppercase_weight,
            lowercase_weight=self.lowercase_weight,
        )


def _install_dataset_for_component_patch() -> None:
    """Monkey-patch ``levanter.data.text.datasets.dataset_for_component`` to
    add a ``DNALmDatasetFormat`` branch.

    The ``register_subclass("dna")`` decorator above handles config-side
    parsing (so ``format: dna`` deserializes), but the train-time dispatch
    in ``dataset_for_component`` is a hard isinstance() chain over the four
    upstream-known format classes (``Text``, ``Supervised``, ``Chat``,
    ``Prebuilt``) and raises ``ValueError: Unknown format DNALmDatasetFormat``
    on anything else. The marin@dna-dev levanter fork adds a DNA branch
    inline; the released ``marin-levanter`` (currently ``0.99.dev20260516``)
    does not, so we patch it in here.

    The patch routes ``DNALmDatasetFormat`` through ``TokenSeqDataset`` with
    ``loss_weights_key="loss_weight"`` to match the per-token weights that
    ``DNABatchTokenizer`` writes into the cache (singular ``loss_weight`` key
    per ``DNABatchTokenizer.__call__`` line 121). The downstream
    ``CausalLmDataset`` then surfaces them as the model's per-position loss
    weights.

    Idempotent: re-importing this module re-applies the patch onto the
    already-patched function harmlessly (we wrap the *original*, not the
    current binding).
    """
    import levanter.data.text.datasets as _datasets

    if getattr(_datasets.dataset_for_component, "_marin_dna_patched", False):
        return  # already patched

    original = _datasets.dataset_for_component

    # Canary: if a future ``marin-levanter`` adds its own DNA branch to
    # ``dataset_for_component``, this monkey-patch becomes redundant (and would
    # silently shadow the upstream implementation). Surface that loudly so the
    # patch gets removed rather than quietly masking upstream behavior. Guarded
    # by a try/except so a source-unavailable build (e.g. zipimport) can't break
    # training over a diagnostic.
    try:
        import inspect

        if "DNALmDatasetFormat" in inspect.getsource(original):
            raise RuntimeError(
                "levanter.data.text.datasets.dataset_for_component now references "
                "DNALmDatasetFormat upstream — the marin_dna monkey-patch in "
                "marin_dna.levanter.formats is likely redundant. Verify and "
                "remove _install_dataset_for_component_patch()."
            )
    except OSError:
        pass  # source not available (zipimport / frozen); skip the canary

    def patched(component, Pos, cache, *, eos_id, block_cross_document_attention):
        fmt = component.format
        if isinstance(fmt, DNALmDatasetFormat):
            return _datasets.CausalLmDataset(
                _datasets.TokenSeqDataset(
                    cache, Pos.size, loss_weights_key="loss_weight"
                ),
                Pos,
                eos_id=eos_id,
                block_cross_document_attention=block_cross_document_attention,
            )
        return original(
            component,
            Pos,
            cache,
            eos_id=eos_id,
            block_cross_document_attention=block_cross_document_attention,
        )

    patched._marin_dna_patched = True  # type: ignore[attr-defined]
    _datasets.dataset_for_component = patched


_install_dataset_for_component_patch()
