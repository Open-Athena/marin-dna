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
