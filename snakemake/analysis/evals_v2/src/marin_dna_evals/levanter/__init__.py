"""DNA-specific glue for marin/levanter.

Importing this module triggers the
``@LmDatasetFormatBase.register_subclass`` decorators inside ``formats.py``,
making ``"dna"`` and ``"rag_dna"`` valid choices for
``levanter.data.text.formats.LmDatasetFormatBase`` consumers.
"""

from marin_dna_evals.levanter import formats  # noqa: F401  side-effect: register formats
