"""DNA-specific glue for marin/levanter.

Importing this module triggers the
``@LmDatasetFormatBase.register_subclass("dna")`` decorator inside
``formats.py``, making ``"dna"`` a valid choice for
``levanter.data.text.formats.LmDatasetFormatBase`` consumers.
"""

from marin_dna.levanter import formats  # noqa: F401  side-effect: register "dna"
