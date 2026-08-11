import pandas as pd
from marin_dna_zoonomia_projection.cli.calibrate_447m_threshold import (
    _build_defined_intervals_per_chrom,
)


def test_build_defined_intervals_complements_undefined_regions() -> None:
    chrom_sizes = pd.DataFrame({"chrom": ["1"], "size": [10]})
    undefined = pd.DataFrame(
        {
            "chrom": ["1", "1"],
            "start": [2, 7],
            "end": [4, 9],
        }
    )

    result = _build_defined_intervals_per_chrom(chrom_sizes, undefined, ["1"])

    assert result["1"].to_dict("records") == [
        {"chrom": "1", "start": 0, "end": 2},
        {"chrom": "1", "start": 4, "end": 7},
        {"chrom": "1", "start": 9, "end": 10},
    ]
