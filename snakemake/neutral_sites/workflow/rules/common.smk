import pandas as pd

from marin_dna.data.genome import Genome
from marin_dna.pipelines.evals.conservation import CONSERVATION_TRACKS
from marin_dna.pipelines.neutral_sites.sites import (
    annotate_pentanucleotide,
    enumerate_positions,
    filter_acgt_window_sites,
    parse_rmsk,
    scan_neutral_intervals,
    subsample_per_context,
)

CHROMS = [str(c) for c in config["chroms"]]

# Resolve the conservation-track URLs from the single source of truth. Fail
# fast at parse time on an unknown track key (typo in config).
for _key in (config["phylop_track"], config["phastcons_track"]):
    assert (
        _key in CONSERVATION_TRACKS
    ), f"unknown conservation track {_key!r}; known: {sorted(CONSERVATION_TRACKS)}"
PHYLOP_URL = CONSERVATION_TRACKS[config["phylop_track"]]
PHASTCONS_URL = CONSERVATION_TRACKS[config["phastcons_track"]]
