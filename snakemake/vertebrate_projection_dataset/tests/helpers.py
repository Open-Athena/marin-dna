from __future__ import annotations

import polars as pl
from marin_dna_vertebrate_projection.manifest import (
    select_family_representatives,
)


def species_manifest() -> pl.DataFrame:
    rows = [
        {
            "alignment_name": "Mus_musculus",
            "scientific_name": "Mus musculus",
            "assembly": "GRCm38",
            "taxonomy_id": 10090,
            "family": "Muridae",
            "clade": "mammals",
            "phylogenetic_rank": 1,
            "backend": "zoonomia_cactus",
            "selection_priority": 10,
            "assembly_level": "Chromosome",
            "contig_n50": 100,
        },
        {
            "alignment_name": "Felis_catus",
            "scientific_name": "Felis catus",
            "assembly": "Felis_catus_9.0",
            "taxonomy_id": 9685,
            "family": "Felidae",
            "clade": "mammals",
            "phylogenetic_rank": 1,
            "backend": "zoonomia_cactus",
            "selection_priority": 10,
            "assembly_level": "Chromosome",
            "contig_n50": 90,
        },
        {
            "alignment_name": "galGal4",
            "scientific_name": "Gallus gallus",
            "assembly": "Gallus_gallus-4.0",
            "taxonomy_id": 9031,
            "family": "Phasianidae",
            "clade": "birds",
            "phylogenetic_rank": 2,
            "backend": "ucsc_multiz100way",
            "selection_priority": 10,
            "assembly_level": "Chromosome",
            "contig_n50": 80,
        },
        {
            "alignment_name": "melGal1",
            "scientific_name": "Meleagris gallopavo",
            "assembly": "Turkey_2.01",
            "taxonomy_id": 9103,
            "family": "Phasianidae",
            "clade": "birds",
            "phylogenetic_rank": 2,
            "backend": "ucsc_multiz100way",
            "selection_priority": 5,
            "assembly_level": "Scaffold",
            "contig_n50": 70,
        },
        {
            "alignment_name": "xenTro7",
            "scientific_name": "Xenopus tropicalis",
            "assembly": "JGI_7.0",
            "taxonomy_id": 8364,
            "family": "Pipidae",
            "clade": "amphibians",
            "phylogenetic_rank": 4,
            "backend": "ucsc_multiz100way",
            "selection_priority": 10,
            "assembly_level": "Chromosome",
            "contig_n50": 60,
        },
    ]
    return select_family_representatives(pl.DataFrame(rows))
