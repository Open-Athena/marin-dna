"""Build the one-genome-per-Metazoan-order animal species list for issue #353.

Re-deduplicates the committed family-level genome universe
(``dataset_creation/config/genomes.parquet``, itself the annotated-Metazoa
RefSeq output of the ``genome_selection`` pipeline) down to one best-quality
genome per NCBI *order*, and writes the committed TSV wired into
``dataset_creation`` as the ``animals_order204`` genome set (Arm A / Arm B
target species for the CDS projection-vs-annotation experiment).

The universe already has the ``--annotated`` / size / exclude filters baked in,
so this is only the coarser order-level dedup; ``select_one_per_rank`` re-applies
the same priority ranking used to build the universe.

Reproduce:
    uv run python scripts/issue353_build_animal_order_species.py
"""

from pathlib import Path

import pandas as pd

from marin_dna.pipelines.training_dataset.genome_selection import (
    select_one_per_rank,
)

# Same priority genomes as the genome_selection pipeline config, so each order's
# winner matches the family universe it is drawn from (keeps human as the
# Primates representative, etc.).
PRIORITY = [
    "GCF_000001405.40",  # Homo sapiens
    "GCF_000001635.27",  # Mus musculus
    "GCF_016699485.2",  # Gallus gallus
    "GCF_000001215.4",  # Drosophila melanogaster
    "GCF_049306965.1",  # Danio rerio
    "GCF_000002985.6",  # Caenorhabditis elegans
]

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "snakemake/training_dataset/dataset_creation/config"
GENOMES = CONFIG / "genomes.parquet"
OUT = CONFIG / "animals_order204.tsv"

COLS = [
    "Assembly Accession",
    "Organism Name",
    "order",
    "class",
    "phylum",
    "Assembly Level",
    "Assembly Stats Total Sequence Length",
]


def main() -> None:
    genomes = pd.read_parquet(GENOMES)
    picked = select_one_per_rank(genomes, "order", priority=PRIORITY)
    out = picked[COLS].sort_values(["phylum", "class", "order"]).reset_index(drop=True)
    assert out["Assembly Accession"].is_unique
    assert out["order"].is_unique
    assert "GCF_000001405.40" in set(out["Assembly Accession"]), (
        "human (Primates representative) missing from the order set"
    )
    OUT.write_text(out.to_csv(sep="\t", index=False))
    print(f"wrote {len(out)} rows -> {OUT.relative_to(REPO)}")
    print(f"phyla: {out['phylum'].nunique()}  classes: {out['class'].nunique()}")
    print(out["phylum"].value_counts().to_string())


if __name__ == "__main__":
    main()
