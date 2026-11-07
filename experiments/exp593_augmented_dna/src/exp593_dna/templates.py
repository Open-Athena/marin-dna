"""Finite, curated semantic components combined into open-ended presentations."""

from __future__ import annotations

import json
from itertools import product
from typing import Any

import numpy as np


def phrase_catalog() -> dict[str, list[str]]:
    """Compile approved grammatical choices; never generate factual prose online."""
    mask = [
        f"{lead}{subject} {verb} {case}."
        for lead, subject, verb, case in product(
            ["", "For this sequence, ", "Assume that ", "In this DNA, "],
            ["repeat-masked bases", "softmasked repeats", "masked repeat nucleotides"],
            ["are written in", "are represented in", "use"],
            ["lowercase", "lower case"],
        )
    ]
    mask += [
        "Repeat mask: lowercase nucleotides.",
        "Softmask includes repeats as lower case nucleotides.",
        "Assume repeat masks are lower case.",
        "Lowercase bases carry the native PlantCAD repeat mask.",
    ]
    unmasked = {
        mode: [
            f"{lead}{noun} {verb} {case}; repeat-mask information is not encoded by case."
            for lead, noun, verb, case in product(
                ["", "For this input, ", "In the presented sequence, "],
                ["all bases", "all nucleotides"],
                ["use", "are written in", "are represented in"],
                variants,
            )
        ]
        for mode, variants in {
            "unmasked_lower": ["lowercase", "lower case"],
            "unmasked_upper": ["uppercase", "upper case"],
        }.items()
    }
    causal = [
        f"{lead}{description}{tail}"
        for lead, description, tail in product(
            ["", "Here is ", "The input is ", "Consider "],
            ["a DNA sequence", "a nucleotide sequence", "a genomic DNA fragment"],
            [".", ":", " for sequence modeling.", " from a plant genome."],
        )
    ]
    fim = [
        f"{lead}{verb} {noun}{context}"
        for lead, verb, noun, context in product(
            [
                "",
                "Using the context, ",
                "For this fragment, ",
                "With both flanks available, ",
            ],
            ["Fill in", "Recover", "Predict", "Complete", "Infer"],
            [
                "the missing DNA",
                "the omitted bases",
                "the missing nucleotides",
                "the gap",
            ],
            [
                " using both flanks.",
                " from the left and right context.",
                ".",
                " using the surrounding sequence.",
            ],
        )
    ]
    return {
        "native_softmask": mask,
        **unmasked,
        "causal_description": causal,
        "fim_instruction": fim,
        "gap_length": [
            "The missing span contains {length} {unit}.",
            "Recover exactly {length} {unit}.",
            "Gap length: {length}.",
            "Fill a gap of {length} {unit}.",
            "Expected answer length = {length} {unit}.",
            "The omitted DNA is {length} {unit} long.",
            "Number of missing nucleotides: {length}.",
            "Return {length} missing {unit}.",
        ],
        "left_label": [
            "Left flank",
            "Upstream context in the presented sequence",
            "Prefix",
            "Sequence before the gap",
            "Left DNA",
            "5-prime flank in this presentation",
        ],
        "right_label": [
            "Right flank",
            "Downstream context in the presented sequence",
            "Suffix",
            "Sequence after the gap",
            "Right DNA",
            "3-prime flank in this presentation",
        ],
        "answer_label": [
            "Missing bases",
            "Fill",
            "Answer",
            "Gap sequence",
            "Recovered DNA",
            "Middle",
        ],
        "dna_label": [
            "DNA",
            "Sequence",
            "Nucleotides",
            "Genomic fragment",
            "Input DNA",
            "Bases",
        ],
    }


KEYS: dict[str, list[str]] = {
    "species": [
        "species",
        "organism",
        "taxon name",
        "source species",
        "plant species",
        "organism name",
    ],
    "taxonomy": [
        "taxonomy",
        "taxonomic lineage",
        "classification",
        "lineage",
        "taxon path",
        "plant lineage",
    ],
    "chrom": [
        "chromosome",
        "sequence record",
        "contig/chromosome",
        "source record",
        "chrom",
        "genomic record",
    ],
    "assembly": [
        "dataset assembly code",
        "source assembly label",
        "assembly code",
        "corpus genome code",
        "genome label",
        "assembly label",
    ],
    "source_strand": [
        "recorded source strand",
        "source strand annotation",
        "stored strand metadata",
        "source feature strand",
        "record strand",
        "strand annotation",
    ],
    "orientation": [
        "presentation orientation",
        "orientation relative to stored window",
        "sequence transformation",
        "input orientation",
        "stored-window transformation",
        "presentation transform",
    ],
}
STRUCTURES = (
    "lines",
    "inline",
    "json",
    "xml",
    "fasta",
    "prose",
    "brackets",
    "comments",
)


def pick(rng: np.random.Generator, choices: list[str] | tuple[str, ...]) -> str:
    """Draw one component with a reproducible integer choice."""
    return choices[int(rng.integers(len(choices)))]


def render_metadata(
    fields: dict[str, str], rng: np.random.Generator
) -> tuple[str, str]:
    """Randomize fields, labels and syntax without changing their values."""
    structure = pick(rng, STRUCTURES)
    items = [(pick(rng, KEYS[k]), v) for k, v in fields.items()]
    rng.shuffle(items)
    if not items:
        return "", structure
    if structure == "json":
        text = json.dumps(dict(items), ensure_ascii=False)
    elif structure == "xml":
        from xml.sax.saxutils import escape

        text = (
            "<metadata>"
            + "".join(
                f"<{k.replace(' ', '_').replace('/', '_')}>{escape(v)}</{k.replace(' ', '_').replace('/', '_')}>"
                for k, v in items
            )
            + "</metadata>"
        )
    elif structure == "prose":
        text = " ".join(f"The {k} is {v}." for k, v in items)
    else:
        separator = {
            "lines": "\n",
            "inline": "; ",
            "fasta": " | ",
            "brackets": " ",
            "comments": "\n",
        }[structure]
        delimiter = pick(rng, [": ", "=", " = ", " -> "])
        entries = [f"{k}{delimiter}{v}" for k, v in items]
        if structure == "brackets":
            entries = [f"[{s}]" for s in entries]
        if structure == "comments":
            entries = [f"# {s}" for s in entries]
        text = separator.join(entries)
        if structure == "fasta":
            text = ">" + text
    return text + "\n", structure


def catalog_manifest() -> dict[str, Any]:
    """Describe the finite vocabulary, distinct from complete presentations."""
    catalog = phrase_catalog()
    return {
        "components": catalog,
        "metadata_keys": KEYS,
        "metadata_structures": list(STRUCTURES),
        "fim_layouts": ["instruction_first", "instruction_after_flanks", "compact"],
        "component_count": sum(map(len, catalog.values()))
        + sum(map(len, KEYS.values())),
        "composition": "Independent component choices, field subsets/order, metadata syntax, lineage depth/name form, gap, crop, length, case availability and RC compose complete presentations. No runtime LLM calls.",
    }
