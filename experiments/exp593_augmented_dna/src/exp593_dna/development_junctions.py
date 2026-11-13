"""Freeze blind splice-motif recovery and matched random controls on disjoint DNA."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from exp593_dna.annotations import Landmark, RowAnnotations, load_annotations
from exp593_dna.audit import file_sha256
from exp593_dna.fim_validation import prompt_and_target
from exp593_dna.inputs import INPUTS_MANIFEST_SHA256
from exp593_dna.pipeline import (
    Encoder,
    Example,
    RawWindow,
    Span,
    load_encoder,
    reverse_complement,
    rng_for,
)

CANONICAL_MOTIFS = {"splice_donor": "GT", "splice_acceptor": "AG"}


def junction_pair(
    raw: RawWindow,
    annotations: RowAnnotations,
    site: Landmark,
    encoder: Encoder,
    *,
    seed: int = 593,
) -> tuple[Example, Example]:
    """Do not reveal the feature class in either rendered prompt."""
    if raw.split != "validation" or annotations.row != raw.row:
        raise ValueError("Junction development requires the frozen validation row")
    annotations.verify(raw.seq, "validation")
    if (
        site not in annotations.landmarks
        or site.kind not in CANONICAL_MOTIFS
        or site.end - site.start != 2
    ):
        raise ValueError("Expected an annotated two-base splice motif")
    if raw.seq[site.start : site.end].upper() != CANONICAL_MOTIFS[site.kind]:
        raise ValueError("Splice motif is noncanonical in the source orientation")
    if not 192 <= site.start < site.end <= 8000:
        raise ValueError("Junction needs at least 128 bases per flank after cropping")
    rng = rng_for(seed, raw.row, "junction_development")
    rc = bool(rng.integers(2))
    oriented = reverse_complement(raw.seq) if rc else raw.seq
    dna = oriented[64:8128].lower()
    gap = (8192 - site.end if rc else site.start) - 64
    starts = np.arange(128, len(dna) - 2 - 128 + 1)
    eligible = np.ones(len(starts), dtype=bool)
    for feature in annotations.landmarks:
        left, right = (
            (8192 - feature.end, 8192 - feature.start)
            if rc
            else (feature.start, feature.end)
        )
        left -= 64
        right -= 64
        eligible &= (starts + 2 < left - 32) | (starts > right + 32)
    eligible &= np.asarray([set(dna[i : i + 2]) <= set("acgt") for i in starts])
    if not np.any(eligible):
        raise ValueError("No matched random control outside landmark neighborhoods")
    control = int(rng.choice(starts[eligible]))
    result = []
    for task, start in ((site.kind, gap), ("random_control", control)):
        spans = (
            Span("[left] ", "text"),
            Span(dna[:start], "left"),
            Span("\n[right] ", "text"),
            Span(dna[start + 2 :], "right"),
            Span("\n[gap:2]\n[fill] ", "text"),
            Span(dna[start : start + 2], "middle", True),
        )
        tokens, weights = encoder.encode(spans)
        if len(tokens) > 8192:
            raise ValueError("Junction presentation overflows its context")
        example = Example(
            tokens,
            weights,
            spans,
            8192,
            {
                "kind": "short_fim",
                "presentation": "minimal",
                "gap_length": 2,
                "gap_start": start,
                "annotation_guided": task != "random_control",
                "development_task": task,
                "source_class": site.kind,
                "raw_row": raw.row,
                "assembly": raw.assembly,
                "rc": rc,
                "case": "unmasked_lower",
                "crop_start_in_oriented_window": 64,
                "dna_tokens": len(dna),
                "input_tokens": len(tokens),
                "use": "evaluation_only",
            },
        )
        prompt_and_target(example)
        result.append(example)
    return result[0], result[1]


def prepare_junctions(
    partitions: Path,
    annotations: Path,
    holdout: Path,
    fim_pairs: Path,
    inputs_manifest: Path,
    output: Path,
    *,
    per_class: int = 32,
) -> dict[str, Any]:
    if file_sha256(inputs_manifest) != INPUTS_MANIFEST_SHA256:
        raise ValueError("Frozen source input manifest changed")
    manifest = json.loads(inputs_manifest.read_text())
    for name, path in (
        ("annotations/validation.parquet", annotations),
        ("audit/validation_disjoint_512.json", holdout),
        ("audit/fim_validation_pairs.jsonl", fim_pairs),
    ):
        if file_sha256(path) != manifest["files"][name]["sha256"]:
            raise ValueError("Frozen held-out source identity changed")
    selected = json.loads(holdout.read_text())["examples"]
    if len(selected) != 512 or per_class < 1:
        raise ValueError("Expected the fixed disjoint source panel")
    hashes = {
        r["source_row"]: r["source_sequence_sha256"]
        for r in (json.loads(line) for line in fim_pairs.read_text().splitlines())
    }
    raw_by_row = {}
    for code in sorted({r["assembly"] for r in selected}):
        chosen = {r["row"]: r for r in selected if r["assembly"] == code}
        for batch in pq.ParquetFile(
            partitions / "validation" / f"{code}.parquet"
        ).iter_batches():
            subset = pc.filter(
                batch,
                pc.is_in(
                    batch.column(batch.schema.get_field_index("row")),
                    value_set=pa.array(sorted(chosen)),
                ),
            )
            for record in subset.to_pylist():
                raw = RawWindow.from_dict(record)
                if (
                    raw.split != "validation"
                    or raw.row in raw_by_row
                    or hashlib.sha256(raw.seq.encode()).hexdigest() != hashes[raw.row]
                    or any(getattr(raw, k) != v for k, v in chosen[raw.row].items())
                ):
                    raise ValueError(
                        "Original held-out row metadata/DNA identity changed"
                    )
                raw_by_row[raw.row] = raw
    if len(raw_by_row) != 512:
        raise ValueError("Missing held-out source DNA")
    bank = load_annotations(annotations, split="validation", rows=set(raw_by_row))
    pools: dict[str, list[tuple[str, int, Landmark]]] = defaultdict(list)
    for row, raw in raw_by_row.items():
        if row not in bank:
            continue
        bank[row].verify(raw.seq, "validation")
        seen = set()
        for site in bank[row].landmarks:
            key = (site.kind, site.start, site.end)
            if key in seen:
                continue
            seen.add(key)
            if (
                site.kind in CANONICAL_MOTIFS
                and site.end - site.start == 2
                and 192 <= site.start < site.end <= 8000
                and raw.seq[site.start : site.end].upper()
                == CANONICAL_MOTIFS[site.kind]
            ):
                rank = hashlib.sha256(
                    f"593:{row}:{site.kind}:{site.start}".encode()
                ).hexdigest()
                pools[site.kind].append((rank, row, site))
    for pool in pools.values():
        pool.sort(reverse=True)
    eligible_counts = {k: len(v) for k, v in pools.items()}
    encoder = load_encoder()
    chosen_rows = set()
    counts: Counter[str] = Counter()
    records = []
    for _ in range(per_class):
        for kind in CANONICAL_MOTIFS:
            while pools[kind] and pools[kind][-1][1] in chosen_rows:
                pools[kind].pop()
            if not pools[kind]:
                raise ValueError("Insufficient independent canonical junction rows")
            _, row, site = pools[kind].pop()
            raw = raw_by_row[row]
            pair = junction_pair(raw, bank[row], site, encoder)
            chosen_rows.add(row)
            counts[kind] += 1
            counts["rc_pairs"] += int(pair[0].attributes["rc"])
            records.append(
                {
                    "source_row": row,
                    "source_sequence_sha256": hashes[row],
                    "annotation": asdict(site),
                    "members": [
                        {
                            "bucket": e.bucket,
                            "attributes": e.attributes,
                            "spans": [asdict(s) for s in e.spans],
                        }
                        for e in pair
                    ],
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    path = output / "junction_development.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    report = {
        "purpose": "evaluation_only; never training or annotation guidance",
        "lm_inputs_manifest_sha256": INPUTS_MANIFEST_SHA256,
        "heldout_manifest_sha256": file_sha256(holdout),
        "annotation_bank_sha256": file_sha256(annotations),
        "tokenizer_sha256": encoder.digest,
        "source_rows": len(chosen_rows),
        "species": len({raw_by_row[r].assembly for r in chosen_rows}),
        "eligible_unique_sites": eligible_counts,
        "counts": dict(counts),
        "presentations": len(records) * 2,
        "panel_sha256": file_sha256(path),
        "source_policy": "Only the existing 512 source-disjoint validation windows; one source row per guided example. Two-base GT/AG source motifs; blind minimal FIM prompts; uniform lowercase, fixed RC Bernoulli draw, 8064-base shared crop; random controls avoid all known landmark neighborhoods by more than 32 bases.",
        "scoring_protocol": "Freeze per-example gap NLL, teacher-forced base accuracy, greedy case-sensitive and case-folded exact match, separated by donor, acceptor and random-control task. Do not use teacher-forced span agreement as greedy exact match.",
    }
    (output / "junction_development_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "partitions",
        "annotations",
        "holdout",
        "fim-pairs",
        "inputs-manifest",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=32)
    print(json.dumps(prepare_junctions(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
