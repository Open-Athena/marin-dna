import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
import pytest

from exp593_dna.annotations import load_annotations
from exp593_dna.build_annotations import map_reference
from exp593_dna.partition_inputs import partition_split
from exp593_dna.pipeline import reverse_complement


def test_metadata_join_preserves_case_and_rejects_row_reordering(tmp_path):
    corpus, metadata, output = (tmp_path / x for x in ("corpus", "metadata", "output"))
    for root in (corpus, metadata):
        (root / "train").mkdir(parents=True)
    dna = ["AaCG" * 2048, "tGcA" * 2048]
    name = "original.parquet"
    pq.write_table(pa.table({"seq": dna}), corpus / "train" / name)
    record = {"uri": "s3://original/train/" + name, "etag": '"immutable"', "rows": 2}
    (corpus / "train" / "original.source.json").write_text(
        json.dumps({**record, "sha256": "source-binding"})
    )
    rows = [
        {
            "split": "train",
            "row": i,
            "assembly": code,
            "chrom": "1",
            "start": 0,
            "end": 12000,
            "strand": "+",
        }
        for i, code in enumerate(("Zmays", "Athaliana"))
    ]
    pq.write_table(pa.Table.from_pylist(rows), metadata / "train" / name)
    (metadata / "train" / "original.sample.jsonl").write_text(
        "".join(json.dumps({"row": i, "seq": seq}) + "\n" for i, seq in enumerate(dna))
    )
    report = partition_split(
        "train", [record], corpus, metadata, output, expected_rows=2
    )
    assert report["rows"] == report["verified_raw_samples"] == 2
    assert pq.read_table(output / "train/Zmays.parquet")["seq"][0].as_py() == dna[0]
    pq.write_table(pa.Table.from_pylist(rows[::-1]), metadata / "train" / name)
    with pytest.raises(ValueError, match="order/split"):
        partition_split(
            "train", [record], corpus, metadata, tmp_path / "bad", expected_rows=2
        )


def test_reference_mapping_preserves_stored_strand_and_feature_coordinates(tmp_path):
    rng = np.random.default_rng(42)
    genome = "".join(rng.choice(list("ACGT"), size=12000))
    fasta = tmp_path / "reference.fa"
    fasta.write_text(">chr1\n" + genome + "\n")
    pysam.faidx(str(fasta))
    gff = tmp_path / "features.gff3"
    gff.write_text(
        "##gff-version 3\n"
        "chr1\tref\tCDS\t5501\t5550\t.\t+\t0\tParent=plus_tx\n"
        "chr1\tref\tCDS\t5581\t5600\t.\t+\t1\tParent=plus_tx\n"
        "chr1\tref\texon\t5401\t5550\t.\t+\t.\tParent=plus_tx\n"
        "chr1\tref\texon\t5581\t5700\t.\t+\t.\tParent=plus_tx\n"
        "chr1\tref\tCDS\t8001\t8100\t.\t-\t0\tParent=minus_tx\n"
    )
    part = tmp_path / "parts/train"
    part.mkdir(parents=True)
    rows = [
        {
            "split": "train",
            "row": i,
            "assembly": "Zmays",
            "chrom": "1",
            "start": 0,
            "end": 12000,
            "strand": strand,
            "seq": seq,
        }
        for i, (strand, seq) in enumerate(
            [
                ("+", genome[:8192]),
                ("-", reverse_complement(genome[-8192:])),
            ]
        )
    ]
    pq.write_table(pa.Table.from_pylist(rows), part / "Zmays.parquet")
    config = {
        "assembly_code": "Zmays",
        "accession": "verified-v1",
        "fasta": str(fasta),
        "assets": {"gff": {"path": str(gff)}},
    }
    out = tmp_path / "sidecars"
    report = map_reference(config, tmp_path / "parts", out, splits=("train",))
    assert report["counts"]["train/exact_DNA_mapping"] == 2
    loaded = load_annotations(out / "Zmays.train.jsonl")
    loaded[0].verify(rows[0]["seq"])
    loaded[1].verify(rows[1]["seq"])
    plus = {
        (m.kind, m.start, m.end)
        for m in loaded[0].landmarks
        if m.feature_id == "plus_tx"
    }
    assert ("translation_start", 5500, 5503) in plus
    assert ("splice_donor", 5550, 5552) in plus
    assert ("splice_acceptor", 5578, 5580) in plus
    minus = {
        (m.kind, m.start, m.end)
        for m in loaded[1].landmarks
        if m.feature_id == "minus_tx"
    }
    assert ("translation_start", 3900, 3903) in minus
    assert (
        loaded[1].sequence_sha256 == hashlib.sha256(rows[1]["seq"].encode()).hexdigest()
    )


def test_trans_spliced_transcripts_are_skipped_without_dropping_other_genes(tmp_path):
    from collections import Counter

    from exp593_dna.build_annotations import parse_features

    path = tmp_path / "features.gff"
    path.write_text(
        "chr1\tRefSeq\tCDS\t101\t200\t.\t+\t0\tParent=normal\n"
        "chr1\tRefSeq\texon\t301\t400\t.\t-\t.\tParent=trans;exception=trans-splicing\n"
        "chr1\tRefSeq\tCDS\t301\t400\t.\t-\t0\tParent=trans\n"
        "chr1\tRefSeq\texon\t501\t600\t.\t+\t.\tParent=trans;exception=trans-splicing\n"
    )
    statistics = Counter()
    result = parse_features(path, {"1": "chr1"}, "reference-v1", statistics)
    assert {m.feature_id for m in result["chr1"]} == {"normal"}
    assert {m.kind for m in result["chr1"]} == {"translation_start", "translation_stop"}
    assert statistics["unsupported_trans_splicing_or_conflicting_transcripts"] == 1
