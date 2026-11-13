import hashlib
import json
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from exp593_dna.annotations import (
    Landmark,
    RowAnnotations,
    candidate_starts,
    choose_guided_gap,
    load_annotations,
)
from exp593_dna.freeze_annotations import freeze_sidecar
from exp593_dna.pipeline import (
    ALPHABET,
    DATASET_REVISION,
    AugmentationConfig,
    Augmenter,
    Encoder,
    RawWindow,
)


def test_gap_coordinates_reverse_complement_and_neighborhood():
    site = Landmark(1000, 1003, "translation_start", "reference-v1")
    selected = choose_guided_gap(
        (site,),
        np.random.default_rng(1),
        reverse_complemented=True,
        crop_start=2000,
        dna_length=6000,
        gap_length=3,
        guided_probability=1,
        overlap_probability=1,
    )
    assert selected.start == 8192 - 1003 - 2000
    assert selected.landmark_class == "translation_start" and selected.guided
    for length in (1, 2, 3):
        starts = candidate_starts(1000, 1003, length, 8192, "nearby")
        assert starts
        for start in starts:
            assert (968 <= start and start + length <= 1000) or (
                1003 <= start and start + length <= 1035
            )
    assert candidate_starts(0, 3, 3, 2048, "overlap") == ()


def test_classes_balanced_before_sites_and_random_controls():
    features = (
        Landmark(1000, 1003, "translation_start", "reference-v1"),
        Landmark(6000, 6003, "translation_stop", "reference-v1"),
        *(
            Landmark(1500 + i * 30, 1502 + i * 30, "splice_donor", "reference-v1")
            for i in range(100)
        ),
    )
    counts = Counter()
    for seed in range(1800):
        choice = choose_guided_gap(
            features,
            np.random.default_rng(seed),
            reverse_complemented=False,
            crop_start=0,
            dna_length=8192,
            gap_length=2,
            guided_probability=1,
        )
        assert choice.guided and choice.eligible
        counts[choice.landmark_class] += 1
    assert set(counts) == {"translation_start", "translation_stop", "splice_donor"}
    assert all(0.29 < n / 1800 < 0.38 for n in counts.values())
    random = choose_guided_gap(
        features,
        np.random.default_rng(0),
        reverse_complemented=False,
        crop_start=0,
        dna_length=8192,
        gap_length=2,
        guided_probability=0,
    )
    assert random.eligible and not random.guided and random.start is None
    assert random.reason == "random_control"


def test_row_annotation_bindings_reject_wrong_dna_or_split():
    dna = "ACGT" * 2048
    sidecar = RowAnnotations(1, hashlib.sha256(dna.encode()).hexdigest(), ())
    sidecar.verify(dna)
    with pytest.raises(ValueError, match="identity"):
        sidecar.verify(dna.lower())
    with pytest.raises(ValueError, match="split"):
        sidecar.verify(dna, "validation")


def test_columnar_annotations_preserve_bindings_and_reject_duplicate_rows(tmp_path):
    source = tmp_path / "train.jsonl"
    dna = "ACGT" * 2048
    record = {
        "dataset_revision": DATASET_REVISION,
        "split": "train",
        "row": 5,
        "sequence_sha256": hashlib.sha256(dna.encode()).hexdigest(),
        "landmarks": [
            {
                "start": 1000,
                "end": 1003,
                "kind": "translation_start",
                "source": "reference-v1",
                "feature_id": "tx1",
            }
        ],
    }
    source.write_text(json.dumps(record) + "\n")
    output = tmp_path / "train.parquet"
    report = freeze_sidecar(source, output, "train")
    assert report["counts"]["annotated_rows"] == 1
    index = load_annotations(output)
    assert len(index) == 1 and list(index) == [5] and index.get(6) is None
    assert index[5].landmarks == (
        Landmark(1000, 1003, "translation_start", "reference-v1", "tx1"),
    )
    index[5].verify(dna)
    assert len(load_annotations(output, rows={6})) == 0
    with pytest.raises(ValueError, match="split"):
        load_annotations(output, split="validation")
    source.write_text((json.dumps(record) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicated"):
        freeze_sidecar(source, output, "train")
    # A rejected rebuild cannot replace a previously verified bank.
    load_annotations(output)[5].verify(dna)


def test_minimal_fim_retains_hints_rc_and_dna_targets():
    vocab = {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(ALPHABET)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    encoder = Encoder(tokenizer)
    raw = RawWindow("Zmays", "1", 100, 20100, "+", ("ACGTacgt" * 1024), 1)
    config = replace(
        AugmentationConfig(),
        kind_probabilities=(0, 0, 1, 0),
        minimal_fim_probability=1,
        gap_length_probability=1,
    )
    pipeline = Augmenter(encoder, config)
    orientations = set()
    for occurrence in range(100):
        e = pipeline.build(raw, occurrence)
        a = e.attributes
        orientations.add(a["rc"])
        assert a["minimal_fim"] and a["gap_length_visible"]
        assert (
            not a["metadata_fields"]
            and not a["description"]
            and not a["mask_description"]
        )
        assert f"[gap:{a['gap_length']}]" in e.text
        assert e.scored_tokens == a["gap_length"]
        assert e.loss_weights.sum() == e.scored_tokens
        assert e.spans[-1].role == "middle"
    assert orientations == {True, False}
    no_hint = Augmenter(encoder, replace(config, gap_length_probability=0)).build(
        raw, 0
    )
    assert "[gap:" not in no_hint.text
    assert not no_hint.attributes["gap_length_visible"]
