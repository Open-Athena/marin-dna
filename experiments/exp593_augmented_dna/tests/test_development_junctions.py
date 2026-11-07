import hashlib
from dataclasses import replace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from exp593_dna.annotations import Landmark, RowAnnotations
from exp593_dna.development_junctions import junction_pair
from exp593_dna.fim_validation import prompt_and_target
from exp593_dna.pipeline import ALPHABET, Encoder, RawWindow, reverse_complement


def test_blind_junction_and_control_share_dna_and_obey_rc_coordinates():
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(ALPHABET)}},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    encoder = Encoder(tokenizer)
    bases = np.random.default_rng(593).choice(list("ACGTacgt"), 8192)
    bases[2048:2050] = list("GT")
    bases[5000:5002] = list("ag")
    dna = "".join(bases)
    donor = Landmark(2048, 2050, "splice_donor", "reference-v1")
    acceptor = Landmark(5000, 5002, "splice_acceptor", "reference-v1")
    landmarks = (
        donor,
        acceptor,
        Landmark(6000, 6003, "translation_stop", "reference-v1"),
    )
    orientations = set()
    for row in range(32):
        raw = RawWindow("Zmays", "1", 0, 12000, "-", dna, row, "validation")
        annotations = RowAnnotations(
            row, hashlib.sha256(dna.encode()).hexdigest(), landmarks, "validation"
        )
        for site in (donor, acceptor):
            guided, control = junction_pair(raw, annotations, site, encoder)
            rc = guided.attributes["rc"]
            orientations.add(rc)
            oriented = reverse_complement(dna) if rc else dna
            for item in (guided, control):
                parts = {s.role: s.text for s in item.spans if s.role != "text"}
                assert (
                    parts["left"] + parts["middle"] + parts["right"]
                    == oriented[64:8128].lower()
                )
                assert len(parts["left"]) >= 128 and len(parts["right"]) >= 128
                assert len(item.token_ids) <= 8192
                assert np.isclose(item.loss_weights.sum(), 1)
                assert len(prompt_and_target(item)[1]) == 2
                text = "".join(s.text for s in item.spans if s.role == "text")
                assert (
                    "donor" not in text
                    and "acceptor" not in text
                    and "Zmays" not in text
                )
            original = dna[site.start : site.end].lower()
            expected = reverse_complement(original) if rc else original
            assert next(s.text for s in guided.spans if s.role == "middle") == expected
            assert control.attributes["rc"] == rc
            start = control.attributes["gap_start"]
            for feature in landmarks:
                left, right = (
                    (8192 - feature.end, 8192 - feature.start)
                    if rc
                    else (feature.start, feature.end)
                )
                left -= 64
                right -= 64
                assert start + 2 < left - 32 or start > right + 32
    assert orientations == {False, True}
    with pytest.raises(ValueError, match="validation row"):
        junction_pair(replace(raw, split="train"), annotations, donor, encoder)
    with pytest.raises(ValueError, match="DNA identity"):
        junction_pair(
            raw, replace(annotations, sequence_sha256="0" * 64), donor, encoder
        )
