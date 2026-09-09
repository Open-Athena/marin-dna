import pytest

from marin_dna_exp550.recipe import (
    BASE_IDS,
    BOS_ID,
    PAD_ID,
    SEQ_ID,
    encode_document,
    resolve_optimizer,
)


def test_human_only_masks_prediction_of_first_pad_and_every_pad():
    result = encode_document("N" * 255)
    assert result["input_ids"][:256] == [BOS_ID] + [BASE_IDS["N"]] * 255
    assert result["input_ids"][256:] == [PAD_ID] * 9984
    assert result["loss_weight"][:255] == [1.0] * 255
    assert result["loss_weight"][255:] == [0.0] * 9985
    assert PAD_ID != BASE_IDS["N"]


def test_full_document_has_one_bos_atomic_separators_and_no_padding():
    result = encode_document("[SEQ]".join(["a" * 255] * 40))
    ids = result["input_ids"]
    assert len(ids) == len(result["loss_weight"]) == 10240
    assert ids.count(BOS_ID) == 1
    assert ids.count(SEQ_ID) == 39
    assert ids[256::256] == [SEQ_ID] * 39
    assert PAD_ID not in ids
    assert result["loss_weight"] == [1.0] * 10239 + [0.0]


@pytest.mark.parametrize(
    "sequence",
    ["A" * 254, "R" * 255, "A" * 255 + "[SEQ]", "[SEQ]".join(["A" * 255] * 41)],
)
def test_invalid_geometry_fails(sequence):
    with pytest.raises(ValueError):
        encode_document(sequence)


def test_reference_optimizer_recovers_reference_and_uses_token_batch_units():
    original = resolve_optimizer(batch_tokens=64 * 4096, total_tokens=2_500_000_000)
    assert original.learning_rate == pytest.approx(0.00630)
    assert original.adam_lr == pytest.approx(0.000656)
    assert original.epsilon == pytest.approx(1.85e-8)
    result = resolve_optimizer()
    assert result.learning_rate == pytest.approx(0.004695897205758308)
    assert result.adam_lr == pytest.approx(0.00020258341260453682)
    assert result.epsilon == pytest.approx(5.990618799422977e-8)
    assert result.beta2 == pytest.approx(0.9992190160617281)
