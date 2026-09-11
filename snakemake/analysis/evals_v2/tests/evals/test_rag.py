"""RAG geometry, canonical routing, and parity with the maintained score kernel."""

import copy
import json
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
import torch
from datasets import Dataset
from marin_dna_evals.model.scoring import compute_variant_score_bundle
from marin_dna_evals.rag import (
    BENCHMARKS,
    _digest,
    compute_combined_rag_scores,
    compute_rag_bundle,
    score_rag_dataset,
    transform_rag,
    transform_rag_cached,
    validate_harness,
)
from transformers import Qwen3Config, Qwen3ForCausalLM


class Tokenizer:
    bos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        tokens = []
        for index, segment in enumerate(text.split("[SEQ]")):
            if index:
                tokens.append(3)
            tokens.extend(
                {"N": 1, "A": 4, "C": 5, "G": 6, "T": 7}[base]
                for base in segment.upper()
            )
        return ([2] if add_special_tokens else []) + tokens


def fixture():
    revisions = {name: "b" * 40 for name in BENCHMARKS}
    canonical, rows = {}, []
    for name in sorted(BENCHMARKS):
        canonical[name] = []
        for index in range(2):
            metadata = {
                "chrom": "1",
                "pos": 300 + index,
                "ref": "T",
                "alt": "A",
                "label": index,
                "subset": "missense_variant",
            }
            metadata["mavedb_urn" if name == "sge" else "match_group"] = (
                "1" if name == "sge" else 1
            )
            canonical[name].append(metadata)
            sequence = "ACGT" * 63 + "ACG"
            rows.append(
                {
                    "benchmark": name,
                    "source_row_index": index,
                    "source_row_id": _digest(name, revisions[name], "train", index),
                    "query_name": "rag_"
                    + _digest("hg38", "chr1", 172 + index, 427 + index),
                    "source_chrom": "chr1",
                    "source_start": 172 + index,
                    "source_end": 427 + index,
                    "split": "train",
                    "ref": "T",
                    "alt": "A",
                    "sequence": ("n" * 255 + "[SEQ]" if index else "") + sequence,
                    "species_order": ["bird", "hg38"] if index else ["hg38"],
                    "metadata_json": json.dumps(metadata),
                    "group_id": _digest(name, "match_group", "1")
                    if name != "sge"
                    else None,
                    "accession_id": _digest(name, "accession", "1")
                    if name == "sge"
                    else None,
                }
            )
    # Deliberately interleave benchmarks and reverse source indices.
    return canonical, revisions, [rows[index] for index in (5, 0, 3, 4, 1, 2)]


def tiny_model():
    torch.manual_seed(0)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=10240,
            attention_dropout=0.0,
        )
    ).eval()


def test_document_geometry_and_segmentwise_rc():
    _, _, rows = fixture()
    row = next(row for row in rows if len(row["species_order"]) == 2)
    fwd = transform_rag(row, tokenizer=Tokenizer(), strand="fwd")
    rc = transform_rag(row, tokenizer=Tokenizer(), strand="rc")
    assert fwd["input_ids"].shape == rc["input_ids"].shape == (10240,)
    assert fwd["human_start"] == rc["human_start"] == 257
    assert fwd["input_ids"][256] == rc["input_ids"][256] == 3
    assert np.all(fwd["input_ids"][1:256] == 1)
    assert np.all(rc["input_ids"][1:256] == 1)
    assert fwd["input_ids"][384] == 7 and rc["input_ids"][384] == 4
    assert fwd["alt_token_id"] == 4 and rc["alt_token_id"] == 7
    assert np.all(fwd["input_ids"][512:] == 0)
    maximum = copy.deepcopy(row)
    maximum["species_order"] = [f"species-{i}" for i in range(39)] + ["hg38"]
    maximum["sequence"] = "[SEQ]".join(
        ["n" * 255] * 39 + [row["sequence"].split("[SEQ]")[-1]]
    )
    full = transform_rag(maximum, tokenizer=Tokenizer(), strand="fwd")
    assert full["human_start"] == 9985 and full["input_ids"][-1] == 6


def test_harness_exact_canonical_routing_and_rejection():
    canonical, revisions, rows = fixture()
    routing = validate_harness(Dataset.from_list(rows), canonical, revisions)
    for name, indices in routing.items():
        assert [rows[index]["source_row_index"] for index in indices] == [0, 1]
        assert all(rows[index]["benchmark"] == name for index in indices)
    cases = [
        ("split", "test"),
        ("source_row_id", "wrong"),
        ("ref", "G"),
        ("source_end", 999),
        ("group_id", "wrong"),
        ("species_order", ["hg38", "bird"]),
    ]
    for field, value in cases:
        bad = copy.deepcopy(rows)
        bad[0][field] = value
        with pytest.raises(ValueError):
            validate_harness(Dataset.from_list(bad), canonical, revisions)
    with pytest.raises(ValueError, match="duplicate"):
        validate_harness(Dataset.from_list(rows + rows[:1]), canonical, revisions)
    with pytest.raises(ValueError, match="missing"):
        validate_harness(Dataset.from_list(rows[:-1]), canonical, revisions)
    held_out = copy.deepcopy(canonical)
    held_out[rows[0]["benchmark"]][rows[0]["source_row_index"]]["chrom"] = "2"
    with pytest.raises(ValueError, match="held-out"):
        validate_harness(Dataset.from_list(rows), held_out, revisions)
    bad = copy.deepcopy(rows)
    bad[0]["sequence"] = bad[0]["sequence"][:-1] + "N"
    with pytest.raises(ValueError, match="ACGT human"):
        validate_harness(Dataset.from_list(bad), canonical, revisions)


def test_padded_variable_positions_match_prefix_kernel_and_human_pool():
    _, _, rows = fixture()
    rows = [
        next(row for row in rows if len(row["species_order"]) == count)
        for count in (1, 2)
    ]
    model = tiny_model()
    nuc = torch.tensor([4, 5, 6, 7])
    for strand in ("fwd", "rc"):
        transformed = [
            transform_rag(row, tokenizer=Tokenizer(), strand=strand) for row in rows
        ]
        ids = torch.tensor(np.stack([row["input_ids"][:512] for row in transformed]))
        alt = torch.tensor([row["alt_token_id"] for row in transformed])
        start = torch.tensor([row["human_start"] for row in transformed])
        kernel = partial(compute_rag_bundle, nuc_token_ids=nuc, return_embeddings=True)
        with torch.inference_mode():
            actual = kernel(model, ids, alt, start)
            for i, item in enumerate(transformed):
                end = item["human_start"] + 255
                expected = compute_variant_score_bundle(
                    model,
                    ids[i : i + 1, :end],
                    alt[i : i + 1],
                    var_pos=item["human_start"] + 127,
                    nuc_token_ids=nuc,
                    return_embeddings=True,
                    pool_lo=item["human_start"],
                    pool_hi=end,
                )
                torch.testing.assert_close(
                    actual[i : i + 1], expected, atol=2e-5, rtol=2e-4
                )
            changed = ids.clone()
            changed[0, 256:] = 7
            torch.testing.assert_close(
                kernel(model, changed, alt, start), actual, atol=2e-5, rtol=2e-4
            )


def test_left_padded_cached_scores_reuse_prefix_and_preserve_outputs(monkeypatch):
    monkeypatch.setattr("marin_dna_evals.rag.MODEL_TOKENS", 512)
    _, _, rows = fixture()
    rows = [next(row for row in rows if len(row["species_order"]) == n) for n in (1, 2)]
    model = tiny_model()
    nuc = torch.tensor([4, 5, 6, 7])
    for strand in ("fwd", "rc"):
        cached = [
            transform_rag_cached(row, tokenizer=Tokenizer(), strand=strand)
            for row in rows
        ]
        ids = torch.tensor(np.stack([row["input_ids"] for row in cached]))
        mask = torch.tensor(np.stack([row["attention_mask"] for row in cached]))
        alt = torch.tensor([row["alt_token_id"] for row in cached])
        assert mask.sum(1).tolist() == [256, 512]
        assert ids[:, 384].tolist() == ([7, 7] if strand == "fwd" else [4, 4])
        calls = []

        def capture(_model, args, kwargs, calls=calls):
            calls.append(
                (
                    tuple(args[0].shape),
                    kwargs.get("use_cache"),
                    kwargs.get("past_key_values") is not None,
                )
            )

        hook = model.register_forward_pre_hook(capture, with_kwargs=True)
        with torch.inference_mode():
            actual = compute_variant_score_bundle(
                model,
                ids,
                alt,
                var_pos=384,
                nuc_token_ids=nuc,
                attention_mask=mask,
                return_embeddings=True,
                pool_lo=257,
                pool_hi=512,
            )
        hook.remove()
        assert calls == [((2, 384), True, False), ((4, 128), False, True)]
        transformed = [
            transform_rag(row, tokenizer=Tokenizer(), strand=strand) for row in rows
        ]
        with torch.inference_mode():
            reference = compute_rag_bundle(
                model,
                torch.tensor(np.stack([row["input_ids"] for row in transformed])),
                alt,
                torch.tensor([row["human_start"] for row in transformed]),
                nuc_token_ids=nuc,
                return_embeddings=True,
            )
            torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-4)
            changed = ids.masked_fill(mask == 0, 6)
            masked = compute_variant_score_bundle(
                model,
                changed,
                alt,
                var_pos=384,
                nuc_token_ids=nuc,
                attention_mask=mask,
                return_embeddings=True,
                pool_lo=257,
                pool_hi=512,
            )
            torch.testing.assert_close(masked, actual, atol=2e-5, rtol=2e-4)


def test_joint_inference_loads_once_and_matches_separate_benchmark_processing(
    tmp_path, monkeypatch
):
    canonical, revisions, rows = fixture()
    path = tmp_path / "harness.parquet"
    Dataset.from_list(rows).to_parquet(path)
    model, tokenizer = tiny_model(), Tokenizer()
    # A short fixed shape exercises the real automatic prediction loop on CPU.
    monkeypatch.setattr("marin_dna_evals.rag.MODEL_TOKENS", 512)
    kwargs = {
        "per_device_eval_batch_size": 4,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "report_to": [],
        "use_cpu": True,
    }
    with patch(
        "marin_dna_evals.rag.load_hf_causal_lm_and_tokenizer",
        return_value=(tokenizer, model),
    ) as loader:
        combined = compute_combined_rag_scores(
            tmp_path, path, canonical, revisions, inference_kwargs=kwargs
        )
        loader.assert_called_once()
    for name in sorted(BENCHMARKS):
        subset = sorted(
            (row for row in rows if row["benchmark"] == name),
            key=lambda row: row["source_row_index"],
        )
        separate = score_rag_dataset(
            model, tokenizer, Dataset.from_list(subset), inference_kwargs=kwargs
        )
        assert combined[name]["label"].tolist() == [0, 1]
        for strand in ("fwd", "rc"):
            np.testing.assert_allclose(
                combined[name][f"llr_{strand}"], separate[strand][:, 0], atol=2e-5
            )
            np.testing.assert_allclose(
                combined[name][f"jsd_{strand}"], separate[strand][:, 1], atol=2e-5
            )
        expected = (separate["fwd"][:, 2:18] + separate["rc"][:, 2:18]) / 2
        np.testing.assert_array_equal(
            np.stack(combined[name]["emb_ref"]), expected.astype(np.float16)
        )
