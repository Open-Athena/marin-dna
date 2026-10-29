import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from exp586_moe.inference.prune_vocab import compact_hf_export


def test_compact_export_selects_exact_embedding_and_head_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    embedding = torch.arange(30, dtype=torch.bfloat16).reshape(10, 3)
    head = torch.arange(40, dtype=torch.bfloat16).reshape(10, 4)
    other = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    save_file(
        {
            "model.embed_tokens.weight": embedding,
            "lm_head.weight": head,
            "model.other.weight": other,
        },
        source / "model-00001-of-00001.safetensors",
        metadata={"format": "pt"},
    )
    names = ["model.embed_tokens.weight", "lm_head.weight", "model.other.weight"]
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 152},
                "weight_map": {
                    name: "model-00001-of-00001.safetensors" for name in names
                },
            }
        )
    )
    (source / "config.json").write_text(json.dumps({"vocab_size": 10}))
    (source / "tokenizer_config.json").write_text("{}")
    original_ids = [5, 7, 2, 9, 4]
    vocab = {f"token_{index}": index for index in range(10)}
    for base, token_id in zip("acgtn", original_ids, strict=True):
        del vocab[f"token_{token_id}"]
        vocab[base] = token_id
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="token_0"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer.save(str(source / "tokenizer.json"))

    output = tmp_path / "compact"
    compact_hf_export(source, output)

    weights = load_file(output / "model-00001-of-00001.safetensors")
    torch.testing.assert_close(
        weights["model.embed_tokens.weight"], embedding[original_ids]
    )
    torch.testing.assert_close(weights["lm_head.weight"], head[original_ids])
    torch.testing.assert_close(weights["model.other.weight"], other)
    config = json.loads((output / "config.json").read_text())
    assert config["vocab_size"] == 5
    assert config["exp586_compact_vocab"]["original_ids"] == original_ids
