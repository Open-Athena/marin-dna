from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

import decoder_neighbors as module


def write_sae(path: Path, decoder: torch.Tensor) -> None:
    path.mkdir()
    save_file({"W_dec": decoder}, path / "sae_weights.safetensors")
    (path / "cfg.json").write_text(
        json.dumps(
            {
                "d_in": decoder.shape[1],
                "d_sae": decoder.shape[0],
                "normalize_activations": "none",
            }
        )
    )


def test_decoder_neighbors_tracks_permutation_and_mutual_match(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "D_IN", 3)
    monkeypatch.setattr(module, "D_SAE", 4)
    monkeypatch.setattr(
        module,
        "REFERENCE_QUERIES",
        (("first", 0), ("second", 1)),
    )
    reference = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    candidate = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    )
    reference_path = tmp_path / "reference"
    candidate_path = tmp_path / "candidate"
    write_sae(reference_path, reference)
    write_sae(candidate_path, candidate)

    frame, summary = module.decoder_neighbors(
        reference_path,
        candidate_path,
        dictionary_name="permuted",
        top_k=2,
    )

    top = frame.filter(frame["candidate_rank"] == 1).sort("concept")
    assert top["candidate_feature_id"].to_list() == [2, 0]
    assert top["mutual_nearest"].to_list() == [True, True]
    assert summary["unique_candidate_features"] >= 2
