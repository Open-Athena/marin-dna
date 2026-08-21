from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from marin_dna_evals.likelihood_dynamics_489 import (
    _prediction_array,
    _read_reference_windows,
    aggregate_token_stats_by_case,
    assemble_token_atoms,
    build_window_metadata,
    compute_per_token_stats_clm,
    leave_one_chrom_7mer_nll,
    parse_window_id,
    validate_pilot_artifacts,
)


class _FakeTwoBit:
    def __init__(self, references: dict[str, str]):
        self.references = references
        self.closed = False

    def chroms(self) -> dict[str, int]:
        return {chrom: len(sequence) for chrom, sequence in self.references.items()}

    def sequence(self, chrom: str, start: int, end: int) -> str:
        return self.references[chrom][start:end]

    def close(self) -> None:
        self.closed = True


def _source_sequences() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["NC_1:0-8", "NC_2:0-8"],
            "seq": ["ACgtACGT", "tgCAACGT"],
        }
    )


def _metadata() -> tuple[pd.DataFrame, dict[str, object]]:
    fake = _FakeTwoBit(
        {
            "NC_1": "AcGTacGT",
            "NC_2": "TGcaACgt",
        }
    )
    module = SimpleNamespace(open=lambda _path, _masked: fake)
    with patch.dict(sys.modules, {"py2bit": module}):
        metadata, manifest = build_window_metadata(
            _source_sequences(),
            region="cds",
            window_size=8,
            reference_kind="twobit",
            reference_path="/unused.2bit",
            assembly="GCF_000001405.40",
            conservation_label_source="phyloP_case",
        )
    assert fake.closed
    return metadata, manifest


def test_parse_window_id_enforces_half_open_width() -> None:
    assert parse_window_id("1:10-18", window_size=8) == ("1", 10, 18)
    with pytest.raises(AssertionError, match="0-based half-open"):
        parse_window_id("1:10-19", window_size=8)


def test_metadata_keeps_conservation_and_repeat_case_independent() -> None:
    metadata, manifest = _metadata()

    np.testing.assert_array_equal(
        metadata.loc[0, "case_is_upper"],
        [True, True, False, False, True, True, True, True],
    )
    np.testing.assert_array_equal(
        metadata.loc[0, "is_repeat"],
        [False, True, False, False, True, True, False, False],
    )
    np.testing.assert_array_equal(
        metadata.loc[0, "is_conserved"],
        [True, True, False, False, True, True, True, True],
    )
    assert manifest["assembly"] == "GCF_000001405.40"
    assert manifest["coordinate_system"] == "0-based half-open"
    assert (
        manifest["conservation_missingness"]
        == "lowercase_conflates_below_threshold_and_missing_alignment"
    )


def test_metadata_fails_on_reference_mismatch() -> None:
    fake = _FakeTwoBit({"NC_1": "TTTTTTTT", "NC_2": "TGCAACGT"})
    module = SimpleNamespace(open=lambda _path, _masked: fake)
    with (
        patch.dict(sys.modules, {"py2bit": module}),
        pytest.raises(AssertionError, match="assembly/coordinate mismatch"),
    ):
        build_window_metadata(
            _source_sequences(),
            region="cds",
            window_size=8,
            reference_kind="twobit",
            reference_path="/unused.2bit",
            assembly="GCF_000001405.40",
            conservation_label_source="phyloP_case",
        )
    assert fake.closed


def test_indexed_fasta_queries_only_requested_byte_ranges(tmp_path) -> None:
    fasta = tmp_path / "reference.fa"
    fasta.write_bytes(b">1\nAcgT\nacGT\n>2\nTTaa\nCCgg\n")
    fasta.with_suffix(".fa.fai").write_text(
        "1\t8\t3\t4\t5\n2\t8\t16\t4\t5\n"
    )

    observed = _read_reference_windows(
        [("1", 1, 7), ("2", 0, 8)],
        reference_kind="fasta",
        reference_path=fasta,
    )

    assert observed == ["cgTacG", "TTaaCCgg"]


def test_7mer_control_is_finite_at_both_edges() -> None:
    values = leave_one_chrom_7mer_nll(
        ["ACGTACGTACGTAC", "TGCATGCATGCATG"],
        ["1", "2"],
    )
    assert all(value.shape == (14,) for value in values)
    assert all(np.isfinite(value).all() for value in values)


class _UniformModel:
    def __call__(self, input_ids: torch.Tensor) -> SimpleNamespace:
        batch, length = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(batch, length, 8))


def test_per_token_kernel_returns_full_vocab_nll_and_four_base_entropy() -> None:
    input_ids = torch.tensor([[7, 0, 1, 2, 3]])
    output = compute_per_token_stats_clm(
        _UniformModel(),
        input_ids,
        nucleotide_token_ids=torch.tensor([0, 1, 2, 3]),
    )
    assert output.shape == (1, 4, 2)
    torch.testing.assert_close(
        output[0, :, 0],
        torch.full((4,), np.log(8), dtype=torch.float32),
    )
    torch.testing.assert_close(
        output[0, :, 1],
        torch.full((4,), np.log(4), dtype=torch.float32),
    )


def test_prediction_array_realigns_flat_trainer_output() -> None:
    values = np.arange(16, dtype=np.float32).reshape(2, 4, 2)
    values[:, :, 0] += 1
    output = _prediction_array(values.ravel(), n_windows=2, window_size=4)
    np.testing.assert_array_equal(output, values)


def test_assemble_token_atoms_has_stable_identity_and_provenance() -> None:
    metadata, _ = _metadata()
    sequences = _source_sequences()
    stats = pd.DataFrame(
        {
            "window_id": sequences["id"],
            "nll": [
                np.linspace(0.1, 0.8, 8, dtype=np.float32),
                np.linspace(0.2, 0.9, 8, dtype=np.float32),
            ],
            "entropy_4nuc": [
                np.ones(8, dtype=np.float32),
                np.full(8, 1.1, dtype=np.float32),
            ],
        }
    )
    atoms, manifest = assemble_token_atoms(
        metadata,
        stats,
        checkpoint="model-step-1",
        checkpoint_order=0,
        stage="m1",
        training_step=1,
        cumulative_tokens=100,
        assembly="GCF_000001405.40",
        conservation_label_source="phyloP_case",
    )

    assert len(atoms) == 16
    assert atoms["token_index"].tolist() == list(range(16))
    assert atoms.loc[7, "target_pos"] == 7
    assert atoms.loc[7, "genomic_pos"] == 7
    assert atoms.loc[2, "conservation_label_status"] == (
        "below_threshold_or_missing"
    )
    assert atoms.loc[0, "repeat_label_source"] == (
        "GCF_000001405.40_soft_mask"
    )
    assert manifest["token_identity"] == ["region", "row_index", "target_pos"]

    aggregate = aggregate_token_stats_by_case(stats, sequences)
    assert aggregate.shape == (2, 4)
    assert np.array_equal(aggregate[:, 2:].sum(axis=1), [8, 8])


def test_validate_pilot_artifacts_requires_identical_token_identity(tmp_path) -> None:
    metadata, _ = _metadata()
    sequences = _source_sequences()
    stats = pd.DataFrame(
        {
            "window_id": sequences["id"],
            "nll": [np.ones(8, dtype=np.float32)] * 2,
            "entropy_4nuc": [np.ones(8, dtype=np.float32)] * 2,
        }
    )
    checkpoints = ["early", "terminal"]
    atom_paths: dict[tuple[str, str], str] = {}
    manifest_paths: dict[tuple[str, str], str] = {}
    for order, checkpoint in enumerate(checkpoints):
        atoms, _ = assemble_token_atoms(
            metadata,
            stats,
            checkpoint=checkpoint,
            checkpoint_order=order,
            stage="m1",
            training_step=order,
            cumulative_tokens=order,
            assembly="GCF_000001405.40",
            conservation_label_source="phyloP_case",
        )
        atom_path = tmp_path / f"{checkpoint}.parquet"
        manifest_path = tmp_path / f"{checkpoint}.json"
        atoms.to_parquet(atom_path, index=False)
        manifest_path.write_text(
            json.dumps(
                {
                    "score_manifest": {
                        "aggregate_gate": {
                            "passed": True,
                            "max_abs_per_window_sum_diff": 0.0,
                        }
                    }
                }
            )
        )
        atom_paths[(checkpoint, "cds")] = str(atom_path)
        manifest_paths[(checkpoint, "cds")] = str(manifest_path)

    report = validate_pilot_artifacts(
        atom_paths,
        manifest_paths,
        checkpoints=checkpoints,
        regions=["cds"],
        expected_windows=2,
        window_size=8,
    )
    assert report["passed"] is True
    assert len(report["cells"]) == 2
