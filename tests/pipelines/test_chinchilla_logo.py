"""Correctness tests for the issue #419 genome-logo path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyBigWig
import pytest
import torch
import torch.nn as nn

import marin_dna.pipelines.chinchilla_logo as chinchilla_logo
from marin_dna.data.dna import reverse_complement
from marin_dna.model.sequence_interpretation import (
    _strand_nucleotide_logits,
    nucleotide_logo,
)
from marin_dna.pipelines.chinchilla_logo import (
    MODEL_REVISION,
    WindowSpec,
    aggregate_strand_logits,
    canonical_runs,
    compute_window_nucleotide_logits,
    list_score_shards,
    load_score_shard,
    logo_from_log_probabilities,
    parse_chrom_sizes,
    score_window_plans,
    sha256_file,
    tile_canonical_run,
    tile_sequence,
    write_bigwig_sets_with_metrics,
    write_dataset_readme,
    write_release_manifest,
    write_score_shard,
    write_ucsc_hub,
    write_window_plans,
)
from marin_dna.tokenizer.char import create_char_tokenizer


class _PrefixSumCausalLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, **kwargs):
        x = input_ids.float()
        prefix = torch.cumsum(x, dim=1)
        position = torch.arange(input_ids.shape[1], dtype=torch.float)
        vocabulary = torch.arange(self.vocab_size, dtype=torch.float)
        logits = torch.sin(
            0.31 * prefix.unsqueeze(-1)
            + 0.17 * position.view(1, -1, 1)
            + 0.53 * vocabulary
        )
        return SimpleNamespace(logits=logits)


def _assert_exact_once(windows: list[WindowSpec]) -> None:
    positions = np.concatenate(
        [np.arange(window.emit_start, window.emit_end) for window in windows]
    )
    assert len(positions) == len(np.unique(positions))
    np.testing.assert_array_equal(positions, np.arange(positions[0], positions[-1] + 1))


def test_canonical_runs_are_case_insensitive_and_split_on_ambiguity():
    assert canonical_runs("NNacGTNARYtT!") == [(2, 6), (7, 8), (10, 12)]
    assert canonical_runs("acgt", offset=10) == [(10, 14)]


@pytest.mark.parametrize(
    ("run_length", "expected_starts", "expected_emits"),
    [
        (255, [0], [(63, 191)]),
        (300, [0, 45], [(63, 191), (191, 236)]),
        (383, [0, 128], [(63, 191), (191, 319)]),
        (400, [0, 128, 145], [(63, 191), (191, 319), (319, 336)]),
    ],
)
def test_tile_canonical_run_tail_is_anchored_without_duplicates(
    run_length, expected_starts, expected_emits
):
    windows = tile_canonical_run("chr1", 0, run_length)
    assert [window.window_start for window in windows] == expected_starts
    assert [
        (window.emit_start, window.emit_end) for window in windows
    ] == expected_emits
    _assert_exact_once(windows)
    assert sum(window.n_emitted for window in windows) == run_length - 127


def test_tile_sequence_reconciles_gaps_short_runs_and_boundaries():
    sequence = "N" * 3 + "a" * 254 + "NN" + "C" * 300 + "!"
    windows, stats = tile_sequence("chr1", sequence)
    assert len(windows) == 2
    assert stats.chrom_size == len(sequence)
    assert stats.canonical_bases == 554
    assert stats.noncanonical_bases == 6
    assert stats.short_run_bases == 254
    assert stats.border_excluded_bases == 127
    assert stats.scored_bases == 173
    assert (
        stats.scored_bases
        + stats.short_run_bases
        + stats.border_excluded_bases
        + stats.noncanonical_bases
        == len(sequence)
    )
    assert windows[0].run_start == 259
    _assert_exact_once(windows)


def test_write_window_plans_reuses_genome_and_keeps_empty_scaffolds(
    tmp_path, monkeypatch
):
    sequences = {"chr1": "A" * 300, "chr2": "N" * 20}
    genome_instances = []

    class _Genome:
        def __init__(self, _path, *, subset_chroms):
            assert subset_chroms == set(sequences)
            self.chroms = {
                chrom: len(sequence) for chrom, sequence in sequences.items()
            }
            genome_instances.append(self)

        def __call__(self, chrom, start, end):
            return sequences[chrom][start:end]

    monkeypatch.setattr(chinchilla_logo, "Genome", _Genome)
    chrom_sizes = tmp_path / "chrom.sizes"
    chrom_sizes.write_text("chr1\t300\nchr2\t20\n")
    plan_dir = tmp_path / "plans"
    stats = write_window_plans(
        tmp_path / "genome.fa",
        chrom_sizes,
        ["chr1", "chr2"],
        plan_dir,
    )

    assert len(genome_instances) == 1
    assert [row.chrom for row in stats] == ["chr1", "chr2"]
    assert stats[0].window_count == 2
    assert stats[1].window_count == 0
    assert (plan_dir / "chr1.parquet").is_file()
    assert (plan_dir / "chr2.parquet").is_file()
    assert (
        json.loads((plan_dir / "chr2.coverage.json").read_text())["coverage"][
            "scored_bases"
        ]
        == 0
    )


def test_shifted_phase_has_shared_positions_from_different_frames():
    production = tile_canonical_run("chr1", 0, 600, phase=0)
    shifted = tile_canonical_run("chr1", 0, 600, phase=64)
    production_by_position = {
        position: window.window_start
        for window in production
        for position in range(window.emit_start, window.emit_end)
    }
    shifted_by_position = {
        position: window.window_start
        for window in shifted
        for position in range(window.emit_start, window.emit_end)
    }
    shared = sorted(production_by_position.keys() & shifted_by_position.keys())
    assert shared
    assert all(
        production_by_position[position] != shifted_by_position[position]
        for position in shared[:-64]
    )


def test_compute_window_logits_matches_single_sequence_readout():
    tokenizer = create_char_tokenizer(bos=True, eos=True)
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    sequence = "ACGTACGTACGTACGT"
    input_ids = torch.tensor([tokenizer.encode(sequence)])
    nucleotide_ids = torch.tensor(
        [tokenizer.encode(base)[1] for base in "ACGT"], dtype=torch.long
    )
    actual = compute_window_nucleotide_logits(
        model,
        input_ids,
        nucleotide_token_ids=nucleotide_ids,
        n_prefix=1,
        context_size=len(sequence),
    )
    expected = _strand_nucleotide_logits(model, tokenizer, sequence)
    torch.testing.assert_close(actual[0], expected)


def test_batch_aggregation_matches_pinned_sequence_explorer_logo():
    tokenizer = create_char_tokenizer(bos=True, eos=True)
    model = _PrefixSumCausalLM(vocab_size=8).eval()
    sequence = "AAAACGTCGATCTTGC"
    forward = _strand_nucleotide_logits(model, tokenizer, sequence).numpy()
    reverse = _strand_nucleotide_logits(
        model, tokenizer, reverse_complement(sequence)
    ).numpy()
    actual = aggregate_strand_logits(forward[None], reverse[None])
    expected = nucleotide_logo(model, tokenizer, sequence)
    np.testing.assert_allclose(
        actual.probabilities[0], expected.probabilities, atol=1e-7
    )
    np.testing.assert_allclose(
        actual.glyph_heights_bits[0], expected.glyph_heights_bits, atol=1e-7
    )


def test_aggregation_is_invariant_to_per_position_additive_offsets():
    rng = np.random.default_rng(7)
    forward = rng.normal(size=(2, 9, 4)).astype(np.float32)
    reverse = rng.normal(size=(2, 9, 4)).astype(np.float32)
    baseline = aggregate_strand_logits(forward, reverse)
    forward_offsets = rng.normal(size=(2, 9, 1)).astype(np.float32)
    reverse_offsets = rng.normal(size=(2, 9, 1)).astype(np.float32)
    shifted = aggregate_strand_logits(
        forward + forward_offsets, reverse + reverse_offsets
    )
    np.testing.assert_allclose(
        baseline.log_probabilities, shifted.log_probabilities, atol=2e-6
    )
    np.testing.assert_allclose(
        baseline.glyph_heights_bits, shifted.glyph_heights_bits, atol=2e-6
    )


def test_logo_from_logp_rejects_non_normalized_values():
    with pytest.raises(AssertionError, match="sum to one"):
        logo_from_log_probabilities(np.zeros((3, 4), dtype=np.float32))


def test_score_shard_round_trip_preserves_mapping_and_float32(tmp_path):
    windows = [
        WindowSpec("chr1", 0, 12, 0, 1, 3),
        WindowSpec("chr1", 0, 12, 4, 5, 8),
    ]
    logits = np.arange(2 * 5 * 4, dtype=np.float32).reshape(2, 5, 4) / 7
    logp = torch.log_softmax(torch.from_numpy(logits), dim=-1).numpy()
    path = tmp_path / "part-000000.npz"
    write_score_shard(
        path,
        windows,
        logp,
        metadata={"context_size": 5, "inference_seconds": 1.5},
    )
    shard = load_score_shard(path)
    assert shard.chrom == "chr1"
    assert shard.log_probabilities.dtype == np.float32
    np.testing.assert_array_equal(shard.score_offsets, [0, 2, 5])
    np.testing.assert_allclose(
        shard.log_probabilities,
        np.concatenate([logp[0, 1:3], logp[1, 1:4]]),
    )


def test_bigwig_sets_round_trip_and_leave_missing_intervals_absent(tmp_path):
    windows = [
        WindowSpec("chr1", 0, 12, 0, 1, 3),
        WindowSpec("chr1", 0, 12, 4, 5, 8),
    ]
    logits = np.arange(2 * 5 * 4, dtype=np.float32).reshape(2, 5, 4) / 9
    logp_windows = torch.log_softmax(torch.from_numpy(logits), dim=-1).numpy()
    shard_path = tmp_path / "shards" / "part-000000.npz"
    write_score_shard(
        shard_path,
        windows,
        logp_windows,
        metadata={"context_size": 5, "inference_seconds": 0.1},
    )
    chrom_sizes = tmp_path / "chrom.sizes"
    chrom_sizes.write_text("chr1\t12\nchrUnused\t4\n")
    output_root = tmp_path / "release"
    runtime_path = output_root / "manifest" / "bigwig_build.json"
    paths = write_bigwig_sets_with_metrics(
        [shard_path], chrom_sizes, output_root, runtime_path
    )
    metrics = json.loads(runtime_path.read_text())
    assert metrics["shard_count"] == 1
    assert metrics["input_shard_bytes"] == shard_path.stat().st_size
    assert metrics["total_bigwig_bytes"] == sum(
        path.stat().st_size for path in paths.values()
    )

    shard = load_score_shard(shard_path)
    expected_logo = logo_from_log_probabilities(shard.log_probabilities)
    for channel, base in enumerate("ACGT"):
        with pyBigWig.open(str(paths[f"logprob/{base}"])) as bigwig:
            values = np.asarray(bigwig.values("chr1", 0, 12))
            assert np.isnan(values[[0, 3, 4, 8, 9, 10, 11]]).all()
            np.testing.assert_allclose(
                values[[1, 2, 5, 6, 7]],
                shard.log_probabilities[:, channel],
                atol=1e-6,
            )
        with pyBigWig.open(str(paths[f"logo/{base}"])) as bigwig:
            values = np.asarray(bigwig.values("chr1", 0, 12))
            np.testing.assert_allclose(
                values[[1, 2, 5, 6, 7]],
                expected_logo.glyph_heights_bits[:, channel],
                atol=1e-6,
            )


def test_ucsc_hub_defaults_and_dataset_readme(tmp_path):
    hub_paths = write_ucsc_hub(tmp_path)
    track_db = hub_paths["trackDb"].read_text()
    assert "aggregate stacked" in track_db
    assert "logo on" in track_db
    assert "viewLimits 0:2" in track_db
    assert "descriptionUrl description.html" in hub_paths["hub"].read_text()
    assert track_db.count("html description.html") == 10
    assert hub_paths["hubDescription"].is_file()
    assert hub_paths["trackDescription"].is_file()
    assert "track marinDnaM51LogProb" in track_db
    assert (
        "track marinDnaM51LogProb\n"
        "shortLabel MarinDNA m5.1 logp\n"
        "longLabel MarinDNA m5.1 canonical A/C/G/T log-probabilities\n"
        "container multiWig\n"
        "aggregate transparentOverlay\n"
        "type bigWig\n"
    ) in track_db
    assert "visibility hide" in track_db
    for color in ("0,128,0", "0,0,255", "255,166,0", "255,0,0"):
        assert color in track_db

    readme = tmp_path / "README.md"
    commit = "a" * 40
    write_dataset_readme(readme, application_commit=commit, scaffolds=["chr1"])
    text = readme.read_text()
    assert f"blob/{commit}/snakemake/analysis/chinchilla_logo" in text
    assert MODEL_REVISION in text
    assert all(f"- {tag}" in text for tag in ("biology", "genomics", "dna"))
    assert "not an LLR" in text
    assert "only the following configured scaffold" in text
    assert "`chr1`" in text
    assert "not a genome-wide" in text

    full_readme = tmp_path / "FULL_README.md"
    write_dataset_readme(
        full_readme,
        application_commit=commit,
        scaffolds=["chr1", "chr2"],
        full_assembly=True,
    )
    full_text = full_readme.read_text()
    assert "genome-wide release covers all 2 sequences" in full_text
    assert "browser-authoritative UCSC" in full_text
    assert "not a genome-wide" not in full_text


def test_chrom_sizes_and_sha256_are_deterministic(tmp_path):
    path = tmp_path / "chrom.sizes"
    path.write_text("chr2\t7\nchr1\t5\n")
    assert parse_chrom_sizes(path) == [("chr2", 7), ("chr1", 5)]
    assert (
        sha256_file(path) == __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    )


def test_score_shard_metadata_is_json(tmp_path):
    window = WindowSpec("chr1", 0, 5, 0, 1, 4)
    logp = np.log(np.full((1, 5, 4), 0.25, dtype=np.float32))
    path = tmp_path / "part.npz"
    metadata = {"context_size": 5, "inference_seconds": 0.5, "model": "m5.1"}
    write_score_shard(path, [window], logp, metadata=metadata)
    with np.load(path, allow_pickle=False) as archive:
        assert json.loads(str(archive["metadata_json"].item())) == metadata


def test_list_score_shards_can_allow_unscoreable_scaffolds(tmp_path):
    scored = tmp_path / "scored"
    empty = tmp_path / "empty"
    scored.mkdir()
    empty.mkdir()
    shard = scored / "part-000000.npz"
    shard.touch()

    assert list_score_shards([scored, empty], allow_empty=True) == [shard]
    with pytest.raises(AssertionError, match="no score shards"):
        list_score_shards([scored, empty])


def test_score_window_plans_loads_one_model_for_multiple_scaffolds(
    tmp_path, monkeypatch
):
    tokenizer = object()
    model = nn.Identity()
    factory_calls = {"tokenizer": 0, "model": 0, "genome": 0}

    def _tokenizer_factory(*_args, **_kwargs):
        factory_calls["tokenizer"] += 1
        return tokenizer

    def _model_factory(*_args, **_kwargs):
        factory_calls["model"] += 1
        return model

    class _Genome:
        def __init__(self, _path, *, subset_chroms):
            factory_calls["genome"] += 1
            self.chroms = {chrom: 300 for chrom in subset_chroms}

    calls = []

    def _score_one(
        plan_path,
        _genome_path,
        _shard_dir,
        runtime_path,
        done_path,
        *,
        expected_chrom,
        model: nn.Module,
        tokenizer,
        genome,
        **_kwargs,
    ):
        assert plan_path == tmp_path / f"{expected_chrom}.parquet"
        assert model is test_model
        assert tokenizer is test_tokenizer
        assert genome.chroms[expected_chrom] == 300
        calls.append(expected_chrom)
        runtime = {
            "chrom": expected_chrom,
            "window_count": 1,
            "logical_scored_sequence_count": 2,
            "scored_base_count": 128,
            "shard_count": 1,
            "resumed_shard_count": 0,
            "model_inference_seconds": 2.0,
            "wall_seconds_this_invocation": 2.1,
            "windows_per_second": 0.5,
            "bases_per_second": 64.0,
            "gpu": "test-gpu",
            "peak_vram_bytes": 123,
            "batch_size": 128,
            "num_workers": 4,
            "torch_compile": True,
            "bf16_full_eval": True,
            "eval_accumulation_steps": None,
            "per_shard": [],
        }
        Path(runtime_path).parent.mkdir(parents=True, exist_ok=True)
        Path(runtime_path).write_text(json.dumps(runtime))
        Path(done_path).write_text(json.dumps({"complete": True, **runtime}))
        return runtime

    test_model = model
    test_tokenizer = tokenizer
    monkeypatch.setattr(
        chinchilla_logo.AutoTokenizer,
        "from_pretrained",
        staticmethod(_tokenizer_factory),
    )
    monkeypatch.setattr(
        chinchilla_logo.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(_model_factory),
    )
    monkeypatch.setattr(chinchilla_logo, "Genome", _Genome)
    monkeypatch.setattr(chinchilla_logo, "score_window_plan", _score_one)

    runtime_path = tmp_path / "shards" / "scope.runtime.json"
    done_path = tmp_path / "shards" / "scope.done.json"
    runtime = score_window_plans(
        [tmp_path / "chr1.parquet", tmp_path / "chr2.parquet"],
        ["chr1", "chr2"],
        tmp_path / "genome.fa",
        tmp_path / "shards",
        runtime_path,
        done_path,
        application_commit="a" * 40,
        num_workers=4,
    )

    assert factory_calls == {"tokenizer": 1, "model": 1, "genome": 1}
    assert calls == ["chr1", "chr2"]
    assert runtime["scaffold_count"] == 2
    assert runtime["window_count"] == 2
    assert runtime["scored_base_count"] == 256
    assert runtime_path.is_file() and done_path.is_file()


def test_release_manifest_reconciles_scope_and_hashes_artifacts(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    (release / "README.md").write_text("dataset card\n")
    upload_cache = release / ".cache" / "huggingface"
    upload_cache.mkdir(parents=True)
    (upload_cache / "README.md.metadata").write_text("transient\n")
    chrom_sizes = tmp_path / "chrom.sizes"
    chrom_sizes.write_text("chr1\t12\nchr2\t5\n")
    plan = tmp_path / "coverage.json"
    plan.write_text(
        json.dumps(
            {
                "coverage": {
                    "chrom": "chr1",
                    "chrom_size": 12,
                    "canonical_bases": 10,
                    "noncanonical_bases": 2,
                    "short_run_bases": 1,
                    "border_excluded_bases": 2,
                    "scored_bases": 7,
                    "canonical_run_count": 2,
                    "scoreable_run_count": 1,
                    "window_count": 1,
                }
            }
        )
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"chrom": "chr1", "gpu": "test"}))

    manifest = write_release_manifest(
        release,
        chrom_sizes,
        [plan],
        [runtime],
        application_commit="b" * 40,
    )
    assert manifest["assembly"]["ucsc_span"] == 17
    assert manifest["assembly"]["scoped_span"] == 12
    assert manifest["assembly"]["out_of_scope_bases"] == 5
    assert manifest["coverage"]["scored_bases"] == 7
    assert manifest["runtime"]["scoring"] == [{"chrom": "chr1", "gpu": "test"}]
    assert manifest["runtime"]["artifact_construction"] == []
    assert manifest["files"]["README.md"]["sha256"] == sha256_file(
        release / "README.md"
    )
    assert not any(path.startswith(".cache/") for path in manifest["files"])
    assert "manifest/release.json" not in manifest["files"]

    second_plan = tmp_path / "coverage2.json"
    second_plan.write_text(
        json.dumps(
            {
                "coverage": {
                    "chrom": "chr2",
                    "chrom_size": 5,
                    "canonical_bases": 5,
                    "noncanonical_bases": 0,
                    "short_run_bases": 5,
                    "border_excluded_bases": 0,
                    "scored_bases": 0,
                    "canonical_run_count": 1,
                    "scoreable_run_count": 0,
                    "window_count": 0,
                }
            }
        )
    )
    aggregate_runtime = tmp_path / "aggregate-runtime.json"
    aggregate_runtime.write_text(
        json.dumps({"scope": "multi-scaffold", "scaffold_count": 2})
    )
    full_manifest = write_release_manifest(
        release,
        chrom_sizes,
        [plan, second_plan],
        [aggregate_runtime],
        application_commit="b" * 40,
    )
    assert full_manifest["assembly"]["scoped_span"] == 17
    assert full_manifest["assembly"]["out_of_scope_bases"] == 0
    assert full_manifest["coverage"]["short_run_bases"] == 6
    assert full_manifest["runtime"]["scoring"] == [
        {"scope": "multi-scaffold", "scaffold_count": 2}
    ]
