import json

import pytest

from marin_dna.experiment_record import (
    RECORD_FILENAME,
    SCHEMA_VERSION,
    ExperimentDep,
    ExperimentRecord,
    ExperimentRun,
    read_experiment_record,
    utc_now_iso,
    write_experiment_record,
)

DATASET_SHA = "d2bea760f6416775772699b821b266d3ae87245e"
CODE_SHA = "eaac2efffb73d33b87ba75bcf5521809af74fec7"


def make_record(**overrides) -> ExperimentRecord:
    """A valid exp417-shaped record; override single fields to probe validation."""
    fields = {
        "name": "dna-exp417",
        "issue": "https://github.com/Open-Athena/marin-dna/issues/417",
        "created_at": "2026-08-11T12:00:00+00:00",
        "config": {"train_batch_size": 8192, "train_steps": 5000, "seed": 0},
        "deps": [
            ExperimentDep("hf-dataset", "marin-dna/vertebrate-v1-cds", DATASET_SHA),
            ExperimentDep("git", "https://github.com/Open-Athena/marin-dna", CODE_SHA),
        ],
        "runs": [
            ExperimentRun(
                run_id="dna-exp417-cds-mammals-only-p255m-b2m-5k",
                output_path="gs://bucket/checkpoints/dna-exp417-cds-mammals-only",
                arm="mammals_only",
            ),
            ExperimentRun(
                run_id="dna-exp417-cds-combined-vertebrates-p255m-b2m-5k",
                output_path="gs://bucket/checkpoints/dna-exp417-cds-combined",
                arm="combined_vertebrates",
                wandb_url="https://wandb.ai/org/marin/runs/abc123",
            ),
        ],
        "provenance": {"experiment_commit": CODE_SHA, "marin_pin": "0.2.39"},
    }
    fields.update(overrides)
    return ExperimentRecord(**fields)


def test_round_trip_local(tmp_path):
    record = make_record()
    prefix = str(tmp_path / "experiments" / "dna-exp417")
    url = write_experiment_record(record, prefix)
    assert url.endswith(f"dna-exp417/{RECORD_FILENAME}")
    assert read_experiment_record(prefix) == record


def test_round_trip_memory_url():
    record = make_record()
    prefix = "memory://records/round-trip/dna-exp417"
    write_experiment_record(record, prefix)
    assert read_experiment_record(prefix) == record


def test_written_file_is_readable_json(tmp_path):
    record = make_record()
    write_experiment_record(record, str(tmp_path))
    payload = json.loads((tmp_path / RECORD_FILENAME).read_text())
    assert payload["name"] == "dna-exp417"
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["runs"][0]["arm"] == "mammals_only"
    assert payload["deps"][0]["revision"] == DATASET_SHA


def test_experiment_number():
    assert make_record().experiment_number == 417


def test_utc_now_iso_is_timezone_aware():
    record = make_record(created_at=utc_now_iso())
    assert record.experiment_number == 417


@pytest.mark.parametrize("name", ["exp417", "dna-exp", "dna-exp0", "dna-exp417x"])
def test_bad_name_rejected(name):
    with pytest.raises(AssertionError):
        make_record(name=name)


def test_issue_number_must_match_experiment_number():
    with pytest.raises(AssertionError, match="must track issue #417"):
        make_record(issue="https://github.com/Open-Athena/marin-dna/issues/416")


def test_issue_must_be_github_issue_url():
    with pytest.raises(AssertionError, match="not a GitHub issue URL"):
        make_record(issue="https://github.com/Open-Athena/marin-dna/pull/417")


def test_naive_created_at_rejected():
    with pytest.raises(AssertionError, match="timezone-aware"):
        make_record(created_at="2026-08-11T12:00:00")


def test_empty_runs_rejected():
    with pytest.raises(AssertionError, match="at least one run"):
        make_record(runs=[])


def test_duplicate_run_ids_rejected():
    run = ExperimentRun(run_id="dna-exp417-a", output_path="gs://b/1")
    dup = ExperimentRun(run_id="dna-exp417-a", output_path="gs://b/2")
    with pytest.raises(AssertionError, match="duplicate run_ids"):
        make_record(runs=[run, dup])


def test_duplicate_output_paths_rejected():
    run_a = ExperimentRun(run_id="dna-exp417-a", output_path="gs://b/same")
    run_b = ExperimentRun(run_id="dna-exp417-b", output_path="gs://b/same")
    with pytest.raises(AssertionError, match="duplicate output_paths"):
        make_record(runs=[run_a, run_b])


def test_run_id_must_carry_experiment_name():
    run = ExperimentRun(run_id="cds-mammals-only", output_path="gs://b/1")
    with pytest.raises(AssertionError, match="does not carry"):
        make_record(runs=[run])


def test_run_id_prefix_collision_rejected():
    """'dna-exp41' must not accept a run tagged 'dna-exp417-...'."""
    run = ExperimentRun(run_id="dna-exp417-arm", output_path="gs://b/1")
    with pytest.raises(AssertionError, match="does not carry"):
        make_record(
            name="dna-exp41",
            issue="https://github.com/Open-Athena/marin-dna/issues/41",
            runs=[run],
        )


def test_bad_dep_kind_rejected():
    with pytest.raises(AssertionError, match="unknown dep kind"):
        ExperimentDep("dataset", "marin-dna/vertebrate-v1-cds", DATASET_SHA)


@pytest.mark.parametrize("revision", ["main", DATASET_SHA[:12], DATASET_SHA.upper()])
def test_unpinned_dep_revision_rejected(revision):
    with pytest.raises(AssertionError, match="40-hex sha"):
        ExperimentDep("hf-dataset", "marin-dna/vertebrate-v1-cds", revision)


def test_empty_run_output_path_rejected():
    with pytest.raises(AssertionError, match="output_path must be non-empty"):
        ExperimentRun(run_id="dna-exp417-a", output_path="")


def test_non_string_provenance_rejected():
    with pytest.raises(AssertionError, match="provenance"):
        make_record(provenance={"steps": 5000})


def test_non_json_config_rejected():
    with pytest.raises(TypeError):
        make_record(config={"tpu_types": {"v5p-8", "v6e-8"}})


def test_wrong_schema_version_rejected():
    with pytest.raises(AssertionError, match="schema_version"):
        make_record(schema_version=SCHEMA_VERSION + 1)


def test_tampered_record_fails_on_read(tmp_path):
    write_experiment_record(make_record(), str(tmp_path))
    record_file = tmp_path / RECORD_FILENAME
    payload = json.loads(record_file.read_text())
    payload["schema_version"] = SCHEMA_VERSION + 1
    record_file.write_text(json.dumps(payload))
    with pytest.raises(AssertionError, match="schema_version"):
        read_experiment_record(str(tmp_path))


def test_read_missing_record_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_experiment_record(str(tmp_path / "nowhere"))
