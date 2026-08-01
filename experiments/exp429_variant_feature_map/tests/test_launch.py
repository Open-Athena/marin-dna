from launch import CPU_CLUSTER, GPU_CLUSTER, sky_command


def test_stage_commands_use_pinned_commit_and_expected_cluster() -> None:
    commit = "a" * 40
    panel = sky_command("panel", commit)
    extract = sky_command("extract", commit)
    analyze = sky_command("analyze", commit)
    spatial = sky_command("spatial", commit)
    assert panel[:4] == ["sky", "launch", "-c", CPU_CLUSTER]
    assert extract[:4] == ["sky", "launch", "-c", GPU_CLUSTER]
    assert analyze[:4] == ["sky", "launch", "-c", CPU_CLUSTER]
    assert spatial[:4] == ["sky", "launch", "-c", GPU_CLUSTER]
    assert f"EXPERIMENT_COMMIT={commit}" in panel
    assert f"EXPERIMENT_COMMIT={commit}" in extract
    assert f"ANALYSIS_COMMIT={commit}" in analyze
    assert f"SPATIAL_COMMIT={commit}" in spatial
