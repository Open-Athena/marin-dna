from exp586_moe.full_eval import FullEvalConfig, centered, task_jobs


def test_full_worker_plan_includes_both_contexts_and_only_full_sv():
    manifest = {
        "plan": [
            {"task": "task", "worker": 0},
            {"task": "sv_impact", "worker": 0},
            {"task": "other", "worker": 1},
        ],
        "af_plan": [{"task": "maize-af", "worker": 0}],
    }
    jobs = task_jobs(manifest, 0)
    assert [(profile, kind, chunk["task"]) for profile, _, kind, chunk in jobs] == [
        ("context-8192", "tasks", "task"),
        ("context-8192", "af", "maize-af"),
        ("context-2048", "tasks", "task"),
        ("context-2048", "af", "maize-af"),
        ("context-8192", "tasks", "sv_impact"),
    ]


def test_full_config_and_center_crop_are_explicit():
    config = FullEvalConfig(
        condition="scratch",
        run_id="exp586-final-eval-smoke-worker-00",
        cluster="cw-rno2a",
        checkpoint="s3://example/checkpoint",
        checkpoint_metadata_digest="a" * 64,
        requests_uri="s3://example/requests.json",
        requests_sha256="b" * 64,
        output_prefix="s3://example/results/scratch",
        worker_index=0,
    )
    assert config.artifact_root == "s3://example/results/scratch"
    assert centered("AAAACCCC", 4) == "AACC"
