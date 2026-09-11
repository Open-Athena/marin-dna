from types import SimpleNamespace

import pytest
from fray.iris_backend import convert_environment
from fray.types import ResourceConfig
from marin_dna_exp550 import launch


def test_child_job_carries_its_setup_instead_of_inheriting_the_image_uv(monkeypatch):
    submitted = []
    completed = []

    def submit(job):
        submitted.append(job)
        return SimpleNamespace(
            wait=lambda *, raise_on_failure: completed.append(raise_on_failure)
        )

    monkeypatch.setattr(
        launch, "current_client", lambda: SimpleNamespace(submit=submit)
    )
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-test-credential")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("MARIN_PREFIX", "gs://test/prefix")
    monkeypatch.setenv("UV_PROJECT", "/app/experiments/exp550_rag_five_regions")
    resources = ResourceConfig.with_tpu("v6e-8", regions=["us-east1"])
    request = launch.TrainingRequest(
        pod=SimpleNamespace(resources=resources), datasets={}, pilot=True
    )

    launch.dispatch(request)

    assert completed == [True]
    assert len(submitted) == 1
    job = submitted[0]
    assert job.resources == resources
    environment = convert_environment(job.environment, resources.device)
    assert environment is not None
    assert environment.extras == ["tpu"]
    assert environment.env_vars["WANDB_API_KEY"] == "synthetic-test-credential"
    assert environment.env_vars["UV_PROJECT"].endswith("exp550_rag_five_regions")
    assert len(environment.setup_scripts) == 1
    setup = environment.setup_scripts[0]
    assert setup.index("/uv/0.11.31/install.sh") < setup.index("uv sync")
    assert "--extra tpu" in setup and "--python 3.12" in setup
    assert "synthetic-test-credential" not in setup


@pytest.mark.parametrize(
    "region,bucket",
    [
        ("us-east1", "marin-us-east1"),
        ("us-east5", "marin-us-east5"),
        ("europe-west4", "marin-eu-west4"),
    ],
)
def test_native_resume_is_required_and_uses_a_separate_pilot_run(
    monkeypatch, region, bucket
):
    prefix = f"gs://{bucket}/MarinDNA/exp550_rag_five_regions"
    source = (
        f"{prefix}/checkpoints/dna-exp550-rag46m-five-regions-v1-pilot-mb5/"
        "2026.09.10.1/checkpoints/step-10"
    )
    monkeypatch.setenv("MARIN_PREFIX", prefix)
    monkeypatch.setattr(launch, "resolve_version", lambda *_: "2026.09.10.2")

    def build(resume=None):
        step = launch.build_training(
            {}, pilot=True, per_device=5, region=region, resume_pilot_from=resume
        )
        config = step.build_config(
            SimpleNamespace(
                output_path=f"{prefix}/{step.name}/2026.09.10.2",
                runtime_arg=lambda key: step.runtime_args[key],
            )
        )
        return config.pod

    original, resumed = build(), build(source)
    assert original.resources == ResourceConfig.with_tpu(
        "v6e-8", regions=[region], cpu=16, ram="48g", disk="80g", preemptible=True
    )
    if region == "europe-west4":
        assert original.train_config.trainer.id.endswith("-europe-west4")
    assert resumed.output_path != original.output_path
    assert resumed.train_config.trainer.id != original.train_config.trainer.id
    assert resumed.train_config.trainer.load_checkpoint is True
    assert resumed.train_config.trainer.load_checkpoint_path == source
    assert original.train_config.trainer.load_checkpoint_path is None
    assert resumed.train_config.optimizer == original.train_config.optimizer
    assert resumed.train_config.trainer.num_train_steps == 20
    with pytest.raises(ValueError, match="synthetic pilot"):
        launch.build_training(
            {}, pilot=False, per_device=5, region=region, resume_pilot_from=source
        )
    with pytest.raises(ValueError, match="intermediate native"):
        build(source.replace("step-10", "step-20"))
    monkeypatch.setenv("MARIN_PREFIX", prefix + "-wrong")
    with pytest.raises(ValueError, match="MARIN_PREFIX must be"):
        build()


def test_four_chip_fallback_preserves_recipe_and_separates_pilot(monkeypatch):
    prefix = "gs://marin-us-east1/MarinDNA/exp550_rag_five_regions"
    monkeypatch.setenv("MARIN_PREFIX", prefix)
    monkeypatch.setattr(launch, "resolve_version", lambda *_: "2026.09.10.8")
    steps = [
        launch.build_training(
            {}, pilot=True, per_device=5, region="us-east1", tpu_variant=variant
        )
        for variant in ("v6e-8", "v6e-4")
    ]
    assert steps[0].name != steps[1].name
    assert steps[1].name.endswith("-v6e-4")
    configs = [
        step.build_config(
            SimpleNamespace(
                output_path=f"{prefix}/checkpoints/test",
                runtime_arg=lambda key, step=step: step.runtime_args[key],
            )
        ).pod.train_config
        for step in steps
    ]
    first, fallback = configs
    assert first.model == fallback.model
    assert first.optimizer == fallback.optimizer
    assert first.data == fallback.data
    assert first.trainer.train_batch_size == fallback.trainer.train_batch_size == 200
    assert (
        first.trainer.per_device_parallelism
        == fallback.trainer.per_device_parallelism
        == 5
    )
    assert first.trainer.num_train_steps == fallback.trainer.num_train_steps == 20
    assert steps[1].runtime_args["train_resources"] == ResourceConfig.with_tpu(
        "v6e-4", regions=["us-east1"], cpu=16, ram="48g", disk="80g", preemptible=True
    )
    with pytest.raises(ValueError, match="supported TPU variants"):
        launch.build_training(
            {}, pilot=True, per_device=5, region="us-east1", tpu_variant="v6e-16"
        )


def test_production_memory_reservation_preserves_training_recipe(monkeypatch):
    monkeypatch.setenv(
        "MARIN_PREFIX", "gs://marin-eu-west4/MarinDNA/exp550_rag_five_regions"
    )
    monkeypatch.setattr(launch, "resolve_version", lambda *_: "2026.09.10.9")
    step = launch.build_training({}, pilot=False, per_device=5, region="europe-west4")
    request = step.build_config(
        SimpleNamespace(
            output_path="gs://marin-eu-west4/MarinDNA/exp550_rag_five_regions/checkpoints/dna-exp550-rag46m-five-regions-v1/2026.09.10.9",
            runtime_arg=lambda key: step.runtime_args[key],
        )
    )
    assert request.pod.resources == ResourceConfig.with_tpu(
        "v6e-8",
        regions=["europe-west4"],
        cpu=16,
        ram="256g",
        disk="80g",
        preemptible=True,
    )
    assert request.pod.train_config.trainer.train_batch_size == 200
    assert request.pod.train_config.trainer.num_train_steps == 100000
    assert request.pod.train_config.trainer.per_device_parallelism == 5
