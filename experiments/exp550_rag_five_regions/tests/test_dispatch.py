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


def test_native_resume_is_required_and_uses_a_separate_pilot_run(monkeypatch):
    prefix = "gs://marin-us-east1/MarinDNA/exp550_rag_five_regions"
    source = (
        f"{prefix}/checkpoints/dna-exp550-rag46m-five-regions-v1-pilot-mb5/"
        "2026.09.10.1/checkpoints/step-10"
    )
    monkeypatch.setenv("MARIN_PREFIX", prefix)
    monkeypatch.setattr(launch, "resolve_version", lambda *_: "2026.09.10.2")

    def build(resume=None):
        step = launch.build_training(
            {}, pilot=True, per_device=5, region="us-east1", resume_pilot_from=resume
        )
        config = step.build_config(
            SimpleNamespace(
                output_path=f"{prefix}/{step.name}/2026.09.10.2",
                runtime_arg=lambda key: step.runtime_args[key],
            )
        )
        return config.pod

    original, resumed = build(), build(source)
    assert resumed.output_path != original.output_path
    assert resumed.train_config.trainer.id != original.train_config.trainer.id
    assert resumed.train_config.trainer.load_checkpoint is True
    assert resumed.train_config.trainer.load_checkpoint_path == source
    assert original.train_config.trainer.load_checkpoint_path is None
    assert resumed.train_config.optimizer == original.train_config.optimizer
    assert resumed.train_config.trainer.num_train_steps == 20
    with pytest.raises(ValueError, match="synthetic pilot"):
        launch.build_training(
            {}, pilot=False, per_device=5, region="us-east1", resume_pilot_from=source
        )
    with pytest.raises(ValueError, match="intermediate native"):
        build(source.replace("step-10", "step-20"))
