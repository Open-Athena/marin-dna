from types import SimpleNamespace

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
