from __future__ import annotations

from typing import Any, cast

from exp479_mntp.train import finish_wandb_run


class FakeRun:
    def __init__(self) -> None:
        self.exit_codes: list[int] = []

    def finish(self, *, exit_code: int) -> None:
        self.exit_codes.append(exit_code)


class FakeLogger:
    def __init__(self) -> None:
        self.experiment = FakeRun()


def test_finish_wandb_run_closes_process_global_run() -> None:
    logger = FakeLogger()
    finish_wandb_run(cast(Any, logger))
    assert logger.experiment.exit_codes == [0]
