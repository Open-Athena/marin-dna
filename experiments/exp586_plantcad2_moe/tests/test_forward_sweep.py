import json
from pathlib import Path

from exp586_moe.inference.forward_sweep import run_plan


def test_resumed_capacity_screen_stops_at_recorded_oom(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            [
                {"name": "fits", "batch_size": 1, "stop_on_oom": True},
                {"name": "limit", "batch_size": 2, "stop_on_oom": True},
                {"name": "unnecessary", "batch_size": 4},
            ]
        )
    )
    output = tmp_path / "results"
    output.mkdir()
    (output / "fits.json").write_text(json.dumps({"status": "passed"}))
    (output / "limit.json").write_text(json.dumps({"status": "oom"}))
    run_plan(plan, tmp_path, output, None)
    assert not (output / "unnecessary.json").exists()
