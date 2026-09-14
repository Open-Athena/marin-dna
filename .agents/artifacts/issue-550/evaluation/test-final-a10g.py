# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3==1.41.5", "PyYAML==6.0.3"]
# ///
"""Bounded launch/cleanup contracts; no cloud access or ML imports."""

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load("run-10k-a10g")
watch = load("watch-final-a10g")


def receipt():
    return {
        "model": watch.MODEL,
        "completed": True,
        "with_probes": True,
        "exit_status": 0,
        "split": "train",
        "canonical_s3_content_sha256_verified": True,
        "files": {
            f"results/{kind}/{watch.MODEL}/{cohort}.{ext}": {
                "bytes": 1,
                "sha256": "a" * 64,
            }
            for kind, ext in [
                ("scores", "parquet"),
                ("metrics", "parquet"),
                ("probe", "parquet"),
                ("probe", "joblib"),
                ("probe_metrics", "parquet"),
            ]
            for cohort in ["mendelian_traits", "complex_traits", "sge"]
        },
    }


class Contracts(unittest.TestCase):
    def test_final_targets_include_all_three_probe_metrics(self):
        targets = driver.evaluation_targets(watch.MODEL, include_probes=True)
        self.assertEqual(len(set(targets)), 6)
        self.assertIn(f"results/probe_metrics/{watch.MODEL}/sge.parquet", targets)
        self.assertIn(
            f"results/metrics/{watch.MODEL}/mendelian_traits.parquet", targets
        )
        self.assertTrue(all("step-100000/" in target for target in targets))

    def test_intermediate_targets_stay_without_probes(self):
        targets = driver.evaluation_targets("intermediate", include_probes=False)
        self.assertEqual(len(targets), 3)
        self.assertTrue(all("/metrics/" in target for target in targets))

    def test_incomplete_or_wrong_completion_cannot_release_disk(self):
        for key, value in [
            ("model", "wrong"),
            ("completed", False),
            ("with_probes", False),
            ("split", "test"),
            ("canonical_s3_content_sha256_verified", False),
            ("files", {}),
        ]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                watch.validate_receipt(receipt() | {key: value})

    def test_copy_precedes_stream_and_no_new_ssh_after_completion(self):
        events = []
        worker = {
            "InstanceId": "i-test",
            "Tags": [
                {"Key": "Name", "Value": "codex-issue550-vep100k-a10g-bf16"},
                {"Key": "issue", "Value": "550"},
            ],
        }

        def copy(*args, **kwargs):
            events.append("copy")

        def stream(*args, **kwargs):
            events.append("stream")
            process = MagicMock()
            process.stdout = io.StringIO(
                "DEVELOPMENT_VEP_OUTPUTS "
                + json.dumps(receipt())
                + "\nVEP_WORKER_EXIT 0\n"
            )
            process.__enter__.return_value = process
            return process

        def aws(command, **kwargs):
            action = command[2]
            events.append(action)
            return json.dumps(
                {"Reservations": [{"Instances": [worker]}]}
                if action == "describe-instances"
                else {}
            )

        with tempfile.TemporaryDirectory() as temporary:
            argv = [
                "watch",
                "--instance",
                "i-test",
                "--host",
                "ubuntu@test",
                "--ssh-key",
                "/test/key",
                "--output",
                temporary,
            ]
            with (
                patch("sys.argv", argv),
                patch.object(watch.subprocess, "run", side_effect=copy),
                patch.object(watch.subprocess, "Popen", side_effect=stream),
                patch.object(watch.subprocess, "check_output", side_effect=aws),
            ):
                watch.main()
            self.assertTrue((Path(temporary) / "step-100000-completed.json").is_file())
        self.assertEqual(
            events, ["copy", "stream", "describe-instances", "terminate-instances"]
        )

    def test_missing_completion_never_terminates_worker(self):
        process = MagicMock()
        process.stdout = io.StringIO("failed before receipt\n")
        process.__enter__.return_value = process
        with tempfile.TemporaryDirectory() as temporary:
            argv = [
                "watch",
                "--instance",
                "i-test",
                "--host",
                "ubuntu@test",
                "--ssh-key",
                "/test/key",
                "--output",
                temporary,
            ]
            with (
                patch("sys.argv", argv),
                patch.object(watch.subprocess, "run"),
                patch.object(watch.subprocess, "Popen", return_value=process),
                patch.object(watch.subprocess, "check_output") as aws,
                self.assertRaises(RuntimeError),
            ):
                watch.main()
            aws.assert_not_called()


if __name__ == "__main__":
    unittest.main()
