# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3==1.41.5", "PyYAML==6.0.3"]
# ///
"""Small transport contract tests; no cloud access or ML imports."""

import base64
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    "staging", Path(__file__).with_name("stage-final-checkpoint.py")
)
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)

CONFIG = {
    "model_type": "qwen3",
    "vocab_size": 8,
    "hidden_size": 640,
    "intermediate_size": 2560,
    "num_hidden_layers": 7,
    "num_attention_heads": 5,
    "num_key_value_heads": 5,
    "head_dim": 128,
    "max_position_embeddings": 10240,
    "tie_word_embeddings": False,
}
SOURCE = (
    "gs://marin-eu-west4/MarinDNA/exp550_rag_five_regions/checkpoints/"
    "dna-exp550-rag46m-five-regions-v1/2026.09.10.9/hf/step-100000"
)
PAYLOADS = {name: b"tiny fixture" for name in staging.FILES}
PAYLOADS["config.json"] = json.dumps(CONFIG).encode()


class FakeS3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.writes = []

    def list_objects_v2(self, **kwargs):
        return {"Contents": [{"Key": staging.PREFIX + name} for name in self.objects]}

    def get_object(self, *, Bucket, Key):
        name = Key.removeprefix(staging.PREFIX)
        if name not in self.objects:
            raise staging.ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[name])}

    def put_object(self, *, Bucket, Key, Body, ContentMD5, IfNoneMatch, **kwargs):
        name = Key.removeprefix(staging.PREFIX)
        assert name not in self.objects
        assert IfNoneMatch == "*"
        value = Body if isinstance(Body, bytes) else Body.read()
        assert ContentMD5 == base64.b64encode(hashlib.md5(value).digest()).decode()
        self.writes.append(name)
        self.objects[name] = value


class TransportTests(unittest.TestCase):
    def test_pressure_interrupt_survives_concurrent_child_cleanup(self):
        child = Mock()

        def cleared_during_poll():
            staging.ACTIVE = None

        child.poll.side_effect = cleared_during_poll
        with (
            patch.object(staging, "ACTIVE", child),
            patch.object(staging.os, "kill") as kill,
        ):
            staging.interrupt_transport()
        child.terminate.assert_called_once_with()
        kill.assert_called_once_with(staging.os.getpid(), staging.signal.SIGINT)

    def test_pressure_interrupt_survives_child_termination_failure(self):
        child = Mock()
        child.poll.return_value = None
        child.terminate.side_effect = ProcessLookupError
        with (
            patch.object(staging, "ACTIVE", child),
            patch.object(staging.os, "kill") as kill,
            self.assertRaises(ProcessLookupError),
        ):
            staging.interrupt_transport()
        kill.assert_called_once_with(staging.os.getpid(), staging.signal.SIGINT)

    def run_stage(self, client, *, payloads=None, corrupt_download=False, apply=True):
        payloads = payloads or PAYLOADS
        registry = {"models": [{"name": staging.MODEL, "gcs_path": SOURCE}]}

        def fake_gcloud(arguments, output=None):
            if arguments[0] == "objects":
                name = arguments[2].rsplit("/", 1)[1]
                value = payloads[name]
                return {
                    "name": name,
                    "generation": "123",
                    "size": str(len(value)),
                    "md5_hash": base64.b64encode(hashlib.md5(value).digest()).decode(),
                }
            assert arguments[0] == "cat" and arguments[1].endswith("#123")
            name = arguments[1].rsplit("/", 1)[1].split("#")[0]
            output.write_bytes(b"corrupt" if corrupt_download else payloads[name])

        with (
            patch.object(
                staging.subprocess,
                "check_output",
                side_effect=["commit\n", json.dumps(registry)],
            ),
            patch.object(staging, "gcloud", side_effect=fake_gcloud),
            patch.object(staging.boto3, "client", return_value=client),
        ):
            return staging.stage(apply=apply)

    def test_corrupt_source_download_prevents_writes(self):
        client = FakeS3()
        with self.assertRaisesRegex(RuntimeError, "bytes differ"):
            self.run_stage(client, corrupt_download=True)
        self.assertEqual(client.writes, [])

    def test_wrong_geometry_prevents_writes(self):
        client = FakeS3()
        payloads = {
            **PAYLOADS,
            "config.json": json.dumps({**CONFIG, "vocab_size": 9}).encode(),
        }
        with self.assertRaisesRegex(RuntimeError, "configuration: vocab_size"):
            self.run_stage(client, payloads=payloads)
        self.assertEqual(client.writes, [])

    def test_later_existing_mismatch_prevents_all_writes(self):
        client = FakeS3({"tokenizer_config.json": b"unrelated checkpoint"})
        with self.assertRaisesRegex(RuntimeError, "bytes differ"):
            self.run_stage(client)
        self.assertEqual(client.writes, [])

    def test_verified_objects_and_marker_are_written_conditionally(self):
        client = FakeS3()
        report = self.run_stage(client)
        self.assertTrue(report["applied"])
        self.assertEqual(client.writes, [*staging.FILES, ".snakemake_timestamp"])
        self.assertEqual(set(report["verified_objects"]), set(staging.FILES))
        for name in staging.FILES:
            self.assertEqual(
                report["verified_objects"][name]["sha256"],
                hashlib.sha256(PAYLOADS[name]).hexdigest(),
            )

    def test_matching_objects_are_reused(self):
        client = FakeS3(PAYLOADS)
        self.run_stage(client)
        self.assertEqual(client.writes, [".snakemake_timestamp"])
        client.writes.clear()
        self.run_stage(client)
        self.assertEqual(client.writes, [])

    def test_unknown_destination_objects_prevent_writes(self):
        client = FakeS3({"unrelated.txt": b"retain"})
        with self.assertRaisesRegex(RuntimeError, "Unexpected objects"):
            self.run_stage(client)
        self.assertEqual(client.writes, [])

    def test_plan_is_read_only(self):
        client = FakeS3()
        report = self.run_stage(client, apply=False)
        self.assertFalse(report["applied"])
        self.assertEqual(client.objects, {})

    def test_failed_guard_records_status_and_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "failure.json"
            with (
                patch("sys.argv", ["stage", "--receipt", str(receipt)]),
                patch.object(
                    staging, "shared_node_guard", side_effect=BlockingIOError("busy")
                ),
                patch("builtins.print"),
                self.assertRaises(SystemExit) as outcome,
            ):
                staging.main()
            self.assertEqual(outcome.exception.code, 1)
            report = json.loads(receipt.read_text())
            self.assertEqual(report["error_type"], "BlockingIOError")
            self.assertFalse(report["applied"])
            self.assertIn("started_at", report)
            self.assertIn("finished_at", report)
            self.assertGreater(report["peak_rss_upper_bound_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
