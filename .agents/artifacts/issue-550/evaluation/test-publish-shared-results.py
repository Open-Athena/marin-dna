# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3==1.41.5"]
# ///
"""Small publication-contract tests; no cloud requests or biological data."""

import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock

from botocore.exceptions import ClientError

spec = importlib.util.spec_from_file_location(
    "publisher", Path(__file__).with_name("publish-shared-results.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture() -> dict:
    commit = "a" * 40
    return {
        "model": module.MODEL,
        "consumer_commit": commit,
        "split": "train",
        "cohort_sizes": module.COHORTS.copy(),
        "checkpoint_sha256": "a4fd7d562c61aade4f86d7a3349c5894d3a18c93357d797b6a1e3cd8171d3dad",
        "harness_sha256": "6631d35f9ae0afc754c623a2e3960682e8a834a28001906279745882f99cb73b",
        "staging_prefix": f"s3://marin-us-east-02a/{module.source_prefix(commit)}/",
        "canonical_prefix": "s3://oa-bolinas/" + module.AWS_PREFIX,
        "completed": True,
        "scores_complete": True,
        "files": {path: {"bytes": 100, "sha256": "b" * 64} for path in module.EXPECTED},
    }


class Contracts(unittest.TestCase):
    def test_exact_completed_receipt(self):
        module.validate_receipt(fixture(), "a" * 40)

    def test_wrong_identity_split_or_destination_rejected(self):
        for key, value in [
            ("model", "another-model"),
            ("consumer_commit", "c" * 40),
            ("split", "test"),
            ("checkpoint_sha256", "d" * 64),
            ("harness_sha256", "d" * 64),
            ("canonical_prefix", "s3://other/"),
            ("staging_prefix", "s3://other/"),
            ("completed", False),
            ("scores_complete", False),
        ]:
            with self.subTest(key=key):
                receipt = fixture()
                receipt[key] = value
                with self.assertRaises(AssertionError):
                    module.validate_receipt(receipt, "a" * 40)

    def test_missing_or_extra_output_rejected(self):
        receipt = fixture()
        receipt["files"].pop(next(iter(receipt["files"])))
        with self.assertRaises(AssertionError):
            module.validate_receipt(receipt, "a" * 40)
        receipt = fixture()
        receipt["files"]["../../other"] = {"bytes": 100, "sha256": "b" * 64}
        with self.assertRaises(AssertionError):
            module.validate_receipt(receipt, "a" * 40)

    def test_invalid_size_hash_or_cohort_rejected(self):
        for mutation in [{"bytes": 0}, {"bytes": 2**30}, {"sha256": "not-a-digest"}]:
            receipt = fixture()
            next(iter(receipt["files"].values())).update(mutation)
            with self.assertRaises(AssertionError):
                module.validate_receipt(receipt, "a" * 40)
        receipt = fixture()
        receipt["cohort_sizes"]["sge"] -= 1
        with self.assertRaises(AssertionError):
            module.validate_receipt(receipt, "a" * 40)

    def test_absent_destination_is_not_a_match(self):
        client = Mock()
        client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadObject"
        )
        self.assertFalse(
            module.existing_matches(
                client, "example", {"bytes": 100, "sha256": "b" * 64}
            )
        )

    def test_existing_destination_requires_server_checksum(self):
        client = Mock()
        item = {"bytes": 100, "sha256": "b" * 64}
        matching = {
            "ContentLength": 100,
            "ChecksumSHA256": module.checksum(item["sha256"]),
        }
        client.head_object.return_value = matching
        self.assertTrue(module.existing_matches(client, "example", item))
        for delta in [{"ContentLength": 101}, {"ChecksumSHA256": "different"}]:
            client.head_object.return_value = matching | delta
            with self.assertRaises(ValueError):
                module.existing_matches(client, "example", item)
        client.head_object.return_value = {
            "ContentLength": 100,
            "Metadata": copy.copy(item),
        }
        with self.assertRaises(ValueError):
            module.existing_matches(client, "example", item)

    def test_access_denied_is_not_treated_as_missing(self):
        client = Mock()
        client.head_object.side_effect = ClientError(
            {"Error": {"Code": "403"}}, "HeadObject"
        )
        with self.assertRaises(ClientError):
            module.existing_matches(
                client, "example", {"bytes": 100, "sha256": "b" * 64}
            )


if __name__ == "__main__":
    unittest.main()
