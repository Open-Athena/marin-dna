"""Operator-side CoreWeave object access without logging credential values."""

from __future__ import annotations

import os
import subprocess
from typing import Any

import boto3
from botocore.config import Config


def coreweave_client(*, refresh_keys: bool = False) -> Any:
    """Use documented CW keys, optionally refreshed on the trusted operator VM."""
    if refresh_keys:
        for variable, secret in [
            ("CW_KEY_ID", "cw-object-storage-key-id"),
            ("CW_KEY_SECRET", "cw-object-storage-key-secret"),
        ]:
            result = subprocess.run(
                [
                    "gcloud",
                    "secrets",
                    "versions",
                    "access",
                    "latest",
                    "--project=hai-gcp-models",
                    "--secret=" + secret,
                ],
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            if result.returncode or not result.stdout.strip():
                raise RuntimeError(f"Cannot refresh required credential {secret}")
            os.environ[variable] = result.stdout.strip()
    return boto3.client(
        "s3",
        endpoint_url="https://cwobject.com",
        aws_access_key_id=os.environ["CW_KEY_ID"],
        aws_secret_access_key=os.environ["CW_KEY_SECRET"],
        region_name="US-EAST-02A",
        config=Config(s3={"addressing_style": "virtual"}, retries={"max_attempts": 5}),
    )
