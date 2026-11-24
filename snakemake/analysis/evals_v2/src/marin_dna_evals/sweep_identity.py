# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml==6.0.3"]
# ///
"""Check local SkyPilot worker identities before dispatching an autodown sweep."""

import os
import sys
from fnmatch import fnmatchcase
from pathlib import Path

import yaml


def configured_identity() -> str | list[dict[str, str]] | None:
    """Load the local AWS identity, applying project overrides to user defaults."""
    sources = [
        ("SKYPILOT_GLOBAL_CONFIG", "~/.sky/config.yaml"),
        ("SKYPILOT_PROJECT_CONFIG", ".sky.yaml"),
    ]
    if "SKYPILOT_CONFIG" in os.environ:
        sources = [("SKYPILOT_CONFIG", "")]

    identity = None
    for variable, default in sources:
        path = Path(os.environ.get(variable, default)).expanduser()
        if variable not in os.environ and not path.exists():
            continue
        config = yaml.safe_load(path.read_text())
        if config is not None and "remote_identity" in config.get("aws", {}):
            identity = config["aws"]["remote_identity"]
    return identity


def check_clusters(clusters: list[str]) -> None:
    identity = configured_identity()
    for cluster in clusters:
        profile = identity
        if isinstance(identity, list):
            profile = next(
                (
                    value
                    for entry in identity
                    for pattern, value in entry.items()
                    if fnmatchcase(cluster, pattern)
                ),
                None,
            )
        if isinstance(profile, str) and profile.casefold() == "bolinasjob":
            raise ValueError(
                f"{cluster} selects BolinasJob, which lacks EC2 teardown permissions. "
                "Launch it separately and run sky down from the launcher."
            )


if __name__ == "__main__":
    try:
        check_clusters(sys.argv[1:])
    except (OSError, ValueError, yaml.YAMLError) as error:
        sys.exit(f"[parallel_sweep] {error}")
