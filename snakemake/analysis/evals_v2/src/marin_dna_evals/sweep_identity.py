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

Identity = str | list[dict[str, str]]


def configured_identities() -> list[Identity]:
    """Collect possible worker identities after applying local config overrides."""
    sources = [
        ("SKYPILOT_GLOBAL_CONFIG", "~/.sky/config.yaml"),
        ("SKYPILOT_PROJECT_CONFIG", ".sky.yaml"),
    ]
    if "SKYPILOT_CONFIG" in os.environ:
        sources = [("SKYPILOT_CONFIG", "")]

    identities: dict[str | None, Identity] = {}
    active_workspace = None
    for variable, default in sources:
        path = Path(os.environ.get(variable, default)).expanduser()
        if variable not in os.environ and not path.exists():
            continue
        config = yaml.safe_load(path.read_text()) or {}
        active_workspace = config.get("active_workspace", active_workspace)
        for workspace, settings in [
            (None, config),
            *config.get("workspaces", {}).items(),
        ]:
            aws = settings.get("aws", {}) if settings else {}
            if "remote_identity" in aws:
                identities[workspace] = aws["remote_identity"]

    if active_workspace:
        identity = identities.get(active_workspace, identities.get(None))
        return [] if identity is None else [identity]
    # A saved server-side workspace preference may select any local workspace.
    return list(identities.values())


def check_clusters(clusters: list[str]) -> None:
    for identity in configured_identities():
        for cluster in clusters:
            check_identity(cluster, identity)


def check_identity(cluster: str, identity: Identity) -> None:
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
