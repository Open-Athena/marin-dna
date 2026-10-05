"""Emit a strict CoreWeave GPU capacity snapshot from Iris federation peers."""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]

PRODUCTION_PEERS = {
    "cw-rno2a": "h100",
    "cw-us-east-02a": "h100",
    "cw-us-east-08a": "gb200",
}
AVAILABILITY_VERSION = 2


class SnapshotError(RuntimeError):
    """Raised when Iris output cannot support an accurate capacity snapshot."""


def _object(value: object, path: str) -> JsonObject:
    if not isinstance(value, dict):
        raise SnapshotError(f"{path} must be an object")
    return value


def _list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise SnapshotError(f"{path} must be a list")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise SnapshotError(f"{path} must be a nonempty string")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise SnapshotError(f"{path} must be a boolean")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool):
        raise SnapshotError(f"{path} must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        try:
            result = int(value)
        except ValueError as error:
            raise SnapshotError(f"{path} must be an integer") from error
    else:
        raise SnapshotError(f"{path} must be an integer")
    if result < 0:
        raise SnapshotError(f"{path} must be nonnegative")
    return result


def _counts(value: object, path: str) -> dict[str, int]:
    raw = _object(value, path)
    result: dict[str, int] = {}
    for key, amount in raw.items():
        if not isinstance(key, str) or not key:
            raise SnapshotError(f"{path} has an invalid resource key")
        normalized = key.casefold()
        if normalized in result:
            raise SnapshotError(f"{path} repeats resource {normalized!r}")
        result[normalized] = _integer(amount, f"{path}.{key}")
    return result


def _run_iris(command: list[str], timeout_seconds: float) -> JsonObject:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise SnapshotError(f"Iris executable not found: {command[0]}") from error
    except subprocess.TimeoutExpired as error:
        raise SnapshotError(
            f"Iris command timed out after {timeout_seconds:g}s"
        ) from error
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise SnapshotError(f"Iris command failed ({result.returncode}): {detail}")
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise SnapshotError("Iris output is not valid JSON") from error
    return _object(response, "response")


def summarize(
    response: JsonObject,
    observed_at: datetime,
    *,
    peers: tuple[str, ...] = tuple(PRODUCTION_PEERS),
    max_age_seconds: float = 90,
) -> JsonObject:
    if observed_at.tzinfo is None:
        raise SnapshotError("observed_at must include a timezone")
    if not peers:
        raise SnapshotError("at least one peer is required")
    if len(set(peers)) != len(peers):
        raise SnapshotError("selected peers contain duplicates")
    if max_age_seconds <= 0:
        raise SnapshotError("max_age_seconds must be positive")
    unknown = set(peers) - PRODUCTION_PEERS.keys()
    if unknown:
        raise SnapshotError(f"unknown production peers: {sorted(unknown)}")

    raw_peers = _list(response.get("peers"), "response.peers")
    peer_by_id: dict[str, JsonObject] = {}
    for index, value in enumerate(raw_peers):
        path = f"response.peers[{index}]"
        peer = _object(value, path)
        peer_id = _string(peer.get("peer_id"), f"{path}.peer_id")
        if peer_id in peer_by_id:
            raise SnapshotError(f"response repeats peer {peer_id!r}")
        peer_by_id[peer_id] = peer

    targets: list[JsonObject] = []
    fleet_by_gpu: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "free_gpus": 0,
            "held_gpus": 0,
            "total_gpus": 0,
            "held_by_band": defaultdict(int),
        }
    )
    now_ms = int(observed_at.astimezone(UTC).timestamp() * 1000)

    for peer_id in peers:
        if peer_id not in peer_by_id:
            raise SnapshotError(f"selected peer {peer_id!r} is missing")
        peer = peer_by_id[peer_id]
        if not _boolean(peer.get("reachable"), f"peer {peer_id}.reachable"):
            raise SnapshotError(f"selected peer {peer_id!r} is unreachable")
        backends = _list(peer.get("backends"), f"peer {peer_id}.backends")
        expected_gpu = PRODUCTION_PEERS[peer_id]
        matched = 0

        for backend_index, value in enumerate(backends):
            path = f"peer {peer_id}.backends[{backend_index}]"
            backend = _object(value, path)
            backend_id = _string(backend.get("backend_id"), f"{path}.backend_id")
            availability = _object(backend.get("availability"), f"{path}.availability")
            version = _integer(
                availability.get("version"), f"{path}.availability.version"
            )
            if version != AVAILABILITY_VERSION:
                raise SnapshotError(
                    f"{path} reports availability version {version}; "
                    f"expected {AVAILABILITY_VERSION}"
                )
            observation_ms = _integer(
                availability.get("observation_epoch_ms"),
                f"{path}.availability.observation_epoch_ms",
            )
            age_seconds = (now_ms - observation_ms) / 1000
            if age_seconds < 0:
                raise SnapshotError(f"{path} availability observation is in the future")
            if age_seconds > max_age_seconds:
                raise SnapshotError(
                    f"{path} availability is stale ({age_seconds:.1f}s > {max_age_seconds:g}s)"
                )

            free = _counts(availability.get("amounts"), f"{path}.availability.amounts")
            total = _counts(
                availability.get("total_amounts"),
                f"{path}.availability.total_amounts",
            )
            if set(free) != set(total):
                raise SnapshotError(f"{path} free and total resource keys disagree")
            if expected_gpu not in free:
                continue
            matched += 1

            held_by_band: dict[str, int] = {}
            for held_index, held_value in enumerate(
                _list(
                    availability.get("held_by_band"),
                    f"{path}.availability.held_by_band",
                )
            ):
                held_path = f"{path}.availability.held_by_band[{held_index}]"
                held = _object(held_value, held_path)
                band = _string(held.get("band"), f"{held_path}.band")
                if band in held_by_band:
                    raise SnapshotError(f"{path} repeats priority band {band!r}")
                amounts = _counts(held.get("amounts"), f"{held_path}.amounts")
                held_by_band[band] = amounts.get(expected_gpu, 0)

            free_gpus = free[expected_gpu]
            total_gpus = total[expected_gpu]
            held_gpus = sum(held_by_band.values())
            if free_gpus > total_gpus:
                raise SnapshotError(f"{path} free {expected_gpu} exceeds total")
            if free_gpus + held_gpus != total_gpus:
                raise SnapshotError(
                    f"{path} {expected_gpu} accounting disagrees: "
                    f"free={free_gpus}, held={held_gpus}, total={total_gpus}"
                )

            pending_tasks = _integer(
                backend.get("pending_task_count"), f"{path}.pending_task_count"
            )
            running_tasks = _integer(
                backend.get("running_task_count"), f"{path}.running_task_count"
            )
            targets.append(
                {
                    "cluster": peer_id,
                    "backend_id": backend_id,
                    "gpu_variant": expected_gpu.upper(),
                    "free_gpus": free_gpus,
                    "held_gpus": held_gpus,
                    "total_gpus": total_gpus,
                    "in_use_percent": round(100 * held_gpus / total_gpus, 1)
                    if total_gpus
                    else 0.0,
                    "held_by_band": dict(sorted(held_by_band.items())),
                    "pending_task_count": pending_tasks,
                    "running_task_count": running_tasks,
                    "observation_epoch_ms": observation_ms,
                    "observation_age_seconds": round(age_seconds, 3),
                }
            )
            aggregate = fleet_by_gpu[expected_gpu.upper()]
            aggregate["free_gpus"] += free_gpus
            aggregate["held_gpus"] += held_gpus
            aggregate["total_gpus"] += total_gpus
            for band, amount in held_by_band.items():
                aggregate["held_by_band"][band] += amount

        if matched == 0:
            raise SnapshotError(
                f"selected peer {peer_id!r} has no {expected_gpu.upper()} availability backend"
            )

    fleet: dict[str, JsonObject] = {}
    for gpu_variant, aggregate in sorted(fleet_by_gpu.items()):
        fleet[gpu_variant] = {
            "free_gpus": aggregate["free_gpus"],
            "held_gpus": aggregate["held_gpus"],
            "total_gpus": aggregate["total_gpus"],
            "held_by_band": dict(sorted(aggregate["held_by_band"].items())),
        }

    targets.sort(
        key=lambda row: (row["gpu_variant"], row["cluster"], row["backend_id"])
    )
    return {
        "observed_at_utc": observed_at.astimezone(UTC).isoformat(),
        "fleet_by_gpu": fleet,
        "targets": targets,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit strict JSON describing CoreWeave GPU fleet capacity."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cluster", help="Named Iris cluster, normally marin")
    source.add_argument("--config", type=Path, help="Exact Iris cluster config path")
    parser.add_argument(
        "--peer",
        action="append",
        choices=tuple(PRODUCTION_PEERS),
        help="Production peer to include; repeat as needed (default: all)",
    )
    parser.add_argument("--iris-bin", default="iris", help="Iris executable path")
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--max-age-seconds", type=float, default=90)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.timeout_seconds <= 0 or args.max_age_seconds <= 0:
        print("ERROR: timeout and max age must be positive", file=sys.stderr)
        return 2
    if args.config is not None and not args.config.is_file():
        print(f"ERROR: config does not exist: {args.config}", file=sys.stderr)
        return 2

    command = [args.iris_bin]
    if args.cluster is not None:
        command.extend(["--cluster", args.cluster])
    else:
        command.extend(["--config", str(args.config)])
    command.extend(["rpc", "controller", "list-peers"])

    try:
        response = _run_iris(command, args.timeout_seconds)
        snapshot = summarize(
            response,
            datetime.now(UTC),
            peers=tuple(args.peer or PRODUCTION_PEERS),
            max_age_seconds=args.max_age_seconds,
        )
    except SnapshotError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    json.dump(snapshot, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
