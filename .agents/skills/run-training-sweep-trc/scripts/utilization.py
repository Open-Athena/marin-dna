"""Emit a strict TPU utilization snapshot from Iris autoscaler status."""

import argparse
import json
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SLICE_STATES = {"requesting", "booting", "initializing", "ready", "failed"}
CAPACITY_STATES = {"available", "in_use", "idle", "degraded"}
AVAILABILITY_STATES = {
    "available",
    "cooldown",
    "requesting",
    "backoff",
    "quota_exceeded",
    "at_max_slices",
}
AVAILABILITY_RANK = {
    "available": 0,
    "cooldown": 1,
    "requesting": 2,
    "backoff": 3,
    "at_max_slices": 4,
    "quota_exceeded": 5,
}
TPU_VARIANT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*-(?P<chips>[1-9][0-9]*)$")

JsonObject = dict[str, Any]


class SnapshotError(RuntimeError):
    """Raised when Iris output cannot support an accurate snapshot."""


@dataclass
class Target:
    region: str
    tpu_slice: str
    chips_per_slice: int
    ready_slices: int = 0
    in_use_slices: int = 0
    ages_ms: list[int] = field(default_factory=list)
    capacity_counts: Counter[str] = field(default_factory=Counter)
    availability_counts: Counter[str] = field(default_factory=Counter)
    availability_reasons: set[str] = field(default_factory=set)


def _object(value: Any, path: str) -> JsonObject:
    if not isinstance(value, dict):
        raise SnapshotError(f"{path} must be an object, got {type(value).__name__}")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SnapshotError(f"{path} must be a list, got {type(value).__name__}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SnapshotError(f"{path} must be a nonempty trimmed string")
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SnapshotError(f"{path} must be an integer")
    if value < 0:
        raise SnapshotError(f"{path} must be nonnegative")
    return value


def _epoch_ms(value: Any, path: str) -> int:
    timestamp = _object(value, path)
    epoch_ms = timestamp.get("epoch_ms")
    if isinstance(epoch_ms, str) and epoch_ms.isdigit():
        epoch_ms = int(epoch_ms)
    return _integer(epoch_ms, f"{path}.epoch_ms")


def _optional_list(parent: JsonObject, key: str, path: str) -> list[Any]:
    if key not in parent:
        return []
    return _list(parent[key], f"{path}.{key}")


def _optional_counts(parent: JsonObject, key: str, path: str) -> Counter[str]:
    if key not in parent:
        return Counter()
    values = _object(parent[key], f"{path}.{key}")
    counts: Counter[str] = Counter()
    for name, count in values.items():
        state = _string(name, f"{path}.{key} key")
        counts[state] = _integer(count, f"{path}.{key}.{state}")
    return counts


def _run_iris(command: list[str], timeout_seconds: float) -> JsonObject:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise SnapshotError(f"Iris executable not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SnapshotError(f"Iris timed out after {timeout_seconds:g}s") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise SnapshotError(
            f"Iris exited {result.returncode}: {shlex.join(command)}\n{detail}"
        )
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)

    try:
        return _object(json.loads(result.stdout), "response")
    except json.JSONDecodeError as exc:
        raise SnapshotError(
            f"Iris stdout is not JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


def _variant(value: Any, path: str) -> tuple[str, int]:
    variant = _string(value, path)
    match = TPU_VARIANT.fullmatch(variant)
    if match is None:
        raise SnapshotError(
            f"{path} is not a TPU slice with a trailing chip count: {variant!r}"
        )
    return variant, int(match.group("chips"))


def _availability(group: JsonObject, path: str) -> tuple[str, str | None]:
    status = _string(group.get("availability_status"), f"{path}.availability_status")
    if status not in AVAILABILITY_STATES:
        raise SnapshotError(
            f"{path}.availability_status has unknown value {status!r}; "
            f"expected one of {sorted(AVAILABILITY_STATES)}"
        )
    reason_value = group.get("availability_reason")
    reason = (
        None
        if reason_value in (None, "")
        else _string(reason_value, f"{path}.availability_reason")
    )
    return status, reason


def _slice_age_and_use(
    slice_info: JsonObject, path: str, now_ms: int
) -> tuple[int | None, bool, str]:
    capacity = _string(slice_info.get("capacity_status"), f"{path}.capacity_status")
    if capacity not in CAPACITY_STATES:
        raise SnapshotError(
            f"{path}.capacity_status has unknown value {capacity!r}; "
            f"expected one of {sorted(CAPACITY_STATES)}"
        )

    vms = _optional_list(slice_info, "vms", path)
    created_at: list[int] = []
    running_tasks = 0
    for vm_index, vm_value in enumerate(vms):
        vm_path = f"{path}.vms[{vm_index}]"
        vm = _object(vm_value, vm_path)
        created_at.append(_epoch_ms(vm.get("created_at"), f"{vm_path}.created_at"))
        running_tasks += _integer(
            vm.get("running_task_count", 0), f"{vm_path}.running_task_count"
        )

    if capacity != "degraded" and not vms:
        raise SnapshotError(f"{path}.vms is empty for ready {capacity!r} slice")
    in_use = running_tasks > 0
    if capacity == "in_use" and not in_use:
        raise SnapshotError(f"{path} says in_use but has no running tasks")
    if capacity in {"available", "idle"} and in_use:
        raise SnapshotError(f"{path} says {capacity!r} but has running tasks")

    if not created_at:
        return None, in_use, capacity
    age_ms = now_ms - min(created_at)
    if age_ms < 0:
        raise SnapshotError(f"{path} has a future VM creation timestamp")
    return age_ms, in_use, capacity


def _group_slices(
    group: JsonObject, path: str, now_ms: int
) -> tuple[int, int, list[int], Counter[str]]:
    slices = _optional_list(group, "slices", path)
    actual_states: Counter[str] = Counter()
    ready: list[tuple[JsonObject, str]] = []

    for slice_index, slice_value in enumerate(slices):
        slice_path = f"{path}.slices[{slice_index}]"
        slice_info = _object(slice_value, slice_path)
        state = _string(slice_info.get("state"), f"{slice_path}.state")
        if state not in SLICE_STATES:
            raise SnapshotError(
                f"{slice_path}.state has unknown value {state!r}; "
                f"expected one of {sorted(SLICE_STATES)}"
            )
        actual_states[state] += 1
        if state == "ready":
            ready.append((slice_info, slice_path))

    declared_states = _optional_counts(group, "slice_state_counts", path)
    unknown_declared = set(declared_states) - SLICE_STATES
    if unknown_declared:
        raise SnapshotError(
            f"{path}.slice_state_counts has unknown states {sorted(unknown_declared)}"
        )
    if actual_states != declared_states:
        raise SnapshotError(
            f"{path} slice counts disagree: slices={dict(actual_states)}, "
            f"slice_state_counts={dict(declared_states)}"
        )

    in_use = 0
    ages_ms: list[int] = []
    capacity_counts: Counter[str] = Counter()
    for slice_info, slice_path in ready:
        age_ms, slice_in_use, capacity = _slice_age_and_use(
            slice_info, slice_path, now_ms
        )
        in_use += int(slice_in_use)
        capacity_counts[capacity] += 1
        if age_ms is not None:
            ages_ms.append(age_ms)
    return len(ready), in_use, ages_ms, capacity_counts


def summarize(response: JsonObject, observed_at: datetime) -> JsonObject:
    status = _object(response.get("status"), "response.status")
    groups = _list(status.get("groups"), "response.status.groups")
    if not groups:
        raise SnapshotError("response.status.groups is empty")

    now_ms = int(observed_at.timestamp() * 1000)
    targets: dict[tuple[str, str], Target] = {}
    saw_tpu_group = False

    for group_index, group_value in enumerate(groups):
        path = f"response.status.groups[{group_index}]"
        group = _object(group_value, path)
        _string(group.get("name"), f"{path}.name")
        device_type = _string(group.get("device_type"), f"{path}.device_type").lower()
        if device_type != "tpu":
            continue
        saw_tpu_group = True

        tpu_slice, chips = _variant(
            group.get("device_variant"), f"{path}.device_variant"
        )
        region = _string(group.get("region"), f"{path}.region")
        availability, reason = _availability(group, path)
        ready, in_use, ages_ms, capacity_counts = _group_slices(group, path, now_ms)
        if ready == 0:
            continue

        key = (region, tpu_slice)
        target = targets.setdefault(key, Target(region, tpu_slice, chips))
        if target.chips_per_slice != chips:
            raise SnapshotError(f"conflicting chip counts for target {key}")
        target.ready_slices += ready
        target.in_use_slices += in_use
        target.ages_ms.extend(ages_ms)
        target.capacity_counts.update(capacity_counts)
        target.availability_counts[availability] += 1
        if reason is not None:
            target.availability_reasons.add(reason)

    if not saw_tpu_group:
        raise SnapshotError("response contains no structured TPU groups")
    if not targets:
        raise SnapshotError("response contains no ready TPU slices")

    output_targets: list[JsonObject] = []
    region_totals: defaultdict[str, Counter[str]] = defaultdict(Counter)
    fleet_ready = 0
    fleet_in_use = 0

    for target in sorted(
        targets.values(),
        key=lambda item: (-item.ready_slices, item.region, item.tpu_slice),
    ):
        if len(target.ages_ms) != target.ready_slices:
            raise SnapshotError(
                f"cannot compute complete average age for "
                f"{target.region}/{target.tpu_slice}: "
                f"{len(target.ages_ms)} of {target.ready_slices} ready slices"
            )
        fleet_ready += target.ready_slices
        fleet_in_use += target.in_use_slices
        region_totals[target.region]["ready_slices"] += target.ready_slices
        region_totals[target.region]["in_use_slices"] += target.in_use_slices
        worst_status = max(
            target.availability_counts, key=AVAILABILITY_RANK.__getitem__
        )
        average_age_seconds = round(sum(target.ages_ms) / len(target.ages_ms) / 1000)
        output_targets.append(
            {
                "region": target.region,
                "tpu_slice": target.tpu_slice,
                "chips_per_slice": target.chips_per_slice,
                "ready_slices": target.ready_slices,
                "in_use_slices": target.in_use_slices,
                "in_use_percent": round(
                    100 * target.in_use_slices / target.ready_slices, 1
                ),
                "average_age_seconds": average_age_seconds,
                "slice_capacity_counts": dict(sorted(target.capacity_counts.items())),
                "autoscaler_status": worst_status,
                "autoscaler_reasons": sorted(target.availability_reasons),
            }
        )

    return {
        "observed_at_utc": observed_at.isoformat(),
        "fleet": {
            "ready_slices": fleet_ready,
            "in_use_slices": fleet_in_use,
            "in_use_percent": round(100 * fleet_in_use / fleet_ready, 1),
            "regions": {
                region: dict(counts)
                for region, counts in sorted(
                    region_totals.items(),
                    key=lambda item: (-item[1]["ready_slices"], item[0]),
                )
            },
        },
        "targets": output_targets,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit strict JSON describing current Iris TPU fleet utilization."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cluster", help="Named Iris cluster, for example marin")
    source.add_argument("--config", type=Path, help="Exact Iris cluster config path")
    parser.add_argument(
        "--iris-bin",
        default="iris",
        help="Iris executable path (default: iris from PATH)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60,
        help="RPC command timeout (default: 60)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.timeout_seconds <= 0:
        print("ERROR: --timeout-seconds must be positive", file=sys.stderr)
        return 2
    if args.config is not None and not args.config.is_file():
        print(f"ERROR: config does not exist: {args.config}", file=sys.stderr)
        return 2

    command = [args.iris_bin]
    if args.cluster is not None:
        command.extend(["--cluster", args.cluster])
    else:
        command.extend(["--config", str(args.config)])
    command.extend(["rpc", "controller", "get-autoscaler-status"])

    try:
        response = _run_iris(command, args.timeout_seconds)
        snapshot = summarize(response, datetime.now(UTC))
    except SnapshotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    json.dump(snapshot, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
