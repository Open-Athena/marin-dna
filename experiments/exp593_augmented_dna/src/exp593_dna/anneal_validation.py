"""Bounded validation with external Iris profiles instead of a traceback timer.

Two native faults coincided with the first watchdog dump during concurrent
input/restore work. The scientifically bound native trainer stays unchanged.
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
from typing import Any

from exp593_dna import anneal_train

_NATIVE_RUN = anneal_train.run
LOGGER = logging.getLogger(__name__)


def require_validation(config: anneal_train.TrainingConfig) -> None:
    """Reject production identities before dispatch or any diagnostic change."""
    if config.validation_updates != 64:
        raise ValueError("This entry point permits bounded validation only")


def _external_profiles_only(*args: Any, **kwargs: Any) -> None:
    LOGGER.info("In-process traceback watchdog disabled; use Iris task profiles")


def run(config: anneal_train.TrainingConfig) -> None:
    """Suppress only the optional watchdog, retaining all checks and updates."""
    require_validation(config)
    original = faulthandler.dump_traceback_later
    try:
        faulthandler.dump_traceback_later = _external_profiles_only
        _NATIVE_RUN(config)
    finally:
        faulthandler.dump_traceback_later = original


def dispatch(config: anneal_train.TrainingConfig) -> None:
    """Reuse the native resource, priority, and environment dispatcher."""
    require_validation(config)
    original = anneal_train.run
    try:
        anneal_train.run = run
        anneal_train.dispatch(config)
    finally:
        anneal_train.run = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--nodes", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    dispatch(
        anneal_train.TrainingConfig(**vars(parser.parse_args()), validation_updates=64)
    )


if __name__ == "__main__":
    main()
