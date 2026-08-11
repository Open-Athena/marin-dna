"""Audit the frozen MarinDNA m5.1 versus Evo 2 40B efficiency claims.

The training calculation reconstructs m5.1's inherited lineage from the
recorded W&B ``throughput/total_*`` fields. The inference calculation retains
the precision of the published benchmark: MarinDNA's rate is exact to the
reported measurement, whereas the archived Evo 2 rate is rounded at source.
"""

from __future__ import annotations

import json

INITIAL_UNIFORM = {
    "description": "uniform three-region run through inherited checkpoint step 20000",
    "run": (
        "https://wandb.ai/eric-czech/marin/runs/"
        "dna-bolinas-mix-v0.9-p1B-i0-uniform-2ba217"
    ),
    "global_step": 20_000,
    "total_tokens": 41_945_137_152,
    "total_gflops": 286_770_406_301.2425,
}
UNIFORM_CONTINUATION = {
    "description": "uniform three-region continuation initialized from step 20000",
    "run": (
        "https://wandb.ai/eric-czech/marin/runs/"
        "dna-bolinas-mix-v0.9-p1B-i18-uniform_to_uniform_1-84cd83"
    ),
    "global_step": 29_579,
    "total_tokens": 62_033_756_160,
    "total_gflops": 424_112_225_308.2722,
}
FIVE_REGION_CONTINUATION = {
    "description": "m5.1 five-region continuation with inherited global step",
    "run": (
        "https://wandb.ai/eric-czech/marin/runs/"
        "dna-bolinas-mix-v0.9-p1B-i24-exp135-zoonomia-m5.1-bef41e"
    ),
    "global_step": 59_158,
    "cumulative_tokens_from_uniform_continuation": 124_065_415_168,
    "cumulative_gflops_from_uniform_continuation": 848_210_112_813.1196,
}
EVO2_TRAINING = {
    "model": "Evo 2 40B",
    "parameters": 40_300_000_000,
    "tokens": 9_300_000_000_000,
    "estimated_flops": 2.25e24,
    "source": ("https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full#T1"),
    "qualification": (
        "The source describes this as an estimate that does not account for "
        "mixed precision or pretraining context length."
    ),
}
INFERENCE = {
    "hardware": "one Lambda NVIDIA GH200",
    "hourly_price_usd": 2.29,
    "marin_variants_per_hour": 1_467_776,
    "marin_seconds_per_million": 2_452.69,
    "evo2_variants_per_hour_rounded": 630,
    "evo2_seconds_per_million_implied_by_rounded_rate": 5_714_286,
    "source": (
        "https://github.com/Open-Athena/marin-dna/issues/354#issuecomment-5154623667"
    ),
    "qualification": (
        "The Evo 2 variants/hour value was rounded at source, so the ratio is "
        "approximate rather than an unrounded measurement."
    ),
}


def audit_efficiency() -> dict[str, object]:
    """Return the claim inputs, arithmetic, and precision qualifications."""
    five_region_tokens = (
        FIVE_REGION_CONTINUATION["cumulative_tokens_from_uniform_continuation"]
        - UNIFORM_CONTINUATION["total_tokens"]
    )
    five_region_gflops = (
        FIVE_REGION_CONTINUATION["cumulative_gflops_from_uniform_continuation"]
        - UNIFORM_CONTINUATION["total_gflops"]
    )
    assert five_region_tokens > 0
    assert five_region_gflops > 0

    marin_total_tokens = (
        INITIAL_UNIFORM["total_tokens"]
        + UNIFORM_CONTINUATION["total_tokens"]
        + five_region_tokens
    )
    marin_total_flops = 1e9 * (
        INITIAL_UNIFORM["total_gflops"]
        + UNIFORM_CONTINUATION["total_gflops"]
        + five_region_gflops
    )
    training_ratio = EVO2_TRAINING["estimated_flops"] / marin_total_flops
    throughput_ratio = (
        INFERENCE["marin_variants_per_hour"]
        / INFERENCE["evo2_variants_per_hour_rounded"]
    )

    assert round(marin_total_tokens / 1e9, 1) == 166.0
    assert round(marin_total_flops / 1e21, 3) == 1.135
    assert round(training_ratio, -1) == 1_980
    assert round(throughput_ratio, -1) == 2_330
    assert (
        abs(
            1e6 * 3600 / INFERENCE["marin_variants_per_hour"]
            - INFERENCE["marin_seconds_per_million"]
        )
        < 0.01
    )

    return {
        "training": {
            "marin_stages": [
                INITIAL_UNIFORM,
                UNIFORM_CONTINUATION,
                {
                    **FIVE_REGION_CONTINUATION,
                    "incremental_tokens": five_region_tokens,
                    "incremental_gflops": five_region_gflops,
                },
            ],
            "marin_total_tokens": marin_total_tokens,
            "marin_total_flops": marin_total_flops,
            "evo2": EVO2_TRAINING,
            "evo2_to_marin_ratio": training_ratio,
            "headline_ratio_rounded": 1_980,
        },
        "inference": {
            **INFERENCE,
            "evo2_to_marin_time_ratio_from_rounded_rate": throughput_ratio,
            "headline_ratio_rounded": 2_330,
        },
    }


def main() -> None:
    print(json.dumps(audit_efficiency(), indent=2))


if __name__ == "__main__":
    main()
