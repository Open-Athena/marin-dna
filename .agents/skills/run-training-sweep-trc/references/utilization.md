# Utilization

Use this optional snapshot for cold starts or when measured placements stop working.
It is a hint, never capacity truth or a source of target eligibility.

Run from an environment that provides the `iris` executable:

```bash
uv run --project <marin-checkout> --package marin-iris \
  python <skill-path>/scripts/utilization.py --cluster marin
```

Pass exactly one of `--cluster <name>` or `--config <path>`. Use `--iris-bin
<path>` only when `iris` is not on `PATH`.

The script emits JSON to stdout with:

- `observed_at_utc`, fleet and per-region ready/in-use totals.
- One target per `(region, TPU slice)` currently represented by ready slices.
- Ready/in-use counts, percent in use, and average slice age per target.
- Slice-capacity counts plus autoscaler status and any reasons.

Priority-band mix is intentionally omitted: it does not guide placement and requires
a second, independently timed scheduler snapshot.

It calls Iris `get-autoscaler-status` and requires structured TPU variant, region,
slice state, and capacity fields. It exits nonzero on CLI failure, malformed JSON,
missing fields, inconsistent counts, or unknown Iris states. Extra fields are safe.

Investigate failures when useful, but do not block placement if an accurate snapshot
cannot be obtained. Continue with measured throughput and grid exploration.
