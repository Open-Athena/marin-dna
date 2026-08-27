# Utilization

Use this snapshot before initial placement and every capacity-changing decision.
It describes current admission headroom; W&B remains the progress authority.

```bash
uv run --project <marin-checkout> --package marin-iris \
  python <skill-path>/scripts/utilization.py --cluster marin
```

Pass exactly one of `--cluster <name>` or `--config <path>`. Repeat `--peer` to
narrow the default production set: `cw-rno2a`, `cw-us-east-02a`, and
`cw-us-east-08a`.

The script calls `iris rpc controller list-peers` and emits:

- UTC observation time and aggregate free/held/total GPUs by variant.
- One target per selected peer backend with GPU variant, free/total GPUs,
  priority-band holds, observation age, and pending/running task counts.

It exits nonzero when a selected peer is missing or unreachable, the availability
metric is unsupported or stale, the expected GPU is absent, or capacity accounting
is inconsistent. Extra response fields are safe.

At batch priority, immediately usable capacity is the reported free amount. Do not
count higher-priority holds, add capacity across separate backends to fit one gang,
or hard-code fleet totals. Reserve planned and newly submitted gangs in the current
fleet plan, then refresh after material actions.

The metric semantics and Backends-page mapping are documented in Iris
[federation placement](https://github.com/marin-community/marin/blob/877bbaddbded9fdb0ffcd7f6f5f00b67d1cb9683/lib/iris/docs/federation.md#L43-L73)
and [observability](https://github.com/marin-community/marin/blob/877bbaddbded9fdb0ffcd7f6f5f00b67d1cb9683/lib/iris/docs/federation.md#L180-L213).
