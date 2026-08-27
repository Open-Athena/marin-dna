# Targets

Translate the operator's GPU scope into the complete candidate grid before
validating placement. By default, include H100 on `cw-us-east-02a` and `cw-rno2a`
and GB200 on `cw-us-east-08a`. Never target the CI cluster `cw-us-west-04a`.

Read current Marin and Iris configuration and inspect the training code's hardware
profiles. Expand the requested scope into exact `(cluster, GPU, nodes, GPUs)` rows.
Use whole nodes—currently eight H100 or four GB200 per node—and only gang sizes the
training code can fit. If profiles are missing, explain the smallest changes needed
and ask before editing training semantics.

For a cold start, equal H100 and GB200 node counts are a useful rough BF16 compute
class: a four-GPU GB200 node has about 1.25 times the dense BF16 peak of an
eight-GPU H100 node. Treat this only as a sizing heuristic. Batch fit, communication,
and measured `target_rate` determine useful placement once evidence exists.

Add every candidate to Operations as `unvalidated`. Only validation or a verified
invalid-target result changes eligibility. Fleet utilization and measured throughput
rank eligible targets; they never define or narrow the candidate grid.
