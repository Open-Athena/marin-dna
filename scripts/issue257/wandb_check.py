"""Compact status of the exp257 online-repro wandb run (for the babysit poller).

Optional argv[1] = display-name regex (default the base repro run; pass
'issue257-mps1$' for the no-packing run).
"""

import sys

import wandb

pat = (
    sys.argv[1] if len(sys.argv) > 1 else "exp232-ccre-step4999-online-repro-issue257$"
)
api = wandb.Api(timeout=20)
rs = list(
    api.runs(
        "gonzalobenegas/marin",
        filters={"display_name": {"$regex": pat}},
        per_page=3,
    )
)
if not rs:
    print("norun")
else:
    r = rs[0]
    d = r.summary.get("lm_eval/mendelian_traits_255/distal/fwd/auprc", "-")
    g = r.summary.get("lm_eval/mendelian_traits_255/_global_/avg/auprc", "-")
    rt = r.summary.get("_runtime", "-")
    print(f"{r.state} distal_fwd={d} global_avg={g} rt={rt}s")
