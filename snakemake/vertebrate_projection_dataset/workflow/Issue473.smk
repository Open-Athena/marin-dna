"""Standalone additive projection and trace entry point for issue #473."""


configfile: "config/config.yaml"


include: "rules/common.smk"
include: "rules/anchors.smk"
include: "rules/staging.smk"
include: "rules/projection.smk"
include: "rules/dataset.smk"
include: "rules/issue_473.smk"
include: "rules/issue_473_fixed.smk"
include: "rules/issue_473_trace.smk"
