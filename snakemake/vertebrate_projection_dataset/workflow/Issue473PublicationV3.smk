"""Additive staged-source publication entry point for issue #473."""


configfile: "config/issue_473_publication.yaml"


include: "rules/issue_473_publication.smk"
include: "rules/issue_473_publication_v3.smk"
