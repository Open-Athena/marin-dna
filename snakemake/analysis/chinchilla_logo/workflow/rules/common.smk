"""Shared issue #419 configuration and exact release paths."""

import subprocess
from pathlib import Path

from marin_dna.data.dna import NUCLEOTIDES
from marin_dna.pipelines.chinchilla_logo import parse_chrom_sizes


def git_commit_sha():
    """Resolve the commit that will be embedded in the dataset card/manifest."""
    override = config.get("commit_sha")
    if override is not None:
        assert len(override) == 40
        return override
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workflow.basedir,
        capture_output=True,
        text=True,
        check=True,
    )
    sha = result.stdout.strip()
    assert len(sha) == 40
    return sha


ASSEMBLY = config["assembly"]["accession"]
REFERENCE_PREFIX = f"results/reference/{ASSEMBLY}"
CHROM_SIZES = f"{REFERENCE_PREFIX}.chrom.sizes.txt"
TWO_BIT = f"{REFERENCE_PREFIX}.2bit"
FASTA = f"{REFERENCE_PREFIX}.fa"
FASTA_INDEX = f"{FASTA}.fai"

FULL_ASSEMBLY = bool(config.get("scaffolds_from_chrom_sizes", False))
RESIDENT_MODEL_ACROSS_SCAFFOLDS = bool(
    config.get("resident_model_across_scaffolds", False)
)
if FULL_ASSEMBLY:
    assert Path(CHROM_SIZES).is_file(), (
        f"full-assembly configuration requires {CHROM_SIZES}; "
        "download that target with the default config before parsing the full DAG"
    )
    SCAFFOLDS = [chrom for chrom, _size in parse_chrom_sizes(CHROM_SIZES)]
else:
    SCAFFOLDS = list(config["scaffolds"])
assert SCAFFOLDS, "configure at least one scaffold"
assert len(SCAFFOLDS) == len(set(SCAFFOLDS)), "scaffolds must be unique"
assert not FULL_ASSEMBLY or RESIDENT_MODEL_ACROSS_SCAFFOLDS, (
    "full-assembly scoring must keep one model resident across scaffolds"
)

CONTEXT_SIZE = int(config["tiling"]["context_size"])
STRIDE = int(config["tiling"]["stride"])
RETAIN_START = int(config["tiling"]["retain_start"])
RETAIN_END = int(config["tiling"]["retain_end"])
PHASE = int(config["tiling"]["phase"])
assert CONTEXT_SIZE == 255, "issue #419 is pinned to the m5.1 255-bp context"
assert RETAIN_END - RETAIN_START == STRIDE

PLAN_OUTPUTS = [f"results/plans/{scaffold}.parquet" for scaffold in SCAFFOLDS]
PLAN_METADATA_OUTPUTS = [
    f"results/plans/{scaffold}.coverage.json" for scaffold in SCAFFOLDS
]
PLAN_SCOPE_DONE = "results/plans/scope.done.json"
if RESIDENT_MODEL_ACROSS_SCAFFOLDS:
    SCORE_DONE_OUTPUTS = ["results/shards/scope.done.json"]
    RUNTIME_OUTPUTS = ["results/shards/scope.runtime.json"]
else:
    SCORE_DONE_OUTPUTS = [
        f"results/shards/{scaffold}.done.json" for scaffold in SCAFFOLDS
    ]
    RUNTIME_OUTPUTS = [
        f"results/shards/{scaffold}.runtime.json" for scaffold in SCAFFOLDS
    ]
BIGWIG_OUTPUTS = [
    f"results/release/bigwig/{kind}/{base}.bw"
    for kind in ("logprob", "logo")
    for base in NUCLEOTIDES
]
GIT_COMMIT_SHA = git_commit_sha()
