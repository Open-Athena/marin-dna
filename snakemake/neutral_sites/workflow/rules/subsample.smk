"""Pentanucleotide annotation, ACGT-window filtering, and per-5-mer subsampling
for cLLR calibration (#267/#270).

All three steps are model-independent (pure functions of genome + position +
window + seed), so the calibration-ready set is built and pinned **here**, once,
and reused by every checkpoint in the GPU eval pipeline (evals_v2) — never
re-filtered per model. The artifact is seeded + kept site-level / alt-agnostic so
the LLR path (3 alts) and the entropy atom (#269, 4-allele marginal) consume one set.

Order matters: filter to scoreable sites **before** subsampling, so each 5-mer's
``n`` sites are drawn from its *clean* sites (a full ``n`` per cell, not ``n`` minus
whatever the filter later removes).

Genome reads go through a plain (uncompressed) FASTA (`stage_genome_fasta`):
pyfaidx random access on the bgzipped reference is pathologically slow for the
genome-wide spans these rules touch (one whole-chromosome read each), so we mmap a
plain file instead. The plain FASTA + its `.fai` are kept in S3 next to the
bgzipped reference (`genome_fasta_path`) so this is a one-time *download*, never a
re-decompress.
"""


rule stage_genome_fasta:
    """Download the plain (uncompressed) reference FASTA from S3 for fast,
    genome-wide pyfaidx access (mmap). The uncompressed reference lives in S3 next
    to the bgzipped one precisely so this is a download, not a per-run `zcat`.
    The `.fai` is **built locally** by the first ``Genome(...)`` open (a few seconds
    for a standard fixed-width FASTA) rather than fetched as a separate S3 object —
    so the index is always consistent with this exact file (no stale/partial-`.fai`
    → wrong-byte-offset risk) and there's no untracked sidecar for `temp` to leak.
    Local-only scratch (`temp`): the ~3 GB file is removed once its consumers finish."""
    output:
        temp("results/genome.fa"),
    shell:
        "aws s3 cp {config[genome_fasta_path]} {output}"


rule annotate_pentanuc:
    """Attach each neutral site's central pentanucleotide (5-mer) from the genome.

Additive (consumes the pinned neutral set, leaves it untouched); reads the plain
FASTA once per chromosome and asserts the 5-mer center == ref."""
    input:
        sites="results/neutral_sites.parquet",
        genome="results/genome.fa",
    output:
        "results/neutral_sites_pentanuc.parquet",
    run:
        sites = pd.read_parquet(input.sites)
        out = annotate_pentanucleotide(sites, Genome(input.genome))
        assert len(out) > 0, "empty pentanuc-annotated neutral set"
        out.to_parquet(output[0], index=False)
        print(
            f"[annotate_pentanuc] {len(out):,} sites, "
            f"{out['pentanuc'].nunique()} distinct 5-mers -> {output[0]}"
        )


rule filter_scoreable:
    """Drop sites whose centered {w}-bp model window isn't all-ACGT (assembly-gap N
    or off-chromosome edge).

The gLM scoring kernel asserts ACGT over the variant's window, so an N trips it;
filtering here (once) lets every checkpoint reuse one scoreable set. `w` is a path
wildcard — use the largest model window (512) so the set is valid for all models (a
window clean at 512 bp is clean at 255/256). Runs before `subsample_neutral`."""
    input:
        sites="results/neutral_sites_pentanuc.parquet",
        genome="results/genome.fa",
    output:
        "results/scoreable/neutral_sites_pentanuc_w{w}.parquet",
    wildcard_constraints:
        w=r"\d+",
    run:
        sites = pd.read_parquet(input.sites)
        out = filter_acgt_window_sites(sites, Genome(input.genome), int(wildcards.w))
        assert len(out) > 0, "every site filtered out — wrong window/genome?"
        out.to_parquet(output[0], index=False)
        print(
            f"[filter_scoreable] w={wildcards.w}: {len(out):,} scoreable sites "
            f"({out['pentanuc'].nunique()} distinct 5-mers) -> {output[0]}"
        )


rule subsample_neutral:
    """Subsample to at most {n} scoreable sites per 5-mer (seeded, deterministic),
    from the window-{w} ACGT-filtered set.

`n` and `w` are path wildcards so several caps / windows coexist; evals_v2 pins one
(by `subsample_n` + `scoreable_window`) — see README."""
    input:
        "results/scoreable/neutral_sites_pentanuc_w{w}.parquet",
    output:
        "results/subsampled/neutral_sites_n{n}_w{w}.parquet",
    wildcard_constraints:
        n=r"\d+",
        w=r"\d+",
    run:
        sites = pd.read_parquet(input[0])
        n_cap = int(wildcards.n)
        out = subsample_per_context(sites, n_cap, config["subsample_seed"])
        per = out.groupby("pentanuc").size()
        n_depleted = int((per < n_cap).sum())  # 5-mers with fewer than n sites
        out.to_parquet(output[0], index=False)
        print(
            f"[subsample_neutral] n={n_cap} w={wildcards.w} "
            f"seed={config['subsample_seed']}: {len(out):,} sites across {len(per)} "
            f"5-mers (<= {per.max()} per 5-mer; {n_depleted} 5-mers below the cap, "
            f"min {per.min()}) -> {output[0]}"
        )
