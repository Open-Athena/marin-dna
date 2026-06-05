"""Pentanucleotide annotation + per-5-mer subsampling for cLLR calibration (#267/#270).

Both steps are model-independent (pure functions of genome + position + seed), so
they live here rather than in the GPU eval pipeline. The subsampled artifact is
pinned + seeded → every checkpoint calibrates on the same neutral sites, and it is
kept site-level / alt-agnostic so the LLR path (3 alts) and the entropy atom
(#269, 4-allele marginal) consume one set.
"""


rule annotate_pentanuc:
    """Attach each neutral site's central pentanucleotide (5-mer) from the genome.

Additive (consumes the pinned neutral set, leaves it untouched); the genome is
already wired into this pipeline, so context extraction belongs here."""
    input:
        "results/neutral_sites.parquet",
    output:
        "results/neutral_sites_pentanuc.parquet",
    run:
        sites = pd.read_parquet(input[0])
        # Genome is the get_seq callable (same as enumerate_positions); reads
        # once per chromosome and asserts the 5-mer center == ref.
        out = annotate_pentanucleotide(sites, Genome(config["genome_path"]))
        assert len(out) > 0, "empty pentanuc-annotated neutral set"
        out.to_parquet(output[0], index=False)
        print(
            f"[annotate_pentanuc] {len(out):,} sites, "
            f"{out['pentanuc'].nunique()} distinct 5-mers -> {output[0]}"
        )


rule subsample_neutral:
    """Subsample to at most {n} neutral sites per 5-mer (seeded, deterministic).

`n` is a path wildcard so several caps coexist (e.g. n100, n1000) and evals_v2
pins one by revision — see README."""
    input:
        "results/neutral_sites_pentanuc.parquet",
    output:
        "results/subsampled/neutral_sites_n{n}.parquet",
    wildcard_constraints:
        n=r"\d+",
    run:
        sites = pd.read_parquet(input[0])
        out = subsample_per_context(
            sites, int(wildcards.n), config["subsample_seed"]
        )
        per = out.groupby("pentanuc").size()
        out.to_parquet(output[0], index=False)
        print(
            f"[subsample_neutral] n={wildcards.n} seed={config['subsample_seed']}: "
            f"{len(out):,} sites across {len(per)} 5-mers "
            f"(<= {per.max()} per 5-mer) -> {output[0]}"
        )
