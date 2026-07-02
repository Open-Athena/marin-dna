"""HuggingFace dataset-card (``README.md``) generators for ``genomes-v*``.

The ``snakemake/training_dataset`` pipeline uploads its shards with
``hf upload-large-folder``, which silently skips top-level files — so the
datasets historically shipped without a card. These generators produce a
recipe-accurate, commit-pinned ``README.md`` that the upload rules push with a
separate ``hf upload … README.md`` call (same trick the zoonomia pipeline
uses).

Two card flavors, because the two dataset families differ in schema *and* in
what the sequence case means:

- **Training** datasets (``…-genome_set-{set}-intervals-{recipe}_{w}_{s}``):
  multi-genome ``.jsonl.zst`` shards; sequence case = **soft-masked repeats**
  carried over from the source assembly; reverse complements included.
- **Validation** datasets (``…-validation-intervals-{recipe}_{w}_{w}``):
  a single human ``validation.parquet``; sequence case is **overwritten to
  encode phyloP conservation** (uppercase = conserved). No reverse complements.

Conflating those two case conventions is the easiest way to misread the data,
so both cards call the distinction out explicitly.
"""

from __future__ import annotations

from pathlib import Path

from marin_dna.pipelines.training_dataset import (
    _GITHUB_PIPELINE_PATH,
    _GITHUB_REPO,
)

# ---------------------------------------------------------------------------
# Per-recipe and per-genome_set descriptive blurbs.
#
# Recipe logic is faithful to
# ``snakemake/training_dataset/dataset_creation/workflow/rules/intervals.smk``.
# Keep these in sync if a recipe's selection logic changes.
# ---------------------------------------------------------------------------

# Short region label, used in titles / pretty_name.
RECIPE_TITLES: dict[str, str] = {
    "v1": "promoters",
    "v5": "CDS",
    "v15": "downstream-of-CDS",
    "v17": "cCRE enhancers",
    "v18": "conserved cCRE enhancers",
    "v20": "segmentation enhancers",
    "v30": "projected conserved enhancers",
    "v31": "projected conserved enhancers (≥50 bp)",
    "v32": "projected phastCons enhancers",
    "v33": "projected phastCons enhancers (≥50 bp)",
}

# One-paragraph description of what each recipe selects.
RECIPE_BLURBS: dict[str, str] = {
    "v1": (
        "Promoter windows around protein-coding transcription start sites: "
        "256 bp upstream + 256 bp downstream of each mRNA (protein-coding) "
        "transcript's TSS (`get_promoters(..., mRNA_only=True)`), clipped to "
        "`defined` (non-`N`) sequence. Comparable to the "
        "`gpn-animal-promoter-dataset` used in the TraitGym paper."
    ),
    "v5": (
        "Coding sequence (CDS): every annotated CDS interval, kept if "
        "20 bp–10 kb long, extended by 20 bp on each side (to capture "
        "splice-site signal at exon boundaries), then expanded to a minimum "
        "length of 256 bp and intersected with `defined`."
    ),
    "v15": (
        "The 256 bp immediately downstream (3′) of each CDS "
        "(`get_downstream_of_CDS(dist=256)`), clipped to `defined`. A "
        "proximal-3′ / terminator-adjacent counterpart to the upstream "
        "promoter recipe (`v1`)."
    ),
    "v17": (
        "ENCODE cCRE V4 enhancer-like signatures (distal dELS + proximal "
        "pELS), each resized to 255 bp and intersected with `defined`. "
        "Defined natively on the human genome (GRCh38)."
    ),
    "v18": (
        "ENCODE cCRE V4 enhancer-like signatures (dELS + pELS) retained only "
        "where ≥20 bp are evolutionarily conserved (phyloP-241way ≥ 2.27), "
        "each resized to 255 bp and intersected with `defined`. Human "
        "(GRCh38)."
    ),
    "v20": (
        "Segmentation-predicted enhancers: 128 bp bins in the top 1% of "
        "per-genome enhancer logit from the whole-genome `EnhancerSegmenter` "
        "model ([issue #118](https://github.com/Open-Athena/marin-dna/issues/118)), "
        "each resized to a 255 bp window centered on the bin, with annotated "
        "transcript exons subtracted (`get_exons_for_masking` — all exons, "
        "minus low-quality biotypes where annotated) and intersected with "
        "`defined`."
    ),
    "v30": (
        "Cross-species enhancers: human ENCODE cCRE ELS with ≥20 conserved bp "
        "(phyloP-241way ≥ 2.27), projected onto each target genome by mmseqs2 "
        "best-hit alignment (`-s 7.5 --max-accept 1`), resized to 255 bp and "
        "intersected with `scannable` (`defined` minus low-quality-excluded "
        "exons). On human this is the native set; on other genomes it is the "
        "alignment projection."
    ),
    "v31": (
        "As `v30`, but the upstream per-cCRE conservation filter requires "
        "≥50 conserved bp (phyloP-241way ≥ 2.27) instead of ≥20."
    ),
    "v32": (
        "As `v30`, but conservation is measured with the Zoonomia 43-primate "
        "phastCons track (≥0.961 per base, calibrated to match phyloP-241way "
        "2.27), requiring ≥20 conserved bp. Note: primate-conserved by "
        "construction, so projection to non-primate mammals may be weaker."
    ),
    "v33": ("As `v32` (phastCons-43p ≥ 0.961) but requiring ≥50 conserved bp."),
}

# Display name + taxonomic / membership description for each genome_set. The
# ``{n_genomes}`` placeholder is filled from the resolved genome list at card
# build time (counts can drift as the curated list grows).
GENOME_SET_TITLES: dict[str, str] = {
    "animals": "Animals",
    "vertebrates": "Vertebrates",
    "mammals": "Mammals",
    "primates": "Primates",
    "humans": "Humans",
    "human_mouse": "Human + mouse",
    "enhancer_seg_mammals_v1": "20 mammals (segmentation)",
    "mammals_seg20": "20 mammals",
    "animals_order204": "204 animals (one per order)",
}

GENOME_SET_BLURBS: dict[str, str] = {
    "animals": (
        "All NCBI RefSeq genomes in kingdom **Metazoa** that pass the "
        "`genome_selection` quality and taxonomic-dedup filters "
        "(**{n_genomes} genomes**). The broadest evolutionary timescale in "
        "the family."
    ),
    "vertebrates": (
        "NCBI RefSeq genomes in phylum **Chordata** (**{n_genomes} "
        "genomes**) — a vertebrate-scoped subset of `animals`."
    ),
    "mammals": ("NCBI RefSeq genomes in class **Mammalia** (**{n_genomes} genomes**)."),
    "primates": (
        "NCBI RefSeq genomes in order **Primates** (**{n_genomes} genomes**)."
    ),
    "humans": ("**Homo sapiens** only (GRCh38 / GCF_000001405.40)."),
    "human_mouse": (
        "**Homo sapiens** (GRCh38) + **Mus musculus** (GRCm39) — a 2-genome "
        "set (GCF_000001405.40, GCF_000001635.27)."
    ),
    "enhancer_seg_mammals_v1": (
        "20 chromosome-level mammalian assemblies, one per order — the "
        "[issue #118](https://github.com/Open-Athena/marin-dna/issues/118) "
        "segmentation-benchmark subset."
    ),
    "mammals_seg20": (
        "20 chromosome-level mammalian assemblies, one per order (the same "
        "set as `enhancer_seg_mammals_v1`)."
    ),
    "animals_order204": (
        "One annotated RefSeq genome per **Metazoan order** (**{n_genomes} "
        "genomes** across 17 phyla) — the order-level dedup of the `animals` "
        "family universe, target species for the "
        "[issue #353](https://github.com/Open-Athena/marin-dna/issues/353) "
        "CDS projection-vs-annotation experiment."
    ),
}

# Recipes that also have a matched conservation-encoded human validation repo
# (``…-validation-intervals-{recipe}_{w}_{w}``).
VALIDATION_RECIPES: frozenset[str] = frozenset({"v1", "v5", "v15", "v17", "v18", "v30"})

_FRONT_MATTER_TAGS = "- biology\n- genomics\n- DNA"


def _permalinks(commit_sha: str, github_repo: str) -> tuple[str, str]:
    """Return ``(permalink, short_sha)`` for the producing pipeline."""
    permalink = (
        f"https://github.com/{github_repo}/tree/{commit_sha}/{_GITHUB_PIPELINE_PATH}"
    )
    return permalink, commit_sha[:12] if len(commit_sha) >= 12 else commit_sha


def _overlap_phrase(window: int, stride: int) -> str:
    """Human-readable description of the window/stride tiling overlap."""
    if stride >= window:
        return "non-overlapping"
    pct = round(100 * (window - stride) / window)
    return f"~{pct}% overlap"


def count_parquet_rows(parquet_paths: list[str] | tuple[str, ...]) -> int:
    """Exact total row count across one or more parquet files.

    Reads only the parquet footer metadata (``pl.scan_parquet(...).select(
    pl.len())``) — no row-group data is fetched, so this is cheap even over
    ``s3://`` URIs. Used to put the *true* sequence count into a dataset card:
    HuggingFace's auto-generated ``num_examples`` is frequently wrong for
    sharded ``JSONL.zst``, and for these datasets the shard row total equals
    the sum of the per-genome parquet rows the shards are built from (verified
    invariant), so we count those instead of decompressing tens of GB.
    """
    import polars as pl

    paths = list(parquet_paths)
    if not paths:
        return 0
    # Single multi-file scan so polars reads the footers in parallel (matters
    # for genome sets with hundreds of per-genome parquets).
    return int(pl.scan_parquet(paths).select(pl.len()).collect().item())


def build_training_readme(
    genome_set: str,
    recipe: str,
    window: int,
    stride: int,
    *,
    hf_prefix: str,
    commit_sha: str,
    n_genomes: int,
    n_samples: int,
    n_shards: int = 64,
    seed: int = 42,
    add_rc: bool = True,
    github_repo: str = _GITHUB_REPO,
) -> str:
    """Build the HF dataset card for a ``genomes-v*`` *training* dataset.

    Args:
        genome_set: e.g. ``"animals"`` — must be in :data:`GENOME_SET_BLURBS`.
        recipe: e.g. ``"v5"`` — must be in :data:`RECIPE_BLURBS`.
        window: window length in bp (e.g. 255).
        stride: window step in bp (e.g. 128).
        hf_prefix: ``output_hf_prefix``, e.g. ``"bolinas-dna/genomes-v5"``.
        commit_sha: full git SHA of the producing pipeline (or ``"main"``).
        n_genomes: number of genomes in ``genome_set`` (for the count text).
        n_samples: exact total row count across all shards (RC included). See
            :func:`count_parquet_rows` — HF's auto-count is unreliable here.
        n_shards: number of ``.jsonl.zst`` shards under ``data/train/``.
        seed: global-shuffle seed.
        add_rc: whether reverse complements were added.
        github_repo: ``owner/name`` for the permalink.

    Returns:
        The full ``README.md`` text (YAML front-matter + body).
    """
    if recipe not in RECIPE_BLURBS:
        raise ValueError(
            f"unknown recipe {recipe!r}; expected one of {sorted(RECIPE_BLURBS)}"
        )
    if genome_set not in GENOME_SET_BLURBS:
        raise ValueError(
            f"unknown genome_set {genome_set!r}; expected one of "
            f"{sorted(GENOME_SET_BLURBS)}"
        )

    repo_name = (
        f"{hf_prefix}-genome_set-{genome_set}-intervals-{recipe}_{window}_{stride}"
    )
    permalink, sha12 = _permalinks(commit_sha, github_repo)
    set_blurb = GENOME_SET_BLURBS[genome_set].format(n_genomes=n_genomes)
    set_title = GENOME_SET_TITLES.get(genome_set, genome_set)
    recipe_blurb = RECIPE_BLURBS[recipe]
    recipe_title = RECIPE_TITLES.get(recipe, recipe)

    rc_note = (
        "every window appears **twice** — the reference-forward sequence "
        "(`id` suffix `_+`) and its reverse complement (`_-`), sharing "
        "identical `chrom:start-end` coordinates. The suffix denotes "
        "forward-vs-RC, **not** the annotation strand of the source region."
        if add_rc
        else "reverse complements were **not** added; each window appears once."
    )

    genome_word = "genome" if n_genomes == 1 else "genomes"
    steps = [
        "Download genome assemblies (soft-masked 2bit) + GTF annotations from "
        f"NCBI RefSeq for every genome in `{genome_set}` ({n_genomes} {genome_word}).",
        f"Build the `{recipe}` interval set per genome — see *Region recipe* above.",
        f"Tile into {window} bp / {stride} bp windows; drop windows overlapping "
        "undefined (`N`) sequence.",
        "Extract sequence with `twoBitToFa` (soft-masking preserved).",
    ]
    if add_rc:
        steps.append("Add reverse complements (`add_rc`).")
    steps.append(
        f"Concatenate across all genomes, globally shuffle (seed {seed}), and "
        f"shard into {n_shards} `.jsonl.zst` files."
    )
    construction_block = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))

    if recipe in VALIDATION_RECIPES:
        val_repo = f"{hf_prefix}-validation-intervals-{recipe}_{window}_{window}"
        matched_validation = (
            f"- Matched validation set (conservation-encoded human): "
            f"[`{val_repo}`](https://huggingface.co/datasets/{val_repo})"
        )
    else:
        matched_validation = "- No matched validation repo for this recipe."

    return f"""---
tags:
{_FRONT_MATTER_TAGS}
pretty_name: "{set_title} · {recipe_title} ({recipe}) · {window} bp training windows"
---

# `{repo_name}`

{set_title} **{recipe_title}** ({recipe}) sequences — {window} bp DNA windows
for genomic language model pretraining.

Part of the **`{hf_prefix}`** training-dataset family produced by the
[`{_GITHUB_PIPELINE_PATH}`]({permalink}) pipeline (commit
[`{sha12}`]({permalink})). Each repo in the family is one
`(genome_set, region-recipe)` combination.

## Size

**{n_samples:,} sequences** across {n_shards} `data/train/*.jsonl.zst` shards
(reverse complements included). This is an exact count over the shards —
HuggingFace's auto-generated row count is frequently wrong for sharded
JSONL.zst, so prefer this number.

## Genome set: `{genome_set}`

{set_blurb}

## Region recipe: `{recipe}`

{recipe_blurb}

Windows are **{window} bp**, tiled with a **{stride} bp** stride
({_overlap_phrase(window, stride)}); windows overlapping undefined (`N`)
sequence are dropped.

## Schema

`data/train/*.jsonl.zst` — {n_shards} zstd-compressed JSON-Lines shards,
globally shuffled (seed {seed}). Each record:

| Field | Type | Description |
|-------|------|-------------|
| `id`  | str  | `{{chrom}}:{{start}}-{{end}}_{{strand}}` — 0-based half-open coordinates, bare RefSeq chrom names (e.g. `NC_000001.11`), with a `_+` / `_-` strand suffix (see *Reverse complements*). |
| `seq` | str  | {window} bp DNA sequence with **soft-masking preserved** (see below). |

### Sequence case = soft-masked repeats

`seq` carries the **soft-masking from the source assembly** (the NCBI RefSeq
2bit), unchanged:

- **Uppercase** `A` / `C` / `G` / `T` — a non-repeat-masked base.
- **Lowercase** `a` / `c` / `g` / `t` — a repeat-masked base (interspersed
  repeats, low-complexity, simple tandem repeats). `N` marks undefined
  sequence.

> ⚠️ This is **not** a conservation encoding. The matched *validation* sets
> (`{hf_prefix}-validation-…`) instead overwrite case to encode phyloP
> conservation (uppercase = conserved). Do not conflate the two conventions.

### Reverse complements

`add_rc` = `{add_rc}`: {rc_note}

## Construction

{construction_block}

This is a **train-only** set (no chromosome holdout); evaluation uses the
separate conservation-encoded validation repos.

## Provenance

- Pipeline: [`{_GITHUB_PIPELINE_PATH}`]({permalink}) @ commit [`{sha12}`]({permalink})
- Genomes: NCBI RefSeq, selected by the `genome_selection` stage (taxonomic +
  assembly-quality filters, taxonomic dedup).
{matched_validation}
"""


def build_validation_readme(
    recipe: str,
    window: int,
    stride: int,
    *,
    hf_prefix: str,
    commit_sha: str,
    n_samples: int,
    phylop_threshold: float,
    max_samples: int,
    seed: int,
    validation_genome: str = "GCF_000001405.40",
    train_stride: int = 128,
    github_repo: str = _GITHUB_REPO,
) -> str:
    """Build the HF dataset card for a ``genomes-v*`` *validation* dataset.

    These are single-parquet, human-only, conservation-case-encoded probes
    matched to the training recipes.

    Args:
        recipe: e.g. ``"v5"`` — must be in :data:`RECIPE_BLURBS`.
        window: window length in bp (e.g. 255).
        stride: window step in bp (validation uses step = window).
        hf_prefix: ``output_hf_prefix``, e.g. ``"bolinas-dna/genomes-v5"``.
        commit_sha: full git SHA of the producing pipeline (or ``"main"``).
        n_samples: exact row count of ``validation.parquet`` (``<= max_samples``).
        phylop_threshold: phyloP-241way cutoff for the uppercase encoding.
        max_samples: subsample cap.
        seed: subsample seed.
        validation_genome: assembly the probe is drawn from (human GRCh38).
        train_stride: stride of the matched training repos (for the link text).
        github_repo: ``owner/name`` for the permalink.

    Returns:
        The full ``README.md`` text (YAML front-matter + body).
    """
    if recipe not in RECIPE_BLURBS:
        raise ValueError(
            f"unknown recipe {recipe!r}; expected one of {sorted(RECIPE_BLURBS)}"
        )

    repo_name = f"{hf_prefix}-validation-intervals-{recipe}_{window}_{stride}"
    permalink, sha12 = _permalinks(commit_sha, github_repo)
    recipe_blurb = RECIPE_BLURBS[recipe]
    recipe_title = RECIPE_TITLES.get(recipe, recipe)
    train_glob = f"{hf_prefix}-genome_set-*-intervals-{recipe}_{window}_{train_stride}"

    return f"""---
tags:
{_FRONT_MATTER_TAGS}
pretty_name: "Validation · {recipe_title} ({recipe}) · {window} bp conservation-encoded"
---

# `{repo_name}`

Conservation-encoded **human** validation probe for the `{recipe}`
({recipe_title}) region recipe, matched to the
`{train_glob}` training sets. Produced by the
[`{_GITHUB_PIPELINE_PATH}`]({permalink}) pipeline (commit
[`{sha12}`]({permalink})).

## Size

**{n_samples:,} sequences** in `validation.parquet` (a subsample of up to
{max_samples:,} human windows). Exact row count — HuggingFace's auto-generated
count can be wrong.

## Region recipe: `{recipe}`

{recipe_blurb}

Built on the human genome (GRCh38 / {validation_genome}).

## Schema

A single `validation.parquet`. Each row:

| Field | Type | Description |
|-------|------|-------------|
| `id`  | str  | `{{chrom}}:{{start}}-{{end}}` — 0-based half-open, bare RefSeq chrom names. |
| `seq` | str  | {window} bp DNA with **conservation-encoded case** (see below). |

### Sequence case = phyloP conservation

Each base's case is **overwritten** to encode conservation (the source
soft-masking is discarded):

- **Uppercase** — phyloP-241way (Cactus 241-way alignment) score ≥
  **{phylop_threshold}** at this position (conserved).
- **Lowercase** — score < {phylop_threshold}, **or** there is no alignment /
  the score is missing (`NaN`, which compares false). Non-conserved and
  unaligned bases are not distinguished.

> ⚠️ This is the **opposite** convention to the training sets, where lowercase
> means repeat-masked, not non-conserved.

## Construction

1. Build the `{recipe}` interval set on the human genome and tile into
   {window} bp / {stride} bp ({_overlap_phrase(window, stride)}) windows.
2. Subsample up to **{max_samples}** windows (seed {seed}).
3. Re-encode each base's case by phyloP-241way ≥ {phylop_threshold}.

## Provenance

- Pipeline: [`{_GITHUB_PIPELINE_PATH}`]({permalink}) @ commit [`{sha12}`]({permalink})
- Matched training family: `{train_glob}`
"""


def write_training_readme(output_path: str | Path, /, **kwargs: object) -> None:
    """Write :func:`build_training_readme` output to ``output_path``."""
    Path(output_path).write_text(build_training_readme(**kwargs))  # type: ignore[arg-type]


def write_validation_readme(output_path: str | Path, /, **kwargs: object) -> None:
    """Write :func:`build_validation_readme` output to ``output_path``."""
    Path(output_path).write_text(build_validation_readme(**kwargs))  # type: ignore[arg-type]
