"""Dashboard-facing catalog of nucleotide-dependency-map artifacts (issue #240).

The evals_v2 pipeline (issues #237 / #238) writes one heatmap SVG per
``(combine, locus, model)`` to S3 under
``snakemake/analysis/evals_v2/results/plots/nuc_dep/{combine}/{locus}/{model}.svg``.
This module turns the pipeline's ``nuc_dep`` config block into the *candidate*
manifest the Observable dashboard loader fetches — pairing each artifact with
its locus metadata, a UCSC Genome Browser deep-link, and (when the file is
committed) a paper-reference screenshot. The loader
(``dashboard/src/data/nuc_dep.zip.py``) does the S3 I/O and drops candidates
whose SVG isn't materialized yet; the pure functions here are unit-tested.

Coordinates are 0-based half-open everywhere (repo convention); the only place
that changes is the *display* layer — ``display_region`` and
``ucsc_browser_url`` emit 1-based inclusive with a ``chr`` prefix (the
UCSC / paper convention) at the tool boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from marin_dna_evals.models import MODELS_YAML, load_models

# Bucket + prefix for the materialized heatmap SVGs (rules/interpretation.smk
# `plot_nuc_dep`). The dashboard CI IAM role grants GetObject on
# `…/analysis/evals_v2/results/*`, which covers this prefix.
S3_BUCKET = "oa-bolinas"
NUC_DEP_PLOT_PREFIX = "snakemake/analysis/evals_v2/results/plots/nuc_dep"
# Embedding-UMAP SVGs (rules/embedding_umap.smk `plot_umap`, issue #246), under
# the same `…/results/*` prefix the dashboard IAM role can read.
UMAP_PLOT_PREFIX = "snakemake/analysis/evals_v2/results/plots/umap"
# The two per-model UMAP panels (GPN-Star Fig 4A / 4B).
UMAP_COLOR_BYS: tuple[str, ...] = ("region", "conservation")

# models.yaml lives at <repo>/dashboard/models.yaml, so its grandparent is the
# repo root — reuse that (models.py already resolves it robustly) to locate the
# evals_v2 pipeline config rather than re-deriving from ``__file__``.
EVALS_V2_CONFIG = (
    MODELS_YAML.parents[1]
    / "snakemake"
    / "analysis"
    / "evals_v2"
    / "config"
    / "config.yaml"
)

# Method papers — cited in the page intro where the dependency map is defined:
# the categorical Jacobian (introduced for protein LMs) and its application to
# genomic LMs.
CATEGORICAL_JACOBIAN_PAPER: dict[str, str] = {
    "citation": "Zhang, Wayment-Steele, Brixi, Wang, Kern & Ovchinnikov, PNAS 2024",
    "url": "https://doi.org/10.1073/pnas.2406285121",
}
NUC_DEP_PAPER: dict[str, str] = {
    "citation": (
        "Tomaz da Silva et al., “Nucleotide dependency analysis of genomic "
        "language models detects functional elements,” Nature Genetics 2025"
    ),
    "url": "https://www.nature.com/articles/s41588-025-02347-3",
    "preprint_url": "https://www.biorxiv.org/content/10.1101/2024.07.27.605418v1",
}
# Source of each locus's reference screenshot (per `LOCUS_META[...]["source"]`).
# The four gene loci are GPN-Star figures; the tRNA panel is from the nuc-dep paper.
GPN_STAR_PAPER: dict[str, str] = {
    "citation": "GPN-Star — Ye, Benegas et al., bioRxiv 2025",
    "url": "https://www.biorxiv.org/content/10.1101/2025.09.21.677619v1",
}
_SOURCE_PAPERS: dict[str, dict[str, str]] = {
    "gpn_star": GPN_STAR_PAPER,
    "nuc_dep": NUC_DEP_PAPER,
}

# Per-locus presentation metadata; keys match the ``nuc_dep.loci`` config block.
# ``figure`` (optional) pins the paper panel; ``note`` surfaces a locus-specific
# finding. Loci absent here still render (title falls back to the config key).
LOCUS_META: dict[str, dict[str, str]] = {
    "LDLR": {
        "title": "LDLR",
        "description": "Low-density lipoprotein receptor promoter (chr19); TF-binding sites (SREBP, SP1) show up as off-diagonal blocks.",
        "source": "gpn_star",
    },
    "HBA1": {
        "title": "HBA1",
        "description": "Hemoglobin subunit alpha 1 (chr16).",
        "source": "gpn_star",
    },
    "TH": {
        "title": "TH",
        "description": "Tyrosine hydroxylase (chr11, − strand).",
        "source": "gpn_star",
    },
    "GRIA4": {
        "title": "GRIA4",
        "description": "Glutamate ionotropic receptor AMPA type subunit 4 (chr11) — a small element; look for the tight ~3-bp-periodic near-diagonal band.",
        "source": "gpn_star",
    },
    "tRNA_Arg_TCT": {
        "title": "tRNA-Arg-TCT-4-1",
        "description": "Arginine tRNA (chr1, − strand) — a structured ncRNA.",
        "figure": "Fig. 7b",
        "source": "nuc_dep",
        "note": (
            "exp135 does not recover this tRNA's cloverleaf base-pairing "
            "(contact AUROC ≈ chance) — verified not a bug; see issue #237."
        ),
    },
}

# Extensions accepted for a committed paper-reference screenshot, in priority order.
_REF_IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".svg")


def _normalize_chrom(chrom: str) -> str:
    """Ensembl-style chrom (no prefix) → bare name, tolerating a stray ``chr``."""
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def ucsc_browser_url(chrom: str, start: int, end: int, *, db: str = "hg38") -> str:
    """UCSC Genome Browser deep-link for a 0-based half-open locus.

    Converts at the boundary to UCSC's 1-based inclusive coordinates with a
    ``chr`` prefix: ``[start, end)`` → ``chr{chrom}:{start+1}-{end}``. GRCh38
    (Ensembl release-115) is assembly-identical to ``hg38``; Ensembl chrom
    names carry no ``chr`` prefix, so we add it (and tolerate a pre-existing one).
    """
    assert end > start, f"end {end} must exceed start {start}"
    assert start >= 0, f"start {start} must be non-negative"
    c = _normalize_chrom(chrom)
    return f"https://genome.ucsc.edu/cgi-bin/hgTracks?db={db}&position=chr{c}:{start + 1}-{end}"


def display_region(chrom: str, start: int, end: int) -> str:
    """Human-facing 1-based inclusive locus string, e.g. ``chr19:11,089,300-11,089,425``."""
    assert end > start, f"end {end} must exceed start {start}"
    c = _normalize_chrom(chrom)
    return f"chr{c}:{start + 1:,}-{end:,}"


def _paper_ref(locus: str, *, refs_dir: Path | None) -> dict[str, Any] | None:
    """Screenshot-source citation (+ optional figure + committed image) for a
    locus, or ``None`` if the locus declares no source.

    The source paper is per-locus (``LOCUS_META[...]["source"]``): GPN-Star for
    the gene loci, the nuc-dep paper for the tRNA. ``image`` is the zip-relative
    key the loader bundles, set only when a screenshot is committed in ``refs_dir``.
    """
    meta = LOCUS_META.get(locus, {})
    source = meta.get("source")
    if source is None:
        return None
    assert source in _SOURCE_PAPERS, (
        f"locus {locus!r}: unknown screenshot source {source!r}; "
        f"expected one of {sorted(_SOURCE_PAPERS)}"
    )
    ref: dict[str, Any] = dict(_SOURCE_PAPERS[source])
    if "figure" in meta:
        ref["figure"] = meta["figure"]
    ref["image"] = None
    if refs_dir is not None:
        for ext in _REF_IMAGE_EXTS:
            if (refs_dir / f"{locus}{ext}").is_file():
                # Zip-relative key: the loader bundles the screenshot into the
                # nuc_dep archive (Observable's build won't copy a file behind a
                # runtime <img src>), and the page reads it back from there.
                ref["image"] = f"refs/{locus}{ext}"
                break
    return ref


def nuc_dep_candidates(
    block: dict[str, Any],
    *,
    model_displays: dict[str, str] | None = None,
    refs_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Candidate ``(combine × locus × model)`` dependency-map artifacts + metadata.

    Pure: enumerates the cartesian product declared in the pipeline's
    ``nuc_dep`` config ``block`` and attaches per-locus metadata, a UCSC link
    and a paper reference. Does no I/O — the loader fetches each ``svg`` key
    from S3 and drops the ones not yet materialized. Order is locus-major, then
    model, then combine (stable + human-scannable).
    """
    combines = block.get("combines", ["mean"])
    models = block.get("models", [])
    loci = block.get("loci", {})
    window = block.get("window_size")  # fixed nuc_dep window, if set (#240)
    assert combines, "nuc_dep config has no `combines`"
    displays = model_displays or {}
    out: list[dict[str, Any]] = []
    for locus, coords in loci.items():
        chrom = str(coords["chrom"])
        start = int(coords["start"])
        end = int(coords["end"])
        strand = coords["strand"]
        assert end > start, f"locus {locus!r}: end {end} must exceed start {start}"
        assert strand in ("+", "-"), f"locus {locus!r}: bad strand {strand!r}"
        assert window is None or (end - start) <= window, (
            f"locus {locus!r}: span {end - start} bp exceeds nuc_dep window_size {window}"
        )
        meta = LOCUS_META.get(locus, {})
        for model in models:
            for combine in combines:
                out.append(
                    {
                        "combine": combine,
                        "locus": locus,
                        "model": model,
                        "model_display": displays.get(model, model),
                        "title": meta.get("title", locus),
                        "description": meta.get("description"),
                        "note": meta.get("note"),
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "span": end - start,
                        "display_region": display_region(chrom, start, end),
                        "ucsc_url": ucsc_browser_url(chrom, start, end),
                        "paper": _paper_ref(locus, refs_dir=refs_dir),
                        "svg": f"{combine}/{locus}/{model}.svg",
                    }
                )
    return out


def load_nuc_dep_block(config_path: Path | None = None) -> dict[str, Any]:
    """Parse the ``nuc_dep`` block from the evals_v2 pipeline config."""
    path = config_path or EVALS_V2_CONFIG
    assert path.is_file(), f"evals_v2 config not found at {path}"
    cfg = yaml.safe_load(path.read_text())
    block = cfg.get("nuc_dep")
    assert block, f"no `nuc_dep` block in {path}"
    return block


def model_display_map() -> dict[str, str]:
    """``model id → display`` from the dashboard model registry (models.yaml)."""
    return {m.id: m.display for m in load_models()}


def umap_candidates(
    block: dict[str, Any],
    *,
    model_displays: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Candidate ``(model, color_by)`` embedding-UMAP artifacts + display names.

    Pure: enumerates the ``models`` declared in the pipeline's ``umap_embeddings``
    config ``block``, each with its ``region`` and ``conservation`` panel. Does
    no I/O — the loader fetches each ``svg`` from S3 and drops the ones not yet
    materialized. Order is model-major (config order), then color_by. Unlike
    ``nuc_dep_candidates`` there is no locus/genomic axis: one global view per
    model.
    """
    models = block.get("models", [])
    displays = model_displays or {}
    out: list[dict[str, Any]] = []
    for model in models:
        for color_by in UMAP_COLOR_BYS:
            out.append(
                {
                    "model": model,
                    "model_display": displays.get(model, model),
                    "color_by": color_by,
                    "svg": f"{model}/{color_by}.svg",
                }
            )
    return out


def load_umap_block(config_path: Path | None = None) -> dict[str, Any]:
    """Parse the ``umap_embeddings`` block from the evals_v2 pipeline config."""
    path = config_path or EVALS_V2_CONFIG
    assert path.is_file(), f"evals_v2 config not found at {path}"
    cfg = yaml.safe_load(path.read_text())
    block = cfg.get("umap_embeddings")
    assert block, f"no `umap_embeddings` block in {path}"
    return block
