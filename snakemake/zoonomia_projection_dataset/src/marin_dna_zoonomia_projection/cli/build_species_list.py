"""One-off: build a rank-deduplicated 447-mammalian species list.

Run once when the alignment changes; commit the resulting TSV. All HTTP
responses are cached on disk so re-runs are idempotent and offline-stable.

``--rank`` picks the dedup rank: ``family`` (default, 108 leaves) or
``order`` (19 leaves — sparser, more deeply diverged). Output defaults to
``snakemake/zoonomia_projection_dataset/config/species_zoonomia_447_<rank>_dedup.tsv``
with columns ``species\\tfamily\\torder\\taccession\\tassembly_level\\tcontig_n50\\tquality_source``.

Sources, in priority order for assembly quality:

1. **Zoonomia ST2** — Supplementary Table 2 of the 241-mammalian alignment
   gives the GCA_/GCF_ accession used for each leaf. The 447-mammalian
   alignment inherits these accessions for non-primate leaves (per the 447
   README's "primate clade replaced and grafted" construction). For each
   ST2 accession we query NCBI Datasets v2 by accession to recover
   ``assembly_level`` + ``contig_n50``. Marked
   ``quality_source = "zoonomia_supp_st2"``.
2. **NCBI taxon proxy** — for 447-only primate leaves (no ST2 entry) AND
   for legacy ST2 accessions that NCBI v2 no longer reports (Mus mm10
   GCF_000001635.26, hg38.p12 GCA_000001405.27, several GCA_004*), fall
   back to taxon-level lookup (NCBI's current best assembly). Marked
   ``quality_source = "ncbi_taxon_proxy"``. The "quality" stats here
   refer to a *different* assembly than what is in the HAL — useful for
   ranking but not for sequence extraction.

Dedup policy: see :func:`marin_dna_zoonomia_projection.projection.taxonomy.dedup_by_rank`.
``Homo_sapiens``, ``Mus_musculus``, and ``Bos_taurus`` are force-included
(belt-and-suspenders; the natural ranking already picks them).

Usage::

    uv run --locked marin-dna-build-species-list

Network: ~9 batched taxonomy calls + ~447 single accession/taxon-genome
calls. ~2 minutes total cold; <5 s warm-cache.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import openpyxl

from marin_dna_zoonomia_projection.projection.taxonomy import (
    DEDUP_RANKS,
    LeafMeta,
    dedup_by_rank,
    normalize_zoonomia_leaf,
    parse_newick_leaves,
)

CACHE = Path.home() / ".cache" / "marin_dna" / "zoonomia"

NEWICK_URL = "https://cgl.gi.ucsc.edu/data/cactus/447-mammalian-2022v1.nh"
SUPP_XLSX_URL = "https://cgl.gi.ucsc.edu/data/cactus/Zoonomia-Supplemental-tables.xlsx"
NCBI_TAX_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/taxonomy/taxon"
NCBI_GENOME_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome"

# Per issue #150: 8 leaves whose Newick name doesn't resolve under NCBI
# Datasets without these alternates.
ALT_NAMES: dict[str, str] = {
    "CanFam4": "Canis lupus familiaris",
    "Canis_lupus_VD": "Canis lupus",
    "Felis_catus_fca126": "Felis catus",
    "Lagothrix_lagothricha": "Lagothrix lagotricha",
    "Leontocebus_illigeri": "Leontocebus",
    "Pithecia_pissinattii": "Pithecia",
    "Propithecus_coquerelli": "Propithecus coquereli",
    "Trachypithecus_melamera": "Trachypithecus pileatus",
}

# Per-rank sanity bounds on the winners set: (count_lo, count_hi,
# min_st2_ratio). Loud failure beats silent corruption (CLAUDE.md).
#   family: ~108 leaves (one per NCBI family), ~85% ST2-true; count bound is
#           loose (~108 ± a handful) since family totals shift with taxonomy.
#   order:  19 leaves (one per order); count bound is tight (±1 around 19 —
#           the extant placental-order count is biologically stable), and the
#           ST2-true gate is looser (real ratio 15/19 ≈ 79%): with only 19
#           winners the always-proxy human + mouse, plus the orders whose best
#           assembly is itself a taxon proxy (e.g. Dermoptera, Proboscidea),
#           weigh far more than they do across 108 families.
RANK_SANITY: dict[str, tuple[int, int, float]] = {
    "family": (100, 115, 0.80),
    "order": (18, 20, 0.70),
}
# Every rank argparse accepts (choices=DEDUP_RANKS) must have sanity bounds, or
# a valid --rank would pass the full ~2-min fetch and then KeyError here. Tie
# the two together at import time so adding a rank fails fast and loud.
assert set(RANK_SANITY) == set(DEDUP_RANKS), (
    f"RANK_SANITY keys {sorted(RANK_SANITY)} != DEDUP_RANKS {sorted(DEDUP_RANKS)}"
)


def _to_query(leaf: str) -> str:
    """Map a raw HAL leaf name to a binomial query for NCBI / ST2 lookup.

    Strips the ``_a`` / ``_b`` duplicate-disambiguator suffix that the
    447-mammalian Newick uses for 4 species (per the README's
    "naming-error fix") so the binomial matches NCBI / ST2. The raw leaf
    name (with ``_a`` / ``_b``) is what the HAL stores and what
    ``halStats`` / ``halLiftover`` expect.
    """
    if leaf in ALT_NAMES:
        return ALT_NAMES[leaf]
    return normalize_zoonomia_leaf(leaf).replace("_", " ")


def default_species_output(rank: str) -> Path:
    """Return the committed species-list location for a deduplication rank."""
    assert rank in DEDUP_RANKS
    return (
        Path(__file__).resolve().parents[3]
        / "config"
        / f"species_zoonomia_447_{rank}_dedup.tsv"
    )


def _http_get_json(url: str, key: str, *, timeout: int = 30) -> dict | None:
    """Cached GET returning parsed JSON or None on failure.

    ``key`` is sanitised to a safe filename — URL-encoded characters
    like ``%20`` and accidental path separators won't escape the cache
    dir or cause ``OSError`` on case-insensitive filesystems.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)
    p = CACHE / f"{safe}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return json.loads(p.read_text())
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as f:
                data = json.loads(f.read())
            p.write_text(json.dumps(data))
            time.sleep(0.2)
            return data
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            print(f"  retry {attempt + 1}/3 on {key}: {e}", file=sys.stderr)
            time.sleep(2)
    return None


def _http_get_bytes(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"Downloading {url} -> {dest}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as f:
        dest.write_bytes(f.read())


def _parse_st2(xlsx_path: Path) -> dict[str, str]:
    """Map species-underscored → first accession from ST2 (Cactus alignment).

    ST2 has 250 rows for 240 species (dog appears twice as Source 1+2).
    Prefer Zoonomia-newly-sequenced ("Source = 1. Zoonomia") over existing
    assemblies when both exist, since that's what Cactus actually used for
    the leaf in the 241-way build.

    Column lookup is by header name (row index 2 of the sheet), so an
    upstream column reorder won't silently corrupt the mapping.
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Supplementary Table 2"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[2]
    cols = {name: i for i, name in enumerate(header) if name}
    for required in ("Species", "Source", "Accession"):
        assert required in cols, (
            f"ST2 header missing {required!r}; got {sorted(cols.keys())}"
        )
    sp_i, src_i, acc_i = cols["Species"], cols["Source"], cols["Accession"]
    out: dict[str, str] = {}
    for r in rows[3:]:
        if r is None or r[sp_i] is None:
            continue
        species, source, accession = r[sp_i], r[src_i], r[acc_i]
        if not species or not accession:
            continue
        sp_us = species.replace(" ", "_")
        if sp_us not in out or "1. Zoonomia" in (source or ""):
            out[sp_us] = accession
    return out


def _fetch_taxonomy_batch(queries: list[str], batch_idx: int) -> dict[str, dict]:
    """Resolve a batch of binomials to NCBI lineage. Returns {query: report}."""
    enc = ",".join(quote(q, safe="") for q in queries)
    url = f"{NCBI_TAX_URL}/{enc}/dataset_report?page_size={len(queries) + 5}"
    data = _http_get_json(url, f"tax_batch_{batch_idx:04d}")
    if data is None:
        return {}
    out: dict[str, dict] = {}
    for r in data.get("reports", []):
        for q in r.get("query", []):
            out[q] = r
    return out


def _resolve_taxonomy(queries: list[str]) -> dict[str, dict]:
    """Batch-fetch taxonomy reports for all unique queries."""
    out: dict[str, dict] = {}
    BATCH = 50
    queries = sorted(set(queries))
    for i in range(0, len(queries), BATCH):
        out |= _fetch_taxonomy_batch(queries[i : i + BATCH], i)
    return out


def _family_order_from_taxonomy(rep: dict | None) -> tuple[str | None, str | None]:
    if rep is None:
        return None, None
    cls = rep.get("taxonomy", {}).get("classification", {})
    fam = cls.get("family") or {}
    order = cls.get("order") or {}
    return fam.get("name"), order.get("name")


def _fetch_assembly_by_accession(accession: str) -> dict | None:
    url = f"{NCBI_GENOME_URL}/accession/{accession}/dataset_report"
    data = _http_get_json(url, f"acc_{accession.replace('.', '_')}")
    if data is None:
        return None
    reports = data.get("reports") or []
    return reports[0] if reports else None


def _fetch_assembly_by_taxon(binomial: str) -> dict | None:
    enc = quote(binomial, safe="")
    url = (
        f"{NCBI_GENOME_URL}/taxon/{enc}/dataset_report"
        f"?filters.reference_only=true&filters.assembly_version=current&page_size=1"
    )
    data = _http_get_json(url, f"taxgenome_{enc}")
    if data is None:
        return None
    reports = data.get("reports") or []
    return reports[0] if reports else None


def _parse_quality(rep: dict | None) -> tuple[str | None, int | None]:
    if rep is None:
        return None, None
    info = rep.get("assembly_info") or {}
    stats = rep.get("assembly_stats") or {}
    n50 = stats.get("contig_n50")
    return info.get("assembly_level"), int(n50) if n50 is not None else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--rank",
        choices=list(DEDUP_RANKS),
        default="family",
        help="Taxonomic rank to deduplicate to (one leaf per group).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output TSV path. Defaults to "
        "config/species_zoonomia_447_<rank>_dedup.tsv.",
    )
    args = p.parse_args()
    if args.output is None:
        args.output = default_species_output(args.rank)

    # 1. Newick leaves — keep raw names (incl. `_a`/`_b` suffix). The HAL
    # stores leaves under these raw names and `halStats` / `halLiftover`
    # need them literally; collapsing to the binomial breaks the HAL
    # lookup. Normalization to a binomial happens only inside `_to_query`
    # for NCBI / ST2 lookups.
    nh_path = CACHE / "447-mammalian-2022v1.nh"
    _http_get_bytes(NEWICK_URL, nh_path)
    leaves = sorted(set(parse_newick_leaves(nh_path.read_text())))
    n_suffixed = sum(1 for leaf in leaves if leaf.endswith(("_a", "_b")))
    print(
        f"Newick: {len(leaves)} leaves ({n_suffixed} with _a/_b suffix kept as-is "
        f"for HAL compatibility)",
        file=sys.stderr,
    )

    # 2. ST2 accessions (indexed by the normalized binomial)
    xlsx_path = CACHE / "Zoonomia-Supplemental-tables.xlsx"
    _http_get_bytes(SUPP_XLSX_URL, xlsx_path)
    st2 = _parse_st2(xlsx_path)
    print(f"ST2: {len(st2)} species → accession", file=sys.stderr)

    # 3. Family/order via NCBI taxonomy
    queries = [_to_query(leaf) for leaf in leaves]
    print(
        f"Resolving taxonomy for {len(set(queries))} unique queries...", file=sys.stderr
    )
    tax = _resolve_taxonomy(queries)
    print(f"  {len(tax)} resolved", file=sys.stderr)

    # 4. Build LeafMeta records — leaf is the raw HAL name, st2 lookup
    # uses the normalized binomial.
    metas: list[LeafMeta] = []
    for leaf in leaves:
        q = _to_query(leaf)
        family, order = _family_order_from_taxonomy(tax.get(q))
        st2_key = normalize_zoonomia_leaf(leaf)
        if st2_key in st2:
            accession: str | None = st2[st2_key]
            rep = _fetch_assembly_by_accession(accession)
            if rep is not None:
                level, n50 = _parse_quality(rep)
                source = "zoonomia_supp_st2"
            else:
                # Legacy accession; fall back to taxon proxy.
                rep = _fetch_assembly_by_taxon(q)
                level, n50 = _parse_quality(rep)
                source = "ncbi_taxon_proxy" if rep is not None else "unknown"
        else:
            rep = _fetch_assembly_by_taxon(q)
            level, n50 = _parse_quality(rep)
            accession = (rep or {}).get("accession")
            source = "ncbi_taxon_proxy" if rep is not None else "unknown"
        metas.append(
            LeafMeta(
                leaf=leaf,
                family=family,
                order=order,
                accession=accession,
                assembly_level=level,
                contig_n50=n50,
                quality_source=source,
            )
        )

    # 5. Dedup to the requested rank
    winners = dedup_by_rank(metas, args.rank)
    winners.sort(key=lambda m: (getattr(m, args.rank) or "", m.leaf))

    # 6. Asserts (loud failure beats silent corruption per CLAUDE.md)
    leaves_won = {w.leaf for w in winners}
    assert {"Homo_sapiens", "Mus_musculus", "Bos_taurus"}.issubset(leaves_won), (
        f"force-include species missing from winners: "
        f"{ {'Homo_sapiens', 'Mus_musculus', 'Bos_taurus'} - leaves_won }"
    )
    n = len(winners)
    count_lo, count_hi, min_st2_ratio = RANK_SANITY[args.rank]
    assert count_lo <= n <= count_hi, (
        f"unexpected {args.rank} count: {n} (expected {count_lo}..{count_hi})"
    )
    n_st2 = sum(1 for w in winners if w.quality_source == "zoonomia_supp_st2")
    assert n_st2 / n >= min_st2_ratio, (
        f"ST2-true ratio too low: {n_st2}/{n} = {n_st2 / n:.2%} "
        f"(expected ≥ {min_st2_ratio:.0%} for rank={args.rank}). "
        f"Reproducer logic may be miscoded."
    )
    n_unknown = sum(1 for w in winners if w.quality_source == "unknown")
    assert n_unknown == 0, (
        f"{n_unknown} winners have quality_source = unknown; "
        f"the accession→taxon fallback should reach all of them."
    )

    # 7. Write TSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.write(
            "species\tfamily\torder\taccession\tassembly_level"
            "\tcontig_n50\tquality_source\n"
        )
        for w in winners:
            f.write(
                f"{w.leaf}\t{w.family or ''}\t{w.order or ''}"
                f"\t{w.accession or ''}\t{w.assembly_level or ''}"
                f"\t{w.contig_n50 if w.contig_n50 is not None else ''}"
                f"\t{w.quality_source}\n"
            )
    print(
        f"\nWrote {args.output} ({n} species, {len({w.family for w in winners})} families, "
        f"{len({w.order for w in winners})} orders)",
        file=sys.stderr,
    )
    print(
        f"  quality_source: {n_st2} st2-true ({n_st2 / n:.0%}) + "
        f"{n - n_st2} ncbi proxy",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
