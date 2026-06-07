"""Enumerate human, nucleotide-level saturation-genome-editing (SGE) datasets in MaveDB.

SGE edits the *endogenous genomic locus* (CRISPR-HDR, usually in haploid HAP1 cells),
so unlike protein-level deep mutational scans it yields nucleotide-resolution scores in
genomic coordinates that include coding *and* noncoding (intronic/splice/UTR) SNVs —
the property that makes it a good genomic-language-model VEP benchmark (cf. Evo2's BRCA1
eval). This script counts distinct human SGE studies/genes via the MaveDB REST API,
deduping the many per-exon / per-replicate score-sets into one row per (gene x study).

Run: python3 scripts/mavedb_sge_inventory.py
"""

import json
import urllib.request

API = "https://api.mavedb.org/api/v1"


def search(text: str) -> list[dict]:
    req = urllib.request.Request(
        f"{API}/score-sets/search",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r).get("scoreSets", [])


def assay_type(s: dict) -> str:
    e = s.get("experiment") or {}
    txt = " ".join(
        [
            s.get("title") or "",
            e.get("title") or "",
            e.get("abstractText") or "",
            e.get("methodText") or "",
        ]
    ).lower()
    if "saturation genome editing" in txt or "saturation genome essential" in txt:
        return "SGE"
    # Require an explicit prime-editing term — a bare "prime" substring also
    # matches "primers" in PCR-based exogenous-DMS methods (e.g. CXCR4/CCR5 NNK
    # SSM libraries), which are NOT endogenous prime editing.
    if any(t in txt for t in ("prime editing", "prime-editing", "prime editor", "pegrna", "clipe")):
        return "prime-editing"
    return "other"


def gene_and_assembly(s: dict) -> tuple[str, str]:
    g = (s.get("targetGenes") or [{}])[0]
    gene = g.get("mappedHgncName") or g.get("name") or "?"
    acc = g.get("targetAccession")
    assembly = acc.get("assembly") if isinstance(acc, dict) else None
    return gene, (assembly or "?")


def main() -> None:
    by_urn: dict[str, dict] = {}
    for term in (
        "saturation genome editing",
        "SGE",
        "saturation prime editing",
        "prime editing",
    ):
        for s in search(term):
            by_urn[s["urn"]] = s

    studies: dict[str, dict] = {}
    for s in by_urn.values():
        e = s.get("experiment") or {}
        eset = e.get("experimentSetUrn") or s["urn"]
        gene, assembly = gene_and_assembly(s)
        pubs = s.get("primaryPublicationIdentifiers") or []
        pub = pubs[0] if pubs else {}
        rec = studies.setdefault(
            eset,
            {
                "gene": gene,
                "assay": "other",
                "nv": 0,
                "assembly": assembly,
                "year": pub.get("publicationYear"),
                "pmid": pub.get("identifier"),
                "deposited": e.get("publishedDate"),
                "title": (pub.get("title") or "")[:55],
            },
        )
        rec["nv"] = max(rec["nv"], s.get("numVariants") or 0)
        if gene != "?":
            rec["gene"] = gene
        if assembly != "?":
            rec["assembly"] = assembly
        # prefer the most-specific assay label seen across this study's score-sets
        rank = {"other": 0, "prime-editing": 1, "SGE": 2}
        if rank[assay_type(s)] > rank[rec["assay"]]:
            rec["assay"] = assay_type(s)

    for label in ("SGE", "prime-editing"):
        grp = sorted(
            (r for r in studies.values() if r["assay"] == label), key=lambda r: -r["nv"]
        )
        genes = sorted({r["gene"] for r in grp})
        print(f"\n=== {label}: {len(grp)} studies across {len(genes)} genes ===")
        print(f"{'GENE':9s} {'#var':>6s} {'asm':7s} {'year/deposit':12s} title")
        for r in grp:
            when = str(r["year"]) if r["year"] else f"dep {r['deposited']}"
            print(
                f"{r['gene']:9s} {r['nv']:6d} {r['assembly']:7s} {when:12s} {r['title']}"
            )
        print(f"genes: {genes}")


if __name__ == "__main__":
    main()
