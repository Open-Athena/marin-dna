"""Post the cross-model per-subset LLR-vs-probe tables to issue #314.

Reads ``best_vs_llr.parquet`` (written by ``iter1_consolidate``) for each model and
emits three factual tables — zero-shot LLR by model, fixed-recipe probe by model, and
the probe−LLR delta — over the Mendelian subsets. The point of interest is the
within-family **exp166 1B↔4B** comparison (same zoonomia-v0.1 recipe, different scale):
genuine missense decay should show as a lower 1B→4B LLR with the probe recovering it.

Run by the overnight orchestrator. It posts **only the factual tables** (clearly marked
automated) — the narrative synthesis is added in a follow-up so an automated post can't
assert a wrong story. Skips any model whose parquet is missing.
"""

import subprocess

import polars as pl

MODELS = {
    "exp135-1B-m5.1": "exp135-1B",
    "scaling-v0.5-1B": "scaling-1B",
    "exp166-v0.1-p1B": "exp166-1B",
    "exp166-v0.1-p4B": "exp166-4B",
}
ORDER = [
    "missense_variant", "distal", "tss_proximal", "splicing",
    "synonymous_variant", "5_prime_UTR_variant",
    "non_coding_transcript_exon_variant", "3_prime_UTR_variant",
]
SHORT = {
    "missense_variant": "missense", "distal": "distal", "tss_proximal": "tss",
    "splicing": "splicing", "synonymous_variant": "synon",
    "5_prime_UTR_variant": "5′UTR", "non_coding_transcript_exon_variant": "ncRNA",
    "3_prime_UTR_variant": "3′UTR",
}
BASE = "s3://oa-bolinas/analysis/issue314/iter1_search"


def _table(data: dict[str, dict], present: list[str], col: str) -> str:
    head = "| subset | " + " | ".join(MODELS[m] for m in present) + " |"
    sep = "|---|" + "---:|" * len(present)
    rows = [head, sep]
    for s in ORDER:
        cells = []
        for m in present:
            v = data[m].get(s, {}).get(col)
            cells.append("—" if v is None else f"{v:.3f}")
        rows.append(f"| {SHORT.get(s, s)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    data, present = {}, []
    for mid in MODELS:
        try:
            df = pl.read_parquet(f"{BASE}/{mid}/best_vs_llr.parquet")
        except Exception:
            continue
        present.append(mid)
        data[mid] = {
            r["subset"]: {"llr": r["llr_auprc"], "probe": r["fixed_auprc"],
                          "delta": r["fixed_delta"]}
            for r in df.iter_rows(named=True)
        }
    assert present, "no best_vs_llr parquets found"

    body = (
        "🤖 **[automated] 4-model per-subset: zero-shot LLR vs fixed-recipe probe.** "
        "Factual tables from the iter1 consolidation across the available models; "
        "synthesis (esp. the within-family `exp166` 1B↔4B decay read) follows in a "
        "reply. All AUPRC, train split, chromosome-grouped OOF, single best-on-average "
        "fixed recipe per model.\n\n"
        "**Zero-shot LLR AUPRC**\n\n" + _table(data, present, "llr") + "\n\n"
        "**Probe AUPRC (fixed recipe)**\n\n" + _table(data, present, "probe") + "\n\n"
        "**Probe − LLR (Δ)**\n\n" + _table(data, present, "delta") + "\n\n"
        "Models: `exp135-1B-m5.1` (mix), `scaling-v0.5-1B` (ladder anchor), "
        "`exp166-v0.1-p1B`/`-p4B` (zoonomia-v0.1, the scale pair). "
        "Probe helps most where the LLR is low; the `exp166` pair isolates scale at "
        "fixed recipe."
    )
    subprocess.run(
        ["gh", "issue", "comment", "314", "--repo", "Open-Athena/marin-dna",
         "--body", body],
        check=True,
    )
    print(f"posted decay tables ({len(present)} models: {present})")


if __name__ == "__main__":
    main()
