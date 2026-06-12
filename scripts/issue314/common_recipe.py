"""issue #314 — clean cross-model comparison at a SINGLE common recipe.

The per-model consolidation picks each model's own best-on-average config, which
confounds cross-model *probe* comparisons (different pooling/combo/C per model). Because
``sample_configs`` is seeded identically, **config IDs denote the same recipe across all
models**, so we can hold the recipe fixed and read its per-(model, subset) AUPRC straight
from the saved ``auprc.parquet`` — a fair cross-model comparison with no re-fitting.

Picks the common config with the highest mean within-(model, subset)-centered AUPRC over
all models and subsets (one recipe that's best on average across the fleet), then prints
zero-shot LLR (from ``best_vs_llr.parquet``) vs that fixed-recipe probe, per model.
Read-only, runs locally in seconds.
"""

import argparse

import polars as pl

MODELS = ["exp135-1B-m5.1", "scaling-v0.5-1B", "exp166-v0.1-p1B", "exp166-v0.1-p4B"]
SHORT = {"exp135-1B-m5.1": "exp135-1B", "scaling-v0.5-1B": "scaling-1B",
         "exp166-v0.1-p1B": "exp166-1B", "exp166-v0.1-p4B": "exp166-4B"}
ORDER = ["missense_variant", "distal", "tss_proximal", "splicing",
         "synonymous_variant", "5_prime_UTR_variant",
         "non_coding_transcript_exon_variant", "3_prime_UTR_variant"]
SH = {"missense_variant": "missense", "distal": "distal", "tss_proximal": "tss",
      "splicing": "splicing", "synonymous_variant": "synon",
      "5_prime_UTR_variant": "5pUTR",
      "non_coding_transcript_exon_variant": "ncRNA", "3_prime_UTR_variant": "3pUTR"}
BASE = "s3://oa-bolinas/analysis/issue314/iter1_search"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config_id", type=int, default=None,
                    help="force a specific common config; default = fleet best-avg")
    args = ap.parse_args()

    auprc = {m: pl.read_parquet(f"{BASE}/{m}/auprc.parquet") for m in MODELS}
    llr = {m: {r["subset"]: r["llr_auprc"]
               for r in pl.read_parquet(f"{BASE}/{m}/best_vs_llr.parquet").iter_rows(named=True)}
           for m in MODELS}

    if args.config_id is None:
        pooled = pl.concat([auprc[m].with_columns(pl.lit(m).alias("model")) for m in MODELS])
        cen = pooled.with_columns(
            (pl.col("auprc") - pl.col("auprc").mean().over("model", "subset")).alias("ce"))
        cfg = int(cen.group_by("config_id").agg(pl.col("ce").mean().alias("mce"))
                  .sort("mce", descending=True).row(0, named=True)["config_id"])
    else:
        cfg = args.config_id
    meta = auprc[MODELS[0]].filter(pl.col("config_id") == cfg).row(0, named=True)
    print(f"COMMON recipe = config {cfg}: {meta['rep']} scaler={int(meta['scaler'])} "
          f"pca={meta['n_pca']} C={meta['c']:.3g}\n")

    def probe(m: str, s: str) -> float | None:
        r = auprc[m].filter((pl.col("config_id") == cfg) & (pl.col("subset") == s))
        return r.row(0, named=True)["auprc"] if len(r) else None

    header = "| subset | " + " | ".join(SHORT[m] for m in MODELS) + " |"
    for title, fn in [("zero-shot LLR", lambda m, s: llr[m].get(s)),
                      (f"probe @ common recipe (cfg {cfg})", probe)]:
        print(f"## {title}\n{header}\n|---|" + "---:|" * len(MODELS))
        for s in ORDER:
            cells = [("—" if (v := fn(m, s)) is None else f"{v:.3f}") for m in MODELS]
            print(f"| {SH[s]} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
