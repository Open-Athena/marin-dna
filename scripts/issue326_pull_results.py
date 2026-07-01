"""exp326 readout (#326): online in-training AUPRC + LL gap, exp326 arms vs exp232 ccre baseline.

All numbers are the ONLINE in-training lm_eval (`lm_eval/mendelian_traits_255/<subset>/avg/auprc`,
FWD/RC-averaged) so exp326 and exp232 are methodology-matched. The OFFICIAL exp232 heatmap
numbers (distal 0.127 etc.) come from the OFFLINE evals_v2 pipeline — kept separate.
"""
import numpy as np
import pandas as pd
import wandb
from scipy import stats

PROJ = "gonzalobenegas/marin"
SUBSETS = ["distal", "splicing", "synonymous_variant", "missense_variant",
           "3_prime_UTR_variant", "5_prime_UTR_variant",
           "non_coding_transcript_exon_variant", "tss_proximal"]
RECIPES = ["val_cds", "val_utr3", "val_ncrna", "val_tss_pc", "val_enhancer"]

RUNS = {
    "exp326_A_noexon": ("dna-exp326-v0.1", "-v4_ccre_noexon-v0.1"),
    "exp326_B_noexon_enh": ("dna-exp326-v0.1", "-v4_ccre_noexon_enhancer-v0.1"),
    "exp232_ccre_baseline": ("dna-exp232-v0.1", "-v4_ccre_non_promoter-v0.1"),
}

api = wandb.Api(timeout=180)


def pick_run(group, name_sub):
    runs = list(api.runs(PROJ, filters={"group": group}, order="-created_at"))
    cands = [r for r in runs if name_sub in r.name]
    cands.sort(key=lambda r: (r.state == "finished", r.created_at), reverse=True)
    return cands[0] if cands else None


auprc_keys = [f"lm_eval/mendelian_traits_255/{s}/avg/auprc" for s in SUBSETS]
auprc_keys += ["lm_eval/mendelian_traits_255/_macro_avg_/avg/auprc",
               "lm_eval/mendelian_traits_255/_global_/avg/auprc"]
ll_keys = [f"eval/{r}_{k}/loss" for r in RECIPES for k in ("functional", "nonfunctional")]

print("=" * 100)
final = {}
traj = {}
for label, (group, sub) in RUNS.items():
    run = pick_run(group, sub)
    if run is None:
        print(f"{label}: NO RUN FOUND in {group} matching {sub}")
        continue
    print(f"{label}: {run.name} (state={run.state}, id={run.id})")
    h = run.history(keys=auprc_keys + ll_keys, samples=10000, pandas=True)
    if "_step" in h:
        h = h.sort_values("_step")
    final[label] = {"run": run.name}
    # final-step values (last non-null per key)
    for s in SUBSETS + ["_macro_avg_", "_global_"]:
        k = f"lm_eval/mendelian_traits_255/{s}/avg/auprc"
        col = h[k].dropna() if k in h else pd.Series(dtype=float)
        final[label][s] = float(col.iloc[-1]) if len(col) else np.nan
    for r in RECIPES:
        f = h[f"eval/{r}_functional/loss"].dropna() if f"eval/{r}_functional/loss" in h else pd.Series(dtype=float)
        n = h[f"eval/{r}_nonfunctional/loss"].dropna() if f"eval/{r}_nonfunctional/loss" in h else pd.Series(dtype=float)
        final[label][f"gap_{r}"] = float(n.iloc[-1] - f.iloc[-1]) if len(f) and len(n) else np.nan
    # trajectory: distal auprc & val_enhancer gap aligned by step
    dk = "lm_eval/mendelian_traits_255/distal/avg/auprc"
    ef, en = "eval/val_enhancer_functional/loss", "eval/val_enhancer_nonfunctional/loss"
    if dk in h and ef in h and en in h:
        t = h[["_step", dk, ef, en]].copy()
        t["gap"] = t[en] - t[ef]
        t = t.dropna(subset=[dk])
        # align gap to nearest step (ffill from eval cadence)
        t["gap"] = t["gap"].ffill()
        t = t.dropna(subset=["gap"])
        traj[label] = t[["_step", dk, "gap"]].rename(columns={dk: "distal_auprc"})

print("\n" + "=" * 100)
print("FINAL-STEP ONLINE AUPRC (avg, FWD/RC) — per subset")
print("=" * 100)
df = pd.DataFrame(final).T
cols = SUBSETS + ["_macro_avg_", "_global_"]
print(df[cols].to_string(float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 100)
print("FINAL-STEP LL GAP (nonfunc_loss - func_loss)  [positive = constraint learned]")
print("=" * 100)
print(df[[f"gap_{r}" for r in RECIPES]].to_string(float_format=lambda x: f"{x:+.4f}"))

print("\n" + "=" * 100)
print("TRAJECTORY: Pearson r  (distal online AUPRC  vs  val_enhancer LL gap)")
print("=" * 100)
for label, t in traj.items():
    if len(t) >= 4:
        r, p = stats.pearsonr(t["distal_auprc"], t["gap"])
        print(f"  {label:22s} n={len(t):2d}  r={r:+.3f}  p={p:.3f}   "
              f"[distal {t['distal_auprc'].iloc[0]:.3f}->{t['distal_auprc'].iloc[-1]:.3f}, "
              f"gap {t['gap'].iloc[0]:+.3f}->{t['gap'].iloc[-1]:+.3f}]")
    else:
        print(f"  {label:22s} n={len(t)} (too few points)")

# dump trajectories for the markdown
for label, t in traj.items():
    print(f"\n--- {label} trajectory (step, distal_auprc, val_enh_gap) ---")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
