"""Appendix: cooldown effect on VEP/loss via the m1 & m3 continuation chains.

The two zoonomia pre_cooldown chains
(`m1 -> m1.1 -> m1.2 -> m1.3`, `m3 -> m3.1 -> m3.2 -> m3.3`) are a series of
overlapping runs that give a natural cooldown counterfactual. Every continuation
warm-starts from its parent *before* that parent's learning-rate cooldown and
keeps training at peak LR; the parent meanwhile finishes its own cooldown on a
path the child never took. So at each fork we can read three evals at matched
cumulative-token levels:

  - pre_cooldown   : the parent's eval AT the fork (the shared warm-start ckpt,
                     still at peak LR).
  - post_cooldown  : the parent's FINAL eval, after it cooled down.
  - continued_peak : the child's eval at the SAME cumulative-token level as the
                     parent's cooled end, reached by continuing at peak LR.

From these we form two deltas per metric:
  - delta_cooldown  = post_cooldown  - pre_cooldown   (effect of cooling down)
  - delta_continued = continued_peak - pre_cooldown   (effect of the same extra
                      tokens spent at peak LR instead of cooling)

Token alignment is exact, not interpolated: all v0.9 1B runs share one batch x
seq = 2,097,152 tokens/step, and a continuation keeps its parent's W&B `_step`
counter, so cumulative tokens = `_step * TOKENS_PER_STEP` across a whole chain
and matching on `_step` matches on cumulative tokens. The fork step, the parent's
cooled end, and the child's matched point all land on the 10%-cadence eval steps
(within <=1 step), so each value is read directly off a logged eval.

Caveat (flagged in the output): the forks step progressively earlier into each
parent (0.8 -> 0.6 -> 0.4 of the parent's own tokens), so the parent's cooldown
*tail* lengthens down the chain while each child's peak-LR runway does not. By the
LAST fork in each chain (`m1.2->m1.3`, `m3.2->m3.3`) the child is the terminal
leaf, which has itself begun cooling down before reaching the parent's final
token level. Its `continued_peak` point is therefore ~29% into the leaf's own
cooldown — not a pure peak-LR counterfactual — so those rows are flagged
`continued_partially_cooled=True`. The pre->post cooldown delta is unaffected.

VEP AUPRCs come from the committed `data/data_mixture_history.csv` (same source
as Figure 10); `eval/loss` is not in that CSV and is fetched live from W&B at the
same eval steps. Outputs (in `plots/output/blog/marin_dna/appendix/`, with the other appendix results):
  - cooldown_effects.csv  : all 10 metrics x 6 forks (raw + deltas).
  - cooldown_effects.md   : summary for eval/loss and macro_avg.

Usage:
  uv run src/figures/appendix/cooldown_effects.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from figures import mixture_lineage as ml  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR = (
    ROOT.parents[2]
    / "plots"
    / "output"
    / "blog"
    / "marin_dna"
    / "appendix"
)
WANDB_PROJECT = "eric-czech/marin"

# Constant across every v0.9 1B run (train_batch_size x train_seq_len).
TOKENS_PER_STEP = 8192 * 256

# Leaves of the two pre_cooldown continuation chains (lineage label, leaf mix).
CHAIN_LEAVES: tuple[tuple[str, str], ...] = (
    ("m1", "exp135-zoonomia-m1.3"),
    ("m3", "exp135-zoonomia-m3.3"),
)

# The 8 VEP subsets (Figure 10 order); macro_avg is their unweighted mean.
SUBSETS: tuple[str, ...] = (
    "missense_variant",
    "tss_proximal",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "splicing",
    "synonymous_variant",
    "distal",
    "non_coding_transcript_exon_variant",
)
MACRO = "macro_avg"
LOSS = "eval_loss"
# Output metric order: loss, macro, then the 8 subsets.
METRICS: tuple[str, ...] = (LOSS, MACRO, *SUBSETS)

# Cooldown is the final 20% of a (combined) run — ResumeBeforeCooldown(0.2).
COOLDOWN_FRACTION = 0.2


def _vep_col(subset: str) -> str:
    return f"lm_eval/traitgym_mendelian_v2_255/{subset}/auprc"


def _short(mix: str) -> str:
    """Compact lineage handle, e.g. 'exp135-zoonomia-m1.2' -> 'm1.2'."""
    return mix.replace("exp135-zoonomia-", "")


def _chain(leaf: str) -> list[str]:
    """Root->leaf `mix` chain (walk parent pointers)."""
    chain: list[str] = []
    mix: str | None = leaf
    while mix is not None:
        chain.append(mix)
        mix = ml.BY_MIX[mix].parent
    return chain[::-1]


def _fetch_eval_loss(mix_to_run: dict[str, str]) -> dict[str, pd.Series]:
    """Per-mix `eval/loss` history, indexed by W&B step (same cadence as VEP)."""
    api = wandb.Api(timeout=300)
    out: dict[str, pd.Series] = {}
    for mix, run_name in mix_to_run.items():
        runs = list(api.runs(WANDB_PROJECT, filters={"display_name": run_name}))
        if not runs:
            raise RuntimeError(f"no W&B run matching display_name {run_name!r}")
        hist = runs[0].history(keys=["eval/loss"], samples=10000, pandas=True)
        sub = hist[["_step", "eval/loss"]].dropna()
        out[mix] = sub.set_index("_step")["eval/loss"].sort_index()
        print(f"  {mix}: {len(out[mix])} eval/loss points")
    return out


def _step_table(mix: str, history: pd.DataFrame, loss: pd.Series) -> pd.DataFrame:
    """Per-step wide table for one run: the 8 subsets + macro_avg + eval_loss."""
    wide = history[history["mix"] == mix].pivot_table(
        index="step", columns="metric", values="value"
    )
    df = pd.DataFrame({s: wide[_vep_col(s)] for s in SUBSETS})
    df[MACRO] = df[list(SUBSETS)].mean(axis=1)
    df[LOSS] = loss  # aligns on the shared step index (VEP & loss co-logged)
    return df.sort_index()


def _nearest(series: pd.Series, target: int) -> tuple[int, float]:
    """(step, value) of the eval nearest `target` in `series` (dropna'd)."""
    s = series.dropna()
    steps = s.index.to_numpy()
    i = int(np.argmin(np.abs(steps - target)))
    return int(steps[i]), float(s.iloc[i])


def build_rows(results: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """One row per (fork, metric): the 3 raw evals and the 2 deltas."""
    mixes = sorted({m for _, leaf in CHAIN_LEAVES for m in _chain(leaf)})
    loss = _fetch_eval_loss({m: results.loc[m, "run_name"] for m in mixes})
    tables = {m: _step_table(m, history, loss[m]) for m in mixes}

    rows: list[dict] = []
    for lineage, leaf in CHAIN_LEAVES:
        chain = _chain(leaf)
        for parent, child in zip(chain, chain[1:]):
            n_child = int(results.loc[child, "num_train_steps"])
            own_child = round(float(results.loc[child, "tokens"]) / TOKENS_PER_STEP)
            fork_step = n_child - own_child  # = child's phase start
            end_step = int(tables[parent].index.max())  # parent's cooled-end eval
            onset_child = (1.0 - COOLDOWN_FRACTION) * n_child
            for metric in METRICS:
                pre_s, pre = _nearest(tables[parent][metric], fork_step)
                post_s, post = _nearest(tables[parent][metric], end_step)
                cont_s, cont = _nearest(tables[child][metric], end_step)
                partial = cont_s > onset_child
                cooled_frac = (
                    max(0.0, (cont_s - onset_child)) / (n_child - onset_child)
                    if partial
                    else 0.0
                )
                rows.append(
                    {
                        "lineage": lineage,
                        "fork": f"{_short(parent)}->{_short(child)}",
                        "parent_mix": parent,
                        "child_mix": child,
                        "fork_step": fork_step,
                        "fork_tokens_b": fork_step * TOKENS_PER_STEP / 1e9,
                        "cooldown_end_step": end_step,
                        "end_tokens_b": end_step * TOKENS_PER_STEP / 1e9,
                        "continued_step": cont_s,
                        "metric": metric,
                        "pre_cooldown": pre,
                        "post_cooldown": post,
                        "continued_peak": cont,
                        "delta_cooldown": post - pre,
                        "delta_continued": cont - pre,
                        "continued_partially_cooled": partial,
                        "continued_cooldown_fraction": cooled_frac,
                    }
                )
    return pd.DataFrame(rows)


def _fmt(metric: str, v: float) -> str:
    return f"{v:+.3f}" if metric == LOSS else f"{v:+.4f}"


def _fmt_raw(metric: str, v: float) -> str:
    return f"{v:.3f}" if metric == LOSS else f"{v:.4f}"


def _speedup(r) -> str:
    """Δ cooldown / Δ continued — how many times larger the cooldown gain is than
    the same tokens at peak LR. Blank when Δ continued is ~0 (ratio undefined)."""
    denom = r["delta_continued"]
    if abs(denom) < 1e-9:
        return "—"
    return f"{r['delta_cooldown'] / denom:.1f}×"


def _avg_row(sub: pd.DataFrame, metric: str, label: str) -> str:
    """A bold lineage-average row over the (clean) forks in `sub`. The raw eval
    cells are left blank (averaging metric values across different token scales is
    not meaningful); only the deltas and their ratio are aggregated."""
    dc, dk = sub["delta_cooldown"].mean(), sub["delta_continued"].mean()
    speed = f"{dc / dk:.1f}×" if abs(dk) > 1e-9 else "—"
    return (
        f"| **{label}** | — | — | — | — | **{_fmt(metric, dc)}** "
        f"| **{_fmt(metric, dk)}** | **{speed}** | mean of {len(sub)} clean forks |"
    )


def _metric_table(df: pd.DataFrame, metric: str) -> list[str]:
    """Markdown table of the per-fork raw values + deltas for one metric, with a
    per-lineage average row (over that lineage's clean forks) appended."""
    sub = df[df["metric"] == metric]
    lines = [
        "| fork | tokens (B) | pre-cooldown | post-cooldown | continued (peak LR) "
        "| Δ cooldown | Δ continued | speedup | note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in sub.iterrows():
        note = (
            f"continued pt {r['continued_cooldown_fraction']:.0%} into leaf cooldown"
            if r["continued_partially_cooled"]
            else ""
        )
        lines.append(
            f"| {r['fork']} | {r['fork_tokens_b']:.0f}→{r['end_tokens_b']:.0f} "
            f"| {_fmt_raw(metric, r['pre_cooldown'])} "
            f"| {_fmt_raw(metric, r['post_cooldown'])} "
            f"| {_fmt_raw(metric, r['continued_peak'])} "
            f"| {_fmt(metric, r['delta_cooldown'])} "
            f"| {_fmt(metric, r['delta_continued'])} | {_speedup(r)} | {note} |"
        )
    # Per-lineage averages over clean forks only (the flagged terminal fork's
    # continued point is partly cooled, so it's excluded — consistent with the
    # overall clean-fork mean below the table).
    for lineage in sub["lineage"].unique():
        clean = sub[(sub["lineage"] == lineage) & ~sub["continued_partially_cooled"]]
        lines.append(_avg_row(clean, metric, f"{lineage}.x mean"))
    return lines


def write_report(df: pd.DataFrame, path: Path) -> None:
    clean = df[~df["continued_partially_cooled"]]
    out: list[str] = [
        "# Cooldown effects (m1 & m3 continuation chains)",
        "",
        "A natural counterfactual from the two zoonomia pre-cooldown chains "
        "(`m1→m1.1→m1.2→m1.3`, `m3→m3.1→m3.2→m3.3`). At each fork a child resumes "
        "from the parent **before** the parent's learning-rate cooldown and keeps "
        "training at peak LR, while the parent finishes its own cooldown on a path "
        "the child never took. This lets us compare, at matched cumulative-token "
        "levels:",
        "",
        "- **pre-cooldown** — the parent's eval at the fork (shared warm-start "
        "checkpoint, peak LR).",
        "- **post-cooldown** — the parent's final eval after it cooled down.",
        "- **continued (peak LR)** — the child's eval at the *same* cumulative "
        "tokens as the parent's cooled end, reached by staying at peak LR.",
        "",
        "**Deltas** (relative to pre-cooldown): `Δ cooldown = post − pre` (effect of "
        "cooling down) and `Δ continued = continued − pre` (effect of spending the "
        "same extra tokens at peak LR instead). For `eval/loss` lower is better, so "
        "an improvement is **negative**; for VEP AUPRC higher is better, so an "
        "improvement is **positive**.",
        "",
        "**speedup** `= Δ cooldown / Δ continued`: how many times larger the gain "
        "from cooling down is than the gain from spending the same extra tokens at "
        "peak LR. A value of e.g. 3× means cooldown achieved 3× the metric change "
        "that continued peak-LR training did over the same token budget. (Both "
        "deltas share a sign per metric, so the ratio is positive.)",
        "",
        "> **Note on the terminal forks** (`m1.2→m1.3`, `m3.2→m3.3`): the child is "
        "the terminal leaf, which begins its own cooldown before reaching the "
        "parent's final token level, so its *continued (peak LR)* point is ~29% "
        "into the leaf's cooldown rather than pure peak LR. Those rows are flagged "
        "and excluded from the clean-fork averages below. The pre→post cooldown "
        "delta is unaffected.",
        "",
        "Full per-metric data (all 8 VEP subsets) is in `cooldown_effects.csv`.",
        "",
        "## Validation loss (`eval/loss`)",
        "",
        *_metric_table(df, LOSS),
        "",
        _summary_line(clean, LOSS),
        "",
        "## VEP macro average (mean of 8 subsets)",
        "",
        *_metric_table(df, MACRO),
        "",
        _summary_line(clean, MACRO),
        "",
    ]
    path.write_text("\n".join(out))


def _summary_line(clean: pd.DataFrame, metric: str) -> str:
    sub = clean[clean["metric"] == metric]
    dc, dk = sub["delta_cooldown"].mean(), sub["delta_continued"].mean()
    speed = f"{dc / dk:.1f}×" if abs(dk) > 1e-9 else "—"
    return (
        f"Mean over the {clean['fork'].nunique()} clean forks: "
        f"Δ cooldown = {_fmt(metric, dc)}, Δ continued = {_fmt(metric, dk)}, "
        f"speedup = {speed}."
    )


def main() -> None:
    results = pd.read_csv(DATA_DIR / "data_mixture_results.csv").set_index("mix")
    history = pd.read_csv(DATA_DIR / "data_mixture_history.csv")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_rows(results, history)
    csv_path = OUT_DIR / "cooldown_effects.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")

    md_path = OUT_DIR / "cooldown_effects.md"
    write_report(df, md_path)
    print(f"Wrote report to {md_path}")


if __name__ == "__main__":
    main()
