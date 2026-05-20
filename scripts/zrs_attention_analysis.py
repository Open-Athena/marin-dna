"""Inspect attention patterns of the 64k segmenter on the ZRS_human window.

Hooks MHABlock.forward to capture per-layer (B, 8, S, S) attention weights,
runs the model on a 64k window centered on ZRS, and produces a multi-panel
SVG annotated with CREs (cCRE-V4), exons (Ensembl functional), repeat fraction
(from soft-masked .2bit sequence), and the model's per-bin prediction.
"""

import io
import math
from pathlib import Path

import boto3
import matplotlib
import numpy as np
import polars as pl
import pyBigWig
import scipy.stats as ss
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from alphagenome_pytorch.attention import MHABlock  # noqa: E402
from alphagenome_pytorch.utils.sequence import sequence_to_onehot  # noqa: E402
from marin_dna.data.genome import Genome  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from alphagenome_pytorch.attention import apply_rope  # noqa: E402

from marin_dna.pipelines.enhancer_segmentation.model import EnhancerSegmenter  # noqa: E402

# --- region setup -----------------------------------------------------------
ZRS_START_BIO, ZRS_END_BIO = 156790115, 156793672
WINDOW_SIZE = 65536
BIN_SIZE = 128
NUM_BINS = WINDOW_SIZE // BIN_SIZE  # 512
center = (ZRS_START_BIO + ZRS_END_BIO) // 2
WIN_START = center - WINDOW_SIZE // 2
WIN_END = WIN_START + WINDOW_SIZE
print(f"window: chr7:{WIN_START}-{WIN_END} (length {WIN_END - WIN_START})")

# --- fetch genome + ckpt from S3 -------------------------------------------
s3 = boto3.client("s3", region_name="us-east-2")
BUCKET = "oa-bolinas"
PFX = "snakemake/enhancer_classification/"


def fetch(key, dest):
    if not Path(dest).exists():
        print(f"fetching {key}")
        s3.download_file(BUCKET, PFX + key, dest)
    return dest


genome_path = fetch("results/genome/homo_sapiens.fa.gz", "/tmp/hsa.fa.gz")
ckpt_path = fetch(
    "results/segmentation/model/xfmr2_w64k_s42/seg_v1_64k/best.ckpt",
    "/tmp/w64k_best.ckpt",
)

# --- load model + patch attention ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
model = EnhancerSegmenter.load_from_checkpoint(ckpt_path, map_location=device)
model.to(device).eval()

captured: list[torch.Tensor] = []  # (per layer)
orig_forward = MHABlock.forward


def patched_forward(self, x, attention_bias, compute_dtype=None):
    B, S, D = x.shape
    if compute_dtype is None:
        compute_dtype = x.dtype
    x = x.to(compute_dtype)
    h = self.norm(x)
    q = self.norm_q(self.q_proj(h).view(B, S, 8, 128))
    k = self.norm_k(self.k_proj(h).view(B, S, 1, 128))
    v = self.norm_v(self.v_proj(h).view(B, S, 1, 192))
    q = apply_rope(q, inplace=True)
    k = apply_rope(k, inplace=True)
    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3)
    att = torch.matmul(q_t, k_t.transpose(-2, -1)).float() / math.sqrt(128.0)
    if attention_bias is not None:
        att = att + attention_bias.float()
    att = torch.tanh(att / 5.0) * 5.0
    attn_weights = F.softmax(att, dim=-1)
    captured.append(attn_weights.detach().cpu().float())
    v_t = v.permute(0, 2, 1, 3)
    y = torch.matmul(attn_weights.to(compute_dtype), v_t).float().to(compute_dtype)
    y = y.permute(0, 2, 1, 3).reshape(B, S, -1)
    y = self.linear_embedding(y)
    return self.final_norm(y)


MHABlock.forward = patched_forward

# --- forward pass -----------------------------------------------------------
g = Genome(Path(genome_path), subset_chroms={"7"})
seq = g("7", WIN_START, WIN_END)  # raw string, soft-masked (lowercase = repeat)
assert len(seq) == WINDOW_SIZE
seq_upper = seq.upper()
onehot = sequence_to_onehot(seq_upper).astype(np.float32)
x = torch.from_numpy(onehot).unsqueeze(0).to(device)

with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
    logits = model(x).float().cpu().numpy()[0]  # (NUM_BINS,)
probs = 1.0 / (1.0 + np.exp(-logits))
print(f"captured {len(captured)} attention tensors, each shape {captured[0].shape}")


# --- annotation tracks ------------------------------------------------------
def s3_pq(key):
    return pl.read_parquet(
        io.BytesIO(s3.get_object(Bucket=BUCKET, Key=PFX + key)["Body"].read())
    ).to_pandas()


# All cCREs (any class) intersecting the window
cre_all = s3_pq("results/cre/homo_sapiens/all.parquet")
cre_in = cre_all[
    (cre_all["chrom"] == "7")
    & (cre_all["end"] > WIN_START)
    & (cre_all["start"] < WIN_END)
].copy()
cre_in["start"] = cre_in["start"].clip(lower=WIN_START)
cre_in["end"] = cre_in["end"].clip(upper=WIN_END)
print(f"cCREs in window: {len(cre_in)}")

# Conserved ELS (the actual labels used for training)
els_cons = s3_pq(
    "results/cre/homo_sapiens/noexon/conserved/phastCons_43p/20/ELS.parquet"
)
els_in = els_cons[
    (els_cons["chrom"] == "7")
    & (els_cons["end"] > WIN_START)
    & (els_cons["start"] < WIN_END)
].copy()
els_in["start"] = els_in["start"].clip(lower=WIN_START)
els_in["end"] = els_in["end"].clip(upper=WIN_END)
print(f"conserved ELS in window: {len(els_in)}")

# Exons
exons = s3_pq("results/annotation/homo_sapiens/exons.parquet")
ex_in = exons[
    (exons["chrom"] == "7") & (exons["end"] > WIN_START) & (exons["start"] < WIN_END)
].copy()
ex_in["start"] = ex_in["start"].clip(lower=WIN_START)
ex_in["end"] = ex_in["end"].clip(upper=WIN_END)
print(f"exons in window: {len(ex_in)}")

# Repeat fraction per bin (lowercase chars in soft-masked seq)
seq_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8).reshape(NUM_BINS, BIN_SIZE)
is_lower = (seq_arr >= ord("a")) & (seq_arr <= ord("z"))
rep_frac = is_lower.mean(axis=1)

# phastCons (43-primate) per-bp values. Threshold 0.961 is the one used for
# the conserved-ELS filter in the data pipeline (see config.yaml).
bw_path = fetch(
    "results/conservation/homo_sapiens/phastCons_43p.bw", "/tmp/phastCons_43p.bw"
)
bw = pyBigWig.open(bw_path)
PHASTCONS_THR = 0.961
# bigwig uses UCSC chrom naming
phastcons_bp = np.asarray(
    bw.values("chr7", WIN_START, WIN_END, numpy=True), dtype=np.float32
)
bw.close()
phastcons_bp = np.nan_to_num(phastcons_bp, nan=0.0)
phastcons_bp = phastcons_bp.reshape(NUM_BINS, BIN_SIZE)
phastcons_mean = phastcons_bp.mean(axis=1)
phastcons_consfrac = (phastcons_bp >= PHASTCONS_THR).mean(axis=1)
print(
    f"phastCons mean: {phastcons_mean.mean():.3f}; conserved-bp fraction: {phastcons_consfrac.mean():.3f}"
)

# --- plot -------------------------------------------------------------------
n_layers = len(captured)
# One track row per cCRE class so each class is visually independent.
cre_classes = sorted(cre_in["cre_class"].unique())
class_to_color = {c: plt.cm.tab10(i % 10) for i, c in enumerate(cre_classes)}

# Layout: build two nested gridspecs so each group can have its own hspace.
# Track group gets breathing room between panels; heatmap group has zero
# hspace and ratios in inches so each cell is square (equal to the data
# column width) and perfectly x-aligned with the tracks above.
fig_w_data = 9.0
cbar_w = 0.28
# Track heights in inches.
h_prob, h_cre_row, h_exon, h_rep, h_phc, track_hspace_in = (
    1.2,
    0.28,
    0.35,
    0.75,
    0.75,
    0.08,
)
track_heights = [h_prob] + [h_cre_row] * len(cre_classes) + [h_exon, h_rep, h_phc]
track_names = ["prob"] + cre_classes + ["exon", "repeat", "phastCons"]
n_track_rows = len(track_heights)
# Heatmap block: n_layers square heatmaps with a thin spacer row between them.
heat_spacer_in = 0.25
h_heat = fig_w_data
# Assemble combined figure height.
total_tracks_h = sum(track_heights) + (n_track_rows - 1) * track_hspace_in
total_heat_h = n_layers * h_heat + (n_layers - 1) * heat_spacer_in
pad_top, pad_bot, pad_mid = 0.12, 0.20, 0.45
fig_h = pad_top + total_tracks_h + pad_mid + total_heat_h + pad_bot
fig_w = fig_w_data + cbar_w + 0.05
fig = plt.figure(figsize=(fig_w, fig_h))

# Outer 3-row grid: top pad (absorbed via top=...), tracks block, gap, heatmap block, bottom pad.
outer = fig.add_gridspec(
    2,
    1,
    height_ratios=[total_tracks_h, total_heat_h],
    hspace=pad_mid / ((total_tracks_h + total_heat_h) / 2),
    left=0.13,
    right=0.995,
    top=1 - pad_top / fig_h,
    bottom=pad_bot / fig_h,
)
# Tracks inner grid
gs_tr = outer[0, 0].subgridspec(
    n_track_rows,
    2,
    width_ratios=[fig_w_data, cbar_w],
    height_ratios=track_heights,
    hspace=track_hspace_in / (sum(track_heights) / n_track_rows),
    wspace=0.0,
)
# Heatmap inner grid: zero hspace between heatmap+cbar, spacer row between layers
heat_rows = []
for li in range(n_layers):
    heat_rows.append(h_heat)
    if li < n_layers - 1:
        heat_rows.append(heat_spacer_in)
gs_ht = outer[1, 0].subgridspec(
    len(heat_rows),
    2,
    width_ratios=[fig_w_data, cbar_w],
    height_ratios=heat_rows,
    hspace=0.0,
    wspace=0.0,
)
xs_bin_centers = WIN_START + (np.arange(NUM_BINS) + 0.5) * BIN_SIZE

trow = 0

# probability
ax0 = fig.add_subplot(gs_tr[trow, 0])
trow += 1
ax0.plot(xs_bin_centers, probs, color="C0", lw=0.8)
ax0.set_ylabel("Pred p", rotation=0, ha="right", va="center", labelpad=8)
ax0.set_xlim(WIN_START, WIN_END)
ax0.set_ylim(0, 1)
ax0.axvspan(ZRS_START_BIO, ZRS_END_BIO, color="orange", alpha=0.25, label="ZRS")
ax0.set_title(f"chr7:{WIN_START:,}-{WIN_END:,}  64k seg model (xfmr2_w64k_s42)")
ax0.legend(loc="upper left", frameon=False)
ax0.tick_params(labelbottom=False)

for cls in cre_classes:
    axc = fig.add_subplot(gs_tr[trow, 0], sharex=ax0)
    trow += 1
    sub = cre_in[cre_in["cre_class"] == cls]
    for _, r in sub.iterrows():
        axc.add_patch(
            Rectangle(
                (r["start"], 0),
                r["end"] - r["start"],
                1,
                color=class_to_color[cls],
                lw=0,
            )
        )
    axc.set_yticks([])
    axc.set_ylim(0, 1)
    axc.set_ylabel(cls, rotation=0, ha="right", va="center", labelpad=8)
    axc.tick_params(labelbottom=False)

axe = fig.add_subplot(gs_tr[trow, 0], sharex=ax0)
trow += 1
for _, r in ex_in.iterrows():
    axe.add_patch(
        Rectangle((r["start"], 0), r["end"] - r["start"], 1, color="firebrick", lw=0)
    )
axe.set_yticks([])
axe.set_ylim(0, 1)
axe.set_ylabel("exon", rotation=0, ha="right", va="center", labelpad=8)
axe.tick_params(labelbottom=False)

axr = fig.add_subplot(gs_tr[trow, 0], sharex=ax0)
trow += 1
axr.fill_between(xs_bin_centers, 0, rep_frac, color="grey", alpha=0.7, lw=0)
axr.set_ylim(0, 1)
axr.set_ylabel("repeat\nfraction", rotation=0, ha="right", va="center", labelpad=8)
axr.tick_params(labelbottom=False)

axp = fig.add_subplot(gs_tr[trow, 0], sharex=ax0)
trow += 1
axp.fill_between(
    xs_bin_centers, 0, phastcons_consfrac, color="seagreen", alpha=0.7, lw=0
)
axp.set_ylim(0, 1)
axp.set_ylabel(
    f"phastCons\nfrac≥{PHASTCONS_THR}", rotation=0, ha="right", va="center", labelpad=8
)
axp.tick_params(labelbottom=False)

# Heatmaps in their own grid. Each heatmap row in gs_ht: 0, 2, 4, ... (even
# indices); 1, 3, ... are spacer rows (no axes created).
for li, attn in enumerate(captured):
    ht_row = li * 2  # skip spacer rows
    ax = fig.add_subplot(gs_ht[ht_row, 0], sharex=ax0)
    a = attn[0].mean(0).numpy()
    im = ax.imshow(
        np.log10(a + 1e-9),
        aspect="auto",
        extent=[WIN_START, WIN_END, WIN_END, WIN_START],
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_ylabel(f"layer {li}\nquery bin (chr7 pos)")
    cax = fig.add_subplot(gs_ht[ht_row, 1])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("log10 attn (mean over heads)")
    if li < n_layers - 1:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("chr7 position")

out = "/tmp/zrs_attention.svg"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")

# === Quantitative analysis: incoming attention per key bin ==================
# For each layer, compute the mean attention received by each key bin
# (averaged over all queries, then over heads). Annotate each key bin with
# genomic features and look for correlations.


# Per-bin annotations (length = NUM_BINS, each bin spans 128bp).
def bin_overlap_mask(intervals_df):
    """Bool array of shape (NUM_BINS,) — True if any interval overlaps the bin."""
    mask = np.zeros(NUM_BINS, dtype=bool)
    for _, r in intervals_df.iterrows():
        # Bin i spans [WIN_START + i*128, WIN_START + (i+1)*128)
        i_start = max(0, (int(r["start"]) - WIN_START) // BIN_SIZE)
        i_end = min(NUM_BINS, (int(r["end"]) - WIN_START + BIN_SIZE - 1) // BIN_SIZE)
        if i_end > i_start:
            mask[i_start:i_end] = True
    return mask


cre_mask = bin_overlap_mask(cre_in)
els_mask = bin_overlap_mask(els_in)
exon_mask = bin_overlap_mask(ex_in)
# rep_frac already per-bin; probs already per-bin
print(
    f"per-bin annotation rates: cre={cre_mask.mean():.3f}  els={els_mask.mean():.3f}  exon={exon_mask.mean():.3f}"
)

# Incoming attention per key per layer: average over heads then over queries.
incoming = np.stack(
    [attn[0].mean(0).numpy().mean(0) for attn in captured]
)  # (n_layers, NUM_BINS)
# Self-attention along the diagonal: subtract self contribution to focus on
# what each bin contributes to *other* bins.
incoming_no_self = []
for li, attn in enumerate(captured):
    a = attn[0].mean(0).numpy()  # (S, S)
    np.fill_diagonal(a, np.nan)
    incoming_no_self.append(np.nanmean(a, axis=0))
incoming_no_self = np.stack(incoming_no_self)

# --- Correlations: single grouped bar plot ---------------------------------
# Per-bin class masks: one mask per cCRE class.
class_masks = {}
for cls in cre_classes:
    m = np.zeros(NUM_BINS, dtype=bool)
    sub = cre_in[cre_in["cre_class"] == cls]
    for _, r in sub.iterrows():
        i_start = max(0, (int(r["start"]) - WIN_START) // BIN_SIZE)
        i_end = min(NUM_BINS, (int(r["end"]) - WIN_START + BIN_SIZE - 1) // BIN_SIZE)
        m[i_start:i_end] = True
    class_masks[cls] = m

# Ordered variables for the bar plot.
variables: list[tuple[str, np.ndarray]] = [
    ("pred enhancer prob", probs),
    ("phastCons frac≥thr", phastcons_consfrac),
    ("exon", exon_mask.astype(int)),
    ("repeat fraction", rep_frac),
]
for cls in cre_classes:
    variables.append((f"cCRE:{cls}", class_masks[cls].astype(int)))

print("\n=== Spearman ρ: incoming attention vs variable ===")
rhos = np.zeros((n_layers, len(variables)))
pvals = np.zeros((n_layers, len(variables)))
print(f"{'variable':28s}  " + "  ".join(f"L{li} rho   p" for li in range(n_layers)))
for vi, (name, x) in enumerate(variables):
    row = []
    for li in range(n_layers):
        s = ss.spearmanr(incoming_no_self[li], x)
        rhos[li, vi] = s.statistic
        pvals[li, vi] = s.pvalue
        row.append(f"{s.statistic:+.3f}  {s.pvalue:.1e}")
    print(f"{name:28s}  " + "  ".join(row))

# Grouped bar plot
fig2, ax2 = plt.subplots(figsize=(max(8, 0.5 * len(variables) + 4), 4.5))
x = np.arange(len(variables))
width = 0.8 / n_layers
colors = plt.cm.tab10(np.arange(n_layers))
for li in range(n_layers):
    offset = (li - (n_layers - 1) / 2) * width
    bars = ax2.bar(
        x + offset,
        rhos[li],
        width,
        label=f"layer {li}",
        color=colors[li],
        edgecolor="black",
        lw=0.4,
    )
    # mark significance (p < 0.05)
    for bi, b in enumerate(bars):
        if pvals[li, bi] < 0.05:
            y = b.get_height()
            ax2.text(
                b.get_x() + b.get_width() / 2,
                y + (0.005 if y >= 0 else -0.02),
                "*",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=10,
            )

ax2.axhline(0, color="black", lw=0.6)
ax2.set_xticks(x)
ax2.set_xticklabels([v[0] for v in variables], rotation=35, ha="right", fontsize=9)
ax2.set_ylabel("Spearman ρ (incoming attention vs variable)")
ax2.set_title(
    "Per-bin correlations with incoming attention\n(ZRS 64k window; * = p<0.05)"
)
ax2.legend(loc="best", frameon=False)
ax2.grid(axis="y", alpha=0.3)

out2 = "/tmp/zrs_attention_quant.svg"
fig2.savefig(out2, bbox_inches="tight")
print(f"\nwrote {out2}")
