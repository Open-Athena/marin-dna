"""Training-region palette, labels, and the variant→region map for Fig 5.

Fig 5 draws one panel per variant type and colors each by the training-data region
dataset most *relevant* to that variant, so a reader can see how the downstream
variant-effect scaling tracks training-data composition. The five region datasets
are color-coded with the blog's own :data:`EARTH_QUAL` qualitative palette in the
fixed order used across the schematics (issue #368 /
``_mixture_lineage.REGION_ORDER``): CDS = rust, Upstream = teal, Downstream = ochre,
ncRNA = slate, Enhancer = olive. Kept out of the vendored ``_style/`` dir (which
stays byte-for-byte re-fetchable) but reuses its palette.
"""

from __future__ import annotations

from matplotlib.lines import Line2D

from plots.blog._style.figure_style import EARTH_QUAL

# The five training-data region datasets, in the fixed EARTH_QUAL slot order.
REGION_ORDER = ("cds", "upstream", "downstream", "ncrna_exon", "ccre_non_promoter")
REGION_COLORS = {r: EARTH_QUAL[i] for i, r in enumerate(REGION_ORDER)}
REGION_LABELS = {
    "cds": "CDS",
    "upstream": "Upstream",
    "downstream": "Downstream",
    "ncrna_exon": "ncRNA",
    "ccre_non_promoter": "Enhancer",
}

# Variant subset → the *most relevant* training-region dataset. Coding variants
# (missense/synonymous) → CDS; the promoter/TSS-proximal variant → the Upstream
# (pre-mRNA) region; the 3' UTR variant → the Downstream (post-mRNA) region. Two are
# judgment calls: ``splicing`` (coding-exon boundaries → CDS) and
# ``5_prime_UTR_variant`` (5' flank of the mRNA → Upstream) — both TENTATIVE; confirm
# the intended assignment.
VARIANT_REGION: dict[str, str] = {
    "missense_variant": "cds",
    "synonymous_variant": "cds",
    "splicing": "cds",
    "tss_proximal": "upstream",
    "5_prime_UTR_variant": "upstream",
    "3_prime_UTR_variant": "downstream",
}

# Shared panel orders for Figs 5–6. Mendelian groups the CDS-relevant variants on
# the first row and regulatory variants on the second; SGE has only its two
# assayed consequences. Keeping these here makes the two figures' ordering an
# explicit invariant rather than two recipes that can silently drift apart.
MENDELIAN_VARIANT_ORDER: tuple[str, ...] = (
    "missense_variant",
    "synonymous_variant",
    "splicing",
    "tss_proximal",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
)
SGE_VARIANT_ORDER: tuple[str, ...] = ("missense_variant", "splicing")


def region_legend_handles(regions: list[str]) -> tuple[list[Line2D], list[str]]:
    """Proxy line artists (colored by region) for a `relevant training region` key.

    ``regions`` is the ordered list of region keys actually drawn; each proxy is a
    marker+line in that region's :data:`REGION_COLORS` color with its
    :data:`REGION_LABELS` label, matching how the panels are drawn.
    """
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=REGION_COLORS[r],
            markeredgecolor="k",
            markeredgewidth=0.4,
            markersize=7,
            linewidth=1.3,
        )
        for r in regions
    ]
    labels = [REGION_LABELS[r] for r in regions]
    return handles, labels
