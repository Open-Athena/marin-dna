"""Frozen constants for issue-435 paired repeat-variant analysis."""

from __future__ import annotations

VARIANT_PANEL_RUN_ID = "dna-exp435-repeat-variant-panel-r1"
SOURCE_PANEL_SHA256 = "a1843d90c81877a8991b2730b1cad412a4cc36c9518aa514fdcb2ba2123d63ea"
SOURCE_DATASET_ID = "bolinas-dna/evals_mendelian_traits"
SOURCE_DATASET_REVISION = "4aed58e50c5dea0b878a665007af2ef9e5108e9f"
EXPECTED_VARIANTS = 16_140
EXPECTED_MATCH_GROUPS = 1_614
MIN_CATEGORY_VARIANTS = 32
REPEAT_INTERIOR_BP = 32

POSITION_STATUSES = (
    "focal_repeat",
    "near_repeat",
    "repeat_free_window",
)
