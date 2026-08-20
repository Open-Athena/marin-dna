import pandas as pd
from marin_dna_carbon_conditioning_vep.report import render_summary


def test_summary_renders_only_selected_analysis_conditions() -> None:
    config = {
        "config_path": "config/config.yaml",
        "analysis": {
            "conditions": ["untagged", "correct"],
            "subset": "tss_proximal",
        },
        "model": {"repo": "model", "revision": "model-revision"},
        "dataset": {
            "repo": "dataset",
            "split": "train",
            "revision": "dataset-revision",
        },
        "reference": {
            "assembly": "GRCh38",
            "ensembl_release": 115,
            "masking": "soft-masked",
        },
        "inference": {"window_size_bp": 8192, "kmer_size": 6},
        "metrics": {"n_bootstrap": 1000, "min_groups_for_macro": 30},
    }
    absolute = pd.DataFrame(
        [
            {
                "condition": condition,
                "subset": subset,
                "auprc": 0.2,
                "n_rows": 2050,
                "n_groups": 205,
                "macro_eligible": True,
                "low_sample": False,
            }
            for subset in ("tss_proximal", "_macro_avg_")
            for condition in ("untagged", "correct")
        ]
    )
    paired = pd.DataFrame(
        [
            {
                "comparison": "correct_minus_untagged",
                "subset": subset,
                "delta": 0.01,
                "ci_low": -0.01,
                "ci_high": 0.03,
                "n_groups": 205,
                "low_sample": False,
            }
            for subset in ("tss_proximal", "_macro_avg_")
        ]
    )
    preflight = {
        "selected_grammar": "corpus_card",
        "rejected_grammar": "model_card",
        "grammar_templates": {
            "corpus_card": "<species>{species}<dna>",
            "model_card": "<{species}><dna>",
        },
        "tokenizer_revision": "model-revision",
        "selected_prefixes": {
            "untagged": "<dna>",
            "correct": "<species>vertebrate_mammalian<dna>",
        },
        "prefix_ids": {"untagged": [1], "correct": [2, 3]},
    }
    runtimes = [
        {
            "condition": condition,
            "rows": 2050,
            "devices": ["NVIDIA GH200 480GB"],
            "elapsed_seconds": 400.0,
            "peak_gpu_memory_bytes": 12 * 1024**3,
            "peak_rss_bytes": 4 * 1024**3,
        }
        for condition in ("untagged", "correct")
    ]

    summary = render_summary(
        config=config,
        preflight=preflight,
        absolute_metrics=absolute,
        paired_deltas=paired,
        exclusions=pd.DataFrame(),
        runtimes=runtimes,
    )

    assert "reports 2 prompt conditions" in summary
    assert "| subset | untagged | correct |" in summary
    assert "near_wrong" not in summary
    assert "far_wrong" not in summary
