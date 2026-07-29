"""Static reproducibility contract for the GPU tutorial notebook."""

from __future__ import annotations

import ast
from pathlib import Path

NOTEBOOK = Path(__file__).parents[2] / "examples" / "model_inference_and_vep.py"
SOURCE_REVISION = "93654118aecf6b767f96fc1859648b2db772303c"
NOTEBOOK_REVISION = "962858557e6a8d7f5c5998fe030c62f8e5447cec"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
DATASET_REVISION = "225d3d1ea32a4af547891b13c33b5e92a5aae849"


def test_notebook_is_valid_python_with_immutable_revisions():
    source = NOTEBOOK.read_text(encoding="utf-8")
    ast.parse(source)
    assert "PIN_AFTER_FIRST_COMMIT" not in source
    assert "mo.code(" not in source
    assert f'SOURCE_REVISION = "{SOURCE_REVISION}"' in source
    assert f'NOTEBOOK_REVISION = "{NOTEBOOK_REVISION}"' in source
    assert f'MODEL_REVISION = "{MODEL_REVISION}"' in source
    assert f'DATASET_REVISION = "{DATASET_REVISION}"' in source
    assert '"s3://broad-references/hg38/v0/Homo_sapiens_assembly38.fasta"' in source


def test_notebook_declares_complete_analysis_stack():
    source = NOTEBOOK.read_text(encoding="utf-8")
    for requirement in (
        "datasets==",
        "ipython==",
        "logomaker==",
        "marimo==",
        "s3fs==",
        "scikit-learn==",
        "seaborn==",
        "torch==",
        "transformers==",
    ):
        assert f'"{requirement}' in source


def test_notebook_bootstraps_missing_runtime_dependencies():
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert "missing_requirements = [" in source
    assert '"aiobotocore": "aiobotocore==2.26.0"' in source
    assert '"aioitertools": "aioitertools==0.13.0"' in source
    assert '"botocore": "botocore==1.41.5"' in source
    assert "runtime_dependencies_ready" in source
    assert "--disable-pip-version-check" in source
    assert "--no-deps" in source
    assert 'os.environ["USE_TF"] = "0"' in source
    assert 'sys.modules["tensorflow"] = None' in source
    assert 'else [sys.executable, "-m", "pip", "install"]' in source


def test_notebook_exposes_required_inference_and_vep_paths():
    source = NOTEBOOK.read_text(encoding="utf-8")
    for required_snippet in (
        "with torch.inference_mode():",
        "output_hidden_states=True",
        "align_sequence_strand_outputs(",
        "aggregate_sequence_strands(",
        "run_variant_score_bundle(",
        "variant_score_bundle_view(raw_variant_bundle)",
        "rc=True",
        "return_embeddings=False",
        '"llr_avg"',
        '"minus_llr_avg"',
        '"official evals-v2 AUPRC"',
        '"absolute delta from official"',
        "parity_tolerance = 1e-3",
        "VEP_BATCH_SIZE = 128",
        "VEP_DATALOADER_WORKERS = 4",
        "VEP_TORCH_COMPILE = True",
        '"torch_compile": VEP_TORCH_COMPILE',
        '"dataloader_num_workers": VEP_DATALOADER_WORKERS',
        'genome("chr17", int(pos) - 1, int(pos))',
        'torch.multiprocessing.set_start_method("spawn", force=True)',
    ):
        assert required_snippet in source
    assert "eval_accumulation_steps" not in source
    assert "mo.ui.run_button" not in source
    assert "mo.stop(" not in source


def test_notebook_presents_exact_th_paper_interval_with_hidden_setup():
    source = NOTEBOOK.read_text(encoding="utf-8")
    th_panel = (
        NOTEBOOK.parents[1] / "dashboard" / "src" / "interpretation" / "refs" / "TH.png"
    )

    assert source.count("@app.cell(hide_code=True)") >= 2
    assert "TH_START = 2_171_682" in source
    assert "TH_END = 2_171_868" in source
    assert "TH_CONTEXT_SIZE = 186" in source
    assert "CONTEXT_SIZE = 255" in source
    assert "4cb54842e787c07df1a718cd05ecf19a41fdf86c" in source
    assert "dashboard/src/interpretation/refs/TH.png" in source
    assert "VEP" in source
    assert "below continues to use the full 255 bp context" in source
    assert th_panel.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_notebook_contains_ecdf_and_official_auprc_parity():
    source = NOTEBOOK.read_text(encoding="utf-8")

    for required_snippet in (
        'kind="ecdf"',
        '"official evals-v2 AUPRC"',
        '"official bootstrap SE"',
        '"absolute delta from official"',
        "parity_tolerance = 1e-3",
    ):
        assert required_snippet in source

    for excluded_snippet in (
        "umap",
        "UMAP",
        "PCA",
        "variant_embedding_diagnostics",
        "pair_feature",
        "author_experiment",
        "return_embeddings=True",
    ):
        assert excluded_snippet not in source
