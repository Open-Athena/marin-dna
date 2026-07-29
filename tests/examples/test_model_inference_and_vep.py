"""Static reproducibility contract for the GPU tutorial notebook."""

from __future__ import annotations

import ast
from pathlib import Path

NOTEBOOK = Path(__file__).parents[2] / "examples" / "model_inference_and_vep.py"
SOURCE_REVISION = "8bf15b6707e495987c16b62bcd4ef93618ffb134"
NOTEBOOK_REVISION = "05fa583f13c59a7b3ff000473ee57c862af5950e"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
DATASET_REVISION = "225d3d1ea32a4af547891b13c33b5e92a5aae849"


def test_notebook_is_valid_python_with_immutable_revisions():
    source = NOTEBOOK.read_text(encoding="utf-8")
    ast.parse(source)
    assert "PIN_AFTER_FIRST_COMMIT" not in source
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
        "umap-learn==",
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
        "rc=True",
        "return_embeddings=True",
        '"llr_avg"',
        '"minus_llr_avg"',
        '"concat_ref_delta"',
        "random_state=409",
        'genome("chr17", int(pos) - 1, int(pos))',
    ):
        assert required_snippet in source
