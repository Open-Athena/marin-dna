"""Static reproducibility contract for the GPU tutorial notebook."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[2] / "examples" / "model_inference_and_vep.py"
SESSION_SNAPSHOT = (
    NOTEBOOK.parent / "__marimo__" / "session" / "model_inference_and_vep.py.json"
)
SOURCE_REVISION = "0242fc668389d02399ff9835a56326dda11ece98"
NOTEBOOK_REVISION = "a036d31d66a86474f79520503a24ec29b6dfd16a"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
DATASET_REVISION = "225d3d1ea32a4af547891b13c33b5e92a5aae849"
BROAD_REFERENCE_FASTA = "s3://broad-references/hg38/v0/Homo_sapiens_assembly38.fasta"
ENSEMBL_REFERENCE_FASTA = "https://huggingface.co/datasets/marin-dna/human-genome/resolve/11b9433582981bb929af333bc6422f10a8fd71b4/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa"


def test_notebook_is_valid_python_with_immutable_revisions():
    source = NOTEBOOK.read_text(encoding="utf-8")
    ast.parse(source)
    assert "PIN_AFTER_FIRST_COMMIT" not in source
    assert "mo.code(" not in source
    assert f'SOURCE_REVISION = "{SOURCE_REVISION}"' in source
    assert f'NOTEBOOK_REVISION = "{NOTEBOOK_REVISION}"' in source
    assert f'MODEL_REVISION = "{MODEL_REVISION}"' in source
    assert f'DATASET_REVISION = "{DATASET_REVISION}"' in source
    assert (
        "marin-dna/human-genome/resolve/11b9433582981bb929af333bc6422f10a8fd71b4"
        in source
    )
    assert 'MODEL_ID = "marin-dna/marin-dna-exp135-m5.1"' in source
    assert 'DATASET_ID = "marin-dna/evals_sge"' in source


def test_notebook_declares_complete_analysis_stack():
    source = NOTEBOOK.read_text(encoding="utf-8")
    for requirement in (
        "datasets==",
        "ipython==",
        "logomaker==",
        "marimo==",
        "fsspec[http]==",
        "scikit-learn==",
        "seaborn==",
        "torch==",
        "transformers==",
    ):
        assert f'"{requirement}' in source


def test_notebook_bootstraps_missing_runtime_dependencies():
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert "missing_requirements = [" in source
    assert '"aiohttp": "aiohttp==3.14.3"' in source
    assert '"s3fs":' not in source
    assert "runtime_dependencies_ready" in source
    assert "--disable-pip-version-check" in source
    assert "--no-deps" in source
    assert 'os.environ["USE_TF"] = "0"' in source
    assert 'sys.modules["tensorflow"] = None' in source
    assert 'else [sys.executable, "-m", "pip", "install"]' in source


def test_notebook_exposes_required_inference_and_vep_paths():
    source = NOTEBOOK.read_text(encoding="utf-8")
    for required_snippet in (
        "run_aligned_sequence_strand(",
        "aggregate_sequence_strands(",
        "run_variant_score_bundle(",
        "hidden_size=model.config.hidden_size",
        "rc=True",
        "return_embeddings=True",
        "ref_embeddings",
        "alt_embeddings",
        "linear probing",
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
        'genome("17", int(pos) - 1, int(pos))',
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

    assert source.count("@app.cell(hide_code=True)") >= 14
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
        "**Main sequence likelihood:**",
        "import umap",
        "umap.UMAP(",
        "from sklearn.decomposition import PCA",
        "PCA(",
        "variant_embedding_diagnostics",
        "pair_feature",
        "author_experiment",
    ):
        assert excluded_snippet not in source

    assert "**Mean sequence log-likelihood (nats/base):**" in source
    assert "mean log-probabilities in nats/base, not probabilities" in source
    assert "matching the official VEP scorer's precision policy" in source


def test_notebook_session_snapshot_contains_complete_rendered_run():
    session = json.loads(SESSION_SNAPSHOT.read_text(encoding="utf-8"))

    assert session["version"] == "1"
    assert session["metadata"]["marimo_version"] == "0.23.15"
    assert len(session["cells"]) == 28
    assert all(len(cell["outputs"]) == 1 for cell in session["cells"])
    assert all(
        output.get("type") != "error"
        for cell in session["cells"]
        for output in cell["outputs"]
    )

    rendered_session = json.dumps(session)
    for expected_output in (
        "Mean sequence log-likelihood (nats/base):",
        "VEP bundle ready:",
        "Official evals-v2 parity:",
        "ref_embeddings",
        "alt_embeddings",
        "AUPRC",
    ):
        assert expected_output in rendered_session
    assert "marin-dna/human-genome" in rendered_session
    assert "Broad/1000 Genomes" not in rendered_session
    assert "Main sequence likelihood:" not in rendered_session


@pytest.mark.skipif(
    os.getenv("MARIN_DNA_VALIDATE_BRCA1_REFERENCE") != "1",
    reason="set MARIN_DNA_VALIDATE_BRCA1_REFERENCE=1 for remote reference validation",
)
def test_public_ensembl_reference_matches_all_notebook_model_inputs():
    from datasets import load_dataset

    from marin_dna.data.genome import Genome

    broad = Genome(BROAD_REFERENCE_FASTA, storage_options={"anon": True})
    ensembl = Genome(ENSEMBL_REFERENCE_FASTA)
    assert broad.chroms["chr11"] == ensembl.chroms["11"] == 135_086_622
    assert broad.chroms["chr17"] == ensembl.chroms["17"] == 83_257_441

    th_start = 2_171_682
    th_end = 2_171_868
    broad_th = broad("chr11", th_start, th_end, strand="-").upper()
    ensembl_th = ensembl("11", th_start, th_end, strand="-").upper()
    assert broad_th == ensembl_th
    assert len(ensembl_th) == th_end - th_start == 186

    sge = load_dataset(
        "marin-dna/evals_sge",
        split="train",
        revision=DATASET_REVISION,
    )
    brca1 = sge.filter(lambda row: row["gene"] == "BRCA1").select_columns(
        ["chrom", "pos", "ref"]
    )
    frame = brca1.to_pandas()
    assert len(frame) == 2_751
    assert frame["chrom"].unique().tolist() == ["17"]

    window_size = 255
    left_flank = window_size // 2
    positions = frame["pos"].astype(int)
    block_start = int(positions.min()) - 1 - left_flank
    block_end = int(positions.max()) - 1 - left_flank + window_size
    broad_block = broad("chr17", block_start, block_end).upper()
    ensembl_block = ensembl("17", block_start, block_end).upper()
    assert broad_block == ensembl_block
    assert len(ensembl_block) == block_end - block_start

    for pos, ref in zip(positions, frame["ref"], strict=True):
        window_start = int(pos) - 1 - left_flank
        offset = window_start - block_start
        context = ensembl_block[offset : offset + window_size]
        assert len(context) == window_size
        assert context[left_flank] == ref
