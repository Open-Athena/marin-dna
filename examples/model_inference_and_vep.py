# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "accelerate==1.13.0",
#     "datasets==3.6.0",
#     "einops==0.8.1",
#     "fsspec[http]==2025.3.0",
#     "huggingface-hub==0.36.2",
#     "ipython==9.15.0",
#     "jaxtyping==0.3.9",
#     "joblib==1.5.3",
#     "logomaker==0.8.7",
#     "marimo==0.23.15",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "pandas==2.3.3",
#     "pyfaidx==0.9.0.4",
#     "scikit-learn==1.8.0",
#     "seaborn==0.13.2",
#     "torch==2.8.0",
#     "transformers==4.57.6",
# ]
# ///

"""Code-visible MarinDNA inference and zero-shot VEP tutorial (issue #409)."""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full", app_title="MarinDNA inference and VEP")


@app.cell(hide_code=True)
def _():
    from importlib import invalidate_caches as _invalidate_caches
    from importlib.util import find_spec as _find_spec
    import subprocess as _subprocess
    import sys as _sys

    required_modules = {
        "accelerate": "accelerate==1.13.0",
        "aiohttp": "aiohttp==3.14.3",
        "datasets": "datasets==3.6.0",
        "dateutil": "python-dateutil==2.9.0.post0",
        "einops": "einops==0.8.1",
        "fsspec": "fsspec==2025.3.0",
        "huggingface_hub": "huggingface-hub==0.36.2",
        "IPython": "ipython==9.15.0",
        "jaxtyping": "jaxtyping==0.3.9",
        "joblib": "joblib==1.5.3",
        "logomaker": "logomaker==0.8.7",
        "matplotlib": "matplotlib==3.10.8",
        "multidict": "multidict==6.7.1",
        "numpy": "numpy==2.4.3",
        "pandas": "pandas==2.3.3",
        "pyfaidx": "pyfaidx==0.9.0.4",
        "sklearn": "scikit-learn==1.8.0",
        "seaborn": "seaborn==0.13.2",
        "transformers": "transformers==4.57.6",
    }
    missing_requirements = [
        requirement
        for module, requirement in required_modules.items()
        if _find_spec(module) is None
    ]
    if missing_requirements:
        _subprocess.run(
            [
                _sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                *missing_requirements,
            ],
            check=True,
        )
        _invalidate_caches()

    assert all(_find_spec(module) is not None for module in required_modules)
    runtime_dependencies_ready = True
    return runtime_dependencies_ready


@app.cell(hide_code=True)
def _(runtime_dependencies_ready):
    import importlib
    import importlib.util
    import os
    import shutil
    import subprocess
    import sys
    from functools import cache
    from pathlib import Path

    assert runtime_dependencies_ready

    # Molab ships an irrelevant TensorFlow build that can crash while probing
    # the attached GPU, so make Transformers skip that backend.
    os.environ["USE_TF"] = "0"
    assert "tensorflow" not in sys.modules
    sys.modules["tensorflow"] = None

    import joblib
    import logomaker
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch

    # The notebook opens the remote FASTA in the parent before VEP. Spawn gives
    # each data-loader worker a clean pyfaidx/fsspec HTTP handle and async loop,
    # matching hosted notebook runtimes.
    torch.multiprocessing.set_start_method("spawn", force=True)
    assert torch.multiprocessing.get_start_method() == "spawn"

    del sys.modules["tensorflow"]

    from datasets import Dataset, load_dataset
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # The notebook is self-contained when opened from a commit-pinned GitHub URL.
    # Install only MarinDNA itself here; every runtime dependency is pinned above.
    SOURCE_REVISION = "0242fc668389d02399ff9835a56326dda11ece98"
    NOTEBOOK_REVISION = "a036d31d66a86474f79520503a24ec29b6dfd16a"
    revision_marker = (
        Path.home()
        / ".cache"
        / "marin_dna"
        / "model_inference_and_vep"
        / "source-revision"
    )
    revision_marker.parent.mkdir(parents=True, exist_ok=True)
    installed_revision = (
        revision_marker.read_text(encoding="utf-8").strip()
        if revision_marker.exists()
        else None
    )
    if (
        importlib.util.find_spec("marin_dna") is None
        or installed_revision != SOURCE_REVISION
    ):
        uv_executable = shutil.which("uv")
        install_command = (
            [
                uv_executable,
                "pip",
                "install",
                "--python",
                sys.executable,
            ]
            if uv_executable is not None
            else [sys.executable, "-m", "pip", "install"]
        )
        subprocess.run(
            [
                *install_command,
                "--no-deps",
                "--reinstall",
                (f"git+https://github.com/Open-Athena/marin-dna.git@{SOURCE_REVISION}"),
            ],
            check=True,
        )
        revision_marker.write_text(SOURCE_REVISION, encoding="utf-8")
        importlib.invalidate_caches()

    from marin_dna.data.dna import NUCLEOTIDES, reverse_complement
    from marin_dna.data.genome import Genome
    from marin_dna.model.runner import run_variant_score_bundle
    from marin_dna.model.sequence_interpretation import (
        aggregate_sequence_strands,
        normalize_dna_sequence,
        run_aligned_sequence_strand,
    )
    from marin_dna.model.variant_interpretation import variant_score_bundle_view

    seaborn = sns
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        Dataset,
        Genome,
        NUCLEOTIDES,
        NOTEBOOK_REVISION,
        Path,
        SOURCE_REVISION,
        StandardScaler,
        aggregate_sequence_strands,
        average_precision_score,
        cache,
        joblib,
        load_dataset,
        logomaker,
        mo,
        normalize_dna_sequence,
        np,
        os,
        pd,
        plt,
        reverse_complement,
        run_aligned_sequence_strand,
        run_variant_score_bundle,
        seaborn,
        torch,
        variant_score_bundle_view,
    )


@app.cell
def _(NOTEBOOK_REVISION):
    MODEL_ID = "marin-dna/marin-dna-exp135-m5.1"
    MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
    MODEL_DOWNLOAD_BYTES = 4_483_112_944
    DATASET_ID = "marin-dna/evals_sge"
    DATASET_REVISION = "225d3d1ea32a4af547891b13c33b5e92a5aae849"
    REFERENCE_FASTA = "https://huggingface.co/datasets/marin-dna/human-genome/resolve/11b9433582981bb929af333bc6422f10a8fd71b4/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa"
    TH_CHROM = "11"
    TH_START = 2_171_682
    TH_END = 2_171_868
    TH_STRAND = "-"
    TH_ORIGINAL_START = 2_171_682
    TH_ORIGINAL_END = 2_171_868
    TH_ORIGINAL_SEQUENCE = (
        "GGGGGCTTTGACGTCAGCTCAGCTTATAAGAGGCTGCTGGGCCAGGGCTGTGGAGACGGAG"
        "CCCGGACCTCCACACTGAGCCATGCCCACCCCCGACGCCACCACGCCACAGGCCAAGGGCTT"
        "CCGCAGGGCCGTGTCTGAGCTGGACGCCAAGCAGGCAGAGGCCATCATGGTAAGAGGGCAGGT"
    )
    CONTEXT_SIZE = 255
    TH_CONTEXT_SIZE = 186
    VEP_BATCH_SIZE = 128
    VEP_DATALOADER_WORKERS = 4
    VEP_TORCH_COMPILE = True
    SOURCE_URL = (
        "https://github.com/Open-Athena/marin-dna/blob/"
        f"{NOTEBOOK_REVISION}/examples/model_inference_and_vep.py"
    )
    return (
        CONTEXT_SIZE,
        DATASET_ID,
        DATASET_REVISION,
        MODEL_DOWNLOAD_BYTES,
        MODEL_ID,
        MODEL_REVISION,
        REFERENCE_FASTA,
        SOURCE_URL,
        TH_CHROM,
        TH_CONTEXT_SIZE,
        TH_END,
        TH_ORIGINAL_END,
        TH_ORIGINAL_SEQUENCE,
        TH_ORIGINAL_START,
        TH_START,
        TH_STRAND,
        VEP_BATCH_SIZE,
        VEP_DATALOADER_WORKERS,
        VEP_TORCH_COMPILE,
    )


@app.cell(hide_code=True)
def _(MODEL_DOWNLOAD_BYTES, MODEL_ID, MODEL_REVISION, SOURCE_URL, mo):
    mo.vstack(
        [
            mo.md(
                f"""
                # MarinDNA model inference and zero-shot VEP

                This linear, code-visible tutorial loads the pinned public
                [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}),
                runs a real 186 bp GRCh38 sequence in both orientations, and scores
                the complete pinned BRCA1 saturation-genome-editing set.

                [Commit-pinned notebook source]({SOURCE_URL})
                """
            ),
            mo.callout(
                mo.md(
                    f"""
                    **Hardware expectation.** Use a CUDA GPU with BF16 support.
                    The immutable model file is approximately
                    **{MODEL_DOWNLOAD_BYTES / 10**9:.2f} GB**. The notebook fails
                    loudly when CUDA is unavailable; running this 1.12B-parameter
                    model and 2,751-variant VEP workload on CPU is not a practical
                    fallback.

                    **Molab:** choose **Server**, then **Configure compute → GPU**,
                    and restart the runtime before running cells. WebAssembly has
                    no CUDA device and cannot execute this notebook.
                    """
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell
def _(torch):
    assert torch.cuda.is_available(), (
        "This tutorial requires a CUDA GPU with BF16 support; CPU fallback is "
        "intentionally disabled."
    )
    assert torch.cuda.is_bf16_supported(), (
        "The selected CUDA device lacks BF16 support."
    )
    device = torch.device("cuda")
    model_dtype = torch.bfloat16
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return device, gpu_name, model_dtype


@app.cell(hide_code=True)
def _(gpu_name, mo, model_dtype, torch):
    mo.callout(
        mo.md(
            f"""
            **Runtime:** PyTorch `{torch.__version__}` · device **{gpu_name}** ·
            dtype **{model_dtype}** · CUDA `{torch.version.cuda}`
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    CONTEXT_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    cache,
    device,
    model_dtype,
):
    @cache
    def load_pinned_model():
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
        )
        loaded_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=model_dtype,
            attn_implementation="sdpa",
        ).to(device)
        loaded_model.eval()
        assert loaded_model.config.max_position_embeddings == CONTEXT_SIZE + 1
        return loaded_model, loaded_tokenizer

    model, tokenizer = load_pinned_model()
    assert model.config.hidden_size == 1920
    assert model.config.num_hidden_layers == 19
    return model, tokenizer


@app.cell(hide_code=True)
def _(
    CONTEXT_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    gpu_name,
    mo,
    model,
    model_dtype,
    pd,
):
    model_details = pd.DataFrame(
        [
            ("model", MODEL_ID),
            ("revision", MODEL_REVISION),
            ("device", gpu_name),
            ("dtype", str(model_dtype)),
            ("training DNA context", f"{CONTEXT_SIZE} bp"),
            ("maximum token context", model.config.max_position_embeddings),
            ("hidden size", model.config.hidden_size),
            ("layers", model.config.num_hidden_layers),
        ],
        columns=["field", "value"],
    )
    mo.vstack([mo.md("## 1. Load the model and reference"), model_details])
    return


@app.cell
def _(Genome, REFERENCE_FASTA):
    genome = Genome(REFERENCE_FASTA)
    assert genome.chroms["11"] == 135_086_622
    assert genome.chroms["17"] == 83_257_441
    return (genome,)


@app.cell(hide_code=True)
def _(REFERENCE_FASTA, mo):
    mo.callout(
        mo.md(
            """
            **Reference asset:** pinned public Ensembl release 115 GRCh38
            soft-masked primary assembly from
            [`marin-dna/human-genome`](https://huggingface.co/datasets/marin-dna/human-genome).
            It uses Ensembl contig names and credential-free HTTP byte-range
            reads with its adjacent `.fai`; this does not download all of GRCh38.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(
    TH_CHROM,
    TH_CONTEXT_SIZE,
    TH_END,
    TH_ORIGINAL_END,
    TH_ORIGINAL_SEQUENCE,
    TH_ORIGINAL_START,
    TH_START,
    TH_STRAND,
    genome,
    normalize_dna_sequence,
    reverse_complement,
):
    th_sequence = genome(TH_CHROM, TH_START, TH_END, strand=TH_STRAND).upper()
    th_sequence = normalize_dna_sequence(
        th_sequence,
        min_length=TH_CONTEXT_SIZE,
        max_length=TH_CONTEXT_SIZE,
    )
    assert len(th_sequence) == TH_CONTEXT_SIZE == TH_END - TH_START

    # Negative-strand oriented coordinates reverse the offset calculation.
    original_oriented_start = TH_END - TH_ORIGINAL_END
    original_oriented_end = TH_END - TH_ORIGINAL_START
    assert (original_oriented_start, original_oriented_end) == (0, 186)
    assert (
        th_sequence[original_oriented_start:original_oriented_end]
        == TH_ORIGINAL_SEQUENCE
    )
    th_reverse_complement = reverse_complement(th_sequence)
    return (
        original_oriented_end,
        original_oriented_start,
        th_reverse_complement,
        th_sequence,
    )


@app.cell(hide_code=True)
def _(
    TH_CHROM,
    TH_END,
    TH_ORIGINAL_END,
    TH_ORIGINAL_START,
    TH_START,
    TH_STRAND,
    mo,
    original_oriented_end,
    original_oriented_start,
    th_sequence,
):
    mo.vstack(
        [
            mo.md(
                f"""
                ### Real 186 bp TH input

                **TH · GRCh38 {TH_CHROM}:{TH_START}-{TH_END} ({TH_STRAND}) ·
                0-based, half-open · 186 bp**

                The sequence below is reverse-complemented into 5′→3′ order on
                the annotated negative strand. This is exactly the interval used
                by the GPN-Star paper reference and the MarinDNA nucleotide-
                dependency dashboard, preserved at oriented slice
                `[{original_oriented_start}, {original_oriented_end})`. The model
                supports inputs shorter than its 255 bp training context; VEP
                below continues to use the full 255 bp context.
                """
            ),
            mo.md(
                "```text\n"
                + "\n".join(
                    th_sequence[offset : offset + 62]
                    for offset in range(0, len(th_sequence), 62)
                )
                + "\n```"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("### Functional context from GPN-Star"),
            mo.image(
                (
                    "https://raw.githubusercontent.com/Open-Athena/marin-dna/"
                    "4cb54842e787c07df1a718cd05ecf19a41fdf86c/"
                    "dashboard/src/interpretation/refs/TH.png"
                ),
                alt=(
                    "Annotated GPN-Star reference panel for the 186 bp "
                    "tyrosine-hydroxylase locus"
                ),
                width="100%",
                rounded=True,
                caption=(
                    "Annotated reference — GPN-Star, Ye, Benegas et al., bioRxiv 2025"
                ),
            ),
            mo.md(
                """
                The annotations make the promoter and functional elements in this
                compact sequence concrete before inspecting model-derived views.
                See the [live MarinDNA nucleotide-dependency dashboard](https://openathena.ai/marin-dna/interpretation/nucleotide-dependency)
                and the [GPN-Star preprint](https://www.biorxiv.org/content/10.1101/2025.09.21.677619v1).
                """
            ),
        ]
    )
    return


@app.cell
def _(
    TH_CONTEXT_SIZE,
    device,
    th_reverse_complement,
    th_sequence,
    tokenizer,
    torch,
):
    forward_input_ids = torch.tensor(
        tokenizer.encode(th_sequence),
        dtype=torch.long,
        device=device,
    )[None, :]
    reverse_input_ids = torch.tensor(
        tokenizer.encode(th_reverse_complement),
        dtype=torch.long,
        device=device,
    )[None, :]
    assert tokenizer.bos_token_id is not None
    assert forward_input_ids[0, 0].item() == tokenizer.bos_token_id
    assert reverse_input_ids[0, 0].item() == tokenizer.bos_token_id
    assert forward_input_ids.shape == reverse_input_ids.shape
    assert forward_input_ids.shape == (1, TH_CONTEXT_SIZE + 1)
    return forward_input_ids, reverse_input_ids


@app.cell(hide_code=True)
def _(forward_input_ids, mo, pd, reverse_input_ids, tokenizer):
    tokenization_details = pd.DataFrame(
        [
            ("forward tensor", str(tuple(forward_input_ids.shape))),
            ("reverse-complement tensor", str(tuple(reverse_input_ids.shape))),
            ("BOS token ID", tokenizer.bos_token_id),
            ("DNA tokens after BOS", forward_input_ids.shape[1] - 1),
        ],
        columns=["field", "value"],
    )
    mo.vstack(
        [
            mo.md(
                """
                ### Tokenize with BOS

                MarinDNA uses one token per nucleotide and prepends BOS. The BOS
                token is essential: its logit causally predicts nucleotide zero.
                """
            ),
            tokenization_details,
        ]
    )
    return


@app.cell
def _(
    TH_CONTEXT_SIZE,
    aggregate_sequence_strands,
    forward_input_ids,
    model,
    reverse_input_ids,
    run_aligned_sequence_strand,
    tokenizer,
):
    forward_aligned = run_aligned_sequence_strand(
        model,
        tokenizer,
        forward_input_ids,
        TH_CONTEXT_SIZE,
        reverse_complemented=False,
    )
    reverse_aligned = run_aligned_sequence_strand(
        model,
        tokenizer,
        reverse_input_ids,
        TH_CONTEXT_SIZE,
        reverse_complemented=True,
    )
    sequence_outputs = aggregate_sequence_strands(
        forward_aligned,
        reverse_aligned,
    )
    assert forward_aligned.nucleotide_logits.shape == (TH_CONTEXT_SIZE, 4)
    assert reverse_aligned.nucleotide_logits.shape == (TH_CONTEXT_SIZE, 4)
    assert sequence_outputs.embeddings.shape == (
        TH_CONTEXT_SIZE,
        model.config.hidden_size,
    )
    return (sequence_outputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Align before averaging

    For nucleotide `i`, the causal prediction is the logit at the token immediately
    before it; its embedding is the final-layer hidden state at the nucleotide token
    itself. Reverse-complement positions are reversed into forward coordinates.
    A/C/G/T logit channels are additionally complemented (`A↔T`, `C↔G`), while
    hidden-state dimensions are not.
    """)
    return


@app.cell
def _(NUCLEOTIDES, logomaker, pd, plt, sequence_outputs):
    logo_frame = pd.DataFrame(
        sequence_outputs.logo.glyph_heights_bits,
        columns=NUCLEOTIDES,
    )
    logo_figure, logo_axis = plt.subplots(figsize=(16, 3.6))
    logomaker.Logo(
        logo_frame,
        ax=logo_axis,
        color_scheme={"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"},
    )
    logo_axis.set(
        xlabel="Sequence-relative position (0-based)",
        ylabel="Information (bits)",
        ylim=(0, 2),
        title="Forward/RC-averaged MarinDNA nucleotide logo",
    )
    logo_figure.tight_layout()
    logo_figure
    return


@app.cell(hide_code=True)
def _(TH_CONTEXT_SIZE, mo, pd, sequence_outputs):
    likelihood_table = pd.DataFrame(
        {
            "strand": ["forward", "reverse complement", "strand mean"],
            "mean log-likelihood (nats/base)": [
                sequence_outputs.forward_log_likelihood_nats_per_base,
                sequence_outputs.reverse_complement_log_likelihood_nats_per_base,
                sequence_outputs.average_log_likelihood_nats_per_base,
            ],
        }
    )
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    "**Mean sequence log-likelihood (nats/base):** "
                    f"forward "
                    f"`{sequence_outputs.forward_log_likelihood_nats_per_base:.4f}` · "
                    f"reverse complement "
                    f"`{sequence_outputs.reverse_complement_log_likelihood_nats_per_base:.4f}` · "
                    f"strand mean "
                    f"`{sequence_outputs.average_log_likelihood_nats_per_base:.4f}`."
                ),
                kind="success",
            ),
            mo.md(
                """
                Log-likelihood uses a **full-vocabulary** `log_softmax`, gathers
                the log-probability of each observed nucleotide target, averages
                over the
                {TH_CONTEXT_SIZE} bases on each
                strand, then averages the two strand log-likelihoods. These are
                mean log-probabilities in nats/base, not probabilities. This
                calculation does not use the logo's four-nucleotide
                renormalization.

                The model forward runs in BF16. Logits are cast to FP32 before
                `log_softmax`, and the per-base and two-strand means accumulate
                in FP32, matching the official VEP scorer's precision policy.
                The score itself is different: this is full-vocabulary sequence
                log-likelihood, whereas VEP reports a four-nucleotide
                log-likelihood ratio.
                """.format(TH_CONTEXT_SIZE=TH_CONTEXT_SIZE)
            ),
            likelihood_table,
        ]
    )
    return


@app.cell
def _(StandardScaler, np, plt, seaborn, sequence_outputs):
    position_embeddings_standardized = StandardScaler().fit_transform(
        sequence_outputs.embeddings
    )
    assert position_embeddings_standardized.shape == sequence_outputs.embeddings.shape
    assert np.isfinite(position_embeddings_standardized).all()

    embedding_figure, embedding_axis = plt.subplots(figsize=(16, 7))
    seaborn.heatmap(
        position_embeddings_standardized.T,
        ax=embedding_axis,
        cmap="vlag",
        center=0,
        vmin=-3,
        vmax=3,
        xticklabels=16,
        yticklabels=False,
        cbar_kws={"label": "standardized activation"},
    )
    embedding_axis.collections[0].set_rasterized(True)
    embedding_axis.set(
        xlabel="Sequence-relative position (0-based)",
        ylabel="Final-layer embedding dimension",
        title=(
            "Per-position final-layer embeddings "
            "(FWD/RC mean; dimensions standardized across positions)"
        ),
    )
    embedding_figure.tight_layout()
    embedding_figure
    return


@app.cell
def _(DATASET_ID, DATASET_REVISION, Dataset, genome, load_dataset, np):
    sge_train = load_dataset(
        DATASET_ID,
        split="train",
        revision=DATASET_REVISION,
    )
    brca1_dataset = sge_train.filter(lambda row: row["gene"] == "BRCA1")
    retained_columns = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "label",
        "subset",
        "gene",
        "mavedb_urn",
    ]
    brca1_frame = brca1_dataset.select_columns(retained_columns).to_pandas()

    assert len(brca1_frame) == 2_751
    assert brca1_frame["subset"].value_counts().to_dict() == {
        "missense_variant": 2_010,
        "splicing": 741,
    }
    assert brca1_frame["label"].value_counts().to_dict() == {
        False: 2_153,
        True: 598,
    }
    assert brca1_frame["mavedb_urn"].unique().tolist() == ["urn:mavedb:00000097-0-2"]
    assert brca1_frame["chrom"].unique().tolist() == ["17"]
    assert brca1_frame["gene"].unique().tolist() == ["BRCA1"]
    assert brca1_frame["label"].nunique() == 2
    assert brca1_frame["ref"].str.len().eq(1).all()
    assert brca1_frame["alt"].str.len().eq(1).all()
    assert brca1_frame["ref"].isin(list("ACGT")).all()
    assert brca1_frame["alt"].isin(list("ACGT")).all()

    # Preserve source fields; they already use the Ensembl contig names.
    scoring_frame = brca1_frame.copy()
    scoring_frame["row_id"] = np.arange(len(scoring_frame), dtype=np.int64)
    scoring_frame["source_chrom"] = scoring_frame["chrom"]
    scoring_frame["source_pos_1based"] = scoring_frame["pos"]
    scoring_frame["chrom"] = scoring_frame["chrom"].astype(str)
    reference_bases = np.array(
        [genome("17", int(pos) - 1, int(pos)).upper() for pos in scoring_frame["pos"]]
    )
    assert np.array_equal(reference_bases, scoring_frame["ref"].to_numpy())
    scoring_dataset = Dataset.from_pandas(scoring_frame, preserve_index=False)
    assert len(scoring_dataset) == len(scoring_frame)
    return brca1_frame, scoring_dataset, scoring_frame


@app.cell(hide_code=True)
def _(DATASET_ID, DATASET_REVISION, brca1_frame, mo):
    dataset_counts = (
        brca1_frame.groupby(["subset", "label"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .replace(
            {
                "subset": {
                    "missense_variant": "missense",
                    "splicing": "splicing",
                },
                "label": {False: "normal", True: "abnormal"},
            }
        )
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ## 2. Zero-shot variant-effect prediction

                Dataset:
                [`{DATASET_ID}`](https://huggingface.co/datasets/{DATASET_ID}/tree/{DATASET_REVISION})
                at immutable revision `{DATASET_REVISION}` · complete BRCA1 subset ·
                **2,751 real SNVs**.

                The source `pos` and the current runner boundary are **VCF-style
                1-based**. Every reference check above explicitly reads the
                0-based, half-open FASTA interval **`[pos - 1, pos)`**. Direct
                `Genome` intervals elsewhere in MarinDNA remain 0-based,
                half-open. Source contig `17` already matches the Ensembl FASTA,
                so no contig renaming is needed at the scoring boundary.
                """
            ),
            dataset_counts,
        ]
    )
    return


@app.cell(hide_code=True)
def _(VEP_BATCH_SIZE, VEP_DATALOADER_WORKERS, VEP_TORCH_COMPILE, mo):
    mo.md(
        f"""
        The complete VEP runs automatically. Its raw bundle is cached outside the
        repository and keyed by model revision, dataset revision, reference,
        context, strand/embedding options, batching, and notebook/source revisions.
        Editing plots or metrics downstream does not rerun model forwards.

        **Execution:** BF16 · `torch.compile={VEP_TORCH_COMPILE}` · batch size
        `{VEP_BATCH_SIZE}` · `{VEP_DATALOADER_WORKERS}` data-loader workers · no
        evaluation accumulation barrier.
        """
    )
    return


@app.cell
def _(
    CONTEXT_SIZE,
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    NOTEBOOK_REVISION,
    Path,
    REFERENCE_FASTA,
    SOURCE_REVISION,
    VEP_BATCH_SIZE,
    VEP_DATALOADER_WORKERS,
    VEP_TORCH_COMPILE,
    joblib,
    os,
    run_variant_score_bundle,
):
    cache_root = Path(
        os.getenv(
            "MARIN_DNA_NOTEBOOK_CACHE",
            str(Path.home() / ".cache" / "marin_dna" / "model_inference_and_vep"),
        )
    )
    score_cache = joblib.Memory(cache_root, verbose=0)
    vep_cache_key = (
        NOTEBOOK_REVISION,
        SOURCE_REVISION,
        MODEL_ID,
        MODEL_REVISION,
        DATASET_ID,
        DATASET_REVISION,
        REFERENCE_FASTA,
        CONTEXT_SIZE,
        True,
        True,
        VEP_BATCH_SIZE,
        VEP_DATALOADER_WORKERS,
        VEP_TORCH_COMPILE,
    )

    @score_cache.cache(ignore=["model", "tokenizer", "dataset", "genome"])
    def run_cached_variant_scoring(
        cache_key,
        *,
        model,
        tokenizer,
        dataset,
        genome,
    ):
        assert cache_key == vep_cache_key
        return run_variant_score_bundle(
            model,
            tokenizer,
            dataset,
            genome,
            CONTEXT_SIZE,
            rc=True,
            return_embeddings=True,
            data_transform_on_the_fly=True,
            inference_kwargs={
                "per_device_eval_batch_size": VEP_BATCH_SIZE,
                "bf16_full_eval": True,
                "torch_compile": VEP_TORCH_COMPILE,
                "dataloader_num_workers": VEP_DATALOADER_WORKERS,
                "remove_unused_columns": False,
                "report_to": [],
            },
        )

    return run_cached_variant_scoring, vep_cache_key


@app.cell
def _(
    genome,
    model,
    np,
    pd,
    run_cached_variant_scoring,
    scoring_dataset,
    scoring_frame,
    tokenizer,
    torch,
    variant_score_bundle_view,
    vep_cache_key,
):
    torch.cuda.reset_peak_memory_stats()
    raw_variant_bundle = run_cached_variant_scoring(
        vep_cache_key,
        model=model,
        tokenizer=tokenizer,
        dataset=scoring_dataset,
        genome=genome,
    )
    bundle_view = variant_score_bundle_view(
        raw_variant_bundle,
        hidden_size=model.config.hidden_size,
    )
    assert bundle_view.ref_embeddings is not None
    assert bundle_view.alt_embeddings is not None
    variant_scores = bundle_view.scores.copy()
    variant_scores["llr_avg"] = (
        variant_scores["llr_fwd"] + variant_scores["llr_rc"]
    ) / 2
    variant_scores["jsd_avg"] = (
        variant_scores["jsd_fwd"] + variant_scores["jsd_rc"]
    ) / 2
    variant_scores["minus_llr_avg"] = -variant_scores["llr_avg"]
    variant_results = pd.concat(
        [scoring_frame.reset_index(drop=True), variant_scores],
        axis=1,
    )
    ref_embeddings = bundle_view.ref_embeddings
    alt_embeddings = bundle_view.alt_embeddings
    assert len(variant_results) == len(scoring_frame)
    assert np.array_equal(
        variant_results["row_id"].to_numpy(),
        np.arange(len(variant_results), dtype=np.int64),
    )
    assert ref_embeddings.shape == alt_embeddings.shape
    assert ref_embeddings.shape == (len(variant_results), model.config.hidden_size)
    assert np.isfinite(variant_scores.to_numpy()).all()
    assert np.isfinite(ref_embeddings).all()
    assert np.isfinite(alt_embeddings).all()
    peak_vram_gib = torch.cuda.max_memory_allocated() / 2**30
    return alt_embeddings, peak_vram_gib, ref_embeddings, variant_results


@app.cell(hide_code=True)
def _(mo, peak_vram_gib, variant_results):
    mo.callout(
        mo.md(
            f"""
            **VEP bundle ready:** {len(variant_results):,} rows in original order ·
            raw `llr_fwd`, `llr_rc`, `jsd_fwd`, and `jsd_rc` retained ·
            peak allocated VRAM observed for this execution: **{peak_vram_gib:.2f} GiB**.

            The FWD/RC-averaged `ref_embeddings` and `alt_embeddings` arrays
            are also available with shape `{(len(variant_results), 1_920)}` for
            downstream analyses such as linear probing. This tutorial returns
            them but deliberately does not apply PCA or UMAP.

            `LLR = log P(ALT) - log P(REF)` is a summed log-likelihood ratio in
            **nats** over the variant and downstream causal predictions in the
            runner's four-nucleotide space. Positive raw LLR favors ALT; a
            disruptive ALT tends to have a negative raw LLR. Consequently,
            `minus_llr_avg = -llr_avg` points upward toward the positive
            (`label=True`, calibrated-abnormal) class. JSD is non-negative and
            is not sign-flipped.
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(seaborn, variant_results):
    ecdf_frame = variant_results.assign(
        label_display=variant_results["label"].map({False: "normal", True: "abnormal"}),
        subset_display=variant_results["subset"].map(
            {"missense_variant": "missense", "splicing": "splicing"}
        ),
    )
    llr_ecdf = seaborn.displot(
        data=ecdf_frame,
        x="llr_avg",
        hue="label_display",
        col="subset_display",
        kind="ecdf",
        stat="proportion",
        palette={"normal": "#4c78a8", "abnormal": "#e45756"},
        height=4,
        aspect=1.35,
    )
    llr_ecdf.set_axis_labels(
        "Raw FWD/RC-mean LLR (nats; ALT − REF)",
        "Within-label cumulative fraction",
    )
    llr_ecdf.set_titles("{col_name} variants")
    llr_ecdf.figure.suptitle(
        "BRCA1 zero-shot LLR ECDFs (each label normalized separately)",
        y=1.04,
    )
    llr_ecdf.figure
    return


@app.cell
def _(SOURCE_REVISION, average_precision_score, mo, pd, variant_results):
    pooled_prevalence = float(variant_results["label"].mean())
    pooled_auprc = float(
        average_precision_score(
            variant_results["label"],
            variant_results["minus_llr_avg"],
        )
    )
    auprc_rows = [
        {
            "scope": "pooled BRCA1",
            "n": len(variant_results),
            "abnormal n": int(variant_results["label"].sum()),
            "prevalence / no-skill": pooled_prevalence,
            "AUPRC from -llr_avg": pooled_auprc,
        }
    ]
    for subset_name, subset_frame in variant_results.groupby("subset", sort=True):
        auprc_rows.append(
            {
                "scope": (
                    "missense" if subset_name == "missense_variant" else subset_name
                ),
                "n": len(subset_frame),
                "abnormal n": int(subset_frame["label"].sum()),
                "prevalence / no-skill": float(subset_frame["label"].mean()),
                "AUPRC from -llr_avg": float(
                    average_precision_score(
                        subset_frame["label"],
                        subset_frame["minus_llr_avg"],
                    )
                ),
            }
        )
    auprc_table = pd.DataFrame(auprc_rows)
    official_evals_v2 = pd.DataFrame(
        [
            {
                "scope": "pooled BRCA1",
                "official evals-v2 AUPRC": 0.3883988428066554,
                "official bootstrap SE": 0.0200169324033533,
            },
            {
                "scope": "missense",
                "official evals-v2 AUPRC": 0.4034596892875675,
                "official bootstrap SE": 0.023155120660151324,
            },
            {
                "scope": "splicing",
                "official evals-v2 AUPRC": 0.307823544062546,
                "official bootstrap SE": 0.0377770810842366,
            },
        ]
    )
    auprc_table = auprc_table.merge(
        official_evals_v2,
        on="scope",
        how="left",
        validate="one_to_one",
    )
    auprc_table["absolute delta from official"] = (
        auprc_table["AUPRC from -llr_avg"] - auprc_table["official evals-v2 AUPRC"]
    ).abs()
    parity_tolerance = 1e-3
    max_auprc_delta = float(auprc_table["absolute delta from official"].max())
    assert max_auprc_delta <= parity_tolerance
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"""
                    **Pooled BRCA1 AUPRC:** `{pooled_auprc:.4f}` ·
                    **positive-class prevalence / no-skill baseline:**
                    `{pooled_prevalence:.4f}`.

                    The ECDF intentionally displays raw `llr_avg`; AUPRC
                    explicitly uses `minus_llr_avg` so a larger score predicts the
                    impactful/calibrated-abnormal (`label=True`) class.

                    **Official evals-v2 parity:** the largest absolute AUPRC
                    delta across pooled, missense, and splicing cells is
                    `{max_auprc_delta:.6f}` (required ≤ `{parity_tolerance:g}`).
                    The reference cells were reproduced from the official
                    embedding-bearing score artifact using the
                    [`minus_llr_avg` transform and SGE aggregation](
                    https://github.com/Open-Athena/marin-dna/blob/{SOURCE_REVISION}/snakemake/analysis/evals_v2/workflow/rules/metrics.smk#L42-L86).
                    Its eager embedding extraction and this compiled, batch-128
                    notebook can perturb BF16 reductions and a few near-tied
                    ranks; the point estimates nevertheless agree to <0.001.
                    """
                ),
                kind="success",
            ),
            auprc_table,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
