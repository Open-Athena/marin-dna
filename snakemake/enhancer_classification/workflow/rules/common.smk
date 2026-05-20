import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.metrics import precision_recall_curve
import pandas as pd
import polars as pl
import pyBigWig
from huggingface_hub import hf_hub_download

import bioframe as bf

from marin_dna.data.intervals import GenomicList, GenomicSet
from marin_dna.data.negative_sampling import compute_gc_content, compute_repeat_fraction, match_by_gc_repeat
from marin_dna.data.utils import ENHANCER_CRE_CLASSES, add_rc, get_ensembl_functional_exons, load_annotation, load_fasta
from marin_dna.pipelines.enhancer_segmentation.labeling import label_windows_by_bin_overlap


SPECIES = list(config["genome_urls"].keys())

DATASET_NAMES = list(config["datasets"].keys())

ALL_INTERVALS: set[str] = set()
ALL_SPLIT_NAMES: set[str] = set()
for _dataset_config in list(config["datasets"].values()) + list(
    config.get("seg_datasets", {}).values()
):
    _intervals_cfg = _dataset_config["intervals"]
    if isinstance(_intervals_cfg, dict):
        ALL_INTERVALS.update(_intervals_cfg.values())
    else:
        ALL_INTERVALS.add(_intervals_cfg)
    _split_config = config["splits"][_dataset_config["split"]]
    for _species_splits in _split_config.values():
        ALL_SPLIT_NAMES.update(_species_splits.keys())

ALL_CONSERVATIONS: set[str] = set()
for _species_cons in config.get("conservation", {}).values():
    ALL_CONSERVATIONS.update(_species_cons.keys())

NEGATIVES_FOR_SAMPLING: dict[str, str] = {
    "random": "negatives",
    "gc_repeat_matched": "negatives_gc_repeat_matched",
}

ALL_NEG_TYPES: set[str] = {"random"}
for _dataset_config in config["datasets"].values():
    ALL_NEG_TYPES.add(_dataset_config.get("negative_sampling", "random"))

TRAIN_SPLIT = "train"


def resolve_intervals(intervals_cfg: str | dict, species: str) -> str:
    """Resolve the intervals path for a species.

    Supports both a single string (all species share the same path) and
    a per-species dict mapping.
    """
    if isinstance(intervals_cfg, dict):
        return intervals_cfg[species]
    return intervals_cfg

MODEL_DEFAULTS = config["models"]["default"]


def get_model_config(model_name: str) -> dict:
    return {**MODEL_DEFAULTS, **config["models"].get(model_name, {})}


def get_all_dataset_outputs() -> list[str]:
    outputs = []
    for dataset_name, dataset_config in config["datasets"].items():
        split_config = config["splits"][dataset_config["split"]]
        split_names: set[str] = set()
        for species_splits in split_config.values():
            split_names.update(species_splits.keys())
        for split_name in split_names:
            outputs.append(f"results/dataset/{dataset_name}/{split_name}.parquet")
    return outputs


def fasta_to_df(path: str, label: int, genome: str) -> pd.DataFrame:
    series = load_fasta(path)
    df = series.to_frame().reset_index(names="id")
    coords = df["id"].str.split(":", expand=True)
    df["chrom"] = coords[0]
    start_end = coords[1].str.split("-", expand=True)
    df["start"] = start_end[0].astype(int)
    df["end"] = start_end[1].astype(int)
    df["strand"] = "+"
    df["label"] = label
    df["genome"] = genome
    return df[["genome", "chrom", "start", "end", "strand", "seq", "label"]]
