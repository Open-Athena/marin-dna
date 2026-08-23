from biofoundation.data import Genome
from biofoundation.model.adapters.hf import HFTokenizer
from biofoundation.model.adapters.gpn import GPNMaskedLM, GPNEmbeddingModel
from biofoundation.inference import run_llr_mlm, run_euclidean_distance
from datasets import Dataset
import gpn.model  # noqa: F401  # Registers the GPN architecture
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel


COORDINATES = ["chrom", "pos", "ref", "alt"]


rule download_genome:
    output:
        "results/genome.fa.gz",
    shell:
        "wget -O {output} {config[genome_url]}"
