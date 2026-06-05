"""Free baselines for issue #262: evaluate precomputed ChromBPNet + Enformer scores
on the standardized caQTL/dsQTL benchmarks (ChromBPNet paper, Synapse syn64126763).

No model runs, no API. Downloads the standardized TSVs (cached), evaluates each
precomputed baseline under the standard protocol (causality auPRC + signed-Pearson
direction over positives), and tabulates vs the papers' reported numbers.

Run (needs SYNAPSE_AUTH_TOKEN):

    uv run python scripts/chrombpnet_eval/standardized_baseline_eval.py
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from marin_dna.pipelines.chrombpnet_eval.standardized_qtl import (
    ENCODE_ASSAY,  # noqa: F401  (re-exported for readers)
    STANDARDIZED_QTL,
    baselines_for,
    evaluate_score_columns,
    load_standardized_qtl,
)

CACHE = Path(
    os.environ.get("CHROMBPNET_EVAL_CACHE", Path.home() / ".cache/chrombpnet_eval")
)

# AlphaGenome Suppl Table 4 (reported), for context. Each: (causality auPRC, direction pearson).
AG_TABLE = {
    "caqtl": {
        "random_auprc": 0.0852,
        "ChromBPNet": (0.4148, 0.6737),
        "Borzoi-ens": (0.4624, 0.6485),
        "AlphaGenome": (0.5643, 0.7368),
    },
    "dsqtl": {
        "random_auprc": 0.0200,
        "ChromBPNet": (0.5432, 0.7722),
        "Borzoi-ens": (0.6056, 0.7922),
        "AlphaGenome": (0.6308, 0.8323),
    },
}


def synapse_download_tsv(syn_id: str, dest_tsv: Path) -> Path:
    """Download a Synapse FileEntity (a zip of one .tsv) and extract the TSV (cached)."""
    if dest_tsv.exists():
        return dest_tsv
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    assert token, "set SYNAPSE_AUTH_TOKEN (a Synapse PAT with Download scope)"
    dest_tsv.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://repo-prod.prod.sagebase.org/repo/v1/entity/{syn_id}/file"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".tsv")]
        assert len(names) == 1, f"{syn_id}: expected 1 .tsv in zip, got {names}"
        with zf.open(names[0]) as src:
            dest_tsv.write_bytes(src.read())
    return dest_tsv


def protocol_table(metrics: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Pick the protocol-official metric per baseline: causality auPRC from the
    causality column, direction Pearson from the direction column."""
    by = metrics.set_index(["score_column", "metric"])
    rows = []
    for b in baselines_for(df):
        rows.append(
            {
                "model": b.model,
                "assay": b.assay,
                "causality_auPRC": by.loc[(b.causality_col, "AUPRC"), "value"],
                "causality_se": by.loc[(b.causality_col, "AUPRC"), "se"],
                "direction_pearson": by.loc[(b.direction_col, "pearson"), "value"],
                "direction_se": by.loc[(b.direction_col, "pearson"), "se"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    all_tables = {}
    for name, spec in STANDARDIZED_QTL.items():
        tsv = synapse_download_tsv(spec.synapse_id, CACHE / f"std_{name}.tsv")
        df = load_standardized_qtl(str(tsv), spec)
        npos = int(df["label"].sum())
        print(
            f"\n=== {name} ({spec.build}) — n={len(df)}, n_pos={npos}, "
            f"n_ctrl={len(df) - npos}, pos_rate={npos / len(df):.4f} "
            f"(AG random {AG_TABLE[name]['random_auprc']}) ==="
        )
        cols = sorted(
            {c for b in baselines_for(df) for c in (b.causality_col, b.direction_col)}
        )
        metrics = evaluate_score_columns(df, cols)
        tbl = protocol_table(metrics, df)
        # attach AG-reported ChromBPNet for the matching model rows
        tbl["ref_AG_ChromBPNet_auPRC"] = [
            AG_TABLE[name]["ChromBPNet"][0] if m == "ChromBPNet" else np.nan
            for m in tbl["model"]
        ]
        tbl["ref_AG_ChromBPNet_pearson"] = [
            AG_TABLE[name]["ChromBPNet"][1] if m == "ChromBPNet" else np.nan
            for m in tbl["model"]
        ]
        with pd.option_context(
            "display.float_format", lambda v: f"{v:.4f}", "display.width", 200
        ):
            print(tbl.to_string(index=False))
        all_tables[name] = tbl

    out = CACHE / "standardized_baseline_eval.csv"
    pd.concat({k: v for k, v in all_tables.items()}, names=["dataset"]).to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
