#!/bin/bash
set -euo pipefail
export PATH=/home/ubuntu/miniforge3/bin:/home/ubuntu/.local/bin:$PATH
export POLARS_MAX_THREADS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
curl -fLsS https://astral.sh/uv/0.11.31/install.sh | sh
curl -fLsS https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /opt/issue550/miniforge.sh
sha256sum /opt/issue550/miniforge.sh
bash /opt/issue550/miniforge.sh -b -p /home/ubuntu/miniforge3
git clone --filter=blob:none --no-checkout https://github.com/Open-Athena/marin-dna.git /opt/issue550/repo
git -C /opt/issue550/repo worktree add --detach /opt/issue550/producer 6b1593c274a886d20f5c0ddf3712916d446f5fed
git -C /opt/issue550/repo worktree add --detach /opt/issue550/publisher 54b6f936467bbc657ae88753a6f24b768bd3363d
cd /opt/issue550/producer/snakemake/vertebrate_projection_dataset
uv sync --locked --group dev
python3 - <<'PY'
import os
from pathlib import Path
import subprocess
commit='6b1593c274a886d20f5c0ddf3712916d446f5fed'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==commit
stamp=int(subprocess.check_output(['git','show','-s','--format=%ct',commit],text=True))
for name in ('chains.tsv','genomes.tsv','species.tsv'):
    path=Path('config/rag_issue550')/name
    assert path.read_bytes()==subprocess.check_output(['git','show',f'{commit}:snakemake/vertebrate_projection_dataset/{path}'])
    os.utime(path,(stamp,stamp))
PY
rag_dataset_dir=/opt/issue550/assets/s3/oa-bolinas/snakemake/vertebrate_projection_dataset/results/rag-five-regions-v1/6b1593c274a886d20f5c0ddf3712916d446f5fed/10ba63bc375ba912909b82cda68143df4677431124fc8e15f6ac42850ce0fca6/full/rag/datasets
mkdir -p "$rag_dataset_dir"
sudo mount -t tmpfs -o size=96G,mode=0755,uid=1000,gid=1000 tmpfs "$rag_dataset_dir"
uv run --locked snakemake rag_all_documents --configfile config/rag_issue550/config.yaml --cores 4 --resources mem_mb=110000 --local-storage-prefix /opt/issue550/assets --rerun-triggers code params input -n > /opt/issue550/final-assembly-dryrun.log 2>&1
date -u +%FT%TZ
