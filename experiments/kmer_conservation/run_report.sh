set -euo pipefail
cd /data/issue568/experiments/kmer_conservation
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=2
.venv/bin/ruff check src tests *.py --fix
.venv/bin/ruff format src tests *.py
uv run --locked pytest -q
uv run --locked python validate_artifacts.py --root /data/issue568/v2
uv run --locked python validate_artifacts.py --root /data/issue568/v2/synthetic
uv run --locked python validate_artifacts.py --root /data/issue568/v2/synthetic-paralogs10
uv run --locked python -m kmer_conservation.report --root /data/issue568/v2 --out /data/issue568/v2/report --figures
