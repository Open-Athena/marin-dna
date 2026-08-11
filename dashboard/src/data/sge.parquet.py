"""Observable Framework data loader: SGE metrics → one tidy parquet.

Pulls the per-(method, score_type, metric, subset, accession) SGE metric rows
from S3 via ``marin_dna_evals.leaderboard.sge_normalized_rows`` and
writes the DataFrame as a parquet blob on stdout for the dashboard to read.

SGE's grid is richer than the other leaderboards' (metric × consequence-subset ×
accession), so it gets its own loader + page (``src/leaderboards/sge.md``)
instead of riding the shared ``leaderboard.parquet`` / ``normalized_rows`` path.
"""

from __future__ import annotations

import sys

from marin_dna_evals.leaderboard import sge_normalized_rows


def main() -> None:
    sge_normalized_rows("sge").write_parquet(sys.stdout.buffer)


if __name__ == "__main__":
    main()
