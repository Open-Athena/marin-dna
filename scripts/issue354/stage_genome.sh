#!/usr/bin/env bash
# Stage the canonical GRCh38 reference (bgzip .fa.gz + .fai + .gzi) from S3 to a
# local dir so scripts/issue354/run.yaml can file_mount it onto the GPU box —
# keeping that box cred-light (no AWS keys needed there). Run on the dev box,
# which has S3 creds, BEFORE `sky launch`.
#
# Usage: scripts/issue354/stage_genome.sh [dest_dir]   (default ~/issue354_genome)
set -euxo pipefail

DEST="${1:-$HOME/issue354_genome}"
SRC="s3://oa-bolinas/data/genomes/homo_sapiens/GRCh38/ensembl-release-115"
BASE="Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"

mkdir -p "$DEST"
for ext in "" ".fai" ".gzi"; do
    aws s3 cp "$SRC/$BASE$ext" "$DEST/$BASE$ext"
done

echo "staged canonical GRCh38 (.fa.gz + .fai + .gzi) to $DEST"
