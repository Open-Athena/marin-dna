"""Stage existing issue-550 inputs and record SHA-256; run on the CPU worker.

This research-branch preflight does no projection and never opens VEP records.
The normal vertebrate pipeline owns downstream biological outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import boto3
import pyarrow.parquet as pq

PREFIX = 'snakemake/vertebrate_projection_dataset/results/'
UNIFORM = PREFIX + 'phylop-uniform-v1/2162b6aa8299a9748eeb8031318b49072bb8c3fc/94d512050de327f96fda1105ce9c6ae5562944e402802516c7cde54795d8cdd1/full/'
FUNCTIONAL = PREFIX + 'functional-v1/e42a4ea1eca760219e0add91004b45cac59b19c9/a104a2756f538a1993405165fb5b50b6d4aeaf0a32810d8bb72907916eef1beb/full/'
ALIAS_KEY = 'issues/523/chain-reader-sampled-validation/f3c1de027c3660a17d9cb8bec9297154fd6d2328/Mus_musculus/source.aliases.sizes'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--anchors-only', action='store_true')
    args = parser.parse_args()
    inventory = json.loads(Path(args.inventory).read_text())
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    client = boto3.client('s3', region_name='us-east-2')
    extras = [UNIFORM + 'anchors/phylop_uniform_catalog.parquet', FUNCTIONAL + 'anchors/training.parquet']
    if not args.anchors_only:
        extras += [FUNCTIONAL + 'reference/hg38.2bit', FUNCTIONAL + 'reference/hg38.chrom.sizes', ALIAS_KEY]
    entries = []
    for key in extras:
        head = client.head_object(Bucket='oa-bolinas', Key=key)
        entries.append({'Key': key, 'Size': head['ContentLength'], 'ETag': head['ETag']})
    if not args.anchors_only:
        for group, values in inventory['inventories'].items():
            if group != 'cached_nonmammal_sequences':
                entries.extend(values['objects'])
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for entry in entries:
        key = entry['Key']
        target = directory / 's3' / 'oa-bolinas' / key
        if key in manifest and target.is_file():
            if manifest[key]['etag'] != entry['ETag'] or target.stat().st_size != entry['Size']:
                raise ValueError(f'staged input identity changed: {key}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.partial')
        started = time.time()
        print(json.dumps({'event': 'start', 'key': key, 'bytes': entry['Size']}), flush=True)
        response = client.get_object(Bucket='oa-bolinas', Key=key, IfMatch=entry['ETag'])
        digest = hashlib.sha256()
        count = 0
        with temporary.open('wb') as handle:
            for chunk in response['Body'].iter_chunks(chunk_size=8 * 1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
                count += len(chunk)
        if count != entry['Size']:
            raise ValueError(f'incomplete input: {key}')
        target_digest = digest.hexdigest()
        if key == ALIAS_KEY and target_digest != '56706a5eeaaa368bfcab17bba94437b5a09885fe41494735a42bbb9118bac214':
            raise ValueError('source alias dictionary SHA-256 mismatch')
        temporary.replace(target)
        manifest[key] = {'uri': f's3://oa-bolinas/{key}', 'local_path': str(target), 'bytes': count, 'etag': entry['ETag'], 'sha256': target_digest, 'seconds': time.time() - started}
        atomic = manifest_path.with_suffix('.json.tmp')
        atomic.write_text(json.dumps(manifest, indent=2) + '\n')
        atomic.replace(manifest_path)
        print(json.dumps({'event': 'done', **manifest[key]}), flush=True)
    summaries = {}
    for kind, key, group_column, chrom_column, allowed in [
        ('uniform', extras[0], 'region_label', 'source_chrom', {'cds', 'utr3', 'tss_region_and_utr5'}),
        ('functional', extras[1], 'source_arm', 'chrom', {'ncrna', 'enhancer'}),
    ]:
        counts = {}
        for batch in pq.ParquetFile(directory / 's3' / 'oa-bolinas' / key).iter_batches(columns=[group_column, chrom_column], batch_size=8192):
            for row in batch.to_pylist():
                group = row[group_column]
                if group not in allowed:
                    continue
                counts.setdefault(group, {'total': 0, 'chr18': 0})
                counts[group]['total'] += 1
                counts[group]['chr18'] += str(row[chrom_column]) in {'18', 'chr18'}
        for group, value in counts.items():
            value['validation'] = min(400, value['chr18'])
            value['train_after_rc'] = 2 * (value['total'] - value['chr18'])
            value['validation_allocated_tokens'] = value['validation'] * 10240
            value['expected_epochs'] = 4_000_000 / value['train_after_rc']
        summaries[kind] = counts
    (directory / 'anchor-counts.json').write_text(json.dumps(summaries, indent=2) + '\n')
    print(json.dumps({'event': 'anchor_counts', 'counts': summaries}), flush=True)


if __name__ == '__main__':
    main()
