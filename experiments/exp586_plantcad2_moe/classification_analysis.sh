#!/usr/bin/env bash
set -euo pipefail
exec python - "$@" <<'PY'
import hashlib,io,os,runpy,sys,tempfile,zipfile
from pathlib import Path
import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret
configure_coreweave_s3()
os.environ['WANDB_API_KEY']=require_secret('WANDB_API_KEY')
wheel_uri,wheel_sha=sys.argv[1:3]
with fsspec.open(wheel_uri,'rb') as source: payload=source.read()
if hashlib.sha256(payload).hexdigest()!=wheel_sha: raise ValueError('Analysis wheel identity differs')
directory=Path(tempfile.mkdtemp(prefix='exp586-probe-analysis-code-'))
with zipfile.ZipFile(io.BytesIO(payload)) as archive: archive.extractall(directory)
sys.path.insert(0,str(directory))
sys.argv=['exp586-classification-analysis',*sys.argv[3:]]
runpy.run_module('exp586_evals.classification_runner',run_name='__main__')
PY
