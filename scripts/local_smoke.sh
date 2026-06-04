#!/usr/bin/env bash
# Smoke-test the local code path end-to-end against a tiny PTB-XL subset.
# Run this before launching Modal to catch obvious bugs without paying GPU $$.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f data/ptbxl_test/ptbxl_database.csv ]; then
  echo "downloading PTB-XL metadata + 10 records (~7 MB)…"
  mkdir -p data/ptbxl_test/records100/00000
  cd data/ptbxl_test
  BASE="https://physionet.org/files/ptb-xl/1.0.3"
  curl -sf -o ptbxl_database.csv "$BASE/ptbxl_database.csv"
  curl -sf -o scp_statements.csv "$BASE/scp_statements.csv"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    id=$(printf "%05d" $i)
    curl -sf -o "records100/00000/${id}_lr.dat" "$BASE/records100/00000/${id}_lr.dat" &
    curl -sf -o "records100/00000/${id}_lr.hea" "$BASE/records100/00000/${id}_lr.hea" &
  done
  wait
  cd ../..
fi

if [ ! -f data/ptbxl_cache_test/labels.csv ]; then
  PYTHONPATH=. python3 -c "
from pathlib import Path
from neurocardio.data import PTBXLConfig, build_cache
build_cache(PTBXLConfig(root=Path('data/ptbxl_test'),
                         label_set='diagnostic_superclass',
                         sampling_rate=100),
            Path('data/ptbxl_cache_test'), indices=range(1, 11))
"
fi

echo
echo "── running 1-epoch training smoke test ──"
PYTHONPATH=. python3 -m neurocardio.train \
  --cache-dir data/ptbxl_cache_test --output-dir runs/local_smoke \
  --epochs 1 --batch-size 4 --num-workers 0 --device mps

echo
echo "── running inference smoke test ──"
PYTHONPATH=. python3 -c "
from neurocardio.infer import CardioInferencer
import numpy as np
inf = CardioInferencer('runs/local_smoke/best.pt')
sig = np.load('data/ptbxl_cache_test/signals_100.npy')[0]
result = inf.infer(sig.astype(np.float32))
print('OK — probs:', result['probs'])
"
echo
echo "── all local smoke tests passed ──"
