#!/usr/bin/env bash
# Full Modal flow: deploy → smoke train → launch 14h detached training.
# Assumes data_prep has already completed (run ./scripts/launch_modal.sh data first).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3 deploying web app =="
modal deploy modal_app/app.py

echo
echo "== 2/3 running smoke training (5 min) =="
modal run modal_app/app.py::train \
  --run-name smoke --epochs 2 --batch-size 64 \
  --num-workers 2 --max-steps-per-epoch 30 --wallclock-hours 0

echo
echo "== 3/3 launching 14-hour detached training =="
modal run --detach modal_app/app.py::train \
  --run-name main --epochs 200 --batch-size 96 \
  --num-workers 4 --wallclock-hours 14

echo
echo "Done. Web app live at:"
echo "  https://$(modal profile current 2>/dev/null | head -1)--neurocardio-web.modal.run"
echo
echo "Tail training logs with:"
echo "  modal app list   # find the latest ap-... for neurocardio"
echo "  modal app logs <id>"
