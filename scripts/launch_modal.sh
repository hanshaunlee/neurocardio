#!/usr/bin/env bash
# Convenience launcher for the NeuroCardio Modal app.
#
#   ./scripts/launch_modal.sh data         download PTB-XL into the Volume
#   ./scripts/launch_modal.sh deploy       deploy the web app + endpoints
#   ./scripts/launch_modal.sh smoke        2-epoch quick training to verify GPU
#   ./scripts/launch_modal.sh train        14-hour detached training
#   ./scripts/launch_modal.sh logs <id>    tail logs of a detached run
#   ./scripts/launch_modal.sh url          print the deployed web URL
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-help}" in
  data)
    modal run --detach modal_app/app.py::data_prep ;;
  deploy)
    modal deploy modal_app/app.py ;;
  smoke)
    modal run modal_app/app.py::train \
      --run-name smoke --epochs 2 --batch-size 64 \
      --num-workers 2 --max-steps-per-epoch 30 --wallclock-hours 0 ;;
  train)
    modal run --detach modal_app/app.py::train \
      --run-name main --epochs 200 --batch-size 96 \
      --num-workers 4 --wallclock-hours 14 ;;
  eval)
    modal run modal_app/app.py::evaluate_test --run-name "${2:-main}" ;;
  list)
    modal run modal_app/app.py::list_runs ;;
  logs)
    [ -n "${2:-}" ] || { echo "usage: $0 logs <app-id>"; exit 1; }
    modal app logs "$2" ;;
  url)
    user=$(modal profile current 2>/dev/null | head -1)
    echo "Web URL after \`./scripts/launch_modal.sh deploy\`:"
    echo "  https://${user}--neurocardio-web.modal.run" ;;
  *)
    echo "usage: $0 {data|deploy|smoke|train|eval [run]|list|logs <id>|url}" ;;
esac
