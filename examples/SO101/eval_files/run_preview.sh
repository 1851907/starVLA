#!/usr/bin/env bash
set -euo pipefail

# This script runs only policy inference. It does not connect to the SO101 arm.
SERVER_HOST="${SERVER_HOST:?Set SERVER_HOST to the GPU server IP or hostname}"
SERVER_PORT="${SERVER_PORT:-10093}"
FRONT_CAMERA_INDEX="${FRONT_CAMERA_INDEX:-0}"
SIDE_CAMERA_INDEX="${SIDE_CAMERA_INDEX:-1}"
UNNORM_KEY="${UNNORM_KEY:-new_embodiment}"
TASK="${TASK:-Grab the yellow ball into the yellow plate}"

python examples/SO101/eval_files/inference_preview.py \
  --server-host "${SERVER_HOST}" \
  --server-port "${SERVER_PORT}" \
  --front-camera-index "${FRONT_CAMERA_INDEX}" \
  --side-camera-index "${SIDE_CAMERA_INDEX}" \
  --unnorm-key "${UNNORM_KEY}" \
  --task "${TASK}"
