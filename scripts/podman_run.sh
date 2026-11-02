#!/usr/bin/env bash
# =============================================================================
# Helper script to run GRAPHY inside rootless Podman with Intel Iris Xe (/dev/dri)
# and persistent volume mounts.
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${GRAPHY_IMAGE:-localhost/graphy:latest}"

mkdir -p "${PROJECT_DIR}/models" \
         "${PROJECT_DIR}/data/schemas" \
         "${PROJECT_DIR}/data/raw" \
         "${PROJECT_DIR}/data/processed" \
         "${PROJECT_DIR}/data/vectors" \
         "${PROJECT_DIR}/data/graphs" \
         "${PROJECT_DIR}/data/scores" \
         "${PROJECT_DIR}/data/uploads" \
         "${PROJECT_DIR}/data/sessions"

TTY_FLAGS="-i"
if [ -t 0 ] && [ -t 1 ]; then
  TTY_FLAGS="-it"
fi

exec podman run --rm ${TTY_FLAGS} \
  --device /dev/dri:/dev/dri:rw \
  --group-add keep-groups \
  --security-opt label=disable \
  --memory 8g \
  -p 127.0.0.1:8080:8080 \
  -v "${PROJECT_DIR}/models:/app/models:rw" \
  -v "${PROJECT_DIR}/data:/app/data:rw" \
  -v "${PROJECT_DIR}/configs:/app/configs:ro" \
  -v "${PROJECT_DIR}/src:/app/src:ro" \
  -v "${PROJECT_DIR}/tests:/app/tests:ro" \
  "${IMAGE_NAME}" "$@"
