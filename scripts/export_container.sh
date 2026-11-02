#!/usr/bin/env bash
# =============================================================================
# GRAPHY & GRAPHY Companion — Podman Image & Container Details Exporter
# Exports container metadata, package versions, and optional OCI image tarball
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${GRAPHY_IMAGE:-localhost/graphy:latest}"
EXPORT_DIR="${PROJECT_DIR}/container-export"

mkdir -p "${EXPORT_DIR}"

echo "[1/4] Exporting Podman image inspect metadata to ${EXPORT_DIR}/image-inspect.json..."
podman inspect "${IMAGE_NAME}" > "${EXPORT_DIR}/image-inspect.json"

echo "[2/4] Exporting installed Arch Linux package manifest to ${EXPORT_DIR}/packages.txt..."
podman run --rm --entrypoint pacman "${IMAGE_NAME}" -Q > "${EXPORT_DIR}/packages.txt"

echo "[3/4] Exporting /opt/llama runtime inventory to ${EXPORT_DIR}/llama-runtime.txt..."
podman run --rm --entrypoint ls "${IMAGE_NAME}" -lh /opt/llama > "${EXPORT_DIR}/llama-runtime.txt"

if [[ "${1:-}" == "--save-tar" ]]; then
  TAR_PATH="${EXPORT_DIR}/graphy-latest-oci.tar"
  echo "[4/4] Saving full Podman OCI image archive to ${TAR_PATH}..."
  podman save --format oci-archive -o "${TAR_PATH}" "${IMAGE_NAME}"
  echo "Saved OCI archive: ${TAR_PATH}"
else
  echo "[4/4] Skipping OCI tarball export (pass --save-tar to export full image archive)."
fi

echo "Done! Container export artifacts are in: ${EXPORT_DIR}"
