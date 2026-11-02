# GRAPHY & GRAPHY Companion — Podman Container & Image Details

This document records the complete Podman container and image specifications, layer manifest, installed runtime packages, hardware passthrough configuration, and instructions to rebuild or export the container image on Arch Linux.

---

## 1. Built Image Identity

| Property | Value |
| :--- | :--- |
| **Image Tag** | `localhost/graphy:latest` |
| **Base Image** | `docker.io/library/archlinux:latest` (`bf2533a05fd2`) |
| **Image ID** | `0e8caf860e031bcf48f057ae43a0f9ebe938c90e4057dfd837001842f78e6ea1` |
| **OCI Digest** | `sha256:a35a986e576a30db0d4c72aad180f875ce91e16dcb897e731d535d2e73a6549f` |
| **Uncompressed Size** | `1.48 GB` (`1,480,389,985 bytes`) |
| **Total Layers** | `13` |
| **Architecture / OS** | `amd64` / `linux` |
| **Podman Version** | `6.1.4` (`crun` OCI runtime, cgroup v2, rootless `pasta` networking) |

---

## 2. Key Installed Packages & Hardware Drivers Inside Image

### System & GPU Compute Drivers (Intel Iris Xe `8086:A7A1` via `/dev/dri/renderD128`)
- `vulkan-icd-loader` (`1.4.363.0-1`)
- `vulkan-intel` (`1:26.2.6-1`) — Mesa ANV Vulkan driver for Intel Iris Xe (`Vulkan0`)
- `vulkan-tools` (`1.4.363.0-1`) — `vulkaninfo`
- `intel-compute-runtime` (`26.35.39758.10-1`) — Intel NEO OpenCL & Level-Zero runtime
- `intel-graphics-compiler` (`1:2.41.5-1`)
- `intel-media-driver` (`26.3.5-1`) & `intel-gmmlib` (`22.10.0-1`)
- `clinfo` (`3.0.25.02.14-1`)
- `blas`, `cblas`, `lapack` (`3.12.1-2`), `sqlite` (`3.54.0-1`)

### Local AI Runtime (`/opt/llama/`)
- `llama-server` (`0.8.0-dev`, commit `39febd2ad`)
- `llama-quantize`
- `libllama.so.0.8.0`, `libllama-common.so.0.8.0`, `libllama-server-impl.so`
- `libmtmd.so.0.8.0` (Multimodal Vision projector runtime for `minicpm-v-mmproj.gguf`)
- `libggml-vulkan.so` (Vulkan GPU compute backend for Intel Iris Xe `Vulkan0`)
- `libggml-cpu-alderlake.so` (Raptor Lake / Alder Lake AVX2 + AVX-VNNI hybrid CPU backend)

### Document Parsing, OCR & Python Stack
- `poppler` (`26.10.0-2`) & `poppler-data` (`0.4.12-2`) — `pdftotext`, `pdfimages`, `pdftoppm`, `pdfinfo`
- `tesseract` (`5.5.4-1`), `tesseract-data-eng`, `tesseract-data-osd`
- `python` (`3.14.9-1`)
- `python-pydantic` (`2.14.2-1`) & `python-pydantic-core` (`3:2.50.2-1`)
- `python-yaml` (`6.0.3-2`), `python-orjson` (`3.13.0-1`), `python-jsonschema` (`4.26.0-1`)
- `python-flask` (`3.1.3-2`), `python-flask-cors` (`6.0.5-1`), `python-waitress` (`3.0.2-2`), `python-jinja` (`1:3.1.6-3`)
- `python-click` (`8.5.0-1`), `python-colorama` (`0.4.6-6`), `python-termcolor` (`3.3.0-1`)
- `python-psutil` (`7.2.2-1`), `python-cairo` (`1.29.2-1`), `python-markdown`, `python-beautifulsoup4`

---

## 3. Persistent Host Volumes Mounted into Container

| Host Path | Container Path | Mode | Purpose |
| :--- | :--- | :--- | :--- |
| `./models` | `/app/models` | `rw` | Quantized GGUF weights (`qwen2-7b-instruct.gguf`, `minicpm-v-mmproj.gguf`, `deepseek-r1-qwen3-8b.gguf`) & `manifest.json` |
| `./data` | `/app/data` | `rw` | Textbooks, Companion problems, Lance/SQLite vectors, GraphAr graphs, JSON scores (`data/scores/<book_id>/*.json`), and study sessions |
| `./configs` | `/app/configs` | `ro` | Central system configuration (`configs/config.yaml`) |

---

## 4. How to Build, Run, Export & Recreate the Container

### A. Build Image from `Containerfile`
```bash
make build
# Or directly with podman:
podman build -t localhost/graphy:latest -f Containerfile .
```

### B. Run Container with Rootless Intel Iris Xe (`/dev/dri`) Passthrough
```bash
# Run any CLI command inside the container via helper script:
./scripts/podman_run.sh hardware
./scripts/podman_run.sh serve --port 8080

# Or via podman run directly:
podman run --rm -it \
  --device /dev/dri:/dev/dri:rw \
  --group-add keep-groups \
  --security-opt label=disable \
  --memory 8g \
  -p 127.0.0.1:8080:8080 \
  -v "$(pwd)/models:/app/models:rw" \
  -v "$(pwd)/data:/app/data:rw" \
  -v "$(pwd)/configs:/app/configs:ro" \
  localhost/graphy:latest serve --port 8080
```

### C. Export Container Details & Full OCI Image Archive
```bash
# Export inspect JSON, package manifest, and /opt/llama inventory:
./scripts/export_container.sh

# Export full portable OCI image tarball (container-export/graphy-latest-oci.tar):
./scripts/export_container.sh --save-tar

# Load on another machine:
podman load -i container-export/graphy-latest-oci.tar
```
