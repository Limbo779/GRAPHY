# Subtask 01: Environment, Rootless Podman Containerization & Hardware Optimization

## Objective
Establish the foundational repository structure, configure rootless Podman containerization with Intel Iris Xe (`/dev/dri`) hardware acceleration on Arch Linux, verify CPU/iGPU resource limits for the i5-1335U (12 threads, 15 GiB RAM), and ensure complete reproducibility without polluting Git with large files.

---

## Scope & Deliverables

### 1. Git & Repository Hygiene
- Repository initialized as `GRAPHY` on branch `main`.
- Strict `.gitignore` preventing virtualenvs, model weights (`*.gguf`, `*.onnx`, `models/`), raw PDFs, vector DB files, and runtime data from being committed.
- Never push to remote (`origin`); all commits remain local for the user to inspect and push.
- Create directory skeleton (`src/graphy/...`, `configs/`, `scripts/`, `data/schemas/`, `models/`, `tests/`).

### 2. Arch Linux Host & Rootless Podman Audit
- Verify `podman`, `podman-compose` (or `podman play kube` / native compose), `slirp4netns` / `pasta`, and `/dev/dri` device permissions (`render` / `video` groups, subuid/subgid mappings) on the Arch Linux host.
- If any host package or configuration fix is needed for rootless Podman GPU passthrough (e.g., `intel-compute-runtime`, `vulkan-intel`, `level-zero-loader`), document and verify it cleanly.

### 3. Container Architecture (`Containerfile` & `compose.yaml`)
- Build a single, modular rootless Podman image (`graphy:latest`) that includes:
  - Python 3.11/3.12 runtime with `uv` / `pip` locked dependencies (`requirements.txt` & `pyproject.toml`).
  - Hardware acceleration libraries for Intel Iris Xe + 13th Gen Intel CPU:
    - Vulkan & Intel OpenCL/Level-Zero runtime drivers inside the container to communicate with `/dev/dri/renderD128`.
    - OpenMP / OpenBLAS / ONNX Runtime / OpenVINO CPU & iGPU runtime libraries tuned for AVX2 + VNNI.
  - PDF & layout processing system dependencies (`poppler`, `tesseract`, `libgl`, `qpdf`).
- Configure `compose.yaml` / Podman run script with:
  - Device passthrough: `--device /dev/dri:/dev/dri:rw` and `--security-opt label=disable` (or appropriate rootless group mappings `--group-add keep-groups`).
  - Persistent bind mounts:
    - `./models:/app/models:Z`
    - `./data:/app/data:Z`
    - `./configs:/app/configs:Z`
  - Port forwarding: `127.0.0.1:8080:8080` for the local Web UI.
  - Memory safeguard: soft limit / reservation awareness so the container never starves the host OS (max ~10 GiB hard ceiling, ~6.5 GiB target working set).

### 4. Hardware Profiler & Configuration (`src/graphy/core/config.py` & `hardware.py`)
- Implement a startup hardware probe that detects:
  - Available RAM (`/proc/meminfo`) before loading any model.
  - CPU topology (distinguishing the 2 P-cores / 4 threads from the 8 E-cores / 8 threads on the i5-1335U).
  - `/dev/dri` Vulkan / OpenVINO GPU availability.
- Centralize all tuneable parameters in `configs/config.yaml`.

---

## Acceptance & Verification Criteria
1. `podman build -t graphy:latest -f Containerfile .` succeeds cleanly and reproducibly.
2. Running the hardware check inside the rootless Podman container (`podman run --rm --device /dev/dri --group-add keep-groups graphy:latest python -m graphy.core.hardware`) detects:
   - 12 CPU threads (i5-1335U AVX2/VNNI).
   - Available system RAM.
   - `/dev/dri/renderD128` Intel Iris Xe device access.
3. `git status` shows only clean source, config, and build files (zero large binaries).
