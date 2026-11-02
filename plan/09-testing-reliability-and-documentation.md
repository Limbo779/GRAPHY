# Subtask 09: End-to-End Verification, Failure-Mode Hardening, Podman Export & Documentation

## Objective
Remembering that **real learners will rely on GRAPHY to study Math, Physics, and Computer Science**, this final subtask performs rigorous end-to-end validation, eliminates edge-case bugs, exports all Podman container/image artifacts as requested in `ideastorm.md`, and documents the entire system from installation to daily usage.

---

## Scope & Deliverables

### 1. Comprehensive End-to-End Test Suite (`tests/e2e/`)
Validate the complete workflow across three representative domains:
1. **Mathematics**: Linear Algebra / Calculus chapter + external problem set (definitions, theorems, proofs, multi-part exercises).
2. **Physics**: Classical Mechanics / Electromagnetism section + assignment with figures and multi-step derivations.
3. **Computer Science**: Data Structures & Algorithms chapter + programming/recurrence problem set with pseudocode and code blocks.

Each E2E test verifies:
- Lossless chunking of equations, proofs, figures, and code blocks.
- **GRAPHY Companion** problem separation (multi-part `a, b, c` cohesion and hint/solution quarantine).
- GraphAr Parquet/YAML persistence and `rustworkx` graph integrity.
- Exact JSON creation and mathematical validity of `chunk_scores.json`, `problem_chunk_scores.json`, and `problem_foundation_scores.json`.
- Full 4-problem study cycle (bare-minimum concepts $\to$ handwritten image OCR preview + text solution $\to$ solution analysis $\to$ dynamic mastery update).

### 2. Reliability & Failure-Mode Hardening
- **Interrupted Pipeline Recovery**: Verify that killing the container mid-graph-build or mid-embedding resumes from the last completed pair/chunk via SQLite & LanceDB checkpoints without duplicating work.
- **Out-of-Memory Guardrails**: Verify that if host RAM spikes (e.g., browser tabs open on Arch Linux), the Model Lifecycle Manager gracefully reduces batch size or swaps models rather than crashing with `OOMKilled`.
- **Malformed Input Handling**: Verify clear error reporting and automatic fallback when encountering corrupted PDF pages, unusual LaTeX macros, or blurry handwriting photos.

### 3. Podman Container & Image Details Export (`CONTAINER_DETAILS.md` & `scripts/export_container.sh`)
Per `ideastorm.md` (*"strictly use containerization through podman and get the container details like files and images to make it after the completion of this project"*):
- Record complete Podman image metadata (`podman inspect`, layer manifest, exact package versions, `Containerfile`, `compose.yaml`, rootless `/dev/dri` passthrough flags, and volume mount specifications).
- Provide `scripts/export_container.sh` to save/load the built image (`podman save` / `podman load`) or rebuild it deterministically from scratch on any Arch Linux machine.

### 4. Complete User & Developer Documentation (`README.md` & `docs/USER_GUIDE.md`)
Write clear, practical documentation covering:
- **Quickstart (Podman One-Liner & Compose)**: How to spin up GRAPHY in seconds.
- **CLI Reference & Web UI Walkthrough**: Step-by-step guide to adding a textbook, running GRAPHY Companion on extra problem sets, and studying in 4-problem batches.
- **Architecture & Mathematical Reference**: How chunking, prerequisite graphs, GraphAr storage, foundationness scoring, and dynamic mastery updates work under the hood.
- **Hardware Tuning Guide**: How to adjust model sizes, thread counts, and iGPU offloading in `configs/config.yaml`.

---

## Acceptance & Verification Criteria
1. `pytest` passes 100% of unit and end-to-end tests inside the rootless Podman container.
2. `CONTAINER_DETAILS.md` and `README.md` provide complete, verified instructions to build, inspect, run, and recreate the Podman container and Local AI environment.
3. Final `git status` and `git log` check confirms a clean, well-structured local `GRAPHY` Git repository ready for the user to push to GitHub manually.
