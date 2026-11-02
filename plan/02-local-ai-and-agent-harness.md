# Subtask 02: Local AI Engine, Model Management & Micro-Agent Harness

## Objective
Build the 100% Local AI runtime layer and the **Agent Harnessing framework** tailored to the i5-1335U + 15 GiB RAM + Intel Iris Xe hardware envelope. This layer powers technical embeddings, prerequisite reasoning, problem cleaning, handwritten math OCR, bare-minimum concept synthesis, and solution analysis without ever running out of memory or hallucinating invalid outputs.

---

## Scope & Deliverables

### 1. Curated Local Model Stack & Automated Downloader (`scripts/setup_models.py`)
All models are stored in the persistent `./models/` host volume (ignored by Git, reproducible via a single command):

| Role | Selected Model Class | Quantization / Runtime | RAM Footprint | Why This Model? |
| :--- | :--- | :--- | :--- | :--- |
| **Reasoning & Tutoring LLM** | **Qwen2.5-7B-Instruct** (or **Qwen2.5-Math/Coder-7B / 3B-Instruct** configurable tier) | `Q4_K_M` GGUF via `llama.cpp` (Vulkan / AVX2) | ~2.2 GiB (3B) / ~4.6 GiB (7B) | Best-in-class open model for undergraduate/graduate Math, Physics, and CS reasoning & structured JSON generation. |
| **Technical Embedding Model** | **BAAI/bge-m3** or **nomic-embed-text-v1.5 / jina-embeddings-v2-base-code** | INT8 / FP16 via **ONNX Runtime / OpenVINO** | ~0.4–0.6 GiB | Long context (up to 8192 tokens), strong retrieval across LaTeX equations, code blocks, and scientific prose. |
| **Handwritten Math & Document OCR** | **Qwen2.5-VL-3B-Instruct** (GGUF vision) + **RapidOCR / Texify (ONNX)** | `Q4_K_M` / INT8 ONNX | ~2.4 GiB (loaded on-demand only) | Reads handwritten math derivations, physics diagrams, and multi-line equations into clean LaTeX/Markdown. |

### 2. Memory-Strict Model Lifecycle Manager (`src/graphy/llm/model_manager.py`)
- Because total system RAM is 15 GiB (~8.5 GiB free), the **Model Lifecycle Manager** enforces a strict mutual-exclusion / memory-budget policy:
  - **Phase A (Embedding / Indexing)**: Loads only the ONNX/OpenVINO Embedding model (~500 MB). Runs at maximum batch speed on Intel Iris Xe / AVX2 CPU, then releases memory.
  - **Phase B (LLM Reasoning / Graph Building / Tutoring)**: Runs the `llama-server` / `llama-cpp-python` engine with KV-cache quantization (`q8_0`) and bounded context (`4096`–`8192` tokens), offloaded to Intel Iris Xe Vulkan/SYCL layers + P-core threads.
  - **Phase C (Handwritten OCR Preview)**: Temporarily swaps/unloads the text LLM if free RAM < 3.5 GiB, runs Vision/Math-OCR on the user's handwritten image, caches the LaTeX transcription, and reloads the reasoning LLM for grading.

### 3. The Agent Harnessing Framework (`src/graphy/llm/harness.py`)
Small and medium Local LLMs fail when given massive, open-ended prompts. Our **Agent Harness** maximizes reliability through 5 mechanisms:
1. **Task Decomposition into Single-Responsibility Micro-Agents**:
   - `ProblemBoundaryAgent` (Companion extraction)
   - `PrerequisiteJudgeAgent` (Chunk $A \to$ Chunk $B$ causality verification)
   - `ConceptDistillerAgent` (Extracts bare-minimum concepts vs. self-discovery concepts)
   - `SolutionAuditorAgent` (Checks each step of user's solution against target self-discovery concepts)
   - `RemedialCoachAgent` (Generates scaffolding topics when a learner struggles)
2. **Grammar-Constrained / Pydantic Schema Enforcement**:
   - Uses GBNF grammars / JSON-schema constrained decoding so the Local LLM is *mathematically guaranteed* to output valid, parseable JSON matching our Pydantic models.
3. **Map-Reduce Context Pruning**:
   - Each micro-agent receives only the exact pruned chunk text or problem statement it needs (< 1,500 tokens per call), keeping prompt evaluation fast on the i5-1335U.
4. **Deterministic Self-Verification & Retry Loop**:
   - Every agent output passes through a deterministic validator (e.g., verifying cited concepts exist in the source chunk, checking DAG consistency, ensuring self-discovery hints do not leak the final formula). If validation fails, the harness re-prompts with the exact validation error.
5. **Disk-Backed Content-Addressable Cache (`data/processed/llm_cache.sqlite`)**:
   - Identical chunk-pair checks or problem extractions are cached by SHA-256 hash of `(model_id, agent_version, prompt_payload)` so interrupted pipelines resume instantaneously.

---

## Acceptance & Verification Criteria
1. `scripts/setup_models.py` downloads, verifies SHA-256 checksums, and configures the quantized embedding, LLM, and OCR models inside `./models/`.
2. The `ModelLifecycleManager` runs inside Podman while keeping container memory usage strictly below 6.0 GiB.
3. Unit and integration tests verify that the `AgentHarness` returns 100% schema-valid Pydantic objects with grammar-constrained decoding and automatic SQLite caching.
