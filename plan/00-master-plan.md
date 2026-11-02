# GRAPHY & GRAPHY Companion — Master Project Plan

## 1. Vision & Core Philosophy
**GRAPHY** is a problem-solving-first, self-discovery learning system built for rigorous technical textbooks (**Mathematics, Physics, and Computer Science**). Instead of passively reading chapters from start to finish, the learner is guided through carefully ranked problems starting from the most foundational concepts, given **only the bare-minimum prerequisites**, and challenged to **self-discover** the core insights through problem solving.

**GRAPHY Companion** is the automated problem-set extraction, cleaning, and separation pipeline that turns both textbook exercise sections and external course materials (assignments, exams, worksheets) into structured, spoiler-free, multi-part-aware problem units ready for GRAPHY.

---

## 2. Confirmed Decisions & Clarifications (Zero Assumptions)
All architectural and mathematical ambiguities have been clarified and locked in:

| Area | Confirmed Decision |
| :--- | :--- |
| **GRAPHY Companion Role** | Automated pipeline that cleans noisy problem sets (from textbooks, assignments, exams) and separates individual problems into structured records. |
| **Multi-Part Problem Handling** | Keep multi-part problems (e.g., `1(a)`, `1(b)`, `1(c)`) together as a single problem unit with shared stem context; strip or quarantine any attached hints/solutions so they never spoil self-discovery. |
| **Supported Input Formats** | **Textbooks & Problem Sets**: Digital PDFs + Markdown / LaTeX.<br>**User Solutions**: Typed Markdown / LaTeX / Code **AND** Handwritten solution images (photos/scans of math/physics work). |
| **Handwritten Solution Workflow** | Local Vision / Math-OCR transcribes the handwritten image into structured LaTeX/Markdown first, displays a preview for the user to confirm or edit, and then passes it to the Local LLM solution analyzer. |
| **Prerequisite Graph Filtering** | Avoid $O(N^2)$ all-pairs LLM bottlenecks by filtering to **Top-$K$ embedding-similar chunks** (above a cosine similarity threshold) before asking the Local LLM whether Chunk $A$ is a prerequisite of Chunk $B$ ($A \to B$). |
| **Foundation Score Math** | 1. Raw chunk score: $S_{\text{raw}}(u) = \sum_{v \in \text{Out}(u)} w(u \to v) - \sum_{v \in \text{In}(u)} w(v \to u)$<br>2. Normalize chunk scores to $S_{\text{norm}}(u) \in [0, 1]$.<br>3. Match each problem to its **Top-$K$ most similar chunks** (above threshold) and multiply chunk similarity by $S_{\text{norm}}(u)$ to compute the problem's net **foundationness score**. |
| **Progression Mechanism** | **Dynamic Mastery Graph Update**: As the user solves problems and self-discovers concepts, mastered chunks are updated in the graph state, dynamically unlocking and re-ranking the next frontier of 4 problems. |
| **Graph, Vector & Score Storage** | - **Graph**: Apache **GraphAr** standard format (Parquet + YAML descriptors) + **`rustworkx`** for high-speed graph computation.<br>- **Vectors**: Embedded **LanceDB** (zero-server, disk-backed Arrow/Lance store).<br>- **Scores**: Strictly stored in **JSON** (`chunk_scores.json`, `problem_chunk_scores.json`, `problem_foundation_scores.json`, and user mastery state). |
| **Local AI & Model Storage** | 100% Local AI using **Agent Harnessing** (micro-agents with strict schema enforcement and focused context windows). Quantized models (`GGUF` / `OpenVINO` / `ONNX`) stored in a managed persistent host volume (`./models/`) mounted into Podman. |
| **User Interface & Containerization** | **CLI Core + Lightweight Local Web UI** (with KaTeX math rendering, problem workspace, and OCR preview editor), running inside **rootless Podman** on Arch Linux with `/dev/dri` Intel Iris Xe GPU passthrough. |
| **Git & Repository Policy** | Local Git repo named `GRAPHY`. Track all source code, configs, `Containerfile`, and `requirements.txt`. Ignore all large weights/PDFs/caches. **Never push to GitHub** (user pushes manually). |

---

## 3. Target Hardware Profile & Optimization Strategy
- **OS**: Arch Linux
- **CPU**: 13th Gen Intel Core i5-1335U (12 logical CPUs: 2 Performance cores / 4 threads + 8 Efficient cores / 8 threads; AVX2 + VNNI)
- **RAM**: 15 GiB shared system memory (~8.8 GiB available under normal desktop load)
- **GPU**: Intel Iris Xe Graphics (Raptor Lake-P integrated GPU, `/dev/dri/renderD128`, shared memory)

### How We Squeeze Maximum Performance Out of This Machine:
1. **Strict Memory Envelope (< 6.5 GiB peak container RAM)**:
   - Because Intel Iris Xe shares system RAM with the CPU, loading an embedding model, a Vision/Math-OCR model, and an LLM simultaneously would trigger swap thrashing.
   - We design an **On-Demand Model Lifecycle Manager** that loads only the active stage's model into memory (or keeps a single lightweight LLM server running while embeddings/OCR run via ultra-compact OpenVINO/ONNX int8 models) and explicitly unloads inactive models.
2. **Intel Iris Xe iGPU + Hybrid CPU Acceleration**:
   - **LLM Inference**: `llama.cpp` compiled with **Vulkan / SYCL** support for Intel Iris Xe (`/dev/dri`) + AVX2 CPU fallback, pinning compute threads to P-cores/E-cores appropriately (`--threads 8 --threads-batch 10`).
   - **Embeddings & Math-OCR**: **OpenVINO / ONNX Runtime** targeting Intel CPU (AVX2/VNNI int8 quantization) and Intel GPU (`GPU.0`), delivering 3x–5x faster embedding throughput than standard PyTorch CPU.
3. **Agent Harnessing over Monolithic Prompts**:
   - Instead of feeding huge textbook chapters into a large model, we break every cognitive task into deterministic micro-agents (Prerequisite Verifier, Bare-Minimum Extractor, Self-Discovery Grader, Remedial Coach) with small, targeted RAG contexts and grammar-constrained JSON outputs.

---

## 4. Target Repository Layout
```text
GRAPHY/
├── Containerfile                  # Reproducible Podman image definition
├── compose.yaml                   # Rootless Podman compose with /dev/dri & volumes
├── pyproject.toml                 # Project metadata & dependencies
├── requirements.txt               # Locked Python dependencies for reproducibility
├── Makefile                       # Helper commands for Podman, CLI, UI, and tests
├── plan/                          # Modular Subtask Plans (00 to 09)
├── configs/
│   └── config.yaml                # Hardware, model, chunking, and scoring parameters
├── src/
│   └── graphy/
│       ├── core/                  # Config, hardware profiler, logging, schemas
│       ├── llm/                   # Model manager, llama.cpp/OpenVINO bridge, Agent Harness
│       ├── parser/                # Textbook PDF/MD/LaTeX parser & semantic chunker
│       ├── companion/             # GRAPHY Companion: automated problem set cleaner & splitter
│       ├── embeddings/            # Technical/Math/Code embedding engine & LanceDB store
│       ├── graph/                 # Candidate filter, LLM prerequisite builder, GraphAr + rustworkx
│       ├── scoring/               # Chunk foundation scores, problem scores & dynamic mastery JSONs
│       ├── tutor/                 # Bare-minimum concepts, Handwritten OCR preview, Solution analyzer
│       ├── cli/                   # Rich terminal CLI interface
│       └── web/                   # FastAPI + lightweight KaTeX local Web UI
├── scripts/
│   ├── setup_models.py            # Automated downloader/verifier for quantized local models
│   └── export_container.sh        # Exports Podman image/container details for archival
├── data/                          # Mounted persistent volume (git-ignored contents except schemas)
│   ├── schemas/                   # JSON schemas for scores & GraphAr YAML templates
│   ├── raw/                       # Input textbooks & problem sets
│   ├── processed/                 # Cleaned chunks & Companion problem sets
│   ├── vectors/                   # LanceDB embedded tables
│   ├── graphs/                    # GraphAr vertex/edge Parquet + YAML files
│   ├── scores/                    # chunk_scores.json, problem_chunk_scores.json, problem_foundation_scores.json
│   └── sessions/                  # Learner progress, OCR previews, submitted solutions
├── models/                        # Mounted persistent volume for GGUF / OpenVINO models (git-ignored)
└── tests/                         # Unit & integration tests per subtask
```

---

## 5. Subtask Execution Roadmap
We will build and verify GRAPHY **one subtask at a time**, in the following order:

1. **[Subtask 01 — Environment, Podman & Hardware Foundation](./01-environment-podman-and-git.md)**
   Git workflow, project skeleton, Arch Linux / Intel Iris Xe passthrough verification, `Containerfile`, and rootless Podman orchestration.
2. **[Subtask 02 — Local AI Engine, Model Manager & Agent Harness](./02-local-ai-and-agent-harness.md)**
   Memory-aware model lifecycle manager, quantized model setup (LLM, Embedding, Math-OCR), and the grammar-constrained Micro-Agent Harness.
3. **[Subtask 03 — Textbook Parsing, Math/Code Chunking & Technical Embeddings](./03-textbook-chunking-and-embedding.md)**
   General-purpose Math/Physics/CS PDF & LaTeX/MD ingestion, theorem/equation/code/figure-preserving semantic chunker, and LanceDB embedding pipeline.
4. **[Subtask 04 — GRAPHY Companion: Problem Set Extraction & Cleaning Pipeline](./04-graphy-companion-problem-extractor.md)**
   Automated extraction of textbook & external problem sets (assignments/exams), noise cleaning, multi-part (`1a, 1b, 1c`) grouping, and solution/spoiler separation.
5. **[Subtask 05 — Prerequisite Discovery Graph & GraphAr Storage](./05-prerequisite-graph-and-graphar.md)**
   Top-$K$ similarity candidate filtering, batched Local LLM prerequisite verification ($A \to B$), `rustworkx` graph assembly, and Apache GraphAr storage.
6. **[Subtask 06 — Foundation Scoring, Problem Ranking & Dynamic Mastery JSONs](./06-foundation-scoring-and-ranking.md)**
   Raw & normalized chunk foundation scores, problem-to-chunk similarity matrix, net problem foundationness calculation, JSON persistence, and dynamic mastery updates.
7. **[Subtask 07 — Learning Loop, Handwritten Solution OCR & Solution Analyzer](./07-learning-loop-and-handwritten-ocr.md)**
   Top-4 problem selector, bare-minimum concept generator, Handwritten image $\to$ LaTeX OCR preview pipeline, and multi-agent solution grading (self-discovery verification vs. remedial coaching).
8. **[Subtask 08 — Usable Interface: CLI & Local KaTeX Web UI](./08-interface-cli-and-web-ui.md)**
   Full terminal CLI + lightweight Local Web UI with instant LaTeX math rendering, interactive OCR transcription editor, and mastery dashboard.
9. **[Subtask 09 — End-to-End Verification, Podman Export & Documentation](./09-testing-reliability-and-documentation.md)**
   Full stress-test with real Math/Physics/CS material, bug hardening, Podman container/image export documentation, and comprehensive user guide.
