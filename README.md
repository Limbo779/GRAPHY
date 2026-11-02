# GRAPHY & GRAPHY Companion

**GRAPHY** is a problem-solving-first, self-discovery learning system built for **Mathematics, Physics, and Computer Science textbooks**. Instead of passively reading chapters, you learn by tackling problems ranked from the most foundational topics upward—receiving **only the bare-minimum prerequisites** before solving so you self-discover the deeper principles yourself.

**GRAPHY Companion** is the automated problem-set extraction, cleaning, and multi-part grouping pipeline that turns textbook exercise sections and external course assignments/exams into clean, spoiler-free problem sets ready for GRAPHY.

---

## System Architecture & Pipeline Flow

```text
[Textbook (.pdf / .tex / .md)] ──► [TechnicalDocumentParser]
                                      ├──► Exposition Sections ──► [SemanticTextbookChunker (Atomic Math/Code/Proof/Figure)]
                                      │                                  │
                                      │                                  ▼
                                      │                      [TechnicalEmbeddingEngine + EmbeddedVectorStore]
                                      │                                  │
                                      │                                  ▼
                                      │                      [Top-K Candidate Pair Filter]
                                      │                                  │
                                      │                                  ▼
                                      │                      [Local LLM PrerequisiteJudgeAgent (A -> B)]
                                      │                                  │
                                      │                                  ▼
                                      │                      [KnowledgeDiGraph + Apache GraphAr Storage]
                                      │                                  │
                                      │                                  ▼
                                      │                      [Chunk Foundation Scorer (W_out - W_in)]
                                      │                         └─► data/scores/<book_id>/chunk_scores.json
                                      │
                                      └──► Exercise Blocks ──┐
                                                             ▼
[External Assignments / Exams] ──────────────────► [GRAPHY Companion Pipeline]
  (.pdf / .tex / .md)                                 • Strips exam/homework boilerplate & point badges
                                                      • Keeps multi-part problems (1a, 1b, 1c) unified
                                                      • Quarantines inline hints & solutions (anti-spoiler)
                                                             │
                                                             ▼
                                                   [Problem-to-Chunk Matcher & Foundationness Ranker]
                                                      ├─► data/scores/<book_id>/problem_chunk_scores.json
                                                      └─► data/scores/<book_id>/problem_foundation_scores.json
                                                             │
                                                             ▼
                                                   [4-Problem Self-Discovery Study Loop]
                                                      1. Select Top-4 Foundational Problems
                                                      2. Distill Bare-Minimum Concepts (Hold back Self-Discovery Topics)
                                                      3. Solve via Typed LaTeX/Code OR Handwritten Image -> OCR LaTeX Preview
                                                      4. Audit Solutions (Full Discovery / Missed Discovery / Remedial Coaching)
                                                      5. Dynamic Mastery Graph Update -> Unlock Next Frontier of 4 Problems
```

---

## Hardware Optimization (13th Gen Intel Core i5-1335U + 15 GiB RAM + Intel Iris Xe)

- **Rootless Podman with `/dev/dri` Passthrough**: Runs inside `localhost/graphy:latest` with both **Vulkan (`Vulkan0`)** and **OpenCL (`intel-compute-runtime`)** active on the integrated Intel Iris Xe GPU (`8086:A7A1`).
- **Hybrid GPU + CPU Inference (`/opt/llama/llama-server`)**:
  - **Embeddings**: Full Vulkan offloading (`--device Vulkan0 -ngl 99`) for **6x faster** batch prompt processing.
  - **Micro-Agent Reasoning**: Hybrid split (`--device Vulkan0 -ngl 18 -t 8 -tb 10`) combining fast iGPU prompt evaluation with Raptor Lake / Alder Lake `AVX2 + AVX-VNNI` physical core token generation.
- **Strict Memory Lifecycle Manager (`< 6.5 GiB` cap)**: Enforces mutual exclusion across `instruct`, `embedding`, `vision`, and `reasoning` roles with `q8_0` KV-cache quantization and content-addressable SQLite caching (`data/processed/llm_cache.sqlite`).

---

## Quickstart

### 1. Build the Podman Container & Verify Models
```bash
make build
make setup-models
make hardware
```

### 2. Run the Full Test Suite Inside Rootless Podman
```bash
make test
```

### 3. CLI Workflow
```bash
# 1. Ingest a textbook (.pdf, .tex, or .md) and auto-route its exercises to GRAPHY Companion
./scripts/podman_run.sh ingest-book data/raw/linear_algebra.pdf --book-id linalg

# 2. (Optional) Add external homework assignments or exams via GRAPHY Companion
./scripts/podman_run.sh ingest-problems data/raw/mit_18_06_hw1.pdf --book-id linalg --source-id hw1

# 3. Build the directed prerequisite graph (GraphAr) and compute all 3 JSON score files
./scripts/podman_run.sh build-graph --book-id linalg

# 4. Inspect foundation rankings & mastery status
./scripts/podman_run.sh status --book-id linalg

# 5. Start a 4-problem self-discovery study session
./scripts/podman_run.sh study-start --book-id linalg

# 6. (Optional) Preview & confirm a handwritten solution image via Math-OCR
./scripts/podman_run.sh study-ocr --book-id linalg --session-id <ses_id> --problem-id <prob_id> --image data/uploads/work.png --confirm

# 7. Submit solutions and receive self-discovery analysis & next batch briefing
./scripts/podman_run.sh study-submit --book-id linalg --session-id <ses_id> --solutions-json solutions.json
```

### 4. Local Web UI Workflow
```bash
./scripts/podman_run.sh serve --port 8080
```
Then open **`http://127.0.0.1:8080`** in your browser to explore chunk and problem foundation scores, view KaTeX-rendered bare-minimum briefings, upload handwritten solution images with interactive LaTeX OCR preview, and submit solutions.

---

## Documentation Index
- **[User & Study Guide (`docs/USER_GUIDE.md`)](./docs/USER_GUIDE.md)** — Detailed walkthrough of every feature and JSON/GraphAr data format.
- **[Podman Container Details (`CONTAINER_DETAILS.md`)](./CONTAINER_DETAILS.md)** — Complete container/image metadata, package inventory, and export instructions.
- **[Master Plan & Subtask Specs (`plan/00-master-plan.md`)](./plan/00-master-plan.md)** — Architectural design and subtask breakdown (`01` through `09`).
