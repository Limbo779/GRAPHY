# GRAPHY & GRAPHY Companion — Complete User Guide

## 1. Core Learning Methodology

GRAPHY flips traditional textbook study on its head:
1. **Foundation-First Ranking**: Every textbook chunk is a node in a directed, weighted prerequisite graph where an edge $A \to B$ means **Chunk $A$ is a prerequisite for Chunk $B$**, weighted by their technical embedding similarity $w(A \to B)$.
2. **Chunk Foundation Score**:
   $$S_{\text{raw}}(u) = \sum_{(u \to v)} w(u \to v) - \sum_{(p \to u)} w(p \to u)$$
   Normalized to $S_{\text{norm}}(u) \in [0.05, 1.0]$. Chunks with high outgoing weight and low incoming weight are the most fundamental topics.
3. **Problem Foundationness Score**:
   Every problem $p$ extracted by **GRAPHY Companion** is matched against its Top-$K$ most similar textbook chunks $\mathcal{C}(p)$, and its net foundationness score is:
   $$F(p) = \sum_{u \in \mathcal{C}(p) \setminus M} \text{sim}(p, u) \cdot S_{\text{norm}}(u)$$
4. **Bare-Minimum Briefing & Self-Discovery**:
   GRAPHY selects the top 4 problems with the highest foundationness score, gathers their textbook chunks, and uses the Local AI `ConceptDistillerAgent` to give you **only the bare-minimum definitions and notation** needed to understand the problems—leaving the deeper theorems, patterns, and synthesis insights for you to **self-discover** while solving.
5. **Adaptive Post-Solution Coaching**:
   - **Solved + Self-Discovered**: Marks the chunks mastered, updates the Dynamic Mastery Graph, and unlocks the bare-minimum briefing for the next 4 foundational problems.
   - **Solved, Missed Self-Discovery**: Explicitly teaches the undiscovered topics you missed so no conceptual insight is lost, then unlocks the next 4 problems.
   - **Failed / Struggled**: Teaches both the undiscovered topics **and** remedial prerequisite scaffolding (pulling from incoming prerequisite chunks in the GraphAr graph) so you will not struggle.

---

## 2. Step-by-Step Usage Guide

### Step 1: Verify Container & Local Models
Ensure the Podman image is built and the persistent `./models/` volume is populated:
```bash
make build
make setup-models
make hardware
```

### Step 2: Ingest a Textbook (`PDF`, `LaTeX`, or `Markdown`)
Place your textbook in `data/raw/` (e.g., `data/raw/griffiths_em.pdf`) and run:
```bash
./scripts/podman_run.sh ingest-book /app/data/raw/griffiths_em.pdf --book-id em_101
```
What happens automatically:
- Digital PDF pages are extracted via Poppler (`pdftotext` & `pdfimages`, with automatic `pdftoppm` + `tesseract` OCR fallback for scanned pages).
- Unicode mathematical symbols (`∀, ∃, ∇, ∂, ∫, λ₁...`) are normalized into standard LaTeX (`\forall, \in, \nabla, \partial, \int, \lambda_{1}...`).
- End-of-chapter exercise blocks are separated into `data/processed/em_101_textbook_exercises.json` and automatically processed by **GRAPHY Companion** into `data/processed/em_101_problems.json`.
- Exposition sections are split by the **Atomic-Block Semantic Chunker** (never splitting display equations, `Theorem + Proof` pairs, `Example + Solution` pairs, or code blocks) and embedded into `data/vectors/em_101/chunks_vectors.sqlite`.

### Step 3: Add External Problem Sets via GRAPHY Companion
To add university homework assignments, recitation sheets, or exams (`.pdf`, `.tex`, or `.md`):
```bash
./scripts/podman_run.sh ingest-problems /app/data/raw/ps1.pdf --book-id em_101 --source-id mit_802_ps1
```
What **GRAPHY Companion** does:
- Strips student name/ID lines, due dates, instructor names, grading rubrics, and point badges (`[15 pts]`, `(10 points)`).
- Keeps multi-part problems (`(a)`, `(b)`, `(c)`) unified inside a single problem unit with their shared stem context.
- Quarantines any inline hints (`[Hint: ...]`) or solutions (`\begin{solution}...\end{solution}`) into `quarantined_hint` and `quarantined_solution` so they never spoil your self-discovery.

### Step 4: Build Prerequisite Graph & Compute JSON Scores
```bash
./scripts/podman_run.sh build-graph --book-id em_101
```
This produces:
1. **Apache GraphAr Prerequisite Graph** in `data/graphs/em_101/` (`graph.gar.yml`, `chunk.vertex.yml`, `chunk_prereq_chunk.edge.yml`, CSR/CSC partitions).
2. **JSON Score Files** in `data/scores/em_101/`:
   - `chunk_scores.json`: Every chunk's `out_weight`, `in_weight`, `raw_foundation_score`, `normalized_foundation_score`, `rank`, and `mastery_status`.
   - `problem_chunk_scores.json`: Per-problem matched chunks, similarity scores, and weighted foundation contributions.
   - `problem_foundation_scores.json`: Every problem ranked by overall `net_foundationness_score`.

### Step 5: Study via Local Web UI or Terminal CLI

#### Option A: Local Web UI (Recommended for Math/Physics/CS Notation)
```bash
./scripts/podman_run.sh serve --port 8080
```
Open **`http://127.0.0.1:8080`**:
1. Select your book (`em_101`) in the left panel to inspect the ranked chunks and problems.
2. Click **Start Top-4 Problem Study Batch**.
3. Read the **Bare-Minimum Concepts Briefing** and work through the 4 problems.
4. For each problem, either:
   - Type your solution in Markdown + LaTeX (`$...$`, `$$...$$`) or code, **OR**
   - Upload a photo/scan of your handwritten solution $\to$ review and edit the transcribed LaTeX in the yellow **Handwritten Math-OCR Preview** box $\to$ click **Confirm OCR Transcription**.
5. Click **Submit & Analyze All Solutions** to receive your self-discovery evaluation report and unlock the next batch of 4 problems.

#### Option B: Terminal CLI
```bash
# Start a 4-problem batch:
./scripts/podman_run.sh study-start --book-id em_101

# Preview & confirm a handwritten solution image:
./scripts/podman_run.sh study-ocr \
  --book-id em_101 \
  --session-id ses_1234567890 \
  --problem-id em_101_prob_001_abcd1234 \
  --image /app/data/uploads/my_work.png \
  --confirm

# Submit solutions JSON & get evaluation report:
./scripts/podman_run.sh study-submit \
  --book-id em_101 \
  --session-id ses_1234567890 \
  --solutions-json '{"em_101_prob_001_abcd1234": "By Gauss law, $$\\oint E \\cdot dA = Q/\\epsilon_0$$"}'
```

---

## 3. Tuning Hardware & Pipeline Parameters (`configs/config.yaml`)
All system parameters can be customized in `configs/config.yaml`:
- `hardware.llm_generation_threads` (default `8`) & `hardware.llm_batch_threads` (default `10`)
- `hardware.max_container_ram_gib` (default `6.5`)
- `models.llm.gpu_layers` (default `18` for hybrid Vulkan + AVX-VNNI inference)
- `chunking.target_tokens` (default `550`)
- `graph.top_k_candidates` (default `12`) & `graph.similarity_threshold` (default `0.45`)
- `scoring.problem_top_k_chunks` (default `5`) & `scoring.batch_problem_count` (default `4`)
