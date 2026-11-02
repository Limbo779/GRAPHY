# Subtask 08: Usable Interface — Terminal CLI & Local KaTeX Web UI

## Objective
Build a fast, zero-bloat, rock-solid dual interface—a full-featured **Terminal CLI** (`graphy`) and a **Lightweight Local Web UI** (`http://127.0.0.1:8080`)—that prioritizes usability, readability of complex Math/Physics/CS notation, and reliability without trading off any system performance.

---

## Scope & Deliverables

### 1. Full-Featured Terminal CLI (`src/graphy/cli/main.py`)
Built with `Typer` + `Rich` so every single pipeline operation can be run directly from the terminal inside or outside Podman:
- `graphy ingest-book <path_to_pdf_or_tex> --book-id <id>`: Runs Subtask 03 (chunking + embeddings) and auto-routes textbook exercises to Companion.
- `graphy ingest-problems <path_to_pdf_or_tex_or_md> --book-id <id>`: Runs Subtask 04 (**GRAPHY Companion**) on external problem sets, exams, or assignments.
- `graphy build-graph --book-id <id>`: Runs Subtask 05 & 06 (Top-$K$ filtering, LLM prerequisite graph, GraphAr storage, and JSON foundation scoring).
- `graphy status --book-id <id>`: Prints graph summary, top foundational chunks, top ranked problems, and current mastery progress in clean terminal tables.
- `graphy study --book-id <id>`: Interactive terminal study loop (displays bare-minimum concepts, shows the 4 problems, accepts Markdown/LaTeX file or handwritten image path, previews OCR transcription in `$EDITOR`, and prints detailed solution analysis).
- `graphy serve --port 8080`: Launches the Local Web UI.

### 2. Lightweight Local Web UI (`src/graphy/web/`)
Because reading multi-line integrals, matrices, and commutative diagrams in raw terminal LaTeX can be fatiguing during long study sessions, the Local Web UI provides a clean, responsive local workspace with near-zero RAM overhead (~40 MB RAM using FastAPI + HTMX/Alpine.js + local **KaTeX**):
1. **Library & Pipeline Manager View**:
   - Upload a textbook PDF/LaTeX and one or more problem sets (assignments/exams), trigger ingestion/graph building with live progress logs.
2. **Knowledge Graph & Scores Explorer**:
   - Inspect chunks sorted by Foundation Score (`chunk_scores.json`), inspect prerequisite edges ($A \to B$), and view ranked problems (`problem_foundation_scores.json`).
3. **Active Study Session Workspace (The 4-Problem Loop)**:
   - **Step 1 — Bare-Minimum Briefing**: Renders the bare-minimum concepts with crisp KaTeX math typesetting and code syntax highlighting.
   - **Step 2 — The 4 Problems**: Displays the current batch of 4 foundational problems (keeping sub-parts `a, b, c` together and displaying any extracted figures).
   - **Step 3 — Solution Input & Handwritten OCR Preview**:
     - Write solutions in a live Markdown + LaTeX split-pane editor (with instant KaTeX preview), **OR**
     - Drag-and-drop / select a photo of handwritten work $\to$ triggers local Math-OCR $\to$ populates the editable LaTeX preview box alongside the original image so the user can verify/edit before clicking **Analyze Solutions**.
   - **Step 4 — Discovery & Remedial Analysis Report**:
     - Displays per-problem feedback, highlights which self-discovery concepts were discovered vs. missed, shows remedial explanations if needed, and updates the mastery bar for the next set of 4 problems.

---

## Acceptance & Verification Criteria
1. All CLI commands execute cleanly with informative progress bars and zero unhandled exceptions.
2. Local Web UI serves on `http://127.0.0.1:8080` from inside the rootless Podman container, renders inline (`$...$`) and block (`$$...$$`) LaTeX equations via bundled offline KaTeX assets, and supports the full 4-problem study and handwritten OCR preview workflow.
3. The Web UI adds < 50 MB RAM overhead, preserving full system resources for the Local AI engine.
