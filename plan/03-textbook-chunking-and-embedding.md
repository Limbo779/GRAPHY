# Subtask 03: Universal Textbook Parsing, Math/Code/Figure-Aware Chunking & Technical Embeddings

## Objective
Build a bulletproof textbook ingestion, semantic chunking, and embedding pipeline designed specifically for **Math, Physics, and Computer Science textbooks** in Digital PDF, Markdown, and LaTeX formats. Existing naive splitters break LaTeX environments, chop proofs in half, corrupt code indentation, and lose figure context—our custom parser and chunker solves this generally for any technical textbook.

---

## Scope & Deliverables

### 1. Multi-Format Technical Document Parser (`src/graphy/parser/document_parser.py`)
Supports `.pdf` (digital textbooks), `.tex` (LaTeX sources), and `.md` (Markdown):
- **Digital PDF Extraction Engine**:
  - Uses `PyMuPDF` (`fitz`) + `pdfplumber` font/layout analysis + math symbol/font map reconstruction (converting mathematical glyphs, Greek letters, superscripts/subscripts, matrices, and display equations into clean inline `$...$` and block `$$...$$` LaTeX).
  - Performs **Selective Formula / Figure OCR Fallback**: If a PDF page embeds an equation or diagram as a vector drawing or bitmap where text extraction is garbled, crops that bounding box and passes it through our local math/figure extractor so zero mathematical content is lost.
  - Strips recurring headers, footers, page numbers, table of contents noise, index pages, and bibliography sections automatically using layout geometry and statistical line repetition.
- **LaTeX & Markdown Native Parser**:
  - Directly parses `\chapter`, `\section`, `\begin{theorem}`, `\begin{proof}`, `\begin{equation}`, `\begin{lstlisting}`, and Markdown headings/fences with lossless structure preservation.
- **Textbook Exercise Section Separator**:
  - Automatically detects end-of-section / end-of-chapter exercise blocks (`Exercises`, `Problems`, `Review Questions`, `Problem Set`) and routes them directly to **GRAPHY Companion** (Subtask 04) so exercises are not mixed into exposition chunks.

### 2. Atomic-Block & Hierarchy-Aware Semantic Chunker (`src/graphy/parser/chunker.py`)
Standard fixed-token chunkers destroy technical books. Our chunker enforces **Atomic Structural Integrity**:
1. **Unbreakable Atomic Regions**:
   - Multi-line display equations (`$$...$$`, `\begin{align}...\end{align}`)
   - Theorem / Lemma / Corollary / Definition statements paired with their immediate Proof
   - Worked Examples (`Example X.Y` + its step-by-step derivation)
   - Code blocks & algorithm pseudocode (`def ...`, `\begin{algorithm}`)
   - Figures & Tables bound to their caption and the paragraph that references them (`Figure 3.2`)
2. **Hierarchical Context Breadcrumbs**:
   - Every chunk carries structural metadata: `book_id`, `chapter_title`, `section_title`, `subsection_title`, `page_range`, `chunk_index`, `chunk_type` (`concept`, `theorem_proof`, `worked_example`, `algorithm`), `extracted_symbols`, and `figure_captions`.
   - Before embedding, the chunk's hierarchical breadcrumb (`[Chapter 4: Linear Operators > 4.2 Eigenvalues and Eigenvectors]`) is prepended so the embedding captures full structural context even if pronouns ("this operator") are used.
3. **Adaptive Semantic Sizing**:
   - Target chunk size: `350–800 tokens` with semantic boundary detection (splitting at structural transitions or low adjacent-sentence cosine similarity, never inside an atomic math/code block).

### 3. Technical Embedding Engine & Embedded Vector Store (`src/graphy/embeddings/`)
- **Formula & Code Enriched Embedding Representation**:
  - Encodes both the raw Markdown/LaTeX text and a normalized semantic header (e.g., key mathematical terms, defined symbols, algorithm names) using our local technical embedding model (`BAAI/bge-m3` / `nomic-embed-text-v1.5` via ONNX/OpenVINO).
  - L2-normalizes all embedding vectors so cosine similarity is a fast dot product.
- **LanceDB Embedded Storage (`data/vectors/`)**:
  - Stores chunk records, metadata, full text, and dense vectors in an embedded LanceDB table (`chunks_<book_id>`).
  - Also supports hybrid lexical (BM25 over mathematical terms/symbols) + dense vector lookup when matching rare notation.

---

## Acceptance & Verification Criteria
1. Test suite feeds sample Math (theorems, integrals, matrices), Physics (derivations, figures), and CS (algorithms, code listings) PDFs and LaTeX files into the pipeline.
2. Zero broken LaTeX environments (`$$`, `\begin{...}`), zero split code blocks, and 100% separation of end-of-chapter exercises from exposition chunks.
3. Embedding similarity sanity check: querying a chunk or concept retrieves semantically related chunks across chapters with high cosine precision.
