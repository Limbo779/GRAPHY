# Subtask 04: GRAPHY Companion — Automated Problem Set Cleaning & Extraction Pipeline

## Objective
Build **GRAPHY Companion**, the dedicated automated pipeline that ingests problem sets from **both** textbook exercise sections and external sources (course assignments, university exams, recitation sheets, worksheets in Digital PDF, Markdown, or LaTeX), strips boilerplate/noise, separates individual problems while **keeping multi-part problems (`1a, 1b, 1c`) together as one unit**, preserves figures/code, and quarantines any solutions or hints so they never spoil self-discovery.

---

## Scope & Deliverables

### 1. Multi-Source Problem Set Ingestion (`src/graphy/companion/ingestor.py`)
Accepts:
1. Exercise sections automatically routed from a textbook during Subtask 03.
2. Standalone external PDFs, `.tex`, or `.md` files (homework PDFs, midterm/final exams, MIT OCW / university problem sets).

### 2. Noise Cleaning & Boilerplate Removal (`src/graphy/companion/cleaner.py`)
External course assignments and exams contain heavy non-problem noise. The cleaner automatically strips:
- Course headers, instructor names, due dates, grading rubrics, point values (`[15 pts]`, `(10 points)`), honor code declarations, and page footers.
- Blank answer boxes / lined workspace regions found in exam PDFs.
- Repairs broken line wraps inside mathematical expressions and normalizes numbering styles (`Problem 1.`, `Q1:`, `Exercise 3.14`, `1.`).

### 3. Two-Stage Problem Segmentation & Multi-Part Grouping (`src/graphy/companion/segmenter.py`)
Combines deterministic structural grammar parsing with the **Agent Harness (`ProblemBoundaryAgent`)**:
1. **Stage 1 — Structural Regex & Layout Segmentation**:
   - Detects top-level problem boundaries (`Problem 1`, `Exercise 2.4`, `\item`) vs. sub-part markers (`(a)`, `(b)`, `(c)`, `(i)`, `(ii)`, `Part A`).
   - **Strict Multi-Part Cohesion Rule**: Sub-parts (`a`, `b`, `c`) and their shared problem stem ("Consider a particle of mass $m$ in a potential $V(x)$...") are **always kept together as a single problem unit**, because later parts build on earlier parts.
2. **Stage 2 — Local LLM Agent Verification & Repair (`ProblemBoundaryAgent`)**:
   - When layout formatting is ambiguous (e.g., numbered steps inside a single problem vs. a new problem), the micro-agent inspects the boundary lines and confirms whether a split should occur, repairs any OCR/PDF math glitches, and structures the problem into clean Markdown + LaTeX.

### 4. Spoiler / Hint / Solution Quarantine (`src/graphy/companion/quarantine.py`)
- Many problem sets or textbooks include inline hints (`[Hint: Use integration by parts]`), answers at the back, or interleaved solutions in solution manuals.
- GRAPHY Companion automatically detects and extracts any hint, answer key, or worked solution attached to a problem and moves it into a restricted field (`reference_hint`, `reference_solution`) that is **hidden from the learner** during problem solving and only accessible to the `SolutionAuditorAgent` during grading.

### 5. Structured Companion Output Schema (`data/processed/problems_<set_id>.json`)
Each extracted problem is saved as a validated Pydantic record:
```json
{
  "problem_id": "prob_0042",
  "source_id": "mit_18_06_ps1",
  "source_type": "external_assignment",
  "original_number": "Problem 3",
  "statement_md": "Let $A$ be an $n \\times n$ real symmetric matrix...\n\n**(a)** Show that all eigenvalues of $A$ are real.\n**(b)** Prove that eigenvectors corresponding to distinct eigenvalues are orthogonal.",
  "subparts": ["a", "b"],
  "figures": ["data/processed/figures/prob_0042_fig1.png"],
  "extracted_symbols_and_terms": ["symmetric matrix", "eigenvalues", "eigenvectors", "orthogonal"],
  "quarantined_hint": "Consider x^T A y and take the complex conjugate.",
  "quarantined_solution": null
}
```

---

## Acceptance & Verification Criteria
1. Running **GRAPHY Companion** on a textbook exercise section and an external multi-page homework/exam PDF extracts 100% of problems into clean Markdown/LaTeX JSON records.
2. Multi-part problems (`1a, 1b, 1c`) remain unified with their main stem context.
3. Point values, due dates, headers, and inline hints/solutions are cleanly stripped or quarantined out of `statement_md`.
