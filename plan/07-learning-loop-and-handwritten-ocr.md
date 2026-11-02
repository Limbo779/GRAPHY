# Subtask 07: Self-Discovery Learning Loop, Handwritten Solution OCR & Solution Analysis Engine

## Objective
Implement the pedagogical heart of **GRAPHY** (lines 11–14 of `ideastorm.md`):
1. Select the top 4 problems from the ranked foundationness list and gather their matched chunk details.
2. Use the Local AI Agent Harness (`ConceptDistillerAgent`) to give the learner **only the bare-minimum topics and concepts needed**, holding back the core synthesis/discovery topics so the learner must self-discover them by solving the problems.
3. Accept user solutions for the 4 problems—supporting both typed Markdown/LaTeX/code **and** handwritten solution images (with an interactive Vision/Math-OCR LaTeX preview before grading).
4. Analyze all solutions via the Local AI Agent Harness (`SolutionAuditorAgent` & `RemedialCoachAgent`) to handle both success/partial-discovery and failure/struggle cases with precision.

---

## Scope & Deliverables

### 1. Batch Selector & Bare-Minimum Concept Generator (`src/graphy/tutor/concept_distiller.py`)
When starting a learning round:
1. Pick the top 4 active problems $(p_1, p_2, p_3, p_4)$ with the highest foundationness score from `problem_foundation_scores.json`.
2. Retrieve all associated textbook chunks $\bigcup_{i=1}^4 \mathcal{C}(p_i)$ and their immediate incoming prerequisites.
3. Run the **`ConceptDistillerAgent`**, which partitions the knowledge inside those chunks into two strict lists:
   - **`bare_minimum_briefing` (Shown to the user BEFORE solving)**:
     - Fundamental definitions, notation, and baseline axioms strictly required to *understand what the 4 problems are asking*.
     - **Anti-Spoiler Validator**: Verified to ensure it does *not* reveal the derivation trick, theorem proof strategy, or direct formula that solves the problems.
   - **`target_self_discovery_topics` (Kept hidden in the session state until solutions are submitted)**:
     - The deeper properties, patterns, identities, or algorithmic insights that the learner is expected to discover on their own while working through the 4 problems.

### 2. Handwritten Solution Image $\to$ LaTeX/Markdown OCR Pipeline (`src/graphy/tutor/handwriting_ocr.py`)
When the user submits one or more photos/scans of handwritten math, physics, or CS work:
1. **Image Preprocessing**:
   - Auto-orientation, contrast normalization / adaptive thresholding (handling phone shadows on notebook paper), and bounding-box layout detection.
2. **Local Vision / Math-OCR Transcription**:
   - Uses our on-demand local Vision-Language / Math-OCR pipeline (`Qwen2.5-VL-3B-Instruct` GGUF + `Texify/RapidOCR` ONNX) to convert handwritten equations, matrices, free-body / graph sketches, and prose steps into structured Markdown + LaTeX.
3. **Interactive Preview & Confirmation Gate**:
   - Returns the extracted Markdown/LaTeX side-by-side with the uploaded image so the user can review, confirm, or tweak any misread symbol before triggering the LLM solution analysis.

### 3. Solution Analysis & Adaptive Pedagogical Feedback (`src/graphy/tutor/solution_analyzer.py`)
Once the user submits their confirmed solutions for the 4 problems, the **`SolutionAuditorAgent`** and **`RemedialCoachAgent`** evaluate each solution against:
- The problem statement & quarantined reference hint/solution (if any),
- The associated textbook chunks, and
- The hidden `target_self_discovery_topics`.

The analyzer classifies the outcome for the batch and produces structured feedback across the exact pedagogical branches defined in `ideastorm.md`:

| Learner Outcome | System Response & Next Step |
| :--- | :--- |
| **Case 1: Solved + Self-Discovered All Target Topics** | - Confirms the self-discovered insights and connects them to the textbook's formal terminology.<br>- Marks the chunks as `mastered`, updates the Dynamic Mastery Graph, and generates the **bare-bones concepts for the next set of 4 problems**. |
| **Case 2: Solved (or Mostly Solved), Put Missed One or More Self-Discovery Topics** *(e.g. used brute force instead of discovering the underlying theorem/property)* | - Explicitly teaches the **undiscovered topics** from the current chunks so the learner doesn't miss the deeper insight.<br>- Marks chunks as `mastered` and provides the **bare-bones concepts for the next set of 4 problems**. |
| **Case 3: Failed / Struggled to Solve the Problems** | - Provides **both**:<br>  1. The **undiscovered topics** from the current chunks, AND<br>  2. Deep **scaffolding & remedial topics** (step-by-step conceptual bridges, worked mini-examples from prerequisite chunks, and common pitfalls) so the learner will not struggle.<br>- Keeps the unmastered chunks in the active frontier so the user can retry or tackle remedial foundational problems before advancing. |

---

## Acceptance & Verification Criteria
1. Given 4 ranked problems and their chunks, `ConceptDistillerAgent` outputs a clear `bare_minimum_briefing` (without leaking the solution) and stores `target_self_discovery_topics`.
2. Handwritten math image test produces accurate LaTeX transcription and pauses at the user preview/confirmation step.
3. Automated test cases for all 3 learner outcomes (Full Discovery, Solved without Discovery, Failed Solution) verify that the exact required pedagogical output is generated and the mastery state updates accurately.
