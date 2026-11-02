# Subtask 06: Foundation Scoring, Problem-Chunk Matching & Dynamic Mastery Ranking (JSON Store)

## Objective
Implement the exact mathematical scoring pipeline specified in `ideastorm.md` and our confirmed clarifications, storing all three core score artifacts in strict **JSON format** (`chunk_scores.json`, `problem_chunk_scores.json`, and `problem_foundation_scores.json`), along with the **Dynamic Mastery Graph Update** mechanism that advances the learner through the curriculum.

---

## Scope & Deliverables

### 1. Chunk Foundation Score Computation (`src/graphy/scoring/chunk_scorer.py`)
For every node (chunk) $u \in V$ in the directed weighted prerequisite graph $G = (V, E, w)$:
1. **Outgoing & Incoming Weights**:
   - Since an edge $u \to v$ means "$u$ is a prerequisite of $v$":
     $$W_{\text{out}}(u) = \sum_{(u \to v) \in E} w(u \to v) \quad \text{(how much later material depends on } u \text{)}$$
     $$W_{\text{in}}(u) = \sum_{(p \to u) \in E} w(p \to u) \quad \text{(how much prior prerequisite material } u \text{ requires)}$$
2. **Raw Foundation Score**:
   $$S_{\text{raw}}(u) = W_{\text{out}}(u) - W_{\text{in}}(u)$$
   - The chunk with the highest $S_{\text{raw}}(u)$ has high outgoing dependency weight and low/zero incoming prerequisite weight—making it the most fundamental topic that must be studied first.
3. **Normalized Foundation Score ($[0, 1]$)**:
   - To ensure positive, well-scaled multiplication with problem similarity scores across all chunks (even advanced ones where $S_{\text{raw}}(u) < 0$), we compute min-max normalization onto $[\epsilon, 1.0]$ (with $\epsilon = 0.05$ so the most advanced chunk still retains a small positive weight):
     $$S_{\text{norm}}(u) = \epsilon + (1 - \epsilon) \cdot \frac{S_{\text{raw}}(u) - \min_{v} S_{\text{raw}}(v)}{\max_{v} S_{\text{raw}}(v) - \min_{v} S_{\text{raw}}(v)}$$
4. **JSON Output #1 — `data/scores/<book_id>/chunk_scores.json`**:
   - Stores for every chunk: `chunk_id`, `chapter`, `section`, `out_weight`, `in_weight`, `raw_foundation_score`, `normalized_foundation_score`, `rank`, and `mastery_status` (`unseen` | `in_progress` | `mastered`).

### 2. Problem-to-Chunk Similarity Matching (`src/graphy/scoring/problem_matcher.py`)
For every extracted problem $p$ from **GRAPHY Companion** (Subtask 04):
1. Compute the technical embedding vector $\mathbf{e}_p$ of the problem statement (`statement_md` + `extracted_symbols_and_terms`).
2. Run cosine similarity against all textbook chunks in LanceDB and select the **Top-$K_p$ most similar chunks** (default $K_p = 5$, minimum similarity threshold $\tau_p = 0.35$).
3. **JSON Output #2 — `data/scores/<book_id>/problem_chunk_scores.json`**:
   - Stores for every problem $p$: each matched `chunk_id`, its cosine similarity score $\text{sim}(p, u)$, the chunk's `raw_foundation_score` and `normalized_foundation_score`, and the product $\text{sim}(p, u) \times S_{\text{norm}}(u)$.

### 3. Net Problem Foundationness Score (`src/graphy/scoring/problem_ranker.py`)
For each problem $p$ with matched active top-$K_p$ chunks $\mathcal{C}(p)$:
1. Multiply each chunk's similarity score for problem $p$ with that chunk's normalized foundation score and aggregate:
   $$F(p) = \sum_{u \in \mathcal{C}(p)} \text{sim}(p, u) \cdot S_{\text{norm}}(u)$$
   *(Also records the similarity-weighted average $\bar{F}(p) = \frac{\sum_{u \in \mathcal{C}(p)} \text{sim}(p, u) \cdot S_{\text{norm}}(u)}{\sum_{u \in \mathcal{C}(p)} \text{sim}(p, u)}$ for diagnostic inspection).*
2. Problems are ranked in descending order of $F(p)$ so the learner starts with problems of the **highest foundationness score**.
3. **JSON Output #3 — `data/scores/<book_id>/problem_foundation_scores.json`**:
   - Stores the ranked list of all problems with their net foundationness score $F(p)$, primary prerequisite chunks, and current problem status (`pending`, `active_batch`, `solved`, `needs_remediation`).

### 4. Dynamic Mastery Graph Update (`src/graphy/scoring/mastery_updater.py`)
When the user completes a batch of 4 problems and demonstrates mastery of a set of chunks $M \subset V$:
1. Mark chunks $u \in M$ as `mastered` in the active learner session.
2. In the active subgraph $G_{\text{active}} = G[V \setminus M]$:
   - Outgoing edges from mastered chunks $u \in M$ to unmastered chunks $v \notin M$ are now **satisfied prerequisites**! Thus, $W_{\text{in}}(v)$ decreases because $v$'s prerequisite $u$ has already been learned.
   - Recompute $S_{\text{raw}}^{\text{active}}(v)$ and $S_{\text{norm}}^{\text{active}}(v)$ on $G_{\text{active}}$, and dynamically update $F^{\text{active}}(p)$ for all remaining unsolved problems (filtering out problems whose chunks are already 100% mastered).
   - The newly unlocked frontier topics automatically rise to the top of the problem foundationness ranking for the next batch of 4 problems!

---

## Acceptance & Verification Criteria
1. All three JSON files (`chunk_scores.json`, `problem_chunk_scores.json`, `problem_foundation_scores.json`) are generated, schema-validated, human-readable, and deterministic.
2. Foundational chapters/chunks (e.g., basic definitions/axioms) consistently rank at the top, and advanced dependent topics rank lower until their prerequisites are marked mastered.
3. Simulating mastery of the top foundational chunks dynamically promotes the next layer of dependent problems to the top of the queue.
