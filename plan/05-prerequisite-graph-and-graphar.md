# Subtask 05: Prerequisite Discovery Graph, Top-K Filtering & GraphAr + rustworkx Storage

## Objective
Construct the directed, weighted knowledge prerequisite graph across all textbook chunks:
1. Compute embedding cosine similarities across chunks and filter to the **Top-$K$ most similar candidate pairs** (above a configurable cosine similarity threshold) to avoid $O(N^2)$ LLM calls.
2. Use the Local AI **`PrerequisiteJudgeAgent`** to determine whether Chunk $A$ is a prerequisite of Chunk $B$ ($A \to B$), Chunk $B$ is a prerequisite of Chunk $A$ ($B \to A$), or neither.
3. Assign the embedding similarity as the directed edge weight $w(A \to B)$.
4. Persist the graph in **Apache GraphAr** standard format (Parquet vertex/edge tables + YAML graph metadata) and use **`rustworkx`** in memory for ultra-fast graph analytics.

---

## Scope & Deliverables

### 1. All-Pairs Similarity Matrix & Top-$K$ Candidate Filter (`src/graphy/graph/candidate_filter.py`)
- Given $N$ normalized chunk embedding vectors $E \in \mathbb{R}^{N \times d}$ from Subtask 03:
  - Compute the cosine similarity matrix $S = E E^\top$ via vectorized BLAS/NumPy matrix multiplication (takes < 50 ms even for $N = 5,000$ chunks).
  - For each chunk $u$, select its **Top-$K$ neighbors** (default $K = 12$, configurable in `config.yaml`) where $S(u, v) \ge \tau_{\text{sim}}$ (default $\tau_{\text{sim}} = 0.45$).
  - Deduplicate unordered pairs $\{u, v\}$ so each candidate pair is evaluated **at most once** by the Local LLM.

### 2. Local LLM Prerequisite Directionality Verification (`src/graphy/graph/prerequisite_builder.py`)
- For each candidate pair $\{A, B\}$ with similarity $s_{AB} = S(A, B)$:
  - Pass a compact summary/excerpt + defined symbols/headers of Chunk $A$ and Chunk $B$ to the **`PrerequisiteJudgeAgent`** (via our grammar-constrained Agent Harness).
  - The micro-agent classifies the relationship into one of three mutually exclusive outcomes:
    - `A_PREREQ_OF_B`: Understanding Chunk $A$ is required to understand Chunk $B$ $\implies$ add directed edge $A \to B$ with weight $w(A \to B) = s_{AB}$.
    - `B_PREREQ_OF_A`: Understanding Chunk $B$ is required to understand Chunk $A$ $\implies$ add directed edge $B \to A$ with weight $w(B \to A) = s_{AB}$.
    - `NONE`: Peer topics or no strict prerequisite dependency $\implies$ no directed edge added.
- **Cycle Prevention / DAG Regularization**:
  - Two chunks that are mutually related might occasionally induce small 2-cycles or $k$-cycles if noisy. Using `rustworkx.simple_cycles` / feedback arc detection weighted by confidence and textbook order, we optionally break spurious cycles or retain pure directed weights while guaranteeing stable foundation scores.

### 3. High-Performance Graph Engine (`rustworkx`) & Apache GraphAr Storage (`src/graphy/graph/graphar_store.py`)
- **In-Memory Computation**:
  - Uses `rustworkx.PyDiGraph` (written in Rust) for sub-millisecond in-degree, out-degree, weighted out-sum, weighted in-sum, topological traversal, and dynamic mastery subgraph updates.
- **On-Disk Persistence in Apache GraphAr Standard Format (`data/graphs/<book_id>/`)**:
  - Follows the GraphAr specification:
    - `graph.gar.yml`: Graph descriptor defining vertex types, edge types, chunk sizes, and file paths.
    - `chunk.vertex.yml` + `vertex/chunk/part0.parquet`: Stores chunk IDs, chapter/section metadata, mastery state, and foundation scores.
    - `prereq.edge.yml` + `edge/chunk_prereq_chunk/ordered_by_source/adj_list/part0.parquet`: CSR/CSC-style partitioned Parquet edge lists storing `source_id`, `target_id`, `similarity_weight`, and `llm_rationale`.

---

## Acceptance & Verification Criteria
1. For a 500-chunk textbook, Top-$K$ filtering reduces 249,500 pairs to ~3,500 unique candidate pairs.
2. The `PrerequisiteJudgeAgent` processes candidate pairs with SQLite caching (so interrupted runs resume seamlessly) and builds a directed weighted `rustworkx.PyDiGraph`.
3. Graph is saved to and loaded from `data/graphs/<book_id>/` in valid GraphAr Parquet + YAML format with zero data loss.
