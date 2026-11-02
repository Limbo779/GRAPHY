"""Prerequisite Discovery Graph Builder: Top-K Filtering -> Local LLM Judge -> GraphAr Persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from graphy.core.config import GraphyConfig, load_config
from graphy.embeddings.vector_store import EmbeddedVectorStore
from graphy.graph.candidate_filter import select_top_k_candidate_pairs
from graphy.graph.graph_engine import KnowledgeDiGraph
from graphy.graph.graphar_store import GraphArStore
from graphy.graph.models import GraphBuildSummary, GraphVertex, PrerequisiteEdge
from graphy.llm.agents import PrerequisiteJudgeAgent, PrerequisitePairInput
from graphy.llm.cache import LLMCache
from graphy.llm.harness import AgentHarness
from graphy.llm.model_manager import ModelLifecycleManager
from graphy.parser.models import TextbookChunk


def build_prerequisite_graph(
    book_id: str,
    config: Optional[GraphyConfig] = None,
    model_manager: Optional[ModelLifecycleManager] = None,
    base_dir: Optional[Path] = None,
) -> GraphBuildSummary:
    """Build the directed weighted prerequisite graph for `book_id` and save it in GraphAr format."""
    cfg = config or load_config()
    root = base_dir or Path(".")
    cfg.ensure_Runtime_dirs(base_dir=root)

    data_root = root / cfg.system.data_dir
    vectors_dir = data_root / "vectors"
    graphs_dir = data_root / "graphs"
    processed_dir = data_root / "processed"

    vstore = EmbeddedVectorStore(book_id=book_id, vectors_root=vectors_dir)
    chunks, sim_matrix = vstore.compute_all_pairs_similarity_matrix()
    n = len(chunks)
    if n == 0:
        raise ValueError(f"No chunks found in vector store for book '{book_id}'. Run `ingest-book` first.")

    # Map chunk_id -> (0-based vertex_index, TextbookChunk)
    chunk_lookup: Dict[str, tuple[int, TextbookChunk]] = {
        c.chunk_id: (idx, c) for idx, c in enumerate(chunks)
    }

    graph = KnowledgeDiGraph(book_id=book_id)
    for idx, c in enumerate(chunks):
        graph.add_vertex(
            GraphVertex(
                vertex_index=idx,
                chunk_id=c.chunk_id,
                chunk_index=c.chunk_index,
                chapter_title=c.chapter_title,
                section_title=c.section_title,
                breadcrumb=c.breadcrumb,
                chunk_type=c.chunk_type,
                token_estimate=c.token_estimate,
                mastery_status="unseen",
            )
        )

    candidates = select_top_k_candidate_pairs(
        chunks=chunks,
        similarity_matrix=sim_matrix,
        top_k=cfg.graph.top_k_candidates,
        similarity_threshold=cfg.graph.similarity_threshold,
    )

    cache = LLMCache(db_path=processed_dir / "llm_cache.sqlite")
    own_mgr = model_manager is None
    mgr = model_manager or ModelLifecycleManager(config=cfg, backend="llama_server")

    try:
        harness = AgentHarness(model_manager=mgr, cache=cache)
        judge_agent = PrerequisiteJudgeAgent()

        for pair in candidates:
            v_idx_a, ca = chunk_lookup[pair.chunk_a_id]
            v_idx_b, cb = chunk_lookup[pair.chunk_b_id]

            decision = harness.run(
                judge_agent,
                PrerequisitePairInput(
                    chunk_a_id=ca.chunk_id,
                    chunk_a_breadcrumb=ca.breadcrumb,
                    chunk_a_text=ca.text_md,
                    chunk_b_id=cb.chunk_id,
                    chunk_b_breadcrumb=cb.breadcrumb,
                    chunk_b_text=cb.text_md,
                ),
            )

            if decision.direction == "A_PREREQ_OF_B":
                graph.add_edge(
                    PrerequisiteEdge(
                        source_id=ca.chunk_id,
                        target_id=cb.chunk_id,
                        source_index=v_idx_a,
                        target_index=v_idx_b,
                        similarity_weight=pair.similarity,
                        llm_confidence=decision.confidence,
                        rationale=decision.rationale,
                        bridging_concepts=decision.bridging_concepts,
                    )
                )
            elif decision.direction == "B_PREREQ_OF_A":
                graph.add_edge(
                    PrerequisiteEdge(
                        source_id=cb.chunk_id,
                        target_id=ca.chunk_id,
                        source_index=v_idx_b,
                        target_index=v_idx_a,
                        similarity_weight=pair.similarity,
                        llm_confidence=decision.confidence,
                        rationale=decision.rationale,
                        bridging_concepts=decision.bridging_concepts,
                    )
                )
    finally:
        if own_mgr:
            mgr.release_all()

    cycles_broken = 0
    if cfg.graph.break_cycles:
        cycles_broken = graph.break_cycles()

    gar_store = GraphArStore(book_id=book_id, graphs_root=graphs_dir)
    saved_dir = gar_store.save(graph)

    total_possible = n * (n - 1) // 2
    return GraphBuildSummary(
        book_id=book_id,
        total_chunks=n,
        total_possible_pairs=total_possible,
        candidate_pairs_evaluated=len(candidates),
        directed_edges_created=len(graph.edges()),
        cycles_broken=cycles_broken,
        graphar_dir=str(saved_dir),
    )
