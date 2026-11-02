"""Foundation Scoring Orchestrator & Dynamic Mastery Graph Updater."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Set

from graphy.companion.models import CompanionProblem
from graphy.core.config import GraphyConfig, load_config
from graphy.embeddings.engine import EmbeddingMode, TechnicalEmbeddingEngine
from graphy.embeddings.vector_store import EmbeddedVectorStore
from graphy.graph.graphar_store import GraphArStore
from graphy.llm.cache import LLMCache
from graphy.llm.model_manager import ModelLifecycleManager
from graphy.scoring.chunk_scorer import (
    compute_chunk_foundation_scores,
    save_chunk_scores_json,
)
from graphy.scoring.models import (
    ChunkScoreRecord,
    ProblemChunkScoreRecord,
    ProblemFoundationScoreRecord,
    ScoringPipelineSummary,
)
from graphy.scoring.problem_scorer import (
    compute_problem_scores,
    save_problem_scores_json,
)


def score_book_and_problems(
    book_id: str,
    config: Optional[GraphyConfig] = None,
    embedding_mode: EmbeddingMode = "hybrid",
    model_manager: Optional[ModelLifecycleManager] = None,
    solved_problem_ids: Optional[Set[str]] = None,
    remedial_problem_ids: Optional[Set[str]] = None,
    base_dir: Optional[Path] = None,
) -> ScoringPipelineSummary:
    """Compute and persist all 3 required JSON score files for `book_id`."""
    cfg = config or load_config()
    root = base_dir or Path(".")
    cfg.ensure_Runtime_dirs(base_dir=root)

    data_root = root / cfg.system.data_dir
    processed_dir = data_root / "processed"
    vectors_dir = data_root / "vectors"
    graphs_dir = data_root / "graphs"
    scores_book_dir = data_root / "scores" / book_id
    scores_book_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Prerequisite Graph from GraphAr store and compute chunk foundation scores
    gar_store = GraphArStore(book_id=book_id, graphs_root=graphs_dir)
    graph = gar_store.load()

    chunk_scores = compute_chunk_foundation_scores(
        graph=graph,
        normalization_epsilon=cfg.scoring.normalization_epsilon,
    )
    chunk_scores_path = save_chunk_scores_json(
        chunk_scores,
        scores_book_dir / "chunk_scores.json",
    )

    # 2. Load extracted Companion problems (if any) and preserve prior solved/remedial states
    problems_file = processed_dir / f"{book_id}_problems.json"
    problems: List[CompanionProblem] = []
    if problems_file.is_file():
        raw_probs = json.loads(problems_file.read_text(encoding="utf-8"))
        problems = [CompanionProblem.model_validate(p) for p in raw_probs]

    existing_pf_file = scores_book_dir / "problem_foundation_scores.json"
    merged_solved: Set[str] = set(solved_problem_ids or set())
    merged_remedial: Set[str] = set(remedial_problem_ids or set())
    if existing_pf_file.is_file():
        prev_records = json.loads(existing_pf_file.read_text(encoding="utf-8"))
        for pr in prev_records:
            pid = pr.get("problem_id", "")
            st = pr.get("status", "pending")
            if st == "solved" and pid not in merged_remedial:
                merged_solved.add(pid)
            elif st == "needs_remediation" and pid not in merged_solved:
                merged_remedial.add(pid)

    vstore = EmbeddedVectorStore(book_id=book_id, vectors_root=vectors_dir)
    cache = LLMCache(db_path=processed_dir / "llm_cache.sqlite")

    # Match embedding dimension stored in the book's vector store
    _, existing_vecs = vstore.get_all_chunks_and_vectors()
    stored_dim = len(existing_vecs[0]) if existing_vecs else 512
    effective_mode: EmbeddingMode = (
        "technical_hash" if stored_dim <= 1024 and embedding_mode == "hybrid" else embedding_mode
    )

    emb_engine = TechnicalEmbeddingEngine(
        mode=effective_mode,
        model_manager=model_manager,
        cache=cache,
        technical_dim=stored_dim if stored_dim <= 1024 else 512,
    )

    pc_records, pf_records = compute_problem_scores(
        problems=problems,
        chunk_scores=chunk_scores,
        vector_store=vstore,
        embedding_engine=emb_engine,
        top_k_chunks=cfg.scoring.problem_top_k_chunks,
        similarity_threshold=cfg.scoring.problem_similarity_threshold,
        solved_problem_ids=merged_solved,
        remedial_problem_ids=merged_remedial,
    )

    pc_path, pf_path = save_problem_scores_json(
        problem_chunk_records=pc_records,
        problem_foundation_records=pf_records,
        problem_chunk_path=scores_book_dir / "problem_chunk_scores.json",
        problem_foundation_path=scores_book_dir / "problem_foundation_scores.json",
    )

    mastered_cnt = sum(1 for c in chunk_scores if c.mastery_status == "mastered")
    unmastered_cnt = len(chunk_scores) - mastered_cnt
    top_chunk_id = chunk_scores[0].chunk_id if chunk_scores else ""
    top_prob_id = pf_records[0].problem_id if pf_records else ""

    return ScoringPipelineSummary(
        book_id=book_id,
        chunks_scored=len(chunk_scores),
        unmastered_chunks=unmastered_cnt,
        mastered_chunks=mastered_cnt,
        problems_scored=len(pf_records),
        top_chunk_id=top_chunk_id,
        top_problem_id=top_prob_id,
        chunk_scores_json_path=str(chunk_scores_path),
        problem_chunk_scores_json_path=str(pc_path),
        problem_foundation_scores_json_path=str(pf_path),
    )


def record_mastery_and_recompute(
    book_id: str,
    mastered_chunk_ids: List[str],
    solved_problem_ids: Optional[List[str]] = None,
    remedial_problem_ids: Optional[List[str]] = None,
    config: Optional[GraphyConfig] = None,
    embedding_mode: EmbeddingMode = "hybrid",
    model_manager: Optional[ModelLifecycleManager] = None,
    base_dir: Optional[Path] = None,
) -> ScoringPipelineSummary:
    """Mark chunks as mastered in GraphAr, update problem statuses, and recompute all JSON scores."""
    cfg = config or load_config()
    root = base_dir or Path(".")
    graphs_dir = root / cfg.system.data_dir / "graphs"

    gar_store = GraphArStore(book_id=book_id, graphs_root=graphs_dir)
    graph = gar_store.load()

    for cid in mastered_chunk_ids:
        v = graph.get_vertex(cid)
        if v is not None:
            v.mastery_status = "mastered"

    gar_store.save(graph)

    return score_book_and_problems(
        book_id=book_id,
        config=cfg,
        embedding_mode=embedding_mode,
        model_manager=model_manager,
        solved_problem_ids=set(solved_problem_ids or []),
        remedial_problem_ids=set(remedial_problem_ids or []),
        base_dir=root,
    )


def load_saved_scores(
    book_id: str,
    base_dir: Optional[Path] = None,
    config: Optional[GraphyConfig] = None,
) -> tuple[
    List[ChunkScoreRecord],
    List[ProblemChunkScoreRecord],
    List[ProblemFoundationScoreRecord],
]:
    """Load the three JSON score files for `book_id` from `data/scores/<book_id>/`."""
    cfg = config or load_config()
    root = base_dir or Path(".")
    scores_book_dir = root / cfg.system.data_dir / "scores" / book_id

    cs_raw = json.loads((scores_book_dir / "chunk_scores.json").read_text(encoding="utf-8"))
    pcs_raw = json.loads((scores_book_dir / "problem_chunk_scores.json").read_text(encoding="utf-8"))
    pfs_raw = json.loads((scores_book_dir / "problem_foundation_scores.json").read_text(encoding="utf-8"))

    return (
        [ChunkScoreRecord.model_validate(x) for x in cs_raw],
        [ProblemChunkScoreRecord.model_validate(x) for x in pcs_raw],
        [ProblemFoundationScoreRecord.model_validate(x) for x in pfs_raw],
    )
