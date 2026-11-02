"""Problem-to-Chunk Similarity Matching & Net Problem Foundationness Scoring (JSON Store)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from graphy.companion.models import CompanionProblem
from graphy.embeddings.engine import TechnicalEmbeddingEngine
from graphy.embeddings.vector_store import EmbeddedVectorStore
from graphy.scoring.models import (
    ChunkScoreRecord,
    ProblemChunkMatchScore,
    ProblemChunkScoreRecord,
    ProblemFoundationScoreRecord,
    ProblemStatus,
)


def compute_problem_scores(
    problems: List[CompanionProblem],
    chunk_scores: List[ChunkScoreRecord],
    vector_store: EmbeddedVectorStore,
    embedding_engine: TechnicalEmbeddingEngine,
    top_k_chunks: int = 5,
    similarity_threshold: float = 0.35,
    solved_problem_ids: Optional[Set[str]] = None,
    remedial_problem_ids: Optional[Set[str]] = None,
) -> Tuple[List[ProblemChunkScoreRecord], List[ProblemFoundationScoreRecord]]:
    """Compute per-problem chunk scores and overall net problem foundationness scores."""
    solved_set = solved_problem_ids or set()
    remedial_set = remedial_problem_ids or set()
    chunk_score_map: Dict[str, ChunkScoreRecord] = {c.chunk_id: c for c in chunk_scores}

    problem_chunk_records: List[ProblemChunkScoreRecord] = []
    problem_foundation_records: List[ProblemFoundationScoreRecord] = []

    for prob in problems:
        q_vec = embedding_engine.embed_query(prob.statement_md)
        matches = vector_store.search(
            query_vector=q_vec,
            top_k=top_k_chunks,
            min_score=similarity_threshold,
        )
        # Fallback to top-1 best matching chunk if all chunks are below threshold
        if not matches:
            matches = vector_store.search(query_vector=q_vec, top_k=1, min_score=-1.0)

        match_scores: List[ProblemChunkMatchScore] = []
        net_norm_score = 0.0
        net_raw_score = 0.0
        sim_sum = 0.0

        for m in matches:
            c_rec = chunk_score_map.get(m.chunk.chunk_id)
            if c_rec is None:
                continue

            sim = round(float(m.similarity), 6)
            norm_contrib = round(sim * c_rec.normalized_foundation_score, 6)
            raw_contrib = round(sim * c_rec.raw_foundation_score, 6)

            match_scores.append(
                ProblemChunkMatchScore(
                    chunk_id=c_rec.chunk_id,
                    breadcrumb=c_rec.breadcrumb,
                    similarity_score=sim,
                    chunk_raw_foundation_score=c_rec.raw_foundation_score,
                    chunk_normalized_foundation_score=c_rec.normalized_foundation_score,
                    weighted_foundation_contribution=norm_contrib,
                    raw_weighted_contribution=raw_contrib,
                    chunk_mastery_status=c_rec.mastery_status,
                )
            )

            if c_rec.mastery_status != "mastered":
                net_norm_score += norm_contrib
                net_raw_score += raw_contrib
                sim_sum += sim

        weighted_avg = round(net_norm_score / sim_sum, 6) if sim_sum > 1e-9 else 0.0
        net_norm_score = round(net_norm_score, 6)
        net_raw_score = round(net_raw_score, 6)

        if prob.problem_id in solved_set:
            status: ProblemStatus = "solved"
        elif prob.problem_id in remedial_set:
            status = "needs_remediation"
        else:
            status = "pending"

        problem_chunk_records.append(
            ProblemChunkScoreRecord(
                problem_id=prob.problem_id,
                book_id=prob.book_id,
                source_id=prob.source_id,
                original_number=prob.original_number,
                matched_chunks=match_scores,
            )
        )

        preview = prob.statement_md.replace("\n", " ").strip()
        if len(preview) > 140:
            preview = preview[:137] + "..."

        problem_foundation_records.append(
            ProblemFoundationScoreRecord(
                problem_id=prob.problem_id,
                book_id=prob.book_id,
                source_id=prob.source_id,
                source_type=prob.source_type,
                original_number=prob.original_number,
                statement_preview=preview,
                subparts=prob.subparts,
                net_foundationness_score=net_norm_score,
                weighted_avg_foundationness=weighted_avg,
                raw_net_foundationness_score=net_raw_score,
                primary_chunk_ids=[ms.chunk_id for ms in match_scores],
                rank=0,
                status=status,
            )
        )

    # Sort problems: pending/needs_remediation first, then descending net_foundationness_score
    problem_foundation_records.sort(
        key=lambda p: (
            1 if p.status == "solved" else 0,
            -p.net_foundationness_score,
            p.problem_id,
        )
    )
    for rank_idx, pf in enumerate(problem_foundation_records, start=1):
        pf.rank = rank_idx

    return problem_chunk_records, problem_foundation_records


def save_problem_scores_json(
    problem_chunk_records: List[ProblemChunkScoreRecord],
    problem_foundation_records: List[ProblemFoundationScoreRecord],
    problem_chunk_path: Path | str,
    problem_foundation_path: Path | str,
) -> Tuple[Path, Path]:
    pc_out = Path(problem_chunk_path)
    pf_out = Path(problem_foundation_path)
    pc_out.parent.mkdir(parents=True, exist_ok=True)
    pf_out.parent.mkdir(parents=True, exist_ok=True)

    pc_out.write_text(
        json.dumps([r.model_dump(mode="json") for r in problem_chunk_records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pf_out.write_text(
        json.dumps([r.model_dump(mode="json") for r in problem_foundation_records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return pc_out, pf_out
