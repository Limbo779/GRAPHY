"""Chunk foundation scoring, problem-chunk similarity matching, and dynamic mastery updates."""

from graphy.scoring.chunk_scorer import (
    compute_chunk_foundation_scores,
    save_chunk_scores_json,
)
from graphy.scoring.mastery_updater import (
    load_saved_scores,
    record_mastery_and_recompute,
    score_book_and_problems,
)
from graphy.scoring.models import (
    ChunkScoreRecord,
    ProblemChunkMatchScore,
    ProblemChunkScoreRecord,
    ProblemFoundationScoreRecord,
    ProblemStatus,
    ScoringPipelineSummary,
)
from graphy.scoring.problem_scorer import (
    compute_problem_scores,
    save_problem_scores_json,
)

__all__ = [
    "ChunkScoreRecord",
    "ProblemChunkMatchScore",
    "ProblemChunkScoreRecord",
    "ProblemFoundationScoreRecord",
    "ProblemStatus",
    "ScoringPipelineSummary",
    "compute_chunk_foundation_scores",
    "compute_problem_scores",
    "load_saved_scores",
    "record_mastery_and_recompute",
    "save_chunk_scores_json",
    "save_problem_scores_json",
    "score_book_and_problems",
]
