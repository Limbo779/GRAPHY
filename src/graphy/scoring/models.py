"""Pydantic schemas for the three required JSON score artifacts and mastery state."""

from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field

from graphy.graph.models import MasteryStatus

ProblemStatus = Literal["pending", "active_batch", "solved", "needs_remediation"]


class ChunkScoreRecord(BaseModel):
    chunk_id: str
    book_id: str
    chunk_index: int
    chapter_title: str
    section_title: str
    breadcrumb: str
    out_weight: float
    in_weight: float
    raw_foundation_score: float
    normalized_foundation_score: float
    rank: int
    mastery_status: MasteryStatus = "unseen"


class ProblemChunkMatchScore(BaseModel):
    chunk_id: str
    breadcrumb: str
    similarity_score: float
    chunk_raw_foundation_score: float
    chunk_normalized_foundation_score: float
    weighted_foundation_contribution: float
    raw_weighted_contribution: float
    chunk_mastery_status: MasteryStatus = "unseen"


class ProblemChunkScoreRecord(BaseModel):
    problem_id: str
    book_id: str
    source_id: str
    original_number: str
    matched_chunks: List[ProblemChunkMatchScore] = Field(default_factory=list)


class ProblemFoundationScoreRecord(BaseModel):
    problem_id: str
    book_id: str
    source_id: str
    source_type: str
    original_number: str
    statement_preview: str
    subparts: List[str] = Field(default_factory=list)
    net_foundationness_score: float
    weighted_avg_foundationness: float
    raw_net_foundationness_score: float
    primary_chunk_ids: List[str] = Field(default_factory=list)
    rank: int
    status: ProblemStatus = "pending"


class ScoringPipelineSummary(BaseModel):
    book_id: str
    chunks_scored: int
    unmastered_chunks: int
    mastered_chunks: int
    problems_scored: int
    top_chunk_id: str
    top_problem_id: str
    chunk_scores_json_path: str
    problem_chunk_scores_json_path: str
    problem_foundation_scores_json_path: str
