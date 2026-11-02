"""Pydantic models for candidate chunk pairs, directed prerequisite edges, and GraphAr metadata."""

from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field

from graphy.parser.models import ChunkType


MasteryStatus = Literal["unseen", "in_progress", "mastered"]


class CandidatePair(BaseModel):
    chunk_a_id: str
    chunk_b_id: str
    chunk_a_index: int
    chunk_b_index: int
    similarity: float


class GraphVertex(BaseModel):
    vertex_index: int
    chunk_id: str
    chunk_index: int
    chapter_title: str
    section_title: str
    breadcrumb: str
    chunk_type: ChunkType
    token_estimate: int
    mastery_status: MasteryStatus = "unseen"


class PrerequisiteEdge(BaseModel):
    source_id: str
    target_id: str
    source_index: int
    target_index: int
    similarity_weight: float
    llm_confidence: float = 1.0
    rationale: str = ""
    bridging_concepts: List[str] = Field(default_factory=list)


class GraphBuildSummary(BaseModel):
    book_id: str
    total_chunks: int
    total_possible_pairs: int
    candidate_pairs_evaluated: int
    directed_edges_created: int
    cycles_broken: int
    graphar_dir: str
