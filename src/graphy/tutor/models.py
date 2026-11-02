"""Pydantic models for the 4-Problem Self-Discovery Study Loop, OCR Preview Gate, and Evaluation Report."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from graphy.llm.agents import ConceptTopicItem, TopicCoachingItem


class StudyProblemView(BaseModel):
    """Learner-facing problem view (strictly excludes quarantined hints and solutions)."""

    problem_id: str
    original_number: str
    source_id: str
    statement_md: str
    subparts: List[str] = Field(default_factory=list)
    figures: List[str] = Field(default_factory=list)
    net_foundationness_score: float
    matched_chunk_ids: List[str] = Field(default_factory=list)


class OCRPreviewRecord(BaseModel):
    """Interactive OCR preview state before the learner confirms handwritten work for grading."""

    problem_id: str
    image_path: str
    raw_ocr_latex_md: str
    edited_latex_md: str
    detected_equations: List[str] = Field(default_factory=list)
    confidence: float = 0.9
    ambiguity_notes: Optional[str] = None
    confirmed_by_user: bool = False


class StudyBatchBriefing(BaseModel):
    """Learner-facing briefing at the start of a 4-problem study round (no spoilers)."""

    session_id: str
    book_id: str
    problems: List[StudyProblemView]
    bare_minimum_topics: List[ConceptTopicItem]
    chunk_breadcrumbs: List[str] = Field(default_factory=list)


class StudyBatchSession(BaseModel):
    """Full server-side study session state (includes hidden target self-discovery topics)."""

    session_id: str
    book_id: str
    status: Literal["active", "completed", "needs_remediation"] = "active"
    problems: List[StudyProblemView]
    target_chunk_ids: List[str]
    bare_minimum_topics: List[ConceptTopicItem]
    hidden_self_discovery_topics: List[ConceptTopicItem]
    ocr_previews: Dict[str, OCRPreviewRecord] = Field(default_factory=dict)
    submitted_solutions_md: Dict[str, str] = Field(default_factory=dict)

    def to_learner_briefing(self) -> StudyBatchBriefing:
        return StudyBatchBriefing(
            session_id=self.session_id,
            book_id=self.book_id,
            problems=self.problems,
            bare_minimum_topics=self.bare_minimum_topics,
            chunk_breadcrumbs=self.target_chunk_ids,
        )


class StudyEvaluationReport(BaseModel):
    """Comprehensive post-submission analysis covering all 3 pedagogical branches from ideastorm.md."""

    session_id: str
    book_id: str
    outcome: Literal["SOLVED_WITH_DISCOVERY", "SOLVED_MISSED_DISCOVERY", "FAILED"]
    score_fraction: float
    discovered_topics: List[str] = Field(default_factory=list)
    undiscovered_topics_to_teach: List[TopicCoachingItem] = Field(default_factory=list)
    remedial_scaffolding_topics: List[TopicCoachingItem] = Field(default_factory=list)
    feedback_summary_md: str
    mastered_chunk_ids: List[str] = Field(default_factory=list)
    next_batch_briefing: Optional[StudyBatchBriefing] = None
