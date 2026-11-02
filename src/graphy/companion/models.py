"""Pydantic models for GRAPHY Companion cleaned & separated problems."""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


ProblemSourceType = Literal["textbook_exercise", "external_assignment", "external_exam"]


class CompanionProblem(BaseModel):
    problem_id: str
    book_id: str
    source_id: str
    source_type: ProblemSourceType = "external_assignment"
    chapter_or_topic: str = "General"
    original_number: str
    statement_md: str
    subparts: List[str] = Field(default_factory=list)
    figures: List[str] = Field(default_factory=list)
    extracted_formulas: List[str] = Field(default_factory=list)
    extracted_symbols_and_terms: List[str] = Field(default_factory=list)
    code_identifiers: List[str] = Field(default_factory=list)
    quarantined_hint: Optional[str] = None
    quarantined_solution: Optional[str] = None


class CompanionIngestionSummary(BaseModel):
    book_id: str
    source_id: str
    source_type: ProblemSourceType
    problems_extracted: int
    total_book_problems: int
    multipart_problems_count: int
    hints_quarantined_count: int
    solutions_quarantined_count: int
    output_json_path: str
