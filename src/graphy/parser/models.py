"""Data models for parsed textbook sections, atomic semantic chunks, and separated exercise blocks."""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


ChunkType = Literal[
    "concept",
    "definition",
    "theorem_proof",
    "worked_example",
    "algorithm_code",
    "figure_concept",
]


class FigureRef(BaseModel):
    figure_id: str
    caption: str
    image_path: Optional[str] = None
    page_number: Optional[int] = None


class TextbookChunk(BaseModel):
    chunk_id: str
    book_id: str
    chunk_index: int
    chapter_title: str = "Chapter 1"
    section_title: str = "General"
    subsection_title: str = ""
    breadcrumb: str
    page_start: int = 1
    page_end: int = 1
    chunk_type: ChunkType = "concept"
    text_md: str
    embedding_text: str
    token_estimate: int
    extracted_formulas: List[str] = Field(default_factory=list)
    extracted_symbols_and_terms: List[str] = Field(default_factory=list)
    code_identifiers: List[str] = Field(default_factory=list)
    figures: List[FigureRef] = Field(default_factory=list)


class ExtractedExerciseBlock(BaseModel):
    book_id: str
    chapter_title: str
    section_title: str
    heading: str
    raw_text: str
    page_start: int = 1
    page_end: int = 1


class SectionNode(BaseModel):
    chapter_title: str
    section_title: str
    subsection_title: str = ""
    page_start: int = 1
    page_end: int = 1
    content_md: str
    figures: List[FigureRef] = Field(default_factory=list)


class DocumentParseResult(BaseModel):
    book_id: str
    source_path: str
    source_format: Literal["pdf", "tex", "md"]
    total_pages: int = 1
    sections: List[SectionNode] = Field(default_factory=list)
    exercise_blocks: List[ExtractedExerciseBlock] = Field(default_factory=list)
    figures: List[FigureRef] = Field(default_factory=list)
