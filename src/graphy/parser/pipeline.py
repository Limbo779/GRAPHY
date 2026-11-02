"""End-to-end Textbook Ingestion Pipeline (Parse -> Separate Exercises -> Atomic Chunk -> Embed -> Store)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

from graphy.core.config import GraphyConfig, load_config
from graphy.embeddings.engine import EmbeddingMode, TechnicalEmbeddingEngine
from graphy.embeddings.vector_store import EmbeddedVectorStore
from graphy.llm.cache import LLMCache
from graphy.llm.model_manager import ModelLifecycleManager
from graphy.parser.chunker import SemanticTextbookChunker
from graphy.parser.document_parser import TechnicalDocumentParser


class BookIngestionSummary(BaseModel):
    book_id: str
    source_path: str
    source_format: str
    total_pages: int
    sections_parsed: int
    chunks_created: int
    exercise_blocks_routed: int
    figures_extracted: int
    chunks_json_path: str
    exercises_json_path: str
    vector_db_path: str


def ingest_textbook(
    file_path: Path | str,
    book_id: str,
    config: Optional[GraphyConfig] = None,
    embedding_mode: EmbeddingMode = "hybrid",
    model_manager: Optional[ModelLifecycleManager] = None,
    base_dir: Optional[Path] = None,
) -> BookIngestionSummary:
    """Ingest a Math, Physics, or CS textbook into atomic chunks, vectors, and separated exercise blocks."""
    cfg = config or load_config()
    root = base_dir or Path(".")
    cfg.ensure_Runtime_dirs(base_dir=root)

    data_root = root / cfg.system.data_dir
    processed_dir = data_root / "processed"
    vectors_dir = data_root / "vectors"
    figures_dir = processed_dir / "figures"

    parser = TechnicalDocumentParser(figures_output_dir=figures_dir)
    parsed_doc = parser.parse(file_path=file_path, book_id=book_id)

    # Save separated textbook exercise blocks for GRAPHY Companion (Subtask 04)
    exercises_json_path = processed_dir / f"{book_id}_textbook_exercises.json"
    exercises_json_path.write_text(
        json.dumps(
            [eb.model_dump(mode="json") for eb in parsed_doc.exercise_blocks],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    chunker = SemanticTextbookChunker(config=cfg.chunking)
    chunks = chunker.chunk_document(parsed_doc)

    cache = LLMCache(db_path=processed_dir / "llm_cache.sqlite")
    emb_engine = TechnicalEmbeddingEngine(
        mode=embedding_mode,
        model_manager=model_manager,
        cache=cache,
    )
    vectors = emb_engine.embed_documents([c.embedding_text for c in chunks])

    vstore = EmbeddedVectorStore(book_id=book_id, vectors_root=vectors_dir)
    vstore.upsert_chunks(chunks, vectors)

    chunks_json_path = processed_dir / f"{book_id}_chunks.json"
    vstore.export_chunks_json(chunks_json_path)

    return BookIngestionSummary(
        book_id=book_id,
        source_path=parsed_doc.source_path,
        source_format=parsed_doc.source_format,
        total_pages=parsed_doc.total_pages,
        sections_parsed=len(parsed_doc.sections),
        chunks_created=len(chunks),
        exercise_blocks_routed=len(parsed_doc.exercise_blocks),
        figures_extracted=len(parsed_doc.figures),
        chunks_json_path=str(chunks_json_path),
        exercises_json_path=str(exercises_json_path),
        vector_db_path=str(vstore.db_path),
    )
