"""Embedded Disk-Backed Vector Store for Textbook Chunks (`data/vectors/<book_id>/`)."""

from __future__ import annotations

from array import array
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from typing import List, Optional, Tuple
from pydantic import BaseModel

from graphy.embeddings.engine import cosine_similarity
from graphy.parser.models import TextbookChunk


class ChunkSimilarityMatch(BaseModel):
    chunk: TextbookChunk
    similarity: float


class EmbeddedVectorStore:
    """Zero-server embedded vector store backed by SQLite + binary IEEE-754 float32 vector blobs."""

    def __init__(self, book_id: str, vectors_root: Path | str = "data/vectors") -> None:
        self.book_id = book_id
        self.store_dir = Path(vectors_root) / book_id
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.store_dir / "chunks_vectors.sqlite"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    chunk_index INTEGER NOT NULL,
                    chapter_title TEXT NOT NULL,
                    section_title TEXT NOT NULL,
                    chunk_type TEXT NOT NULL,
                    chunk_json TEXT NOT NULL,
                    vector_dim INTEGER NOT NULL,
                    vector_f32 BLOB NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_index ON chunks(chunk_index);"
            )

    def upsert_chunks(self, chunks: List[TextbookChunk], vectors: List[List[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch between chunks count ({len(chunks)}) and vectors count ({len(vectors)})."
            )

        rows = []
        for chunk, vec in zip(chunks, vectors):
            f32_blob = array("f", vec).tobytes()
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.chunk_index,
                    chunk.chapter_title,
                    chunk.section_title,
                    chunk.chunk_type,
                    chunk.model_dump_json(),
                    len(vec),
                    f32_blob,
                )
            )

        with closing(self._connect()) as conn, conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, chunk_index, chapter_title, section_title, chunk_type, chunk_json, vector_dim, vector_f32)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get_all_chunks_and_vectors(self) -> Tuple[List[TextbookChunk], List[List[float]]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT chunk_json, vector_f32 FROM chunks ORDER BY chunk_index ASC"
            ).fetchall()

        chunks: List[TextbookChunk] = []
        vectors: List[List[float]] = []
        for chunk_json, blob in rows:
            chunks.append(TextbookChunk.model_validate_json(chunk_json))
            arr = array("f")
            arr.frombytes(blob)
            vectors.append(list(arr))
        return chunks, vectors

    def get_chunk(self, chunk_id: str) -> Optional[TextbookChunk]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT chunk_json FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                return None
            return TextbookChunk.model_validate_json(row[0])

    def count(self) -> int:
        with closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[ChunkSimilarityMatch]:
        """Find Top-K chunks with highest cosine similarity to `query_vector`."""
        chunks, vectors = self.get_all_chunks_and_vectors()
        matches: List[ChunkSimilarityMatch] = []
        for chunk, vec in zip(chunks, vectors):
            sim = cosine_similarity(query_vector, vec)
            if sim >= min_score:
                matches.append(ChunkSimilarityMatch(chunk=chunk, similarity=round(sim, 6)))

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:top_k]

    def compute_all_pairs_similarity_matrix(self) -> Tuple[List[TextbookChunk], List[List[float]]]:
        """Compute symmetric N x N cosine similarity matrix across all stored chunks."""
        chunks, vectors = self.get_all_chunks_and_vectors()
        n = len(chunks)
        matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
            vi = vectors[i]
            for j in range(i + 1, n):
                sim = round(cosine_similarity(vi, vectors[j]), 6)
                matrix[i][j] = sim
                matrix[j][i] = sim
        return chunks, matrix

    def export_chunks_json(self, output_path: Path | str) -> Path:
        """Save all chunks metadata and text to a clean JSON file."""
        chunks, _ = self.get_all_chunks_and_vectors()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([c.model_dump(mode="json") for c in chunks], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out
