"""Disk-backed content-addressable SQLite cache for Local AI Agent Harness."""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional


class LLMCache:
    """Content-addressable SQLite cache keyed by SHA-256 of (agent, schema, model, payload)."""

    def __init__(self, db_path: Path | str = "data/processed/llm_cache.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS agent_cache (
                    cache_key TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def compute_key(
        agent_name: str,
        schema_version: str,
        model_id: str,
        payload: Dict[str, Any],
    ) -> str:
        canonical = json.dumps(
            {
                "agent": agent_name,
                "version": schema_version,
                "model": model_id,
                "payload": payload,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT output_json FROM agent_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])

    def put(
        self,
        cache_key: str,
        agent_name: str,
        model_id: str,
        payload: Dict[str, Any],
        output_data: Dict[str, Any],
    ) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_cache
                (cache_key, agent_name, model_id, input_json, output_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    agent_name,
                    model_id,
                    json.dumps(payload, sort_keys=True, ensure_ascii=False),
                    json.dumps(output_data, sort_keys=True, ensure_ascii=False),
                    time.time(),
                ),
            )

    def get_embedding(self, model_id: str, text: str) -> Optional[list[float]]:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = hashlib.sha256(f"{model_id}:{text_hash}".encode("utf-8")).hexdigest()
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT vector_json FROM embedding_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])

    def put_embedding(self, model_id: str, text: str, vector: list[float]) -> None:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = hashlib.sha256(f"{model_id}:{text_hash}".encode("utf-8")).hexdigest()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embedding_cache
                (cache_key, model_id, text_sha256, vector_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, model_id, text_hash, json.dumps(vector), time.time()),
            )

    def stats(self) -> Dict[str, int]:
        with closing(self._connect()) as conn:
            agent_cnt = conn.execute("SELECT COUNT(*) FROM agent_cache").fetchone()[0]
            emb_cnt = conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
        return {"agent_cache_entries": int(agent_cnt), "embedding_cache_entries": int(emb_cnt)}
