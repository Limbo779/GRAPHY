"""Chunk Foundation Score Calculator (`OutWeight - InWeight` and Normalized [epsilon, 1.0])."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

from graphy.graph.graph_engine import KnowledgeDiGraph
from graphy.scoring.models import ChunkScoreRecord


def compute_chunk_foundation_scores(
    graph: KnowledgeDiGraph,
    normalization_epsilon: float = 0.05,
) -> List[ChunkScoreRecord]:
    """Compute raw (`W_out - W_in`) and normalized foundation scores for all chunks in `graph`."""
    vertices = graph.vertices()
    if not vertices:
        return []

    mastered_ids: Set[str] = {
        v.chunk_id for v in vertices if v.mastery_status == "mastered"
    }

    raw_map: Dict[str, tuple[float, float, float]] = {}
    unmastered_raw_values: List[float] = []

    for v in vertices:
        # Active weights ignore already-mastered prerequisites so newly unlocked topics rise
        w_out = graph.outgoing_weight_sum(v.chunk_id, excluded_nodes=mastered_ids)
        w_in = graph.incoming_weight_sum(v.chunk_id, excluded_nodes=mastered_ids)
        s_raw = round(w_out - w_in, 6)
        raw_map[v.chunk_id] = (w_out, w_in, s_raw)
        if v.chunk_id not in mastered_ids:
            unmastered_raw_values.append(s_raw)

    min_raw = min(unmastered_raw_values) if unmastered_raw_values else 0.0
    max_raw = max(unmastered_raw_values) if unmastered_raw_values else 0.0
    span = max_raw - min_raw

    records: List[ChunkScoreRecord] = []
    for v in vertices:
        w_out, w_in, s_raw = raw_map[v.chunk_id]
        if v.chunk_id in mastered_ids:
            s_norm = 0.0
        elif span <= 1e-9:
            s_norm = 1.0
        else:
            scaled = (s_raw - min_raw) / span
            s_norm = round(normalization_epsilon + (1.0 - normalization_epsilon) * scaled, 6)

        records.append(
            ChunkScoreRecord(
                chunk_id=v.chunk_id,
                book_id=graph.book_id,
                chunk_index=v.chunk_index,
                chapter_title=v.chapter_title,
                section_title=v.section_title,
                breadcrumb=v.breadcrumb,
                out_weight=w_out,
                in_weight=w_in,
                raw_foundation_score=s_raw,
                normalized_foundation_score=s_norm,
                rank=0,
                mastery_status=v.mastery_status,
            )
        )

    # Rank unmastered chunks first by descending raw_foundation_score (tie-break by earlier chunk_index)
    records.sort(
        key=lambda r: (
            1 if r.mastery_status == "mastered" else 0,
            -r.raw_foundation_score,
            r.chunk_index,
        )
    )
    for rank_idx, rec in enumerate(records, start=1):
        rec.rank = rank_idx

    return records


def save_chunk_scores_json(records: List[ChunkScoreRecord], output_path: Path | str) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([r.model_dump(mode="json") for r in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out
