"""Top-K Embedding Similarity Candidate Filter to avoid O(N^2) LLM calls."""

from __future__ import annotations

from typing import List, Set, Tuple

from graphy.graph.models import CandidatePair
from graphy.parser.models import TextbookChunk


def select_top_k_candidate_pairs(
    chunks: List[TextbookChunk],
    similarity_matrix: List[List[float]],
    top_k: int = 12,
    similarity_threshold: float = 0.45,
) -> List[CandidatePair]:
    """Select deduplicated unordered candidate pairs {u, v} where v is in Top-K neighbors of u."""
    n = len(chunks)
    if n <= 1:
        return []

    seen_pairs: Set[Tuple[int, int]] = set()
    candidates: List[CandidatePair] = []

    for i in range(n):
        # Collect all j != i meeting threshold
        neighbors: List[Tuple[int, float]] = []
        row = similarity_matrix[i]
        for j in range(n):
            if i == j:
                continue
            sim = row[j]
            if sim >= similarity_threshold:
                neighbors.append((j, sim))

        # Sort descending by similarity and keep top_k
        neighbors.sort(key=lambda item: item[1], reverse=True)
        for j, sim in neighbors[:top_k]:
            low_idx, high_idx = (i, j) if i < j else (j, i)
            if (low_idx, high_idx) in seen_pairs:
                continue
            seen_pairs.add((low_idx, high_idx))

            ca = chunks[low_idx]
            cb = chunks[high_idx]
            candidates.append(
                CandidatePair(
                    chunk_a_id=ca.chunk_id,
                    chunk_b_id=cb.chunk_id,
                    chunk_a_index=ca.chunk_index,
                    chunk_b_index=cb.chunk_index,
                    similarity=round(float(sim), 6),
                )
            )

    # Sort deterministically by (chunk_a_index, chunk_b_index)
    candidates.sort(key=lambda c: (c.chunk_a_index, c.chunk_b_index))
    return candidates
