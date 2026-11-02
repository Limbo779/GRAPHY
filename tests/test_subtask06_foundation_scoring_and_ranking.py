"""Subtask 06 verification tests: Chunk foundation scores, problem-chunk similarity matching, JSON persistence, and dynamic mastery updates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from graphy.companion.pipeline import ingest_problem_set
from graphy.core.config import load_config
from graphy.graph.prerequisite_builder import build_prerequisite_graph
from graphy.llm.model_manager import ModelLifecycleManager
from graphy.parser.pipeline import ingest_textbook
from graphy.scoring.mastery_updater import (
    load_saved_scores,
    record_mastery_and_recompute,
    score_book_and_problems,
)


SAMPLE_THREE_CHAPTER_BOOK = """
# Chapter 1: Sets and Vector Spaces
## 1.1 Vector Space Axioms
**Definition 1.1 (Vector Space).** A **vector space** $V$ over $\\mathbb{R}$ is a set closed under vector addition $u + v$ and scalar multiplication $c v$.

# Chapter 2: Linear Transformations and Matrices
## 2.1 Linear Maps and Matrix Representation
**Definition 2.1 (Linear Map).** Given a vector space $V$, a **linear transformation** $T: V \\to V$ satisfies $T(u + cv) = T(u) + cT(v)$ and is represented by a matrix $A$.

# Chapter 3: Spectral Theory and Eigenvalues
## 3.1 Eigenvalues and Characteristic Polynomial
**Theorem 3.1 (Eigenvalue Equation).** For a linear transformation matrix $A$, a scalar $\\lambda$ is an **eigenvalue** with eigenvector $v \\neq 0$ if:
$$A v = \\lambda v \\iff \\det(A - \\lambda I) = 0$$
**Proof.** The null space of $(A - \\lambda I)$ is non-trivial if and only if its determinant vanishes.
"""

SAMPLE_PROBLEM_SET = """
Problem 1. Verify whether the set of polynomials of degree at most 2 forms a **vector space** over $\\mathbb{R}$ under standard addition $u + v$ and scalar multiplication $c v$.

Problem 2. Let $T: V \\to V$ be a **linear transformation** satisfying $T(u + cv) = T(u) + cT(v)$. Find its matrix $A$ in the standard basis.

Problem 3. Compute the **eigenvalue** $\\lambda$ and eigenvector $v$ of matrix $A$ using the characteristic polynomial $\\det(A - \\lambda I) = 0$.
"""


class TestSubtask06FoundationScoringAndRanking(unittest.TestCase):
    def test_foundation_scoring_json_files_and_dynamic_mastery_progression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            book_file = tmp_path / "linalg_course.md"
            probs_file = tmp_path / "linalg_pset.md"
            book_file.write_text(SAMPLE_THREE_CHAPTER_BOOK, encoding="utf-8")
            probs_file.write_text(SAMPLE_PROBLEM_SET, encoding="utf-8")

            cfg = load_config("configs/config.yaml")
            cfg.chunking.target_tokens = 60
            cfg.chunking.min_tokens = 20
            cfg.graph.similarity_threshold = 0.01
            cfg.scoring.problem_top_k_chunks = 1
            cfg.scoring.problem_similarity_threshold = 0.05

            # Step 1: Ingest textbook and external problem set
            ingest_textbook(
                file_path=book_file,
                book_id="linalg",
                config=cfg,
                embedding_mode="technical_hash",
                base_dir=tmp_path,
            )
            ingest_problem_set(
                file_path=probs_file,
                book_id="linalg",
                source_id="pset1",
                config=cfg,
                base_dir=tmp_path,
            )

            # Step 2: Build prerequisite graph where Chunk 1 -> Chunk 2 -> Chunk 3
            # (and Chunk 1 -> Chunk 3)
            def mock_chain_prereq(schema_name: str, req: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "direction": "A_PREREQ_OF_B",
                    "confidence": 0.95,
                    "rationale": "Foundational chapter is prerequisite for later chapter.",
                    "bridging_concepts": ["vector space", "linear transformation"],
                }

            with ModelLifecycleManager(config=cfg, backend="mock", mock_handler=mock_chain_prereq) as mgr:
                build_prerequisite_graph(
                    book_id="linalg",
                    config=cfg,
                    model_manager=mgr,
                    base_dir=tmp_path,
                )

            # Step 3: Compute all 3 JSON score files
            summary = score_book_and_problems(
                book_id="linalg",
                config=cfg,
                embedding_mode="technical_hash",
                base_dir=tmp_path,
            )

            self.assertEqual(summary.chunks_scored, 3)
            self.assertEqual(summary.unmastered_chunks, 3)
            self.assertEqual(summary.mastered_chunks, 0)
            self.assertEqual(summary.problems_scored, 3)

            # Verify all 3 JSON files exist on disk
            self.assertTrue(Path(summary.chunk_scores_json_path).is_file())
            self.assertTrue(Path(summary.problem_chunk_scores_json_path).is_file())
            self.assertTrue(Path(summary.problem_foundation_scores_json_path).is_file())

            chunk_scores, prob_chunk_scores, prob_foundation_scores = load_saved_scores(
                book_id="linalg",
                base_dir=tmp_path,
                config=cfg,
            )

            # Verify Chunk 1 (Vector Spaces) has highest foundation score (Out - In > 0, In = 0)
            self.assertEqual(chunk_scores[0].chunk_id, "linalg_chunk_0001")
            self.assertEqual(chunk_scores[0].in_weight, 0.0)
            self.assertGreater(chunk_scores[0].out_weight, 0.0)
            self.assertEqual(chunk_scores[0].normalized_foundation_score, 1.0)
            self.assertEqual(chunk_scores[0].rank, 1)

            # Verify Chunk 3 (Eigenvalues) has lowest foundation score (Out = 0, In > 0 -> raw < 0)
            self.assertEqual(chunk_scores[2].chunk_id, "linalg_chunk_0003")
            self.assertEqual(chunk_scores[2].out_weight, 0.0)
            self.assertGreater(chunk_scores[2].in_weight, 0.0)
            self.assertLess(chunk_scores[2].raw_foundation_score, 0.0)
            self.assertAlmostEqual(
                chunk_scores[2].normalized_foundation_score,
                cfg.scoring.normalization_epsilon,
                places=5,
            )

            # Verify Problem 1 (about Vector Spaces) is ranked #1 highest foundationness!
            self.assertEqual(prob_foundation_scores[0].original_number, "Problem 1")
            self.assertEqual(prob_foundation_scores[0].rank, 1)
            self.assertEqual(prob_foundation_scores[2].original_number, "Problem 3")

            # Verify problem_chunk_scores.json structure
            self.assertEqual(len(prob_chunk_scores), 3)
            for pcs in prob_chunk_scores:
                self.assertEqual(len(pcs.matched_chunks), 1)
                mc = pcs.matched_chunks[0]
                self.assertAlmostEqual(
                    mc.weighted_foundation_contribution,
                    round(mc.similarity_score * mc.chunk_normalized_foundation_score, 6),
                    places=5,
                )

            # Step 4: Dynamic Mastery Graph Update!
            # Mark Chunk 1 ("linalg_chunk_0001") as mastered and Problem 1 as solved.
            prob1_id = prob_foundation_scores[0].problem_id
            updated_summary = record_mastery_and_recompute(
                book_id="linalg",
                mastered_chunk_ids=["linalg_chunk_0001"],
                solved_problem_ids=[prob1_id],
                config=cfg,
                embedding_mode="technical_hash",
                base_dir=tmp_path,
            )
            self.assertEqual(updated_summary.mastered_chunks, 1)
            self.assertEqual(updated_summary.unmastered_chunks, 2)

            new_chunks, _, new_probs = load_saved_scores(
                book_id="linalg",
                base_dir=tmp_path,
                config=cfg,
            )

            # Now Chunk 2 (Linear Transformations) has its prerequisite Chunk 1 satisfied (in_weight = 0.0!)
            # and automatically becomes Rank #1 among active chunks!
            self.assertEqual(new_chunks[0].chunk_id, "linalg_chunk_0002")
            self.assertEqual(new_chunks[0].in_weight, 0.0)
            self.assertEqual(new_chunks[0].normalized_foundation_score, 1.0)
            self.assertEqual(new_chunks[0].rank, 1)

            # And Problem 2 (Linear Transformations) automatically advances to Rank #1!
            self.assertEqual(new_probs[0].original_number, "Problem 2")
            self.assertEqual(new_probs[0].rank, 1)
            self.assertEqual(new_probs[0].status, "pending")
            # Problem 1 is now at the bottom with status="solved"
            self.assertEqual(new_probs[-1].problem_id, prob1_id)
            self.assertEqual(new_probs[-1].status, "solved")


if __name__ == "__main__":
    unittest.main()
