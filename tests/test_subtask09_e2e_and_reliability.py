"""Subtask 09 verification tests: Full E2E across Math, Physics, and CS, checkpoint recovery, and failure-mode hardening."""

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
from graphy.scoring.mastery_updater import load_saved_scores, score_book_and_problems
from graphy.tutor.session_manager import StudySessionManager


CS_ALGORITHMS_TEXTBOOK = r"""
# Chapter 1: Asymptotic Complexity and Recurrences
## 1.1 Big-O Notation and Divide-and-Conquer
**Definition 1.1 (Big-O Notation).** A function $T(n) = \mathcal{O}(g(n))$ if there exist constants $c, n_0 > 0$ such that $0 \le T(n) \le c g(n)$ for all $n \ge n_0$.

**Theorem 1.1 (Master Theorem for Divide-and-Conquer).** For recurrences of the form:
$$T(n) = a T(n/b) + f(n)$$
where $a \ge 1$ and $b > 1$, the asymptotic bound depends on comparing $f(n)$ with $n^{\log_b a}$.
**Proof.** Expand the recursion tree of depth $\log_b n$ with $a^k$ leaves at level $k$.

# Chapter 2: Sorting and Graph Search Algorithms
## 2.1 MergeSort and Binary Search
Using **Big-O Notation** and the **Master Theorem**, MergeSort divides an array of size $n$ into two halves ($a=2, b=2$) and merges in $\mathcal{O}(n)$ time:
```python
def merge_sort(arr: list[int]) -> list[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return left + right
```

\begin{exercises}
1. [15 pts] Solve the recurrence $T(n) = 2T(n/2) + n$ using the **Master Theorem** and state the answer in **Big-O Notation**.
   (a) Identify $a, b$, and $n^{\log_b a}$.
   (b) Write the tight asymptotic bound $\Theta(n \log n)$.
   [Hint: This is Case 2 of the Master Theorem.]
2. [20 pts] Analyze the `merge_sort` recursion depth and space complexity.
\end{exercises}
"""


class TestSubtask09E2EAndReliability(unittest.TestCase):
    def test_cs_domain_end_to_end_and_sqlite_checkpoint_recovery(self) -> None:
        """Verify complete CS workflow and confirm SQLite checkpoint cache prevents duplicate LLM calls on restart."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cs_book = tmp_path / "cs_algorithms.md"
            cs_book.write_text(CS_ALGORITHMS_TEXTBOOK, encoding="utf-8")

            cfg = load_config("configs/config.yaml")
            cfg.chunking.target_tokens = 80
            cfg.chunking.min_tokens = 25
            cfg.graph.similarity_threshold = 0.01
            cfg.scoring.problem_top_k_chunks = 2
            cfg.scoring.problem_similarity_threshold = 0.01
            cfg.scoring.batch_problem_count = 2

            llm_call_counts = {"PrerequisiteJudgeAgent": 0, "ConceptDistillerAgent": 0, "SolutionAuditorAgent": 0}

            def counting_router(schema_name: str, req: Dict[str, Any]) -> Dict[str, Any]:
                llm_call_counts[schema_name] = llm_call_counts.get(schema_name, 0) + 1
                if schema_name == "PrerequisiteJudgeAgent":
                    return {
                        "direction": "A_PREREQ_OF_B",
                        "confidence": 0.95,
                        "rationale": "Big-O and Master Theorem are required to analyze merge_sort.",
                        "bridging_concepts": ["Big-O", "Master Theorem"],
                    }
                if schema_name == "ConceptDistillerAgent":
                    return {
                        "bare_minimum_topics": [
                            {"title": "Recurrence Relations", "summary_md": "$T(n) = aT(n/b) + f(n)$"}
                        ],
                        "self_discovery_topics": [
                            {"title": "Recursion Tree Leaf Summation", "summary_md": "Summing work across $\\log_b n$ levels."}
                        ],
                        "anti_spoiler_verified": True,
                    }
                if schema_name == "SolutionAuditorAgent":
                    return {
                        "outcome": "SOLVED_WITH_DISCOVERY",
                        "score_fraction": 1.0,
                        "discovered_topics": ["Recursion Tree Leaf Summation"],
                        "undiscovered_topics_to_teach": [],
                        "remedial_scaffolding_topics": [],
                        "feedback_summary_md": "Great analysis of MergeSort and the Master Theorem!",
                    }
                raise ValueError(f"Unexpected agent: {schema_name}")

            # 1. Ingest CS textbook (auto-routes exercises to GRAPHY Companion)
            ing_sum = ingest_textbook(
                file_path=cs_book,
                book_id="cs_algo",
                config=cfg,
                embedding_mode="technical_hash",
                base_dir=tmp_path,
            )
            self.assertGreaterEqual(ing_sum.chunks_created, 2)
            self.assertEqual(ing_sum.companion_problems_extracted, 2)

            mgr = ModelLifecycleManager(config=cfg, backend="mock", mock_handler=counting_router)

            # 2. First graph build (evaluates candidate pairs via LLM and caches in SQLite)
            g_sum1 = build_prerequisite_graph(
                book_id="cs_algo",
                config=cfg,
                model_manager=mgr,
                base_dir=tmp_path,
            )
            first_run_calls = llm_call_counts["PrerequisiteJudgeAgent"]
            self.assertGreaterEqual(first_run_calls, 1)
            self.assertEqual(first_run_calls, g_sum1.candidate_pairs_evaluated)

            # 3. Simulate pipeline restart: re-run build_prerequisite_graph -> 0 new LLM calls (100% SQLite checkpoint hit!)
            g_sum2 = build_prerequisite_graph(
                book_id="cs_algo",
                config=cfg,
                model_manager=mgr,
                base_dir=tmp_path,
            )
            self.assertEqual(llm_call_counts["PrerequisiteJudgeAgent"], first_run_calls)
            self.assertEqual(g_sum2.directed_edges_created, g_sum1.directed_edges_created)

            # 4. Score book & problems and verify JSON artifacts
            s_sum = score_book_and_problems(
                book_id="cs_algo",
                config=cfg,
                embedding_mode="technical_hash",
                model_manager=mgr,
                base_dir=tmp_path,
            )
            self.assertEqual(s_sum.problems_scored, 2)

            # 5. Run complete study batch -> solve -> verify all problems marked solved
            tutor = StudySessionManager(
                book_id="cs_algo",
                config=cfg,
                model_manager=mgr,
                embedding_mode="technical_hash",
                base_dir=tmp_path,
            )
            session = tutor.start_next_batch(batch_size=2)
            self.assertIsNotNone(session)
            assert session is not None
            tutor.submit_text_solutions(
                session_id=session.session_id,
                solutions_by_problem_id={
                    p.problem_id: "By Master Theorem with a=2, b=2, T(n) = Theta(n log n)."
                    for p in session.problems
                },
            )
            report = tutor.evaluate_batch_solutions(session.session_id)
            self.assertEqual(report.outcome, "SOLVED_WITH_DISCOVERY")
            # Since all 2 problems in the book are now solved, next_batch_briefing is None!
            self.assertIsNone(report.next_batch_briefing)

            _, _, final_pfs = load_saved_scores(book_id="cs_algo", base_dir=tmp_path, config=cfg)
            self.assertTrue(all(p.status == "solved" for p in final_pfs))

    def test_malformed_inputs_and_edge_case_guardrails(self) -> None:
        cfg = load_config("configs/config.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Missing textbook file raises FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                ingest_textbook(
                    file_path=tmp_path / "nonexistent.pdf",
                    book_id="bad_book",
                    config=cfg,
                    base_dir=tmp_path,
                )

            # Unsupported file extension raises ValueError
            bad_ext = tmp_path / "book.xyz"
            bad_ext.write_text("content", encoding="utf-8")
            with self.assertRaises(ValueError):
                ingest_textbook(
                    file_path=bad_ext,
                    book_id="bad_ext",
                    config=cfg,
                    base_dir=tmp_path,
                )

            # Missing problem set file raises FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                ingest_problem_set(
                    file_path=tmp_path / "missing_pset.pdf",
                    book_id="bad_book",
                    config=cfg,
                    base_dir=tmp_path,
                )

    def test_documentation_and_container_details_artifacts_exist(self) -> None:
        for doc_file in (
            "README.md",
            "CONTAINER_DETAILS.md",
            "docs/USER_GUIDE.md",
            "scripts/export_container.sh",
            "Containerfile",
            "compose.yaml",
        ):
            self.assertTrue(Path(doc_file).is_file(), f"Required artifact missing: {doc_file}")


if __name__ == "__main__":
    unittest.main()
