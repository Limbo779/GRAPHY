"""Subtask 07 verification tests: 4-problem study loop, bare-minimum concepts, Handwritten OCR preview gate, and all 3 grading outcomes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
import cairo

from graphy.companion.pipeline import ingest_problem_set
from graphy.core.config import load_config
from graphy.graph.prerequisite_builder import build_prerequisite_graph
from graphy.llm.model_manager import ModelLifecycleManager
from graphy.parser.pipeline import ingest_textbook
from graphy.scoring.mastery_updater import score_book_and_problems
from graphy.tutor.session_manager import StudySessionManager


BOOK_CONTENT = """
# Chapter 1: Vector Spaces
## 1.1 Vector Space Axioms
**Definition 1.1 (Vector Space).** A **vector space** $V$ is closed under addition $u+v$ and scalar multiplication $cv$.

# Chapter 2: Inner Products and Orthogonality
## 2.1 Inner Product and Cauchy-Schwarz
**Theorem 2.1 (Cauchy-Schwarz Inequality).** In an **inner product** space, $|\\langle u, v \\rangle| \\le \\|u\\| \\|v\\|$.
**Proof.** Project $u$ orthogonally onto $v$ and use non-negativity of norm squared.
"""

PROBLEM_SET_CONTENT = """
Problem 1. Verify that $\\mathbb{R}^n$ is a **vector space** under componentwise addition $u+v$ and scalar multiplication $cv$.
[Hint: Check closure axioms.]

Problem 2. Show that the zero vector in a **vector space** $V$ is unique.

Problem 3. Prove the **Cauchy-Schwarz Inequality** $|\\langle u, v \\rangle| \\le \\|u\\| \\|v\\|$ using the **inner product** properties.

Problem 4. Using the **inner product**, show that orthogonal vectors $u \\perp v$ satisfy the Pythagorean theorem $\\|u+v\\|^2 = \\|u\\|^2 + \\|v\\|^2$.
"""


class TestSubtask07LearningLoopAndOCR(unittest.TestCase):
    def _setup_course(self, tmp_path: Path, audit_mode: str = "SOLVED_MISSED_DISCOVERY"):
        book_file = tmp_path / "book.md"
        pset_file = tmp_path / "pset.md"
        book_file.write_text(BOOK_CONTENT, encoding="utf-8")
        pset_file.write_text(PROBLEM_SET_CONTENT, encoding="utf-8")

        cfg = load_config("configs/config.yaml")
        cfg.chunking.target_tokens = 60
        cfg.chunking.min_tokens = 20
        cfg.graph.similarity_threshold = 0.01
        cfg.scoring.problem_top_k_chunks = 1
        cfg.scoring.problem_similarity_threshold = 0.01
        cfg.scoring.batch_problem_count = 2

        ingest_textbook(
            file_path=book_file,
            book_id="ip_space",
            config=cfg,
            embedding_mode="technical_hash",
            base_dir=tmp_path,
        )
        ingest_problem_set(
            file_path=pset_file,
            book_id="ip_space",
            source_id="hw1",
            config=cfg,
            base_dir=tmp_path,
        )

        def mock_llm_router(schema_name: str, req: Dict[str, Any]) -> Dict[str, Any]:
            if schema_name == "PrerequisiteJudgeAgent":
                return {
                    "direction": "A_PREREQ_OF_B",
                    "confidence": 0.95,
                    "rationale": "Vector spaces are prerequisite to inner product spaces.",
                    "bridging_concepts": ["vector space"],
                }
            if schema_name == "ConceptDistillerAgent":
                return {
                    "bare_minimum_topics": [
                        {
                            "title": "Vector Addition & Scalar Multiplication",
                            "summary_md": "Closure under $u + v$ and $c v$.",
                        }
                    ],
                    "self_discovery_topics": [
                        {
                            "title": "Additive Inverse Cancellation",
                            "summary_md": "Using $v + (-v) = 0$ to prove uniqueness of identity.",
                        }
                    ],
                    "anti_spoiler_verified": True,
                }
            if schema_name == "HandwritingOCRAgent":
                return {
                    "transcribed_latex_md": "Let $0_1, 0_2 \\in V$. Then $$0_1 = 0_1 + 0_2 = 0_2$$.",
                    "detected_equations": ["0_1 = 0_1 + 0_2 = 0_2"],
                    "confidence": 0.94,
                    "ambiguity_notes": None,
                }
            if schema_name == "SolutionAuditorAgent":
                if audit_mode == "SOLVED_WITH_DISCOVERY":
                    return {
                        "outcome": "SOLVED_WITH_DISCOVERY",
                        "score_fraction": 1.0,
                        "discovered_topics": ["Additive Inverse Cancellation"],
                        "undiscovered_topics_to_teach": [],
                        "remedial_scaffolding_topics": [],
                        "feedback_summary_md": "Excellent work! You discovered Additive Inverse Cancellation.",
                    }
                if audit_mode == "SOLVED_MISSED_DISCOVERY":
                    return {
                        "outcome": "SOLVED_MISSED_DISCOVERY",
                        "score_fraction": 0.85,
                        "discovered_topics": [],
                        "undiscovered_topics_to_teach": [
                            {
                                "topic": "Additive Inverse Cancellation",
                                "explanation_md": "By adding $(-v)$ to both sides, cancellation follows immediately.",
                            }
                        ],
                        "remedial_scaffolding_topics": [],
                        "feedback_summary_md": "You solved the problems but missed the general cancellation law.",
                    }
                return {
                    "outcome": "FAILED",
                    "score_fraction": 0.20,
                    "discovered_topics": [],
                    "undiscovered_topics_to_teach": [
                        {
                            "topic": "Cauchy-Schwarz Orthogonal Projection",
                            "explanation_md": "Write $u = \\text{proj}_v(u) + w$ where $w \\perp v$.",
                        }
                    ],
                    "remedial_scaffolding_topics": [
                        {
                            "topic": "Inner Product Bilinearity",
                            "explanation_md": "Review $\\langle u + w, u + w \\rangle = \\|u\\|^2 + 2\\langle u,w \\rangle + \\|w\\|^2$.",
                        }
                    ],
                    "feedback_summary_md": "Let us strengthen your prerequisite foundation so you do not struggle.",
                }
            raise ValueError(f"Unknown schema_name: {schema_name}")

        mgr = ModelLifecycleManager(config=cfg, backend="mock", mock_handler=mock_llm_router)
        build_prerequisite_graph(
            book_id="ip_space",
            config=cfg,
            model_manager=mgr,
            base_dir=tmp_path,
        )
        score_book_and_problems(
            book_id="ip_space",
            config=cfg,
            embedding_mode="technical_hash",
            model_manager=mgr,
            base_dir=tmp_path,
        )
        return cfg, mgr

    def test_handwritten_ocr_preview_gate_and_solved_missed_discovery_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg, mgr = self._setup_course(tmp_path, audit_mode="SOLVED_MISSED_DISCOVERY")

            tutor = StudySessionManager(
                book_id="ip_space",
                config=cfg,
                model_manager=mgr,
                embedding_mode="technical_hash",
                base_dir=tmp_path,
            )

            # Start first batch (top 2 foundational problems: Problem 1 & Problem 2)
            session = tutor.start_next_batch(batch_size=2)
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(len(session.problems), 2)

            # Verify learner briefing hides self-discovery topics
            briefing = session.to_learner_briefing()
            self.assertEqual(len(briefing.bare_minimum_topics), 1)
            self.assertNotIn("hidden_self_discovery_topics", briefing.model_dump())

            # Create a synthetic handwritten solution image for Problem 1
            handwritten_png = tmp_path / "handwritten_p1.png"
            surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 400, 120)
            ctx = cairo.Context(surf)
            ctx.set_source_rgb(1, 1, 1)
            ctx.paint()
            ctx.set_source_rgb(0, 0, 0)
            ctx.set_font_size(16)
            ctx.move_to(20, 60)
            ctx.show_text("0_1 = 0_1 + 0_2 = 0_2")
            surf.write_to_png(str(handwritten_png))

            p1_id = session.problems[0].problem_id
            p2_id = session.problems[1].problem_id

            # Step 1: Run OCR preview on handwritten image
            preview = tutor.preview_handwritten_solution(
                session_id=session.session_id,
                problem_id=p1_id,
                image_path=handwritten_png,
                use_vlm=True,
            )
            self.assertFalse(preview.confirmed_by_user)
            self.assertIn("0_1 = 0_1 + 0_2 = 0_2", preview.raw_ocr_latex_md)

            # Step 2: Verify attempting to evaluate before confirming OCR raises ValueError
            with self.assertRaises(ValueError):
                tutor.evaluate_batch_solutions(session.session_id)

            # Step 3: User confirms/edits the OCR preview & submits text solution for Problem 2
            confirmed = tutor.confirm_handwritten_solution(
                session_id=session.session_id,
                problem_id=p1_id,
                edited_latex_md="Let $0_1, 0_2 \\in V$. Then $$0_1 = 0_1 + 0_2 = 0_2$$ (verified).",
            )
            self.assertTrue(confirmed.confirmed_by_user)
            tutor.submit_text_solutions(
                session_id=session.session_id,
                solutions_by_problem_id={p2_id: "Closure holds under componentwise operations."},
            )

            # Step 4: Evaluate batch solutions -> Branch SOLVED_MISSED_DISCOVERY
            report = tutor.evaluate_batch_solutions(session.session_id)
            self.assertEqual(report.outcome, "SOLVED_MISSED_DISCOVERY")
            # Must reveal the undiscovered topic AND provide the next batch's bare-minimum briefing!
            self.assertEqual(len(report.undiscovered_topics_to_teach), 1)
            self.assertEqual(report.undiscovered_topics_to_teach[0].topic, "Additive Inverse Cancellation")
            self.assertIsNotNone(report.next_batch_briefing)
            assert report.next_batch_briefing is not None
            self.assertEqual(len(report.next_batch_briefing.problems), 2)

    def test_failed_solution_branch_provides_undiscovered_and_remedial_scaffolding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg, mgr = self._setup_course(tmp_path, audit_mode="FAILED")

            tutor = StudySessionManager(
                book_id="ip_space",
                config=cfg,
                model_manager=mgr,
                embedding_mode="technical_hash",
                base_dir=tmp_path,
            )
            session = tutor.start_next_batch(batch_size=2)
            assert session is not None
            # Force target chunk to include chunk 2 (which has incoming prerequisite edge from chunk 1)
            session.target_chunk_ids = ["ip_space_chunk_0002"]
            tutor.save_session(session)

            tutor.submit_text_solutions(
                session_id=session.session_id,
                solutions_by_problem_id={p.problem_id: "I am stuck." for p in session.problems},
            )
            report = tutor.evaluate_batch_solutions(session.session_id)
            self.assertEqual(report.outcome, "FAILED")
            # Must provide BOTH undiscovered topics AND remedial scaffolding (including prerequisite chunk bridge)
            self.assertGreaterEqual(len(report.undiscovered_topics_to_teach), 1)
            self.assertGreaterEqual(len(report.remedial_scaffolding_topics), 2)
            self.assertTrue(
                any("Prerequisite Foundation" in r.topic for r in report.remedial_scaffolding_topics)
            )
            # Should NOT advance to next batch when failed
            self.assertIsNone(report.next_batch_briefing)
            self.assertEqual(report.mastered_chunk_ids, [])


if __name__ == "__main__":
    unittest.main()
