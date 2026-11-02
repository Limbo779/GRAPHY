"""Subtask 08 verification tests: Local Web UI REST API, HTML workspace, Handwritten OCR upload, and CLI commands."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
import cairo

from graphy.core.config import load_config
from graphy.llm.model_manager import ModelLifecycleManager
from graphy.web.app import create_web_app


SAMPLE_MD_BOOK = """
# Chapter 1: Graph Theory Foundations
## 1.1 Graphs, Vertices, and Handshaking Lemma
**Definition 1.1 (Graph).** A **graph** $G = (V, E)$ consists of vertices $V$ and edges $E$.
**Theorem 1.1 (Handshaking Lemma).** In any undirected graph $G = (V, E)$:
$$\\sum_{v \\in V} \\deg(v) = 2|E|$$
**Proof.** Each edge $e = \\{u, v\\} \\in E$ contributes $1$ to $\\deg(u)$ and $1$ to $\\deg(v)$, so the sum of degrees is $2|E|$.

# Chapter 2: Trees and Spanning Trees
## 2.1 Properties of Trees
**Theorem 2.1 (Edges in a Tree).** Using the **Handshaking Lemma**, any tree $T = (V, E)$ on $n = |V|$ vertices has $|E| = n - 1$ edges.
**Proof.** By induction on $n$ and degree sum $2|E| = 2(n-1)$.
"""

SAMPLE_MD_PROBLEMS = """
Problem 1. [10 pts] Let $G = (V, E)$ be a **graph** where every vertex has degree 3 and $|V| = 8$.
(a) Use the **Handshaking Lemma** $\\sum_{v \\in V} \\deg(v) = 2|E|$ to find $|E|$.
(b) Can such a graph have 7 vertices? Explain why or why not.
[Hint: Consider parity of $2|E|$.]

Problem 2. [15 pts] Let $T = (V, E)$ be a tree with $|V| = 10$ vertices. Find $\\sum_{v \\in V} \\deg(v)$.
"""


class TestSubtask08CLIAndWebUI(unittest.TestCase):
    def test_web_ui_end_to_end_study_and_handwritten_ocr_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            book_path = tmp_path / "graphs_book.md"
            probs_path = tmp_path / "graphs_pset.md"
            book_path.write_text(SAMPLE_MD_BOOK, encoding="utf-8")
            probs_path.write_text(SAMPLE_MD_PROBLEMS, encoding="utf-8")

            cfg = load_config("configs/config.yaml")
            cfg.chunking.target_tokens = 60
            cfg.chunking.min_tokens = 20
            cfg.graph.similarity_threshold = 0.01
            cfg.scoring.problem_top_k_chunks = 1
            cfg.scoring.problem_similarity_threshold = 0.01
            cfg.scoring.batch_problem_count = 2

            def mock_router(schema_name: str, req: Dict[str, Any]) -> Dict[str, Any]:
                if schema_name == "PrerequisiteJudgeAgent":
                    return {
                        "direction": "A_PREREQ_OF_B",
                        "confidence": 0.96,
                        "rationale": "Handshaking lemma is prerequisite to tree degree sum.",
                        "bridging_concepts": ["Handshaking Lemma"],
                    }
                if schema_name == "ConceptDistillerAgent":
                    return {
                        "bare_minimum_topics": [
                            {"title": "Vertex Degree", "summary_md": "$\\deg(v)$ is the number of incident edges."}
                        ],
                        "self_discovery_topics": [
                            {"title": "Odd Degree Parity", "summary_md": "The number of odd-degree vertices is always even."}
                        ],
                        "anti_spoiler_verified": True,
                    }
                if schema_name == "HandwritingOCRAgent":
                    return {
                        "transcribed_latex_md": "$$2|E| = 8 \\times 3 = 24 \\implies |E| = 12$$",
                        "detected_equations": ["2|E| = 8 \\times 3 = 24 \\implies |E| = 12"],
                        "confidence": 0.97,
                        "ambiguity_notes": None,
                    }
                if schema_name == "SolutionAuditorAgent":
                    return {
                        "outcome": "SOLVED_WITH_DISCOVERY",
                        "score_fraction": 1.0,
                        "discovered_topics": ["Odd Degree Parity"],
                        "undiscovered_topics_to_teach": [],
                        "remedial_scaffolding_topics": [],
                        "feedback_summary_md": "Spot on! You discovered the odd-degree parity corollary.",
                    }
                raise ValueError(f"Unexpected schema_name: {schema_name}")

            mgr = ModelLifecycleManager(config=cfg, backend="mock", mock_handler=mock_router)
            app = create_web_app(
                config=cfg,
                model_manager=mgr,
                embedding_mode="technical_hash",
                base_dir=tmp_path,
            )
            client = app.test_client()

            # 1. Verify HTML Workspace & Health API
            res_html = client.get("/")
            self.assertEqual(res_html.status_code, 200)
            self.assertIn(b"GRAPHY &amp; GRAPHY Companion", res_html.data)

            res_health = client.get("/api/health")
            self.assertEqual(res_health.status_code, 200)
            self.assertEqual(res_health.get_json()["status"], "ok")

            # 2. Ingest textbook & external problem set via REST API
            res_ing_book = client.post(
                "/api/ingest-book",
                json={"file_path": str(book_path), "book_id": "gt_101"},
            )
            self.assertEqual(res_ing_book.status_code, 200)
            self.assertGreaterEqual(res_ing_book.get_json()["chunks_created"], 2)

            res_ing_probs = client.post(
                "/api/ingest-problems",
                json={"file_path": str(probs_path), "book_id": "gt_101", "source_id": "hw1"},
            )
            self.assertEqual(res_ing_probs.status_code, 200)
            self.assertEqual(res_ing_probs.get_json()["problems_extracted"], 2)

            # 3. Build Prerequisite Graph & Compute JSON Scores via REST API
            res_graph = client.post("/api/build-graph/gt_101")
            self.assertEqual(res_graph.status_code, 200)
            self.assertEqual(res_graph.get_json()["scoring"]["problems_scored"], 2)

            # 4. Inspect book list & scores via REST API
            res_books = client.get("/api/books")
            self.assertIn("gt_101", res_books.get_json()["books"])

            res_scores = client.get("/api/books/gt_101/scores")
            self.assertEqual(res_scores.status_code, 200)
            scores_data = res_scores.get_json()
            self.assertGreaterEqual(len(scores_data["chunk_scores"]), 2)
            self.assertEqual(len(scores_data["problem_foundation_scores"]), 2)

            # 5. Start Study Batch via REST API
            res_start = client.post("/api/books/gt_101/study/start")
            self.assertEqual(res_start.status_code, 200)
            briefing = res_start.get_json()
            session_id = briefing["session_id"]
            p1_id = briefing["problems"][0]["problem_id"]
            p2_id = briefing["problems"][1]["problem_id"]

            # 6. Upload a handwritten PNG image via multipart POST to /ocr-preview
            surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 300, 100)
            png_buf = io.BytesIO()
            surf.write_to_png(png_buf)
            png_buf.seek(0)

            res_ocr = client.post(
                f"/api/books/gt_101/study/{session_id}/ocr-preview",
                data={"problem_id": p1_id, "image": (png_buf, "handwritten.png")},
                content_type="multipart/form-data",
            )
            self.assertEqual(res_ocr.status_code, 200)
            ocr_data = res_ocr.get_json()
            self.assertFalse(ocr_data["confirmed_by_user"])
            self.assertIn("|E| = 12", ocr_data["edited_latex_md"])

            # 7. Confirm OCR transcription via /ocr-confirm
            res_conf = client.post(
                f"/api/books/gt_101/study/{session_id}/ocr-confirm",
                json={
                    "problem_id": p1_id,
                    "edited_latex_md": "$$2|E| = 24 \\implies |E| = 12$$, and $7 \\times 3 = 21$ is odd so impossible.",
                },
            )
            self.assertEqual(res_conf.status_code, 200)
            self.assertTrue(res_conf.get_json()["confirmed_by_user"])

            # 8. Submit & Evaluate solutions via /evaluate
            res_eval = client.post(
                f"/api/books/gt_101/study/{session_id}/evaluate",
                json={"solutions": {p2_id: "$$2|E| = 2(10 - 1) = 18$$"}},
            )
            self.assertEqual(res_eval.status_code, 200)
            eval_report = res_eval.get_json()
            self.assertEqual(eval_report["outcome"], "SOLVED_WITH_DISCOVERY")
            self.assertIn("Odd Degree Parity", eval_report["discovered_topics"])


if __name__ == "__main__":
    unittest.main()
