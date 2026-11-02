"""Subtask 02 verification tests: Model setup, Memory Lifecycle Manager, SQLite Cache, and Micro-Agent Harness."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from graphy.core.config import load_config
from graphy.llm.agents import (
    ConceptDistillerAgent,
    ConceptDistillerInput,
    PrerequisiteJudgeAgent,
    PrerequisitePairInput,
    ProblemBoundaryAgent,
    ProblemBoundaryInput,
    SolutionAuditInput,
    SolutionAuditorAgent,
)
from graphy.llm.cache import LLMCache
from graphy.llm.harness import AgentHarness, ContextPruner, strip_thinking_blocks
from graphy.llm.model_manager import ModelLifecycleManager
from graphy.llm.model_setup import inspect_gguf_header, setup_models_volume


class TestSubtask02LLMAndHarness(unittest.TestCase):
    def test_models_volume_and_gguf_headers(self) -> None:
        manifest = setup_models_volume("models")
        self.assertEqual(len(manifest.entries), 3)
        for entry in manifest.entries:
            self.assertTrue(entry.ready, f"Model {entry.filename} should be ready in ./models/")
            self.assertIsNotNone(entry.header)
            assert entry.header is not None
            self.assertTrue(entry.header.valid_magic)
            self.assertGreaterEqual(entry.header.version, 2)
            self.assertGreater(entry.header.tensor_count, 10)

    def test_context_pruner_preserves_atomic_math_and_code(self) -> None:
        long_intro = "A " * 500
        equation = "$$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$"
        tail = " B" * 500
        text = long_intro + equation + tail
        # Cut right inside where the equation lives
        cut_limit = len(long_intro) + 15
        pruned = ContextPruner.prune_text(text, max_chars=cut_limit)
        # Equation must remain complete and uncorrupted
        self.assertIn(equation, pruned)
        self.assertTrue(pruned.endswith("...[pruned]..."))

    def test_strip_thinking_blocks_and_code_fences(self) -> None:
        raw = "<think>\nLet me analyze step by step...\n</think>\n```json\n{\"direction\": \"A_PREREQ_OF_B\"}\n```"
        cleaned = strip_thinking_blocks(raw)
        self.assertEqual(cleaned, '{"direction": "A_PREREQ_OF_B"}')

    def test_sqlite_llm_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_file = Path(tmp) / "test_cache.sqlite"
            cache = LLMCache(db_path=db_file)
            key = cache.compute_key("PrerequisiteJudgeAgent", "1.0", "qwen2-7b", {"a": "1", "b": "2"})
            self.assertIsNone(cache.get(key))

            cache.put(
                cache_key=key,
                agent_name="PrerequisiteJudgeAgent",
                model_id="qwen2-7b",
                payload={"a": "1", "b": "2"},
                output_data={"direction": "A_PREREQ_OF_B", "confidence": 0.9, "rationale": "ok"},
            )
            hit = cache.get(key)
            self.assertIsNotNone(hit)
            assert hit is not None
            self.assertEqual(hit["direction"], "A_PREREQ_OF_B")

            # Embedding cache check
            cache.put_embedding("qwen2-7b", "vector space", [0.5, -0.5, 0.707])
            vec = cache.get_embedding("qwen2-7b", "vector space")
            self.assertEqual(vec, [0.5, -0.5, 0.707])
            self.assertEqual(cache.stats(), {"agent_cache_entries": 1, "embedding_cache_entries": 1})

    def test_agent_harness_validation_and_retry_loop(self) -> None:
        attempts = {"count": 0}

        def faulty_then_valid_handler(schema_name: str, req: Dict[str, Any]) -> Dict[str, Any]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                # First attempt violates ConceptDistillerAgent disjointness rule
                return {
                    "bare_minimum_topics": [{"title": "Eigenvalues", "summary_md": "Def"}],
                    "self_discovery_topics": [{"title": "Eigenvalues", "summary_md": "Same title!"}],
                    "anti_spoiler_verified": True,
                }
            return {
                "bare_minimum_topics": [
                    {"title": "Linear Operators", "summary_md": "A linear map $T: V \\to V$."}
                ],
                "self_discovery_topics": [
                    {"title": "Spectral Theorem", "summary_md": "Orthogonal diagonalization of symmetric matrices."}
                ],
                "anti_spoiler_verified": True,
            }

        with tempfile.TemporaryDirectory() as tmp:
            cache = LLMCache(db_path=Path(tmp) / "retry_cache.sqlite")
            mgr = ModelLifecycleManager(backend="mock", mock_handler=faulty_then_valid_handler)
            harness = AgentHarness(model_manager=mgr, cache=cache, max_retries=2)

            agent = ConceptDistillerAgent()
            inp = ConceptDistillerInput(
                problem_statements=["Show that real symmetric matrix A has real eigenvalues."],
                chunk_excerpts=["Chapter 6: Spectral Theorem for real symmetric matrices."],
            )
            out = harness.run(agent, inp)
            self.assertEqual(attempts["count"], 2)
            self.assertEqual(out.bare_minimum_topics[0].title, "Linear Operators")
            self.assertEqual(out.self_discovery_topics[0].title, "Spectral Theorem")

            # Second call should hit SQLite cache without invoking the model again
            out_cached = harness.run(agent, inp)
            self.assertEqual(attempts["count"], 2)
            self.assertEqual(out_cached.self_discovery_topics[0].title, "Spectral Theorem")

    def test_all_micro_agents_semantic_validators(self) -> None:
        def mock_router(schema_name: str, req: Dict[str, Any]) -> Dict[str, Any]:
            if schema_name == "ProblemBoundaryAgent":
                return {
                    "problems": [
                        {
                            "original_number": "Problem 1",
                            "statement_md": "Let $A$ be symmetric.\n**(a)** Find $A^T$.\n**(b)** Find $\\det(A)$.",
                            "subparts": ["a", "b"],
                            "quarantined_hint": "Use transpose properties.",
                            "quarantined_solution": None,
                        }
                    ]
                }
            if schema_name == "SolutionAuditorAgent":
                return {
                    "outcome": "FAILED",
                    "score_fraction": 0.25,
                    "discovered_topics": [],
                    "undiscovered_topics_to_teach": [
                        {"topic": "Characteristic Polynomial", "explanation_md": "$\\det(A - \\lambda I) = 0$"}
                    ],
                    "remedial_scaffolding_topics": [
                        {"topic": "2x2 Determinants", "explanation_md": "$ad - bc$"}
                    ],
                    "feedback_summary_md": "Review determinants first.",
                }
            raise ValueError(f"Unexpected schema_name: {schema_name}")

        with tempfile.TemporaryDirectory() as tmp:
            cache = LLMCache(db_path=Path(tmp) / "agents.sqlite")
            with ModelLifecycleManager(backend="mock", mock_handler=mock_router) as mgr:
                harness = AgentHarness(model_manager=mgr, cache=cache)

                pb_out = harness.run(
                    ProblemBoundaryAgent(),
                    ProblemBoundaryInput(raw_block_text="1. (a) Find A^T (b) Find det(A) [Hint: transpose]"),
                )
                self.assertEqual(pb_out.problems[0].subparts, ["a", "b"])
                self.assertEqual(pb_out.problems[0].quarantined_hint, "Use transpose properties.")

                sa_out = harness.run(
                    SolutionAuditorAgent(),
                    SolutionAuditInput(
                        problem_statements=["Find eigenvalues of A."],
                        target_self_discovery_topics=["Characteristic Polynomial"],
                        learner_solutions_md=["I do not know how to start."],
                    ),
                )
                self.assertEqual(sa_out.outcome, "FAILED")
                self.assertEqual(len(sa_out.remedial_scaffolding_topics), 1)

    def test_live_llama_server_vulkan_prerequisite_and_memory_budget(self) -> None:
        """Run a real inference + embedding check on Intel Iris Xe Vulkan0 within the 6.5 GiB RAM cap."""
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            cache = LLMCache(db_path=Path(tmp) / "live_cache.sqlite")
            with ModelLifecycleManager(config=cfg, backend="llama_server") as mgr:
                harness = AgentHarness(model_manager=mgr, cache=cache)
                agent = PrerequisiteJudgeAgent()
                inp = PrerequisitePairInput(
                    chunk_a_id="c1",
                    chunk_a_breadcrumb="Chapter 1: Limits and Continuity",
                    chunk_a_text="The limit of f(x) as x approaches a is L, written \\lim_{x \\to a} f(x) = L.",
                    chunk_b_id="c2",
                    chunk_b_breadcrumb="Chapter 2: The Derivative",
                    chunk_b_text="The derivative is defined via the limit f'(x) = \\lim_{h \\to 0} \\frac{f(x+h)-f(x)}{h}.",
                )
                decision = harness.run(agent, inp)
                self.assertEqual(decision.direction, "A_PREREQ_OF_B")
                self.assertGreater(decision.confidence, 0.5)

                mem_status = mgr.get_memory_status()
                self.assertTrue(
                    mem_status.within_budget,
                    f"Container RSS {mem_status.container_rss_gib} GiB exceeded budget {mem_status.max_budget_gib} GiB",
                )


if __name__ == "__main__":
    unittest.main()
