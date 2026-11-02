"""Subtask 05 verification tests: Top-K candidate filtering, LLM prerequisite graph, cycle breaking, and GraphAr storage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
import yaml

from graphy.core.config import load_config
from graphy.graph.candidate_filter import select_top_k_candidate_pairs
from graphy.graph.graph_engine import KnowledgeDiGraph
from graphy.graph.graphar_store import GraphArStore
from graphy.graph.models import GraphVertex, PrerequisiteEdge
from graphy.graph.prerequisite_builder import build_prerequisite_graph
from graphy.llm.model_manager import ModelLifecycleManager
from graphy.parser.models import TextbookChunk
from graphy.parser.pipeline import ingest_textbook


class TestSubtask05PrerequisiteGraphAndGraphAr(unittest.TestCase):
    def test_top_k_candidate_filter_scales_on_500_chunk_book(self) -> None:
        """Verify 500 chunks (124,750 unordered pairs) filter down to <= 3,500 deduplicated pairs."""
        n = 500
        chunks = [
            TextbookChunk(
                chunk_id=f"c_{i:04d}",
                book_id="big_book",
                chunk_index=i,
                breadcrumb=f"[Chapter {i // 50 + 1}]",
                text_md=f"Chunk {i}",
                embedding_text=f"Chunk {i}",
                token_estimate=100,
            )
            for i in range(1, n + 1)
        ]
        # Construct synthetic band similarity matrix where each chunk is similar to ~14 nearby chunks
        sim_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            sim_matrix[i][i] = 1.0
            for offset in range(1, 15):
                if i + offset < n:
                    s = round(0.85 - offset * 0.025, 4)
                    sim_matrix[i][i + offset] = s
                    sim_matrix[i + offset][i] = s

        candidates = select_top_k_candidate_pairs(
            chunks=chunks,
            similarity_matrix=sim_matrix,
            top_k=6,
            similarity_threshold=0.50,
        )
        self.assertGreater(len(candidates), 500)
        self.assertLessEqual(len(candidates), 3500)
        # Ensure all pairs are strictly ordered (chunk_a_index < chunk_b_index) and unique
        pair_tuples = [(c.chunk_a_id, c.chunk_b_id) for c in candidates]
        self.assertEqual(len(pair_tuples), len(set(pair_tuples)))
        for c in candidates:
            self.assertLess(c.chunk_a_index, c.chunk_b_index)
            self.assertGreaterEqual(c.similarity, 0.50)

    def test_knowledge_digraph_weights_and_cycle_breaking(self) -> None:
        g = KnowledgeDiGraph(book_id="cycle_test")
        for idx in range(3):
            g.add_vertex(
                GraphVertex(
                    vertex_index=idx,
                    chunk_id=f"c{idx}",
                    chunk_index=idx + 1,
                    chapter_title="Ch 1",
                    section_title="Sec 1",
                    breadcrumb="[Ch 1 > Sec 1]",
                    chunk_type="concept",
                    token_estimate=120,
                )
            )

        # Create a 3-cycle: c0 -> c1 (0.9) -> c2 (0.85) -> c0 (weak backward edge 0.50)
        g.add_edge(PrerequisiteEdge(source_id="c0", target_id="c1", source_index=0, target_index=1, similarity_weight=0.90, llm_confidence=0.95))
        g.add_edge(PrerequisiteEdge(source_id="c1", target_id="c2", source_index=1, target_index=2, similarity_weight=0.85, llm_confidence=0.90))
        g.add_edge(PrerequisiteEdge(source_id="c2", target_id="c0", source_index=2, target_index=0, similarity_weight=0.50, llm_confidence=0.55))

        broken = g.break_cycles()
        self.assertEqual(broken, 1)
        remaining_pairs = [(e.source_id, e.target_id) for e in g.edges()]
        self.assertEqual(remaining_pairs, [("c0", "c1"), ("c1", "c2")])

        # Verify outgoing and incoming weights
        self.assertAlmostEqual(g.outgoing_weight_sum("c0"), 0.90)
        self.assertAlmostEqual(g.incoming_weight_sum("c0"), 0.0)
        self.assertAlmostEqual(g.outgoing_weight_sum("c1"), 0.85)
        self.assertAlmostEqual(g.incoming_weight_sum("c1"), 0.90)
        self.assertAlmostEqual(g.outgoing_weight_sum("c2"), 0.0)
        self.assertAlmostEqual(g.incoming_weight_sum("c2"), 0.85)

    def test_end_to_end_prerequisite_graph_builder_and_graphar_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            md_book = tmp_path / "calculus.md"
            md_book.write_text(
                """
# Chapter 1: Foundations of Calculus

## 1.1 Limits of Functions
**Definition 1.1 (Limit).** The limit $\\lim_{x \\to a} f(x) = L$ means $f(x)$ approaches $L$ as $x$ approaches $a$.

## 1.2 The Derivative via Limits
**Definition 1.2 (Derivative).** Given a function $f(x)$ with a well-defined limit, the derivative is:
$$f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}$$

## 1.3 Chain Rule and Differentiation Rules
**Theorem 1.1 (Chain Rule).** Using the derivative limit definition, for composite functions $(g \\circ f)'(x) = g'(f(x)) f'(x)$.
**Proof.** By applying the limit definition of the derivative to $g(f(x+h)) - g(f(x))$, we obtain the product of derivatives.
""",
                encoding="utf-8",
            )

            cfg = load_config("configs/config.yaml")
            cfg.chunking.target_tokens = 60
            cfg.chunking.min_tokens = 20
            cfg.graph.similarity_threshold = 0.05

            ingest_textbook(
                file_path=md_book,
                book_id="calc_101",
                config=cfg,
                embedding_mode="technical_hash",
                base_dir=tmp_path,
            )

            def mock_prereq_judge(schema_name: str, req: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "direction": "A_PREREQ_OF_B",
                    "confidence": 0.92,
                    "rationale": "Earlier calculus concept is required for later concept.",
                    "bridging_concepts": ["limit", "derivative"],
                }

            with ModelLifecycleManager(config=cfg, backend="mock", mock_handler=mock_prereq_judge) as mgr:
                summary = build_prerequisite_graph(
                    book_id="calc_101",
                    config=cfg,
                    model_manager=mgr,
                    base_dir=tmp_path,
                )

            self.assertEqual(summary.total_chunks, 3)
            self.assertGreaterEqual(summary.directed_edges_created, 2)
            self.assertEqual(summary.cycles_broken, 0)

            gar_dir = Path(summary.graphar_dir)
            self.assertTrue((gar_dir / "graph.gar.yml").is_file())
            self.assertTrue((gar_dir / "chunk.vertex.yml").is_file())
            self.assertTrue((gar_dir / "chunk_prereq_chunk.edge.yml").is_file())

            # Inspect YAML descriptor compliance
            gar_meta = yaml.safe_load((gar_dir / "graph.gar.yml").read_text(encoding="utf-8"))
            self.assertEqual(gar_meta["version"], "gar/v1")
            self.assertEqual(gar_meta["vertices"], ["chunk.vertex.yml"])

            # Load back via GraphArStore and verify 100% lossless round-trip
            store = GraphArStore(book_id="calc_101", graphs_root=tmp_path / "data" / "graphs")
            loaded_graph = store.load()
            self.assertEqual(len(loaded_graph.vertices()), 3)
            self.assertEqual(len(loaded_graph.edges()), summary.directed_edges_created)
            self.assertEqual(loaded_graph.edges()[0].bridging_concepts, ["limit", "derivative"])


if __name__ == "__main__":
    unittest.main()
