"""Subtask 03 verification tests: Textbook parsing (PDF/LaTeX/MD), atomic math/code/figure chunking, and embeddings."""

from __future__ import annotations

import cairo
import tempfile
import unittest
from pathlib import Path

from graphy.core.config import ChunkingConfig, load_config
from graphy.embeddings.engine import TechnicalEmbeddingEngine
from graphy.embeddings.vector_store import EmbeddedVectorStore
from graphy.parser.chunker import SemanticTextbookChunker
from graphy.parser.document_parser import TechnicalDocumentParser
from graphy.parser.math_normalizer import normalize_math_unicode
from graphy.parser.pipeline import ingest_textbook


SAMPLE_LATEX_TEXTBOOK = r"""
\documentclass{book}
\begin{document}

\chapter{Linear Algebra Foundations}

\section{Vector Spaces and Subspaces}

\begin{definition}[Vector Space]
A \textbf{vector space} $V$ over a field $\mathbb{R}$ is a set equipped with vector addition and scalar multiplication satisfying closure, associativity, and distributivity.
\end{definition}

\begin{theorem}[Uniqueness of Zero Vector]
Every vector space $V$ contains a unique additive identity $0 \in V$ such that $v + 0 = v$ for all $v \in V$.
\end{theorem}

\begin{proof}
Suppose $0_1$ and $0_2$ are both additive identities in $V$. Then:
$$
0_1 = 0_1 + 0_2 = 0_2
$$
Hence the zero vector is unique.
\end{proof}

\section{Spectral Theory and Algorithms}

Consider the eigenvalue equation for a real symmetric matrix $A \in \mathbb{R}^{n \times n}$:
\begin{align}
A v &= \lambda v \\
\det(A - \lambda I) &= 0
\end{align}

Figure 1.1: Geometric action of an eigenvector stretching along its principal axis by factor $\lambda$.

As shown in Figure 1.1, eigenvectors remain invariant in direction under transformation $A$.

```python
def power_iteration(A, num_steps: int = 100):
    b_k = [1.0 for _ in range(len(A))]
    for _ in range(num_steps):
        # Multiply matrix A by vector b_k
        b_k1 = [sum(row[j] * b_k[j] for j in range(len(row))) for row in A]
        norm = sum(x * x for x in b_k1) ** 0.5
        b_k = [x / norm for x in b_k1]
    return b_k
```

\begin{exercises}
1. (a) Show that the intersection of two subspaces $U \cap W$ is a subspace of $V$.
   (b) Give a counterexample showing $U \cup W$ need not be a subspace.
2. Compute the eigenvalues of $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$.
\end{exercises}

\end{document}
"""


class TestSubtask03ChunkingAndEmbedding(unittest.TestCase):
    def test_unicode_math_normalization(self) -> None:
        raw = "∀x ∈ ℝ, ∇ × E = −∂B/∂t and λ₁ ≤ λ₂"
        norm = normalize_math_unicode(raw)
        self.assertIn(r"\forall", norm)
        self.assertIn(r"\in", norm)
        self.assertIn(r"\mathbb{R}", norm)
        self.assertIn(r"\nabla", norm)
        self.assertIn(r"\partial", norm)
        self.assertIn(r"\lambda", norm)
        self.assertIn("_{1}", norm)
        self.assertIn(r"\le", norm)

    def test_latex_parsing_atomic_chunking_and_exercise_separation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tex_file = Path(tmp) / "linear_algebra.tex"
            tex_file.write_text(SAMPLE_LATEX_TEXTBOOK, encoding="utf-8")

            parser = TechnicalDocumentParser()
            parsed = parser.parse(tex_file, book_id="lin_alg")

            # Verify sections vs separated exercises
            self.assertEqual(len(parsed.sections), 2)
            self.assertEqual(len(parsed.exercise_blocks), 1)
            self.assertIn("intersection of two subspaces", parsed.exercise_blocks[0].raw_text)

            # Use tight target_tokens so Theorem+Proof and Code blocks are tested for atomic integrity
            chunker = SemanticTextbookChunker(
                config=ChunkingConfig(target_tokens=90, min_tokens=40, max_tokens=450)
            )
            chunks = chunker.chunk_document(parsed)
            self.assertGreaterEqual(len(chunks), 3)

            # 1. Verify exercises NEVER leak into exposition chunks
            for c in chunks:
                self.assertNotIn("Give a counterexample showing", c.text_md)

            # 2. Verify Theorem and its Proof stay bound together in a single chunk
            thm_chunks = [c for c in chunks if "Uniqueness of Zero Vector" in c.text_md]
            self.assertEqual(len(thm_chunks), 1)
            self.assertIn("Suppose $0_1$ and $0_2$", thm_chunks[0].text_md)
            self.assertIn("0_1 = 0_1 + 0_2 = 0_2", thm_chunks[0].text_md)
            self.assertEqual(thm_chunks[0].chunk_type, "theorem_proof")

            # 3. Verify multi-line \begin{align}...\end{align} is never broken
            align_chunks = [c for c in chunks if r"\begin{align}" in c.text_md]
            self.assertEqual(len(align_chunks), 1)
            self.assertIn(r"\end{align}", align_chunks[0].text_md)

            # 4. Verify Python code block is uncorrupted and code identifier is extracted
            code_chunks = [c for c in chunks if "def power_iteration" in c.text_md]
            self.assertEqual(len(code_chunks), 1)
            self.assertIn("return b_k", code_chunks[0].text_md)
            self.assertIn("power_iteration", code_chunks[0].code_identifiers)
            self.assertEqual(code_chunks[0].chunk_type, "algorithm_code")

            # 5. Verify Figure 1.1 metadata is bound to the chunk referencing Figure 1.1
            fig_chunks = [c for c in chunks if c.figures]
            self.assertGreaterEqual(len(fig_chunks), 1)
            self.assertEqual(fig_chunks[0].figures[0].figure_id, "Figure 1.1")

    def test_digital_pdf_parsing_and_end_to_end_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pdf_path = tmp_path / "physics_maxwell.pdf"

            # Generate a multi-page digital PDF using Cairo PDFSurface
            surface = cairo.PDFSurface(str(pdf_path), 595, 842)
            ctx = cairo.Context(surface)

            # Page 1: Exposition on Gauss's Law and Faraday's Law
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(14)
            ctx.move_to(50, 50)
            ctx.show_text("Chapter 1: Electrodynamics and Maxwell's Equations")

            ctx.set_font_size(12)
            ctx.move_to(50, 85)
            ctx.show_text("1.1 Gauss's Law and Faraday's Law")

            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(11)
            lines_p1 = [
                "**Theorem 1.1 (Faraday's Law of Induction).** A time-varying magnetic field",
                "induces a curling electric field according to the differential equation:",
                "$$\\nabla \\times E = -\\frac{\\partial B}{\\partial t}$$",
                "**Proof.** By applying Stokes' theorem to the electromotive force around",
                "a closed loop C bounding surface S, the integral form converts to the curl equation.",
                "Figure 1.2: Magnetic flux lines passing through a conducting loop.",
                "As seen in Figure 1.2, changing magnetic flux induces current.",
            ]
            y = 120
            for ln in lines_p1:
                ctx.move_to(50, y)
                ctx.show_text(ln)
                y += 22
            surface.show_page()

            # Page 2: End-of-chapter Exercises
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(12)
            ctx.move_to(50, 50)
            ctx.show_text("Exercises")

            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(11)
            ctx.move_to(50, 85)
            ctx.show_text("1. Compute the induced EMF in a circular loop of radius R when B(t) = B0 cos(omega t).")
            surface.show_page()
            surface.finish()

            cfg = load_config("configs/config.yaml")
            summary = ingest_textbook(
                file_path=pdf_path,
                book_id="maxwell_phys",
                config=cfg,
                embedding_mode="technical_hash",
                base_dir=tmp_path,
            )

            self.assertEqual(summary.source_format, "pdf")
            self.assertEqual(summary.total_pages, 2)
            self.assertGreaterEqual(summary.chunks_created, 1)
            self.assertEqual(summary.exercise_blocks_routed, 1)
            self.assertTrue(Path(summary.chunks_json_path).is_file())
            self.assertTrue(Path(summary.exercises_json_path).is_file())
            self.assertTrue(Path(summary.vector_db_path).is_file())

    def test_technical_embedding_retrieval_precision_across_math_physics_cs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tex_file = Path(tmp) / "book.tex"
            tex_file.write_text(SAMPLE_LATEX_TEXTBOOK, encoding="utf-8")

            parser = TechnicalDocumentParser()
            parsed = parser.parse(tex_file, book_id="test_retr")
            chunker = SemanticTextbookChunker(
                config=ChunkingConfig(target_tokens=90, min_tokens=40, max_tokens=450)
            )
            chunks = chunker.chunk_document(parsed)

            engine = TechnicalEmbeddingEngine(mode="technical_hash", technical_dim=512)
            vecs = engine.embed_documents([c.embedding_text for c in chunks])

            store = EmbeddedVectorStore(book_id="test_retr", vectors_root=Path(tmp) / "vectors")
            store.upsert_chunks(chunks, vecs)
            self.assertEqual(store.count(), len(chunks))

            # Query 1: Eigenvalues & determinant characteristic equation
            q_eig = engine.embed_query(r"Find eigenvalues using \det(A - \lambda I) = 0 for symmetric matrix")
            top_eig = store.search(q_eig, top_k=1)[0]
            self.assertIn(r"\det(A - \lambda I)", top_eig.chunk.text_md)

            # Query 2: Python algorithm for dominant eigenvector
            q_code = engine.embed_query("iterative algorithm power_iteration in Python to normalize vector")
            top_code = store.search(q_code, top_k=1)[0]
            self.assertIn("def power_iteration", top_code.chunk.text_md)

            # Query 3: Vector space additive identity zero vector proof
            q_zero = engine.embed_query("uniqueness of zero vector additive identity proof in vector space")
            top_zero = store.search(q_zero, top_k=1)[0]
            self.assertIn("Uniqueness of Zero Vector", top_zero.chunk.text_md)

            # Also verify all-pairs similarity matrix dimensions & symmetry
            all_chunks, sim_mat = store.compute_all_pairs_similarity_matrix()
            self.assertEqual(len(all_chunks), len(chunks))
            self.assertEqual(len(sim_mat), len(chunks))
            for i in range(len(chunks)):
                self.assertAlmostEqual(sim_mat[i][i], 1.0, places=5)
                for j in range(len(chunks)):
                    self.assertAlmostEqual(sim_mat[i][j], sim_mat[j][i], places=6)


if __name__ == "__main__":
    unittest.main()
