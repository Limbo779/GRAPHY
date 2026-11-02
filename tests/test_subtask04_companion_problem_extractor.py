"""Subtask 04 verification tests: GRAPHY Companion problem set cleaner, multi-part cohesion, and spoiler quarantine."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
import cairo

from graphy.companion.cleaner import ProblemSetCleaner
from graphy.companion.models import CompanionProblem
from graphy.companion.pipeline import ingest_problem_set
from graphy.companion.segmenter import CompanionSegmenter
from graphy.llm.cache import LLMCache
from graphy.llm.harness import AgentHarness
from graphy.llm.model_manager import ModelLifecycleManager
from graphy.parser.pipeline import ingest_textbook


SAMPLE_NOISY_COURSE_ASSIGNMENT = r"""
Name: _______________________
Student ID: _________________
Due Date: Friday, November 14 at 11:59 PM
Instructor: Prof. Gilbert Strang
Maximum Score: 100 points
Show all your work to receive full credit.
Do not open this exam booklet until instructed.
________________________________________________________

Problem 1. [25 pts] Let $A$ be an $n \times n$ real symmetric matrix satisfying $A^2 = A$ (an orthogonal projection matrix).
(a) [10 pts] Prove that every eigenvalue $\lambda$ of $A$ satisfies $\lambda \in \{0, 1\}$.
(b) [15 pts] Show that $\text{rank}(A) = \text{trace}(A)$.
[Hint: Use the spectral theorem $A = Q \Lambda Q^T$ and cyclic invariance of trace.]

\begin{solution}
Since $A^2 = A$, for any eigenvector $v$ with $Av = \lambda v$, we have $\lambda^2 v = \lambda v$, so $\lambda \in \{0,1\}$.
\end{solution}

Problem 2. (30 points) Consider the 1D quantum harmonic oscillator with Hamiltonian:
$$
\hat{H} = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + \frac{1}{2}m\omega^2 x^2
$$
(a) Express $\hat{H}$ in terms of ladder operators $a$ and $a^\dagger$.
(b) Determine the ground state energy $E_0$.
**Hint:** Recall the commutation relation $[a, a^\dagger] = 1$.

Problem 3. [20 marks] Implement the `dijkstra_shortest_path` algorithm in Python and state its time complexity using a binary heap.
```python
def dijkstra_shortest_path(adj_list, source: int):
    # Initialize distances
    pass
```
**Solution:** Using a binary min-heap, the time complexity is $\mathcal{O}((V + E) \log V)$.
"""


class TestSubtask04CompanionProblemExtractor(unittest.TestCase):
    def test_cleaner_strips_exam_and_homework_boilerplate(self) -> None:
        cleaned = ProblemSetCleaner.clean_raw_text(SAMPLE_NOISY_COURSE_ASSIGNMENT)
        self.assertNotIn("Student ID:", cleaned)
        self.assertNotIn("Due Date:", cleaned)
        self.assertNotIn("Prof. Gilbert Strang", cleaned)
        self.assertNotIn("Do not open this exam", cleaned)
        self.assertNotIn("[25 pts]", cleaned)
        self.assertNotIn("[10 pts]", cleaned)
        self.assertNotIn("(30 points)", cleaned)
        self.assertNotIn("[20 marks]", cleaned)
        # Ensure Python `# Initialize distances` comment inside code block is preserved!
        self.assertIn("# Initialize distances", cleaned)

    def test_multipart_problem_cohesion_and_spoiler_quarantine(self) -> None:
        segmenter = CompanionSegmenter()
        problems = segmenter.segment_text(
            raw_text=SAMPLE_NOISY_COURSE_ASSIGNMENT,
            book_id="lin_alg",
            source_id="mit_18_06_hw3",
            source_type="external_assignment",
        )
        # Exactly 3 top-level problems (subparts (a) and (b) must NEVER be split into separate problems!)
        self.assertEqual(len(problems), 3)

        p1, p2, p3 = problems[0], problems[1], problems[2]

        # Check Problem 1: Multi-part (a), (b) kept together + Hint & Solution quarantined
        self.assertEqual(p1.original_number, "Problem 1")
        self.assertEqual(p1.subparts, ["a", "b"])
        self.assertIn("**(a)** Prove that every eigenvalue", p1.statement_md)
        self.assertIn("**(b)** Show that", p1.statement_md)
        # Hint and Solution must be quarantined out of statement_md
        self.assertNotIn("Hint:", p1.statement_md)
        self.assertNotIn("spectral theorem", p1.statement_md)
        self.assertNotIn(r"\lambda^2 v = \lambda v", p1.statement_md)
        self.assertIsNotNone(p1.quarantined_hint)
        self.assertIn("spectral theorem", p1.quarantined_hint or "")
        self.assertIsNotNone(p1.quarantined_solution)
        self.assertIn(r"\lambda^2 v = \lambda v", p1.quarantined_solution or "")

        # Check Problem 2: Display equation + subparts (a), (b) + quarantined hint
        self.assertEqual(p2.original_number, "Problem 2")
        self.assertEqual(p2.subparts, ["a", "b"])
        self.assertIn(r"\hat{H}", p2.statement_md)
        self.assertIsNotNone(p2.quarantined_hint)
        self.assertIn("[a, a^\\dagger] = 1", p2.quarantined_hint or "")

        # Check Problem 3: Code block + code identifier + quarantined solution
        self.assertEqual(p3.original_number, "Problem 3")
        self.assertIn("def dijkstra_shortest_path", p3.statement_md)
        self.assertIn("dijkstra_shortest_path", p3.code_identifiers)
        self.assertNotIn("binary min-heap", p3.statement_md)
        self.assertIn("binary min-heap", p3.quarantined_solution or "")

    def test_external_pdf_and_markdown_ingestion_pipeline_and_merging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # 1. Ingest a textbook with routed exercises first
            tex_book = tmp_path / "book.tex"
            tex_book.write_text(
                r"""
                \chapter{Vector Calculus}
                \section{Divergence and Curl}
                Gauss's divergence theorem relates flux to divergence:
                $$\oint_{\partial V} \mathbf{F} \cdot d\mathbf{A} = \int_V (\nabla \cdot \mathbf{F}) dV$$
                \begin{exercises}
                1. (a) Compute $\nabla \cdot \mathbf{r}$ for $\mathbf{r} = (x, y, z)$.
                   (b) Compute $\nabla \times \mathbf{r}$.
                \end{exercises}
                """,
                encoding="utf-8",
            )
            book_sum = ingest_textbook(
                file_path=tex_book,
                book_id="vec_calc",
                embedding_mode="technical_hash",
                base_dir=tmp_path,
            )
            self.assertEqual(book_sum.companion_problems_extracted, 1)

            # 2. Ingest an external PDF exam for the same book
            exam_pdf = tmp_path / "midterm_exam.pdf"
            surface = cairo.PDFSurface(str(exam_pdf), 595, 842)
            ctx = cairo.Context(surface)
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(11)
            lines = [
                "Name: _______________",
                "Due Date: Oct 20",
                "Problem 1. [20 pts] Verify Stokes' theorem for F = (-y, x, 0) on the unit disk.",
                "(a) Compute the line integral around the boundary circle.",
                "(b) Compute the surface integral of curl F.",
                "[Hint: Use polar coordinates r and theta.]",
            ]
            y = 60
            for ln in lines:
                ctx.move_to(50, y)
                ctx.show_text(ln)
                y += 24
            surface.show_page()
            surface.finish()

            comp_sum = ingest_problem_set(
                file_path=exam_pdf,
                book_id="vec_calc",
                source_id="midterm_2026",
                source_type="external_exam",
                base_dir=tmp_path,
            )
            self.assertEqual(comp_sum.problems_extracted, 1)
            self.assertEqual(comp_sum.total_book_problems, 2)
            self.assertEqual(comp_sum.multipart_problems_count, 1)
            self.assertEqual(comp_sum.hints_quarantined_count, 1)

            saved_problems = json.loads(Path(comp_sum.output_json_path).read_text(encoding="utf-8"))
            self.assertEqual(len(saved_problems), 2)
            validated = [CompanionProblem.model_validate(p) for p in saved_problems]
            self.assertEqual(validated[0].subparts, ["a", "b"])
            self.assertEqual(validated[1].subparts, ["a", "b"])

    def test_llm_agent_fallback_for_unnumbered_problem_sheet(self) -> None:
        def mock_boundary_handler(schema_name: str, req: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "problems": [
                    {
                        "original_number": "Problem 1",
                        "statement_md": "Prove that $\\sqrt{2}$ is irrational.",
                        "subparts": [],
                        "quarantined_hint": "Assume p/q in lowest terms.",
                        "quarantined_solution": None,
                    },
                    {
                        "original_number": "Problem 2",
                        "statement_md": "Show there are infinitely many primes.",
                        "subparts": [],
                        "quarantined_hint": None,
                        "quarantined_solution": None,
                    },
                ]
            }

        with tempfile.TemporaryDirectory() as tmp:
            cache = LLMCache(db_path=Path(tmp) / "comp.sqlite")
            with ModelLifecycleManager(backend="mock", mock_handler=mock_boundary_handler) as mgr:
                harness = AgentHarness(model_manager=mgr, cache=cache)
                segmenter = CompanionSegmenter(harness=harness, use_llm_refinement=True)
                probs = segmenter.segment_text(
                    raw_text="Prove that sqrt(2) is irrational. Also show there are infinitely many primes.",
                    book_id="num_theory",
                    source_id="unnumbered_sheet",
                )
                self.assertEqual(len(probs), 2)
                self.assertEqual(probs[0].quarantined_hint, "Assume p/q in lowest terms.")


if __name__ == "__main__":
    unittest.main()
