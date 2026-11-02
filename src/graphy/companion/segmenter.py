"""Two-Stage Problem Segmenter with strict Multi-Part Cohesion (`1(a), 1(b), 1(c)` kept together)."""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Tuple

from graphy.companion.cleaner import ProblemSetCleaner
from graphy.companion.models import CompanionProblem, ProblemSourceType
from graphy.companion.quarantine import quarantine_spoilers
from graphy.llm.agents import ProblemBoundaryAgent, ProblemBoundaryInput
from graphy.llm.harness import AgentHarness
from graphy.parser.math_normalizer import (
    extract_code_identifiers,
    extract_formulas,
    extract_technical_terms_and_symbols,
)

# Top-level problem start markers (never matches sub-parts like (a), (b), a), i))
TOP_LEVEL_PROBLEM_START_RE = re.compile(
    r"^\s*(?:"
    r"(?:\*\*|#+\s*)?(Problem|Exercise|Question|Prob\.?|Ex\.?|Q)\s*(\d+(?:\.\d+)?)\s*[\.:\)\-\*]*\s*"
    r"|"
    r"(\\item(?:\[\s*\d+[\.\)]?\s*\])?)\s+"
    r"|"
    r"(\d+(?:\.\d+)?)\s*[\.\)]\s+(?=[A-Z$(\\*\"'`])"
    r")(.*)$",
    re.IGNORECASE,
)

# Sub-part markers inside a problem: (a), (b), (c), a), b), **(a)**, Part (a), \subitem
SUBPART_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*|\*)?(?:Part\s+)?[\(\[]?([a-hj-z]|i{1,3}|iv|v|vi{0,3})[\)\]\.](\*\*|\*)?\s+(?=[A-Z$(\\*\"'`a-z0-9])"
)


class CompanionSegmenter:
    """Segments cleaned problem set text into multi-part-cohesive CompanionProblem records."""

    def __init__(self, harness: Optional[AgentHarness] = None, use_llm_refinement: bool = False) -> None:
        self.harness = harness
        self.use_llm_refinement = use_llm_refinement

    def segment_text(
        self,
        raw_text: str,
        book_id: str,
        source_id: str,
        source_type: ProblemSourceType = "external_assignment",
        chapter_or_topic: str = "General",
    ) -> List[CompanionProblem]:
        cleaned = ProblemSetCleaner.clean_raw_text(raw_text)
        if not cleaned:
            return []

        raw_blocks = self._split_top_level_problems(cleaned)

        # If no explicit numbering was found and LLM refinement is enabled, ask ProblemBoundaryAgent
        if len(raw_blocks) <= 1 and self.use_llm_refinement and self.harness is not None:
            return self._segment_via_llm_agent(
                cleaned_text=cleaned,
                book_id=book_id,
                source_id=source_id,
                source_type=source_type,
                chapter_or_topic=chapter_or_topic,
            )

        problems: List[CompanionProblem] = []
        for idx, (orig_num, raw_body) in enumerate(raw_blocks, start=1):
            stmt_clean, hint, sol = quarantine_spoilers(raw_body)
            if not stmt_clean:
                continue

            subparts, normalized_stmt = self._detect_and_format_subparts(stmt_clean)
            formulas = extract_formulas(normalized_stmt)
            terms = extract_technical_terms_and_symbols(normalized_stmt, formulas)
            code_idents = extract_code_identifiers(normalized_stmt)

            prob_hash = hashlib.sha256(f"{book_id}:{source_id}:{orig_num}:{normalized_stmt[:120]}".encode("utf-8")).hexdigest()[:8]
            problem_id = f"{book_id}_prob_{idx:03d}_{prob_hash}"

            problems.append(
                CompanionProblem(
                    problem_id=problem_id,
                    book_id=book_id,
                    source_id=source_id,
                    source_type=source_type,
                    chapter_or_topic=chapter_or_topic,
                    original_number=orig_num,
                    statement_md=normalized_stmt,
                    subparts=subparts,
                    extracted_formulas=formulas,
                    extracted_symbols_and_terms=terms,
                    code_identifiers=code_idents,
                    quarantined_hint=hint,
                    quarantined_solution=sol,
                )
            )

        return problems

    @staticmethod
    def _split_top_level_problems(cleaned_text: str) -> List[Tuple[str, str]]:
        """Split into top-level `(original_number, body)` tuples while keeping sub-parts (a, b, c) inside."""
        blocks: List[Tuple[str, List[str]]] = []
        current_num: Optional[str] = None
        current_lines: List[str] = []

        in_code_fence = False
        in_math_env = False
        item_counter = 0

        for line in cleaned_text.splitlines():
            stripped = line.strip()

            # Track code blocks
            if stripped.startswith("```") or re.match(
                r"^\\(?:begin|end)\{(?:lstlisting|verbatim|algorithm|algorithmic)\}",
                stripped,
            ):
                in_code_fence = not in_code_fence
                if current_num is not None:
                    current_lines.append(line)
                continue

            # Track multi-line LaTeX math / solution environments so internal lines aren't split
            if re.match(r"^\\begin\{(?:align|equation|gather|bmatrix|pmatrix|cases|solution|answer|hint)\*?\}", stripped):
                in_math_env = True
            elif re.match(r"^\\end\{(?:align|equation|gather|bmatrix|pmatrix|cases|solution|answer|hint)\*?\}", stripped):
                in_math_env = False

            if in_code_fence or in_math_env:
                if current_num is not None:
                    current_lines.append(line)
                continue

            # Ignore \begin{enumerate} / \end{enumerate} wrapper lines
            if re.match(r"^\\(?:begin|end)\{(?:enumerate|itemize)\}", stripped):
                continue

            m = TOP_LEVEL_PROBLEM_START_RE.match(line)
            if m:
                if current_num is not None and current_lines:
                    blocks.append((current_num, current_lines))

                if m.group(1) and m.group(2):
                    label_word = m.group(1).capitalize()
                    if label_word.lower().startswith("prob"):
                        label_word = "Problem"
                    elif label_word.lower().startswith("ex"):
                        label_word = "Exercise"
                    elif label_word.lower().startswith("q"):
                        label_word = "Question"
                    current_num = f"{label_word} {m.group(2)}"
                elif m.group(3):
                    item_counter += 1
                    current_num = f"Problem {item_counter}"
                elif m.group(4):
                    current_num = f"Problem {m.group(4)}"
                else:
                    item_counter += 1
                    current_num = f"Problem {item_counter}"

                first_line = (m.group(5) or "").strip()
                current_lines = [first_line] if first_line else []
            else:
                if current_num is not None:
                    current_lines.append(line)
                elif stripped:
                    # Preamble / shared context before Problem 1 if no problem header seen yet
                    current_num = "Problem 1"
                    current_lines = [line]

        if current_num is not None and current_lines:
            blocks.append((current_num, current_lines))

        return [(num, "\n".join(lines).strip()) for num, lines in blocks if "\n".join(lines).strip()]

    @staticmethod
    def _detect_and_format_subparts(statement_md: str) -> Tuple[List[str], str]:
        """Identify sub-part labels (`a, b, c` or `i, ii, iii`) while keeping them inside `statement_md`."""
        subparts: List[str] = []

        def _norm_subpart(m: re.Match[str]) -> str:
            lbl = m.group(1).lower()
            if lbl not in subparts:
                subparts.append(lbl)
            prefix_nl = "\n\n" if m.group(0).startswith("\n") else ""
            return f"{prefix_nl}**({lbl})** "

        formatted = SUBPART_MARKER_RE.sub(_norm_subpart, statement_md)
        return subparts, formatted.strip()

    def _segment_via_llm_agent(
        self,
        cleaned_text: str,
        book_id: str,
        source_id: str,
        source_type: ProblemSourceType,
        chapter_or_topic: str,
    ) -> List[CompanionProblem]:
        assert self.harness is not None
        agent = ProblemBoundaryAgent()
        out = self.harness.run(
            agent,
            ProblemBoundaryInput(raw_block_text=cleaned_text, source_name=source_id),
        )
        results: List[CompanionProblem] = []
        for idx, item in enumerate(out.problems, start=1):
            stmt_clean, extra_hint, extra_sol = quarantine_spoilers(item.statement_md)
            formulas = extract_formulas(stmt_clean)
            terms = extract_technical_terms_and_symbols(stmt_clean, formulas)
            code_idents = extract_code_identifiers(stmt_clean)
            prob_hash = hashlib.sha256(f"{book_id}:{source_id}:{item.original_number}".encode("utf-8")).hexdigest()[:8]
            results.append(
                CompanionProblem(
                    problem_id=f"{book_id}_prob_{idx:03d}_{prob_hash}",
                    book_id=book_id,
                    source_id=source_id,
                    source_type=source_type,
                    chapter_or_topic=chapter_or_topic,
                    original_number=item.original_number,
                    statement_md=stmt_clean,
                    subparts=item.subparts,
                    extracted_formulas=formulas,
                    extracted_symbols_and_terms=terms,
                    code_identifiers=code_idents,
                    quarantined_hint=item.quarantined_hint or extra_hint,
                    quarantined_solution=item.quarantined_solution or extra_sol,
                )
            )
        return results
