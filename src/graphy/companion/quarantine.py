"""Spoiler, Hint, and Solution Quarantine for GRAPHY Companion."""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Matches LaTeX or Markdown solution blocks attached to a problem
SOLUTION_BLOCK_RE = re.compile(
    r"(?:"
    r"\\begin\{(?:solution|answer|sol)\}(.*?)\\end\{(?:solution|answer|sol)\}"
    r"|"
    r"(?:^|\n)\s*(?:\*\*(?:Solution|Answer|Worked\s+Solution)[\s.:]*\*\*|(?:Solution|Answer)\s*:)\s*(.+?)(?=\n\s*\n(?:\*\*(?:Hint|Problem)|$)|\Z)"
    r")",
    re.DOTALL | re.IGNORECASE,
)

# Matches inline or block hints like "[Hint: Use Stokes' theorem.]" or "(Hint: ...)" or "**Hint:** ..."
HINT_BLOCK_RE = re.compile(
    r"(?:"
    r"\\begin\{hint\}(.*?)\\end\{hint\}"
    r"|"
    r"[\[\(]\s*Hint\s*:\s*([^\]\)]+)[\]\)]"
    r"|"
    r"(?:^|\n)\s*(?:\*\*Hint[\s.:]*\*\*|\*Hint[\s.:]*\*|Hint\s*:)\s*([^\n]+)"
    r")",
    re.DOTALL | re.IGNORECASE,
)


def quarantine_spoilers(statement_text: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Strip hints and solutions from `statement_text` and return `(clean_statement, hint, solution)`."""
    hints: list[str] = []
    solutions: list[str] = []

    def _extract_solution(m: re.Match[str]) -> str:
        sol_body = (m.group(1) or m.group(2) or "").strip()
        if sol_body:
            solutions.append(sol_body)
        return ""

    text_no_sol = SOLUTION_BLOCK_RE.sub(_extract_solution, statement_text)

    def _extract_hint(m: re.Match[str]) -> str:
        hint_body = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        if hint_body:
            hints.append(hint_body)
        return ""

    clean_stmt = HINT_BLOCK_RE.sub(_extract_hint, text_no_sol)
    clean_stmt = re.sub(r"\n{3,}", "\n\n", clean_stmt).strip()

    quarantined_hint = " | ".join(hints) if hints else None
    quarantined_solution = "\n\n".join(solutions) if solutions else None
    return clean_stmt, quarantined_hint, quarantined_solution
