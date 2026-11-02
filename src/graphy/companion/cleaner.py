"""GRAPHY Companion Noise Cleaner: strips exam/homework boilerplate, point values, and formatting noise."""

from __future__ import annotations

import re
from typing import List

from graphy.parser.math_normalizer import normalize_math_unicode

# Boilerplate lines commonly found in university course assignments, exams, and worksheets
BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^\s*(?:name|student\s*id|roll\s*no|section|signature|date)\s*:\s*[_\.\s]*$", re.IGNORECASE),
    re.compile(r"^\s*(?:due\s*date|due\s*on|due|deadline|hand\s*in|submit\s*by)\s*:.*$", re.IGNORECASE),
    re.compile(r"^\s*(?:instructor|professor|lecturer|ta|teaching\s*assistant)\s*:.*$", re.IGNORECASE),
    re.compile(r"^\s*(?:time\s*limit|duration|total\s*points|max\s*marks|maximum\s*score)\s*:.*$", re.IGNORECASE),
    re.compile(r"^\s*do\s+not\s+open\s+this\s+(?:exam|test|booklet).*$", re.IGNORECASE),
    re.compile(r"^\s*show\s+all\s+(?:your\s+)?work\s+to\s+receive\s+full\s+credit.*$", re.IGNORECASE),
    re.compile(r"^\s*write\s+your\s+solutions?\s+in\s+the\s+spaces?\s+provided.*$", re.IGNORECASE),
    re.compile(r"^\s*honor\s+code\s*:.*$", re.IGNORECASE),
    re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*[_=\-]{8,}\s*$"),
]

# Point value badges like "[15 pts]", "(10 points)", "[20 marks]", "(5 pt each)"
POINT_BADGE_RE = re.compile(
    r"[\(\[]\s*\d+(?:\.\d+)?\s*(?:pts?\.?|points?|marks?)(?:\s+each|\s+total)?\s*[\)\]]\s*",
    re.IGNORECASE,
)

# LaTeX exam spacing / answer box commands
LATEX_NOISE_RE = re.compile(
    r"\\(?:vspace|hspace)\*?\{[^}]+\}|\\makeemptybox\{[^}]+\}|\\fillin(?:\[[^\]]*\])?",
    re.IGNORECASE,
)


class ProblemSetCleaner:
    """Cleans noisy homework, exam, and textbook exercise text before problem segmentation."""

    @classmethod
    def clean_raw_text(cls, raw_text: str) -> str:
        text = normalize_math_unicode(raw_text)
        text = LATEX_NOISE_RE.sub("", text)

        cleaned_lines: List[str] = []
        in_code_fence = False

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```") or re.match(
                r"^\\(?:begin|end)\{(?:lstlisting|verbatim|algorithm|algorithmic)\}",
                stripped,
            ):
                in_code_fence = not in_code_fence
                cleaned_lines.append(line)
                continue

            if in_code_fence:
                cleaned_lines.append(line)
                continue

            if any(pat.match(stripped) for pat in BOILERPLATE_LINE_PATTERNS):
                continue

            # Remove point value badges from problem/subpart headers
            line_no_pts = POINT_BADGE_RE.sub("", line)
            cleaned_lines.append(line_no_pts.rstrip())

        joined = "\n".join(cleaned_lines)
        # Collapse 3+ consecutive blank lines into 2
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return joined.strip()
