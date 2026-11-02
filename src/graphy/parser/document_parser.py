"""Multi-format Technical Textbook Parser (Digital PDF, LaTeX, Markdown) for Math, Physics, and CS."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

from graphy.parser.math_normalizer import normalize_math_unicode
from graphy.parser.models import (
    DocumentParseResult,
    ExtractedExerciseBlock,
    FigureRef,
    SectionNode,
)

CHAPTER_HEADING_RE = re.compile(
    r"^(?:#\s+|\\chapter\*?\{|(?:CHAPTER|Chapter)\s+\d+[\s.:—\-]*)(.+?)\}?$",
    re.IGNORECASE,
)
SECTION_HEADING_RE = re.compile(
    r"^(?:##\s+|\\section\*?\{|(\d+\.\d+)\s+[A-Z])(.*)$"
)
SUBSECTION_HEADING_RE = re.compile(
    r"^(?:###\s+|\\subsection\*?\{|(\d+\.\d+\.\d+)\s+[A-Z])(.*)$"
)
EXERCISE_HEADING_RE = re.compile(
    r"^(?:#{1,4}\s*|\\(?:section|subsection)\*?\{|\\begin\{)?\s*"
    r"(Exercises|Problems|Review\s+Problems|Problem\s+Set|Practice\s+Problems|Chapter\s+\d+\s+Exercises)"
    r"(?:\s+for\s+[\w.\s]+)?\}?\s*:?\s*$",
    re.IGNORECASE,
)
TOC_LEADER_LINE_RE = re.compile(r"^.{4,80}\s*(?:\.\s*){4,}\s*\d+\s*$")
FIGURE_CAPTION_RE = re.compile(
    r"^(?:Figure|Fig\.)\s+(\d+(?:\.\d+)?)[\s:.\-]+(.+)$",
    re.IGNORECASE,
)


class TechnicalDocumentParser:
    """Parses Digital PDFs, LaTeX (.tex), and Markdown (.md) textbooks into structured sections & exercise blocks."""

    def __init__(self, figures_output_dir: Optional[Path] = None) -> None:
        self.figures_output_dir = figures_output_dir

    def parse(self, file_path: Path | str, book_id: str) -> DocumentParseResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Textbook file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages_text, extracted_figs = self._extract_pdf_pages_and_figures(path, book_id)
            return self._parse_paged_text(
                pages_text=pages_text,
                book_id=book_id,
                source_path=str(path),
                source_format="pdf",
                extracted_figs=extracted_figs,
            )
        if suffix in (".tex", ".latex"):
            raw = path.read_text(encoding="utf-8", errors="ignore")
            converted = self._preprocess_latex_to_hybrid_md(raw)
            return self._parse_paged_text(
                pages_text=[converted],
                book_id=book_id,
                source_path=str(path),
                source_format="tex",
                extracted_figs={},
            )
        if suffix in (".md", ".markdown", ".txt"):
            raw = path.read_text(encoding="utf-8", errors="ignore")
            return self._parse_paged_text(
                pages_text=[raw],
                book_id=book_id,
                source_path=str(path),
                source_format="md",
                extracted_figs={},
            )

        raise ValueError(f"Unsupported textbook file extension: {suffix}")

    def _extract_pdf_pages_and_figures(
        self,
        pdf_path: Path,
        book_id: str,
    ) -> Tuple[List[str], Dict[int, List[str]]]:
        """Extract per-page text from digital PDF using Poppler pdftotext (with OCR fallback for image pages)."""
        if not shutil.which("pdftotext"):
            raise RuntimeError("Poppler `pdftotext` is required for PDF textbook parsing.")

        proc = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        raw_pages = proc.stdout.split("\f")
        if raw_pages and not raw_pages[-1].strip():
            raw_pages.pop()

        # Check for pages with empty/garbled extraction and apply selective OCR fallback if pdftoppm+tesseract exist
        pages: List[str] = []
        for idx, page_txt in enumerate(raw_pages, start=1):
            cleaned = page_txt.strip()
            if len(cleaned) < 20 and shutil.which("pdftoppm") and shutil.which("tesseract"):
                ocr_txt = self._ocr_pdf_page(pdf_path, idx)
                pages.append(ocr_txt if len(ocr_txt.strip()) > len(cleaned) else page_txt)
            else:
                pages.append(page_txt)

        extracted_figs: Dict[int, List[str]] = {}
        if self.figures_output_dir and shutil.which("pdfimages"):
            fig_dir = self.figures_output_dir / book_id
            fig_dir.mkdir(parents=True, exist_ok=True)
            prefix = fig_dir / "fig"
            subprocess.run(
                ["pdfimages", "-png", "-p", str(pdf_path), str(prefix)],
                capture_output=True,
                check=False,
            )
            for img_file in sorted(fig_dir.glob("fig-*.png")):
                # pdfimages -p names files fig-<page>-<num>.png
                parts = img_file.stem.split("-")
                if len(parts) >= 3 and parts[1].isdigit():
                    pno = int(parts[1])
                    extracted_figs.setdefault(pno, []).append(str(img_file))

        return pages, extracted_figs

    @staticmethod
    def _ocr_pdf_page(pdf_path: Path, page_num: int) -> str:
        """Selective OCR fallback for scanned or vector-bitmap pages in a PDF."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            img_prefix = Path(tmp) / "page"
            subprocess.run(
                [
                    "pdftoppm",
                    "-png",
                    "-r",
                    "200",
                    "-f",
                    str(page_num),
                    "-l",
                    str(page_num),
                    str(pdf_path),
                    str(img_prefix),
                ],
                capture_output=True,
                check=False,
            )
            imgs = list(Path(tmp).glob("page*.png"))
            if not imgs:
                return ""
            proc = subprocess.run(
                ["tesseract", str(imgs[0]), "stdout", "-l", "eng"],
                capture_output=True,
                text=True,
                check=False,
            )
            return proc.stdout

    @staticmethod
    def _strip_headers_footers_and_toc(pages: List[str]) -> List[str]:
        """Remove repeating page headers/footers, isolated page numbers, and TOC dot-leader lines."""
        if len(pages) <= 2:
            cleaned_pages: List[str] = []
            for p in pages:
                lines = [
                    ln
                    for ln in p.splitlines()
                    if not TOC_LEADER_LINE_RE.match(ln.strip())
                ]
                cleaned_pages.append("\n".join(lines))
            return cleaned_pages

        top_Bottom_counter: Counter[str] = Counter()
        for p in pages:
            non_empty = [ln.strip() for ln in p.splitlines() if ln.strip()]
            if non_empty:
                top_Bottom_counter[re.sub(r"\d+", "#", non_empty[0])] += 1
                if len(non_empty) > 1:
                    top_Bottom_counter[re.sub(r"\d+", "#", non_empty[-1])] += 1

        threshold = max(3, int(len(pages) * 0.35))
        recurring_patterns = {k for k, cnt in top_Bottom_counter.items() if cnt >= threshold}

        cleaned_pages = []
        for p in pages:
            raw_lines = p.splitlines()
            filtered: List[str] = []
            for i, ln in enumerate(raw_lines):
                stripped = ln.strip()
                if not stripped:
                    filtered.append("")
                    continue
                if TOC_LEADER_LINE_RE.match(stripped):
                    continue
                if (i == 0 or i == len(raw_lines) - 1) and stripped.isdigit():
                    continue
                norm_pat = re.sub(r"\d+", "#", stripped)
                if (i <= 1 or i >= len(raw_lines) - 2) and norm_pat in recurring_patterns:
                    continue
                filtered.append(ln)
            cleaned_pages.append("\n".join(filtered))

        return cleaned_pages

    @staticmethod
    def _preprocess_latex_to_hybrid_md(latex_text: str) -> str:
        """Convert structural LaTeX headings and theorem/example/exercise environments into canonical markers."""
        txt = latex_text
        # Strip preamble before \begin{document} if present
        if r"\begin{document}" in txt:
            _, _, txt = txt.partition(r"\begin{document}")
        if r"\end{document}" in txt:
            txt, _, _ = txt.partition(r"\end{document}")

        # Headings
        txt = re.sub(r"\\chapter\*?\{([^}]+)\}", r"\n# Chapter: \1\n", txt)
        txt = re.sub(r"\\section\*?\{([^}]+)\}", r"\n## \1\n", txt)
        txt = re.sub(r"\\subsection\*?\{([^}]+)\}", r"\n### \1\n", txt)

        # Theorem-like environments -> structured bold markers while keeping internal LaTeX intact
        for env, label in [
            ("definition", "Definition"),
            ("theorem", "Theorem"),
            ("lemma", "Lemma"),
            ("proposition", "Proposition"),
            ("corollary", "Corollary"),
            ("example", "Example"),
            ("proof", "Proof"),
        ]:
            pattern = re.compile(
                rf"\\begin\{{{env}\}}(?:\[([^\]]+)\])?(.*?)\\end\{{{env}\}}",
                re.DOTALL | re.IGNORECASE,
            )

            def _repl(m: re.Match[str], lbl: str = label) -> str:
                opt_title = f" ({m.group(1).strip()})" if m.group(1) else ""
                body = m.group(2).strip()
                return f"\n**{lbl}{opt_title}.** {body}\n"

            txt = pattern.sub(_repl, txt)

        # Exercises environment
        txt = re.sub(
            r"\\begin\{(?:exercises|problems)\}(.*?)\\end\{(?:exercises|problems)\}",
            r"\n## Exercises\n\1\n",
            txt,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return txt

    def _parse_paged_text(
        self,
        pages_text: List[str],
        book_id: str,
        source_path: str,
        source_format: str,
        extracted_figs: Dict[int, List[str]],
    ) -> DocumentParseResult:
        cleaned_pages = self._strip_headers_footers_and_toc(pages_text)

        current_chapter = "Chapter 1"
        current_section = "Introduction"
        current_subsection = ""
        in_exercises = False
        in_code_fence = False
        exercise_heading = "Exercises"

        sections: List[SectionNode] = []
        exercise_blocks: List[ExtractedExerciseBlock] = []
        all_figures: List[FigureRef] = []

        sec_lines: List[str] = []
        sec_figs: List[FigureRef] = []
        sec_page_start = 1
        ex_lines: List[str] = []
        ex_page_start = 1

        def flush_section(end_page: int) -> None:
            nonlocal sec_lines, sec_figs
            body = "\n".join(sec_lines).strip()
            if body:
                sections.append(
                    SectionNode(
                        chapter_title=current_chapter,
                        section_title=current_section,
                        subsection_title=current_subsection,
                        page_start=sec_page_start,
                        page_end=end_page,
                        content_md=body,
                        figures=list(sec_figs),
                    )
                )
            sec_lines = []
            sec_figs = []

        def flush_exercises(end_page: int) -> None:
            nonlocal ex_lines
            body = "\n".join(ex_lines).strip()
            if body:
                exercise_blocks.append(
                    ExtractedExerciseBlock(
                        book_id=book_id,
                        chapter_title=current_chapter,
                        section_title=current_section,
                        heading=exercise_heading,
                        raw_text=body,
                        page_start=ex_page_start,
                        page_end=end_page,
                    )
                )
            ex_lines = []

        for page_idx, raw_page in enumerate(cleaned_pages, start=1):
            norm_page = normalize_math_unicode(raw_page)
            page_img_paths = list(extracted_figs.get(page_idx, []))
            img_cursor = 0

            for line in norm_page.splitlines():
                stripped = line.strip()

                # Track code fences so Python/shell `# comments` are never mistaken for headings
                if stripped.startswith("```") or re.match(
                    r"^\\(?:begin|end)\{(?:lstlisting|verbatim|algorithm|algorithmic)\}",
                    stripped,
                ):
                    in_code_fence = not in_code_fence
                    if in_exercises:
                        ex_lines.append(line)
                    else:
                        sec_lines.append(line)
                    continue

                if in_code_fence:
                    if in_exercises:
                        ex_lines.append(line)
                    else:
                        sec_lines.append(line)
                    continue

                if re.match(r"^\\end\{(?:exercises|problems)\}", stripped, re.IGNORECASE):
                    if in_exercises:
                        flush_exercises(page_idx)
                        in_exercises = False
                    continue

                # Check if this line starts an Exercise / Problem Set block
                ex_match = EXERCISE_HEADING_RE.match(stripped)
                if ex_match:
                    if in_exercises:
                        flush_exercises(page_idx)
                    else:
                        flush_section(page_idx)
                    in_exercises = True
                    exercise_heading = ex_match.group(1).strip()
                    ex_page_start = page_idx
                    continue

                # Check Chapter heading
                chap_match = CHAPTER_HEADING_RE.match(stripped)
                if chap_match and not stripped.startswith("##"):
                    if in_exercises:
                        flush_exercises(page_idx)
                        in_exercises = False
                    else:
                        flush_section(page_idx)
                    title_clean = chap_match.group(1).strip().rstrip("}")
                    current_chapter = (
                        title_clean
                        if title_clean.lower().startswith("chapter")
                        else f"Chapter: {title_clean}"
                    )
                    current_section = "Overview"
                    current_subsection = ""
                    sec_page_start = page_idx
                    continue

                # Check Subsection heading before Section heading
                sub_match = SUBSECTION_HEADING_RE.match(stripped)
                if sub_match:
                    if in_exercises:
                        flush_exercises(page_idx)
                        in_exercises = False
                    else:
                        flush_section(page_idx)
                    prefix = f"{sub_match.group(1)} " if sub_match.group(1) else ""
                    current_subsection = (prefix + sub_match.group(2)).strip().rstrip("}")
                    sec_page_start = page_idx
                    continue

                sec_match = SECTION_HEADING_RE.match(stripped)
                if sec_match and not stripped.startswith("###"):
                    heading_Candidate = (
                        (f"{sec_match.group(1)} " if sec_match.group(1) else "")
                        + sec_match.group(2)
                    ).strip().rstrip("}")
                    # Check if this section heading is actually an Exercises section
                    if EXERCISE_HEADING_RE.match(heading_Candidate):
                        if in_exercises:
                            flush_exercises(page_idx)
                        else:
                            flush_section(page_idx)
                        in_exercises = True
                        exercise_heading = heading_Candidate
                        ex_page_start = page_idx
                        continue

                    if in_exercises:
                        flush_exercises(page_idx)
                        in_exercises = False
                    else:
                        flush_section(page_idx)
                    current_section = heading_Candidate
                    current_subsection = ""
                    sec_page_start = page_idx
                    continue

                # Detect Figure captions
                fig_match = FIGURE_CAPTION_RE.match(stripped)
                if fig_match:
                    fig_id = f"Figure {fig_match.group(1)}"
                    caption_txt = fig_match.group(2).strip()
                    img_path = page_img_paths[img_cursor] if img_cursor < len(page_img_paths) else None
                    if img_path:
                        img_cursor += 1
                    fref = FigureRef(
                        figure_id=fig_id,
                        caption=caption_txt,
                        image_path=img_path,
                        page_number=page_idx,
                    )
                    all_figures.append(fref)
                    if not in_exercises:
                        sec_figs.append(fref)

                if in_exercises:
                    ex_lines.append(line)
                else:
                    sec_lines.append(line)

        total_p = max(1, len(cleaned_pages))
        if in_exercises:
            flush_exercises(total_p)
        else:
            flush_section(total_p)

        return DocumentParseResult(
            book_id=book_id,
            source_path=source_path,
            source_format=source_format,  # type: ignore[arg-type]
            total_pages=total_p,
            sections=sections,
            exercise_blocks=exercise_blocks,
            figures=all_figures,
        )
