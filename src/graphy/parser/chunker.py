"""Atomic-Block & Hierarchy-Aware Semantic Chunker for Math, Physics, and CS Textbooks."""

from __future__ import annotations

import re
from typing import List, Tuple

from graphy.core.config import ChunkingConfig
from graphy.parser.math_normalizer import (
    estimate_tokens,
    extract_code_identifiers,
    extract_formulas,
    extract_technical_terms_and_symbols,
)
from graphy.parser.models import (
    ChunkType,
    DocumentParseResult,
    FigureRef,
    SectionNode,
    TextbookChunk,
)

# Regex matching atomic blocks that must NEVER be split across chunk boundaries
ATOMIC_REGION_RE = re.compile(
    r"("
    # 1. Theorem / Lemma / Proposition / Corollary immediately followed by its Proof
    r"(?:\*\*(?:Theorem|Lemma|Proposition|Corollary)[^*]*\*\*.*?(?:\n\s*\n\*\*Proof[^*]*\*\*.*?(?=\n\s*\n(?:\*\*|#|$))|\Z))"
    r"|"
    # 2. Worked Example immediately followed by its Solution/Derivation
    r"(?:\*\*Example[^*]*\*\*.*?(?:\n\s*\n\*\*(?:Solution|Resolution)[^*]*\*\*.*?(?=\n\s*\n(?:\*\*|#|$))|\Z))"
    r"|"
    # 3. Fenced or LaTeX Code / Algorithm Blocks
    r"(?:```[\w+-]*\n.*?```|\\begin\{(?:lstlisting|verbatim|algorithm|algorithmic)\}.*?\\end\{(?:lstlisting|verbatim|algorithm|algorithmic)\})"
    r"|"
    # 4. Multi-line Display Math Environments
    r"(?:\$\$.*?\$\$|\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}.*?\\end\{(?:equation|align|gather|multline|eqnarray)\*?\})"
    r")",
    re.DOTALL,
)

THEOREM_MARKER_RE = re.compile(r"\*\*(?:Theorem|Lemma|Proposition|Corollary)", re.IGNORECASE)
DEFINITION_MARKER_RE = re.compile(r"\*\*Definition|\bDefinition\s+\d+", re.IGNORECASE)
EXAMPLE_MARKER_RE = re.compile(r"\*\*Example|\bExample\s+\d+", re.IGNORECASE)
PROOF_START_RE = re.compile(r"^\s*(?:\*\*Proof|Proof[\s.:])", re.IGNORECASE)
SOLUTION_START_RE = re.compile(r"^\s*(?:\*\*Solution|Solution[\s.:])", re.IGNORECASE)
CODE_MARKER_RE = re.compile(r"```|\\begin\{(?:lstlisting|verbatim|algorithm|algorithmic)\}")
FIG_REF_RE = re.compile(r"\b(?:Figure|Fig\.)\s+(\d+(?:\.\d+)?)", re.IGNORECASE)


class SemanticTextbookChunker:
    """Splits parsed textbook sections into semantically coherent, atomic-preserving chunks."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_document(self, parsed: DocumentParseResult) -> List[TextbookChunk]:
        chunks: List[TextbookChunk] = []
        chunk_idx = 0

        for section in parsed.sections:
            atomic_blocks = self._split_into_atomic_blocks(section.content_md)
            packed_texts = self._pack_blocks_into_chunks(atomic_blocks)

            for raw_chunk_text in packed_texts:
                text_md = raw_chunk_text.strip()
                if not text_md:
                    continue

                chunk_idx += 1
                chunk_id = f"{parsed.book_id}_chunk_{chunk_idx:04d}"
                breadcrumb = self._build_breadcrumb(section)
                matched_figs = self._match_figures_for_chunk(text_md, section.figures)
                chunk_type = self._classify_chunk_type(text_md, matched_figs)

                formulas = extract_formulas(text_md)
                code_idents = extract_code_identifiers(text_md)
                terms = extract_technical_terms_and_symbols(text_md, formulas)

                embedding_text = self._build_enriched_embedding_text(
                    breadcrumb=breadcrumb,
                    chunk_type=chunk_type,
                    text_md=text_md,
                    formulas=formulas,
                    terms=terms,
                    code_idents=code_idents,
                    figures=matched_figs,
                )

                chunks.append(
                    TextbookChunk(
                        chunk_id=chunk_id,
                        book_id=parsed.book_id,
                        chunk_index=chunk_idx,
                        chapter_title=section.chapter_title,
                        section_title=section.section_title,
                        subsection_title=section.subsection_title,
                        breadcrumb=breadcrumb,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        chunk_type=chunk_type,
                        text_md=text_md,
                        embedding_text=embedding_text,
                        token_estimate=estimate_tokens(text_md),
                        extracted_formulas=formulas,
                        extracted_symbols_and_terms=terms,
                        code_identifiers=code_idents,
                        figures=matched_figs,
                    )
                )

        return chunks

    def _split_into_atomic_blocks(self, content_md: str) -> List[str]:
        """Split section text into paragraphs while keeping equations, code, and Theorem+Proof blocks intact."""
        # First protect atomic multi-paragraph spans (code blocks, display equations, theorem+proof)
        placeholders: List[Tuple[str, str]] = []

        def _protect(match: re.Match[str]) -> str:
            token = f"__ATOMIC_SPAN_{len(placeholders)}__"
            placeholders.append((token, match.group(0).strip()))
            return f"\n\n{token}\n\n"

        protected_text = ATOMIC_REGION_RE.sub(_protect, content_md)
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", protected_text) if p.strip()]

        # Restore placeholders
        restored: List[str] = []
        placeholder_map = dict(placeholders)
        for para in raw_paragraphs:
            for tok, orig in placeholder_map.items():
                if tok in para:
                    para = para.replace(tok, orig)
            restored.append(para)

        # Second pass: merge any standalone Proof with preceding Theorem/Lemma,
        # merge standalone Solution with preceding Example, and merge introductory ":" lines with Display Math
        merged: List[str] = []
        for block in restored:
            if merged and (PROOF_START_RE.match(block) or SOLUTION_START_RE.match(block)):
                merged[-1] = f"{merged[-1]}\n\n{block}"
            elif merged and (block.startswith("$$") or block.startswith("\\begin{")) and merged[-1].endswith(":"):
                merged[-1] = f"{merged[-1]}\n{block}"
            elif merged and FIGURE_CAPTION_Matches(block):
                # Bind Figure caption to the preceding paragraph if it references the figure
                merged[-1] = f"{merged[-1]}\n\n{block}"
            else:
                merged.append(block)

        return merged

    def _pack_blocks_into_chunks(self, blocks: List[str]) -> List[str]:
        """Pack atomic blocks into chunks targeting `target_tokens` without breaking atomic units."""
        if not blocks:
            return []

        packed: List[str] = []
        current_blocks: List[str] = []
        current_tokens = 0

        for block in blocks:
            b_tokens = estimate_tokens(block)

            if not current_blocks:
                current_blocks.append(block)
                current_tokens = b_tokens
                continue

            # If adding this block stays within target_tokens, or current chunk is smaller than min_tokens
            # and adding it stays within max_tokens, merge them
            if (current_tokens + b_tokens <= self.config.target_tokens) or (
                current_tokens < self.config.min_tokens
                and current_tokens + b_tokens <= self.config.max_tokens
            ):
                current_blocks.append(block)
                current_tokens += b_tokens
            else:
                packed.append("\n\n".join(current_blocks))
                current_blocks = [block]
                current_tokens = b_tokens

        if current_blocks:
            # If trailing chunk is tiny (< min_tokens // 2) and can fit in previous chunk under max_tokens, merge
            if (
                packed
                and current_tokens < (self.config.min_tokens // 2)
                and estimate_tokens(packed[-1]) + current_tokens <= self.config.max_tokens
            ):
                packed[-1] = packed[-1] + "\n\n" + "\n\n".join(current_blocks)
            else:
                packed.append("\n\n".join(current_blocks))

        return packed

    @staticmethod
    def _build_breadcrumb(section: SectionNode) -> str:
        parts = [section.chapter_title, section.section_title]
        if section.subsection_title:
            parts.append(section.subsection_title)
        return "[" + " > ".join(p for p in parts if p) + "]"

    @staticmethod
    def _match_figures_for_chunk(text_md: str, section_figures: List[FigureRef]) -> List[FigureRef]:
        if not section_figures:
            return []
        referenced_nums = {m.group(1) for m in FIG_REF_RE.finditer(text_md)}
        matched: List[FigureRef] = []
        for fig in section_figures:
            fig_num = fig.figure_id.replace("Figure", "").replace("Fig.", "").strip()
            if fig_num in referenced_nums or fig.caption in text_md:
                matched.append(fig)
        return matched

    @staticmethod
    def _classify_chunk_type(text_md: str, figures: List[FigureRef]) -> ChunkType:
        if CODE_MARKER_RE.search(text_md):
            return "algorithm_code"
        if THEOREM_MARKER_RE.search(text_md):
            return "theorem_proof"
        if EXAMPLE_MARKER_RE.search(text_md):
            return "worked_example"
        if DEFINITION_MARKER_RE.search(text_md):
            return "definition"
        if figures:
            return "figure_concept"
        return "concept"

    @staticmethod
    def _build_enriched_embedding_text(
        breadcrumb: str,
        chunk_type: ChunkType,
        text_md: str,
        formulas: List[str],
        terms: List[str],
        code_idents: List[str],
        figures: List[FigureRef],
    ) -> str:
        """Construct rich semantic representation for technical embedding accuracy."""
        header_lines = [f"{breadcrumb} (Type: {chunk_type})"]
        if terms:
            header_lines.append("Concepts & Mathematical Terms: " + ", ".join(terms[:15]))
        if code_idents:
            header_lines.append("Code & Algorithm Identifiers: " + ", ".join(code_idents))
        if figures:
            fig_desc = "; ".join(f"{f.figure_id}: {f.caption}" for f in figures)
            header_lines.append("Figures: " + fig_desc)
        if formulas:
            header_lines.append("Key Equations: " + " ; ".join(formulas[:8]))

        return "\n".join(header_lines) + "\n\n" + text_md


def FIGURE_CAPTION_Matches(block: str) -> bool:
    return bool(re.match(r"^(?:Figure|Fig\.)\s+\d+", block.strip(), re.IGNORECASE))
