"""Handwritten Solution Image -> LaTeX/Markdown OCR Preview & User Confirmation Pipeline."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Optional

from graphy.llm.agents import HandwritingOCRAgent, HandwritingOCRInput
from graphy.llm.harness import AgentHarness
from graphy.parser.math_normalizer import extract_formulas, normalize_math_unicode
from graphy.tutor.models import OCRPreviewRecord


def transcribe_handwritten_solution(
    problem_id: str,
    problem_statement: str,
    image_path: Path | str,
    harness: Optional[AgentHarness] = None,
    use_vlm: bool = True,
) -> OCRPreviewRecord:
    """Transcribe a handwritten math/physics/CS solution image into an editable LaTeX/Markdown preview."""
    img = Path(image_path)
    if not img.is_file():
        raise FileNotFoundError(f"Handwritten solution image not found: {img}")

    if use_vlm and harness is not None:
        agent = HandwritingOCRAgent()
        vlm_out = harness.run(
            agent,
            HandwritingOCRInput(
                problem_statement=problem_statement,
                image_path=str(img),
            ),
        )
        norm_md = normalize_math_unicode(vlm_out.transcribed_latex_md)
        equations = vlm_out.detected_equations or extract_formulas(norm_md)
        return OCRPreviewRecord(
            problem_id=problem_id,
            image_path=str(img),
            raw_ocr_latex_md=norm_md,
            edited_latex_md=norm_md,
            detected_equations=equations,
            confidence=vlm_out.confidence,
            ambiguity_notes=vlm_out.ambiguity_notes,
            confirmed_by_user=False,
        )

    # Local Tesseract OCR fallback when VLM is disabled
    raw_txt = ""
    if shutil.which("tesseract"):
        proc = subprocess.run(
            ["tesseract", str(img), "stdout", "-l", "eng"],
            capture_output=True,
            text=True,
            check=False,
        )
        raw_txt = proc.stdout.strip()

    if not raw_txt:
        raw_txt = "% [OCR could not detect text; please type or edit your LaTeX solution below]"

    norm_md = normalize_math_unicode(raw_txt)
    equations = extract_formulas(norm_md)
    return OCRPreviewRecord(
        problem_id=problem_id,
        image_path=str(img),
        raw_ocr_latex_md=norm_md,
        edited_latex_md=norm_md,
        detected_equations=equations,
        confidence=0.80,
        ambiguity_notes="Please review mathematical symbols and subscripts before submitting.",
        confirmed_by_user=False,
    )


def confirm_ocr_preview(
    preview: OCRPreviewRecord,
    user_edited_latex_md: Optional[str] = None,
) -> OCRPreviewRecord:
    """Mark an OCR preview as reviewed/confirmed by the user, applying any edits."""
    if user_edited_latex_md is not None and user_edited_latex_md.strip():
        preview.edited_latex_md = normalize_math_unicode(user_edited_latex_md.strip())
        preview.detected_equations = extract_formulas(preview.edited_latex_md)
    preview.confirmed_by_user = True
    return preview
