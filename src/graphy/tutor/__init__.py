"""Bare-minimum concept distillation, Handwritten Math-OCR preview, and solution analysis."""

from graphy.tutor.handwriting_ocr import (
    confirm_ocr_preview,
    transcribe_handwritten_solution,
)
from graphy.tutor.models import (
    OCRPreviewRecord,
    StudyBatchBriefing,
    StudyBatchSession,
    StudyEvaluationReport,
    StudyProblemView,
)
from graphy.tutor.session_manager import StudySessionManager

__all__ = [
    "OCRPreviewRecord",
    "StudyBatchBriefing",
    "StudyBatchSession",
    "StudyEvaluationReport",
    "StudyProblemView",
    "StudySessionManager",
    "confirm_ocr_preview",
    "transcribe_handwritten_solution",
]
