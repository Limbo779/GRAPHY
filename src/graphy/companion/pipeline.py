"""GRAPHY Companion Automated Pipeline: Ingests textbook exercises & external problem sets."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional

from graphy.companion.models import (
    CompanionIngestionSummary,
    CompanionProblem,
    ProblemSourceType,
)
from graphy.companion.segmenter import CompanionSegmenter
from graphy.core.config import GraphyConfig, load_config
from graphy.llm.harness import AgentHarness
from graphy.parser.models import ExtractedExerciseBlock


def _read_problem_source_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        if not shutil.which("pdftotext"):
            raise RuntimeError("Poppler `pdftotext` is required to extract PDF problem sets.")
        proc = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(file_path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout
    if suffix in (".tex", ".latex", ".md", ".markdown", ".txt"):
        return file_path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported problem set file format: {suffix}")


def _save_and_merge_problems(
    new_problems: List[CompanionProblem],
    problems_json_path: Path,
) -> List[CompanionProblem]:
    existing: List[CompanionProblem] = []
    if problems_json_path.is_file():
        raw_list = json.loads(problems_json_path.read_text(encoding="utf-8"))
        existing = [CompanionProblem.model_validate(item) for item in raw_list]

    by_key = {(p.source_id, p.original_number): p for p in existing}
    for p in new_problems:
        by_key[(p.source_id, p.original_number)] = p

    merged = list(by_key.values())
    problems_json_path.parent.mkdir(parents=True, exist_ok=True)
    problems_json_path.write_text(
        json.dumps([p.model_dump(mode="json") for p in merged], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return merged


def ingest_problem_set(
    file_path: Path | str,
    book_id: str,
    source_id: Optional[str] = None,
    source_type: ProblemSourceType = "external_assignment",
    chapter_or_topic: str = "External Problem Set",
    config: Optional[GraphyConfig] = None,
    harness: Optional[AgentHarness] = None,
    use_llm_refinement: bool = False,
    base_dir: Optional[Path] = None,
) -> CompanionIngestionSummary:
    """Run GRAPHY Companion on an external PDF, LaTeX, or Markdown problem set / exam."""
    cfg = config or load_config()
    root = base_dir or Path(".")
    cfg.ensure_Runtime_dirs(base_dir=root)

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Problem set file not found: {path}")

    src_id = source_id or path.stem
    raw_text = _read_problem_source_file(path)

    segmenter = CompanionSegmenter(harness=harness, use_llm_refinement=use_llm_refinement)
    extracted = segmenter.segment_text(
        raw_text=raw_text,
        book_id=book_id,
        source_id=src_id,
        source_type=source_type,
        chapter_or_topic=chapter_or_topic,
    )

    problems_json_path = root / cfg.system.data_dir / "processed" / f"{book_id}_problems.json"
    merged = _save_and_merge_problems(extracted, problems_json_path)

    return CompanionIngestionSummary(
        book_id=book_id,
        source_id=src_id,
        source_type=source_type,
        problems_extracted=len(extracted),
        total_book_problems=len(merged),
        multipart_problems_count=sum(1 for p in extracted if p.subparts),
        hints_quarantined_count=sum(1 for p in extracted if p.quarantined_hint),
        solutions_quarantined_count=sum(1 for p in extracted if p.quarantined_solution),
        output_json_path=str(problems_json_path),
    )


def ingest_routed_textbook_exercises(
    book_id: str,
    config: Optional[GraphyConfig] = None,
    harness: Optional[AgentHarness] = None,
    use_llm_refinement: bool = False,
    base_dir: Optional[Path] = None,
) -> CompanionIngestionSummary:
    """Process textbook exercise blocks previously separated into `<book_id>_textbook_exercises.json`."""
    cfg = config or load_config()
    root = base_dir or Path(".")
    processed_dir = root / cfg.system.data_dir / "processed"
    routed_path = processed_dir / f"{book_id}_textbook_exercises.json"

    if not routed_path.is_file():
        raise FileNotFoundError(f"No routed textbook exercises file found at: {routed_path}")

    raw_blocks = json.loads(routed_path.read_text(encoding="utf-8"))
    exercise_blocks = [ExtractedExerciseBlock.model_validate(b) for b in raw_blocks]

    segmenter = CompanionSegmenter(harness=harness, use_llm_refinement=use_llm_refinement)
    all_extracted: List[CompanionProblem] = []

    for blk_idx, blk in enumerate(exercise_blocks, start=1):
        src_id = f"{book_id}_exercises_{blk_idx:02d}"
        topic = f"{blk.chapter_title} > {blk.section_title}"
        probs = segmenter.segment_text(
            raw_text=blk.raw_text,
            book_id=book_id,
            source_id=src_id,
            source_type="textbook_exercise",
            chapter_or_topic=topic,
        )
        all_extracted.extend(probs)

    problems_json_path = processed_dir / f"{book_id}_problems.json"
    merged = _save_and_merge_problems(all_extracted, problems_json_path)

    return CompanionIngestionSummary(
        book_id=book_id,
        source_id=f"{book_id}_textbook_exercises",
        source_type="textbook_exercise",
        problems_extracted=len(all_extracted),
        total_book_problems=len(merged),
        multipart_problems_count=sum(1 for p in all_extracted if p.subparts),
        hints_quarantined_count=sum(1 for p in all_extracted if p.quarantined_hint),
        solutions_quarantined_count=sum(1 for p in all_extracted if p.quarantined_solution),
        output_json_path=str(problems_json_path),
    )
