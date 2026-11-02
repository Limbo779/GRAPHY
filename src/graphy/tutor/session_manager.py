"""Self-Discovery Study Loop Manager: 4-Problem Selector, Bare-Minimum Distiller, and Solution Analyzer."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from graphy.companion.models import CompanionProblem
from graphy.core.config import GraphyConfig, load_config
from graphy.embeddings.engine import EmbeddingMode
from graphy.embeddings.vector_store import EmbeddedVectorStore
from graphy.graph.graphar_store import GraphArStore
from graphy.llm.agents import (
    ConceptDistillerAgent,
    ConceptDistillerInput,
    SolutionAuditInput,
    SolutionAuditorAgent,
    TopicCoachingItem,
)
from graphy.llm.cache import LLMCache
from graphy.llm.harness import AgentHarness
from graphy.llm.model_manager import ModelLifecycleManager
from graphy.scoring.mastery_updater import (
    load_saved_scores,
    record_mastery_and_recompute,
)
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


class StudySessionManager:
    """Orchestrates the 4-problem self-discovery loop, OCR previews, solution grading, and progression."""

    def __init__(
        self,
        book_id: str,
        config: Optional[GraphyConfig] = None,
        model_manager: Optional[ModelLifecycleManager] = None,
        embedding_mode: EmbeddingMode = "hybrid",
        base_dir: Optional[Path] = None,
    ) -> None:
        self.book_id = book_id
        self.config = config or load_config()
        self.root = base_dir or Path(".")
        self.config.ensure_Runtime_dirs(base_dir=self.root)
        self.model_manager = model_manager
        self.embedding_mode: EmbeddingMode = embedding_mode

        self.data_root = self.root / self.config.system.data_dir
        self.sessions_dir = self.data_root / "sessions" / book_id
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.cache = LLMCache(db_path=self.data_root / "processed" / "llm_cache.sqlite")

    def start_next_batch(self, batch_size: Optional[int] = None) -> Optional[StudyBatchSession]:
        """Select top 4 problems by foundationness score and distill bare-minimum vs self-discovery concepts."""
        k_probs = batch_size or self.config.scoring.batch_problem_count
        _, _, problem_foundation_scores = load_saved_scores(
            book_id=self.book_id,
            base_dir=self.root,
            config=self.config,
        )

        # Pick top unsolved problems (pending or needs_remediation) already sorted by descending foundationness
        active_ranked = [p for p in problem_foundation_scores if p.status != "solved"]
        if not active_ranked:
            return None

        selected_pf = active_ranked[:k_probs]

        # Load full CompanionProblem records
        problems_file = self.data_root / "processed" / f"{self.book_id}_problems.json"
        raw_probs = json.loads(problems_file.read_text(encoding="utf-8"))
        prob_map: Dict[str, CompanionProblem] = {
            p["problem_id"]: CompanionProblem.model_validate(p) for p in raw_probs
        }

        # Load associated textbook chunks from EmbeddedVectorStore
        vstore = EmbeddedVectorStore(
            book_id=self.book_id,
            vectors_root=self.data_root / "vectors",
        )

        study_problems: List[StudyProblemView] = []
        target_chunk_ids: List[str] = []
        chunk_excerpts: List[str] = []

        for pf in selected_pf:
            cp = prob_map[pf.problem_id]
            study_problems.append(
                StudyProblemView(
                    problem_id=cp.problem_id,
                    original_number=cp.original_number,
                    source_id=cp.source_id,
                    statement_md=cp.statement_md,
                    subparts=cp.subparts,
                    figures=cp.figures,
                    net_foundationness_score=pf.net_foundationness_score,
                    matched_chunk_ids=pf.primary_chunk_ids,
                )
            )
            for cid in pf.primary_chunk_ids:
                if cid not in target_chunk_ids:
                    target_chunk_ids.append(cid)
                    chunk_obj = vstore.get_chunk(cid)
                    if chunk_obj is not None:
                        chunk_excerpts.append(f"{chunk_obj.breadcrumb}\n{chunk_obj.text_md}")

        own_mgr = self.model_manager is None
        mgr = self.model_manager or ModelLifecycleManager(config=self.config, backend="llama_server")
        try:
            harness = AgentHarness(model_manager=mgr, cache=self.cache)
            distiller = ConceptDistillerAgent()
            distilled = harness.run(
                distiller,
                ConceptDistillerInput(
                    problem_statements=[sp.statement_md for sp in study_problems],
                    chunk_excerpts=chunk_excerpts,
                ),
            )
        finally:
            if own_mgr:
                mgr.release_all()

        sid_seed = f"{self.book_id}:{','.join(p.problem_id for p in study_problems)}:{int(time.time() * 1000)}"
        session_id = "ses_" + hashlib.sha256(sid_seed.encode("utf-8")).hexdigest()[:10]

        session = StudyBatchSession(
            session_id=session_id,
            book_id=self.book_id,
            status="active",
            problems=study_problems,
            target_chunk_ids=target_chunk_ids,
            bare_minimum_topics=distilled.bare_minimum_topics,
            hidden_self_discovery_topics=distilled.self_discovery_topics,
        )
        self.save_session(session)
        return session

    def save_session(self, session: StudyBatchSession) -> Path:
        out_path = self.sessions_dir / f"{session.session_id}.json"
        out_path.write_text(
            json.dumps(session.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out_path

    def load_session(self, session_id: str) -> StudyBatchSession:
        path = self.sessions_dir / f"{session_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Study session not found: {path}")
        return StudyBatchSession.model_validate_json(path.read_text(encoding="utf-8"))

    def preview_handwritten_solution(
        self,
        session_id: str,
        problem_id: str,
        image_path: Path | str,
        use_vlm: bool = True,
    ) -> OCRPreviewRecord:
        """Run OCR on a handwritten solution image and store the preview in the session for user confirmation."""
        session = self.load_session(session_id)
        prob_stmt = next(
            (p.statement_md for p in session.problems if p.problem_id == problem_id),
            "",
        )

        own_mgr = self.model_manager is None
        mgr = self.model_manager or ModelLifecycleManager(config=self.config, backend="llama_server")
        try:
            harness = AgentHarness(model_manager=mgr, cache=self.cache)
            preview = transcribe_handwritten_solution(
                problem_id=problem_id,
                problem_statement=prob_stmt,
                image_path=image_path,
                harness=harness,
                use_vlm=use_vlm,
            )
        finally:
            if own_mgr:
                mgr.release_all()

        session.ocr_previews[problem_id] = preview
        self.save_session(session)
        return preview

    def confirm_handwritten_solution(
        self,
        session_id: str,
        problem_id: str,
        edited_latex_md: Optional[str] = None,
    ) -> OCRPreviewRecord:
        """Confirm (and optionally edit) a handwritten OCR preview and register it as the problem's solution."""
        session = self.load_session(session_id)
        if problem_id not in session.ocr_previews:
            raise KeyError(f"No OCR preview found for problem '{problem_id}' in session '{session_id}'.")

        preview = confirm_ocr_preview(session.ocr_previews[problem_id], edited_latex_md)
        session.ocr_previews[problem_id] = preview
        session.submitted_solutions_md[problem_id] = preview.edited_latex_md
        self.save_session(session)
        return preview

    def submit_text_solutions(
        self,
        session_id: str,
        solutions_by_problem_id: Dict[str, str],
    ) -> StudyBatchSession:
        """Record typed Markdown/LaTeX/code solutions for problems in the active batch."""
        session = self.load_session(session_id)
        for pid, sol_md in solutions_by_problem_id.items():
            session.submitted_solutions_md[pid] = sol_md.strip()
        self.save_session(session)
        return session

    def evaluate_batch_solutions(self, session_id: str) -> StudyEvaluationReport:
        """Audit all solutions for the 4-problem batch and execute adaptive mastery progression."""
        session = self.load_session(session_id)

        # Enforce confirmation gate if any handwritten OCR preview was uploaded but not yet confirmed
        for pid, prev in session.ocr_previews.items():
            if not prev.confirmed_by_user and pid not in session.submitted_solutions_md:
                raise ValueError(
                    f"Handwritten OCR preview for problem '{pid}' has not been confirmed yet."
                )

        # Load quarantined reference hints/solutions for the auditor
        problems_file = self.data_root / "processed" / f"{self.book_id}_problems.json"
        raw_probs = json.loads(problems_file.read_text(encoding="utf-8"))
        prob_map: Dict[str, CompanionProblem] = {
            p["problem_id"]: CompanionProblem.model_validate(p) for p in raw_probs
        }

        problem_statements: List[str] = []
        learner_solutions: List[str] = []
        ref_hints_and_sols: List[str] = []

        for sp in session.problems:
            problem_statements.append(f"{sp.original_number}: {sp.statement_md}")
            sol_txt = session.submitted_solutions_md.get(sp.problem_id, "[No solution submitted]")
            learner_solutions.append(f"{sp.original_number} Solution:\n{sol_txt}")

            full_prob = prob_map.get(sp.problem_id)
            if full_prob:
                ref_parts = []
                if full_prob.quarantined_hint:
                    ref_parts.append(f"Hint: {full_prob.quarantined_hint}")
                if full_prob.quarantined_solution:
                    ref_parts.append(f"Reference Solution: {full_prob.quarantined_solution}")
                if ref_parts:
                    ref_hints_and_sols.append(f"{sp.original_number}: " + " | ".join(ref_parts))

        target_topics_titles = [
            f"{t.title}: {t.summary_md}" for t in session.hidden_self_discovery_topics
        ]

        own_mgr = self.model_manager is None
        mgr = self.model_manager or ModelLifecycleManager(config=self.config, backend="llama_server")
        try:
            harness = AgentHarness(model_manager=mgr, cache=self.cache)
            auditor = SolutionAuditorAgent()
            audit_out = harness.run(
                auditor,
                SolutionAuditInput(
                    problem_statements=problem_statements,
                    target_self_discovery_topics=target_topics_titles,
                    learner_solutions_md=learner_solutions,
                    reference_hints_or_solutions=ref_hints_and_sols,
                ),
            )
        finally:
            if own_mgr:
                mgr.release_all()

        batch_prob_ids = [p.problem_id for p in session.problems]
        next_briefing: Optional[StudyBatchBriefing] = None
        mastered_cids: List[str] = []

        if audit_out.outcome in ("SOLVED_WITH_DISCOVERY", "SOLVED_MISSED_DISCOVERY"):
            # Mark target chunks as mastered and problems as solved; recompute dynamic mastery graph
            session.status = "completed"
            mastered_cids = list(session.target_chunk_ids)
            record_mastery_and_recompute(
                book_id=self.book_id,
                mastered_chunk_ids=mastered_cids,
                solved_problem_ids=batch_prob_ids,
                config=self.config,
                embedding_mode=self.embedding_mode,
                model_manager=self.model_manager,
                base_dir=self.root,
            )
            self.save_session(session)

            # Automatically prepare the next batch of 4 problems & bare-bones concepts
            next_session = self.start_next_batch()
            if next_session is not None:
                next_briefing = next_session.to_learner_briefing()

        else:
            # Outcome == "FAILED":
            # Enrich remedial_scaffolding_topics with prerequisite chunk context from GraphAr if needed
            session.status = "needs_remediation"
            extra_remedial = self._gather_prerequisite_scaffolding(session.target_chunk_ids)
            existing_remedial_titles = {r.topic.lower() for r in audit_out.remedial_scaffolding_topics}
            for item in extra_remedial:
                if item.topic.lower() not in existing_remedial_titles:
                    audit_out.remedial_scaffolding_topics.append(item)

            record_mastery_and_recompute(
                book_id=self.book_id,
                mastered_chunk_ids=[],
                remedial_problem_ids=batch_prob_ids,
                config=self.config,
                embedding_mode=self.embedding_mode,
                model_manager=self.model_manager,
                base_dir=self.root,
            )
            self.save_session(session)

        return StudyEvaluationReport(
            session_id=session.session_id,
            book_id=self.book_id,
            outcome=audit_out.outcome,
            score_fraction=audit_out.score_fraction,
            discovered_topics=audit_out.discovered_topics,
            undiscovered_topics_to_teach=audit_out.undiscovered_topics_to_teach,
            remedial_scaffolding_topics=audit_out.remedial_scaffolding_topics,
            feedback_summary_md=audit_out.feedback_summary_md,
            mastered_chunk_ids=mastered_cids,
            next_batch_briefing=next_briefing,
        )

    def _gather_prerequisite_scaffolding(self, chunk_ids: List[str]) -> List[TopicCoachingItem]:
        """Look up incoming prerequisite chunks in GraphAr to provide remedial scaffolding."""
        gar_store = GraphArStore(
            book_id=self.book_id,
            graphs_root=self.data_root / "graphs",
        )
        vstore = EmbeddedVectorStore(
            book_id=self.book_id,
            vectors_root=self.data_root / "vectors",
        )
        try:
            graph = gar_store.load()
        except FileNotFoundError:
            return []

        scaffolding: List[TopicCoachingItem] = []
        seen_prereqs: set[str] = set()
        for cid in chunk_ids:
            for in_edge in graph.incoming_edges(cid):
                prereq_id = in_edge.source_id
                if prereq_id in seen_prereqs:
                    continue
                seen_prereqs.add(prereq_id)
                chunk_obj = vstore.get_chunk(prereq_id)
                if chunk_obj is not None:
                    scaffolding.append(
                        TopicCoachingItem(
                            topic=f"Prerequisite Foundation: {chunk_obj.breadcrumb}",
                            explanation_md=chunk_obj.text_md[:500],
                        )
                    )
        return scaffolding
